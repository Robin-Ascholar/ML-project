#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_all import (  # noqa: E402
    CANONICAL_AA,
    aa_composition,
    aa_composition_l1,
    max_duplicate_fraction,
    mean_bool,
    mean_pairwise_diversity,
    nearest_train,
    needleman_wunsch,
)
from src.fasta import FastaRecord, read_fasta  # noqa: E402


AA_CLASSES = {
    "hydrophobic": set("AILMFWYV"),
    "charged": set("DEKRH"),
    "positive": set("KRH"),
    "negative": set("DE"),
    "polar": set("STNQCY"),
    "tiny": set("AGST"),
    "aromatic": set("FWY"),
    "beta_favoring": set("VIFYWT"),
}
KYTE_DOOLITTLE = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fidelity of generated cyanobacterial GvpA protein sequences."
    )
    parser.add_argument("--generated", required=True, help="Generated FASTA to evaluate.")
    parser.add_argument("--train", default="data/processed/dataset_v1/train.fasta")
    parser.add_argument("--valid", default="data/processed/dataset_v1/valid.fasta")
    parser.add_argument("--test", default="data/processed/dataset_v1/test.fasta")
    parser.add_argument("--reference-all", default="data/processed/dataset_v1/all.fasta")
    parser.add_argument("--reference-msa", default="data/processed/dataset_v1/all.mafft.fasta")
    parser.add_argument("--pf-profile", default="data/processed/dataset_v1/profiles/PF00741.hmm")
    parser.add_argument("--core-profile", default="data/processed/dataset_v1/profiles/GvpA_core95.hmm")
    parser.add_argument("--hmmsearch", default="hmmsearch")
    parser.add_argument("--profile-evalue", type=float, default=1e-3)
    parser.add_argument("--min-profile-coverage", type=float, default=0.60)
    parser.add_argument("--conserved-occupancy", type=float, default=0.90)
    parser.add_argument("--conserved-dominance", type=float, default=0.80)
    parser.add_argument("--novelty-threshold", type=float, default=0.95)
    parser.add_argument("--max-exact-copy-rate", type=float, default=0.05)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--no-markdown", action="store_true", help="Do not write fidelity_report.md.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = read_fasta(args.generated)
    train = read_fasta(args.train)
    valid = read_fasta(args.valid)
    test = read_fasta(args.test)
    reference_all = read_fasta(args.reference_all)
    reference_msa = read_fasta(args.reference_msa) if Path(args.reference_msa).exists() else []

    run_id = args.run_id or infer_run_id(args.generated, generated)
    reference_sequences = [record.sequence for record in train + valid + test]
    train_sequences = [record.sequence for record in train]
    train_lengths = [len(sequence) for sequence in train_sequences]
    reference_features = feature_distribution([record.sequence for record in reference_all])
    conserved_profile = build_conservation_profile(
        reference_msa,
        occupancy_threshold=args.conserved_occupancy,
        dominance_threshold=args.conserved_dominance,
    )

    pf_hits, pf_status = run_hmm_profile(
        generated_path=args.generated,
        generated=generated,
        output_dir=output_dir,
        profile_path=args.pf_profile,
        label="pf00741",
        hmmsearch=args.hmmsearch,
        evalue=args.profile_evalue,
        min_coverage=args.min_profile_coverage,
    )
    core_hits, core_status = run_hmm_profile(
        generated_path=args.generated,
        generated=generated,
        output_dir=output_dir,
        profile_path=args.core_profile,
        label="gvpa_core95",
        hmmsearch=args.hmmsearch,
        evalue=args.profile_evalue,
        min_coverage=args.min_profile_coverage,
    )

    alignment_by_id = {record.record_id: record.sequence for record in reference_msa}
    raw_by_id = {record.record_id: record.sequence for record in reference_all}
    per_sequence = []
    for record in generated:
        nearest = nearest_train(record.sequence, train)
        nearest_alignment = alignment_by_id.get(nearest["record_id"])
        nearest_raw = raw_by_id.get(nearest["record_id"])
        conservation = score_conservation(record.sequence, nearest_raw, nearest_alignment, conserved_profile)
        features = sequence_features(record.sequence)
        pf_hit = pf_hits.get(record.record_id)
        core_hit = core_hits.get(record.record_id)
        legal = all(aa in CANONICAL_AA for aa in record.sequence)
        length_in_range = min(train_lengths) <= len(record.sequence) <= max(train_lengths)
        exact_copy_train = record.sequence in set(train_sequences)
        exact_copy_any = record.sequence in set(reference_sequences)
        basic_pass = legal and length_in_range and not exact_copy_train
        hard_pass = hard_fidelity_pass(
            legal=legal,
            length_in_range=length_in_range,
            exact_copy_train=exact_copy_train,
            pf_hit=pf_hit,
            core_hit=core_hit,
            pf_status=pf_status,
            core_status=core_status,
        )
        per_sequence.append(
            {
                "sequence_id": record.record_id,
                "length": len(record.sequence),
                "legal_chars": legal,
                "length_in_train_range": length_in_range,
                "exact_copy_train": exact_copy_train,
                "exact_copy_any_real": exact_copy_any,
                "nearest_train_id": nearest["record_id"],
                "nearest_train_identity": nearest["identity"],
                "nearest_train_query_coverage": nearest["query_coverage"],
                "nearest_train_target_coverage": nearest["target_coverage"],
                "novel_at_threshold": nearest["identity"] < args.novelty_threshold,
                "pf00741_hit": none_if_not_run(pf_hit, pf_status),
                "pf00741_evalue": profile_value(pf_hits, record.record_id, "evalue"),
                "pf00741_score": profile_value(pf_hits, record.record_id, "score"),
                "pf00741_coverage": profile_value(pf_hits, record.record_id, "coverage"),
                "gvpa_core95_hit": none_if_not_run(core_hit, core_status),
                "gvpa_core95_evalue": profile_value(core_hits, record.record_id, "evalue"),
                "gvpa_core95_score": profile_value(core_hits, record.record_id, "score"),
                "gvpa_core95_coverage": profile_value(core_hits, record.record_id, "coverage"),
                "conserved_sites_scored": conservation["sites_scored"],
                "conservation_exact_fidelity": conservation["exact_fidelity"],
                "conservation_property_fidelity": conservation["property_fidelity"],
                "hydrophobic_fraction": features["hydrophobic_fraction"],
                "charged_fraction": features["charged_fraction"],
                "polar_fraction": features["polar_fraction"],
                "aromatic_fraction": features["aromatic_fraction"],
                "beta_favoring_fraction": features["beta_favoring_fraction"],
                "net_charge_per_residue": features["net_charge_per_residue"],
                "mean_hydropathy": features["mean_hydropathy"],
                "basic_fidelity_pass": basic_pass,
                "hard_fidelity_pass": hard_pass,
            }
        )

    metrics = summarize_fidelity(
        run_id=run_id,
        generated=generated,
        per_sequence=per_sequence,
        train_lengths=train_lengths,
        reference_sequences=reference_sequences,
        reference_features=reference_features,
        conserved_profile=conserved_profile,
        pf_status=pf_status,
        core_status=core_status,
        novelty_threshold=args.novelty_threshold,
        max_exact_copy_rate=args.max_exact_copy_rate,
        min_profile_coverage=args.min_profile_coverage,
    )
    write_csv(output_dir / "per_sequence_fidelity.csv", per_sequence)
    (output_dir / "fidelity_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    )
    if not args.no_markdown:
        (output_dir / "fidelity_report.md").write_text(format_markdown_report(metrics), encoding="utf-8")


def infer_run_id(generated_path: str, records: list[FastaRecord]) -> str:
    if records:
        for field in records[0].description.split():
            if field.startswith("run_id="):
                return field.split("=", 1)[1]
    return Path(generated_path).stem


def run_hmm_profile(
    generated_path: str,
    generated: list[FastaRecord],
    output_dir: Path,
    profile_path: str | None,
    label: str,
    hmmsearch: str,
    evalue: float,
    min_coverage: float,
) -> tuple[dict[str, Any], str]:
    if not profile_path:
        return {}, "not_run"
    profile = Path(profile_path)
    if not profile.exists():
        return {}, "profile_missing"
    tblout = output_dir / f"{label}.tblout"
    domtblout = output_dir / f"{label}.domtblout"
    command = [
        hmmsearch,
        "--noali",
        "--tblout",
        str(tblout),
        "--domtblout",
        str(domtblout),
        "-E",
        str(evalue),
        profile_path,
        generated_path,
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return {}, "hmmsearch_unavailable"
    if completed.returncode not in (0, 1):
        raise RuntimeError(f"hmmsearch failed for {label}: {completed.stderr.strip()}")

    hits: dict[str, Any] = {
        record.record_id: {"hit": False, "evalue": None, "score": None, "coverage": None}
        for record in generated
    }
    parse_tblout(tblout, hits, evalue)
    parse_domtblout(domtblout, hits)
    for hit in hits.values():
        if hit["hit"] and hit["coverage"] is not None and hit["coverage"] < min_coverage:
            hit["hit"] = False
    return hits, "ok"


def parse_tblout(path: Path, hits: dict[str, Any], evalue_threshold: float) -> None:
    if not path.exists():
        return
    with path.open() as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split()
            if len(fields) < 6 or fields[0] not in hits:
                continue
            evalue = float(fields[4])
            score = float(fields[5])
            if evalue <= evalue_threshold:
                current = hits[fields[0]]
                if current["evalue"] is None or evalue < current["evalue"]:
                    current.update({"hit": True, "evalue": evalue, "score": score})


def parse_domtblout(path: Path, hits: dict[str, Any]) -> None:
    if not path.exists():
        return
    with path.open() as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split()
            if len(fields) < 19 or fields[0] not in hits:
                continue
            qlen = int(fields[5])
            hmm_from = int(fields[15])
            hmm_to = int(fields[16])
            coverage = (hmm_to - hmm_from + 1) / max(1, qlen)
            current = hits[fields[0]]
            current["coverage"] = max(current["coverage"] or 0.0, coverage)


def profile_value(profile_hits: dict[str, Any], record_id: str, key: str) -> Any:
    value = profile_hits.get(record_id)
    if isinstance(value, dict):
        return value.get(key)
    return None


def none_if_not_run(value: Any, status: str) -> bool | None:
    if status != "ok":
        return None
    if isinstance(value, dict):
        return bool(value.get("hit"))
    return bool(value)


def build_conservation_profile(
    msa_records: list[FastaRecord],
    occupancy_threshold: float,
    dominance_threshold: float,
) -> dict[int, dict[str, Any]]:
    if not msa_records:
        return {}
    width = max(len(record.sequence) for record in msa_records)
    profile: dict[int, dict[str, Any]] = {}
    for column_index in range(width):
        residues = [record.sequence[column_index] for record in msa_records if column_index < len(record.sequence)]
        nongap = [aa for aa in residues if aa != "-"]
        occupancy = len(nongap) / max(1, len(residues))
        if occupancy < occupancy_threshold or not nongap:
            continue
        counts = Counter(nongap)
        dominant_aa, dominant_count = counts.most_common(1)[0]
        dominance = dominant_count / len(nongap)
        dominant_classes = aa_classes_for(dominant_aa)
        property_support = {
            name: sum(1 for aa in nongap if aa in members) / len(nongap)
            for name, members in AA_CLASSES.items()
        }
        dominant_property = max(property_support, key=property_support.get)
        if dominance >= dominance_threshold or property_support[dominant_property] >= dominance_threshold:
            profile[column_index] = {
                "dominant_aa": dominant_aa,
                "dominance": dominance,
                "dominant_classes": dominant_classes,
                "dominant_property": dominant_property,
                "property_support": property_support[dominant_property],
                "occupancy": occupancy,
            }
    return profile


def score_conservation(
    generated_sequence: str,
    nearest_raw: str | None,
    nearest_alignment: str | None,
    conserved_profile: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    if not nearest_raw or not nearest_alignment or not conserved_profile:
        return {"sites_scored": 0, "exact_fidelity": None, "property_fidelity": None}
    aligned_reference, aligned_generated = align_reference_to_generated(nearest_raw, generated_sequence)
    raw_to_generated = map_reference_positions_to_generated(aligned_reference, aligned_generated)
    raw_position_by_column = msa_column_to_raw_position(nearest_alignment)
    exact_matches = 0
    property_matches = 0
    sites_scored = 0
    for column_index, spec in conserved_profile.items():
        raw_position = raw_position_by_column.get(column_index)
        if raw_position is None:
            continue
        generated_aa = raw_to_generated.get(raw_position)
        if generated_aa is None or generated_aa == "-":
            continue
        sites_scored += 1
        if generated_aa == spec["dominant_aa"]:
            exact_matches += 1
        if aa_classes_for(generated_aa) & set(spec["dominant_classes"]):
            property_matches += 1
    return {
        "sites_scored": sites_scored,
        "exact_fidelity": exact_matches / sites_scored if sites_scored else None,
        "property_fidelity": property_matches / sites_scored if sites_scored else None,
    }


def align_reference_to_generated(reference: str, generated: str) -> tuple[str, str]:
    return needleman_wunsch(reference, generated)


def map_reference_positions_to_generated(aligned_reference: str, aligned_generated: str) -> dict[int, str]:
    mapping: dict[int, str] = {}
    reference_pos = -1
    for reference_aa, generated_aa in zip(aligned_reference, aligned_generated):
        if reference_aa == "-":
            continue
        reference_pos += 1
        mapping[reference_pos] = generated_aa
    return mapping


def msa_column_to_raw_position(aligned_reference: str) -> dict[int, int | None]:
    mapping: dict[int, int | None] = {}
    raw_position = -1
    for column_index, aa in enumerate(aligned_reference):
        if aa == "-":
            mapping[column_index] = None
        else:
            raw_position += 1
            mapping[column_index] = raw_position
    return mapping


def aa_classes_for(aa: str) -> set[str]:
    return {name for name, members in AA_CLASSES.items() if aa in members}


def sequence_features(sequence: str) -> dict[str, float | None]:
    legal = [aa for aa in sequence if aa in CANONICAL_AA]
    if not legal:
        return {
            "hydrophobic_fraction": None,
            "charged_fraction": None,
            "polar_fraction": None,
            "aromatic_fraction": None,
            "beta_favoring_fraction": None,
            "net_charge_per_residue": None,
            "mean_hydropathy": None,
        }
    length = len(legal)
    return {
        "hydrophobic_fraction": fraction_in_class(legal, "hydrophobic"),
        "charged_fraction": fraction_in_class(legal, "charged"),
        "polar_fraction": fraction_in_class(legal, "polar"),
        "aromatic_fraction": fraction_in_class(legal, "aromatic"),
        "beta_favoring_fraction": fraction_in_class(legal, "beta_favoring"),
        "net_charge_per_residue": (sum(1 for aa in legal if aa in AA_CLASSES["positive"]) - sum(1 for aa in legal if aa in AA_CLASSES["negative"])) / length,
        "mean_hydropathy": statistics.fmean(KYTE_DOOLITTLE[aa] for aa in legal),
    }


def fraction_in_class(sequence: Iterable[str], class_name: str) -> float:
    residues = list(sequence)
    return sum(1 for aa in residues if aa in AA_CLASSES[class_name]) / max(1, len(residues))


def feature_distribution(sequences: list[str]) -> dict[str, dict[str, float | None]]:
    rows = [sequence_features(sequence) for sequence in sequences]
    keys = list(rows[0]) if rows else []
    result: dict[str, dict[str, float | None]] = {}
    for key in keys:
        values = [row[key] for row in rows if row[key] is not None]
        result[key] = {
            "mean": statistics.fmean(values) if values else None,
            "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        }
    return result


def hard_fidelity_pass(
    legal: bool,
    length_in_range: bool,
    exact_copy_train: bool,
    pf_hit: Any,
    core_hit: Any,
    pf_status: str,
    core_status: str,
) -> bool | None:
    if pf_status != "ok" or core_status != "ok":
        return None
    pf_ok = bool(pf_hit.get("hit"))
    core_ok = bool(core_hit.get("hit"))
    return legal and length_in_range and not exact_copy_train and pf_ok and core_ok


def summarize_fidelity(
    run_id: str,
    generated: list[FastaRecord],
    per_sequence: list[dict[str, Any]],
    train_lengths: list[int],
    reference_sequences: list[str],
    reference_features: dict[str, dict[str, float | None]],
    conserved_profile: dict[int, dict[str, Any]],
    pf_status: str,
    core_status: str,
    novelty_threshold: float,
    max_exact_copy_rate: float,
    min_profile_coverage: float,
) -> dict[str, Any]:
    sequences = [record.sequence for record in generated]
    identities = [row["nearest_train_identity"] for row in per_sequence]
    generated_feature_dist = feature_distribution(sequences)
    feature_distances = feature_mean_distances(reference_features, generated_feature_dist)
    exact_copy_train_rate = mean_bool(row["exact_copy_train"] for row in per_sequence)
    profile_evidence_complete = pf_status == "ok" and core_status == "ok"
    hard_metrics = {
        "legal_char_rate": mean_bool(row["legal_chars"] for row in per_sequence),
        "length_in_train_range_rate": mean_bool(row["length_in_train_range"] for row in per_sequence),
        "pf00741_hit_rate": mean_optional_bool(row["pf00741_hit"] for row in per_sequence),
        "gvpa_core95_hit_rate": mean_optional_bool(row["gvpa_core95_hit"] for row in per_sequence),
        "exact_copy_train_rate": exact_copy_train_rate,
        "basic_fidelity_pass_rate": mean_bool(row["basic_fidelity_pass"] for row in per_sequence),
        "profile_required_hard_fidelity_pass_rate": mean_optional_bool(
            row["hard_fidelity_pass"] for row in per_sequence
        ),
        "hard_fidelity_pass_rate": mean_optional_bool(row["hard_fidelity_pass"] for row in per_sequence),
    }
    pattern_metrics = {
        "n_conserved_columns": len(conserved_profile),
        "mean_conservation_exact_fidelity": mean_optional_number(
            row["conservation_exact_fidelity"] for row in per_sequence
        ),
        "mean_conservation_property_fidelity": mean_optional_number(
            row["conservation_property_fidelity"] for row in per_sequence
        ),
    }
    distribution_metrics = {
        "aa_composition_l1_distance": aa_composition_l1(
            aa_composition(sequences), aa_composition(reference_sequences)
        ),
        "feature_mean_absolute_distance": feature_distances["mean_absolute_distance"],
        "feature_distances": feature_distances["by_feature"],
    }
    novelty_diversity = {
        "mean_nearest_train_identity": statistics.fmean(identities) if identities else None,
        "max_nearest_train_identity": max(identities) if identities else None,
        "novelty_threshold": novelty_threshold,
        "novel_sequence_rate": mean_bool(row["novel_at_threshold"] for row in per_sequence),
        "unique_ratio": len(set(sequences)) / len(sequences) if sequences else None,
        "mean_pairwise_diversity": mean_pairwise_diversity(sequences),
        "max_duplicate_fraction": max_duplicate_fraction(sequences),
    }
    return {
        "schema_version": "gvpa_fidelity_v0.2",
        "run_id": run_id,
        "n_generated": len(sequences),
        "train_length_min": min(train_lengths),
        "train_length_max": max(train_lengths),
        "mean_length": statistics.fmean(len(sequence) for sequence in sequences) if sequences else None,
        "profile_status": {"pf00741": pf_status, "gvpa_core95": core_status},
        "evaluation_completeness": {
            "profile_evidence_complete": profile_evidence_complete,
            "conservation_evidence_complete": bool(conserved_profile),
            "embedding_evidence_complete": False,
            "functional_prediction_complete": False,
        },
        "thresholds": {
            "novelty_identity": novelty_threshold,
            "max_exact_copy_rate": max_exact_copy_rate,
            "min_profile_coverage": min_profile_coverage,
        },
        "hard_fidelity": hard_metrics,
        "pattern_fidelity": pattern_metrics,
        "distribution_fidelity": distribution_metrics,
        "novelty_and_diversity_context": novelty_diversity,
        "fidelity_summary": classify_run(
            hard_metrics,
            pattern_metrics,
            distribution_metrics,
            pf_status,
            core_status,
            max_exact_copy_rate,
        ),
    }


def feature_mean_distances(
    reference: dict[str, dict[str, float | None]], generated: dict[str, dict[str, float | None]]
) -> dict[str, Any]:
    distances: dict[str, float | None] = {}
    for key, ref_stats in reference.items():
        ref_mean = ref_stats.get("mean")
        gen_mean = generated.get(key, {}).get("mean")
        distances[key] = abs(gen_mean - ref_mean) if ref_mean is not None and gen_mean is not None else None
    valid = [value for value in distances.values() if value is not None and not math.isnan(value)]
    return {
        "mean_absolute_distance": statistics.fmean(valid) if valid else None,
        "by_feature": distances,
    }


def classify_run(
    hard_metrics: dict[str, Any],
    pattern_metrics: dict[str, Any],
    distribution_metrics: dict[str, Any],
    pf_status: str,
    core_status: str,
    max_exact_copy_rate: float,
) -> dict[str, Any]:
    basic_pass = hard_metrics.get("basic_fidelity_pass_rate")
    strict_pass = hard_metrics.get("profile_required_hard_fidelity_pass_rate")
    exact_copy_rate = hard_metrics.get("exact_copy_train_rate")
    property_fidelity = pattern_metrics.get("mean_conservation_property_fidelity")
    aa_l1 = distribution_metrics.get("aa_composition_l1_distance")
    warnings = []
    if pf_status != "ok" or core_status != "ok":
        warnings.append("profile_evidence_unavailable_or_incomplete")
    if basic_pass is not None and basic_pass < 0.8:
        warnings.append("basic_fidelity_pass_rate_below_0.8")
    if strict_pass is not None and strict_pass < 0.8:
        warnings.append("profile_required_hard_fidelity_pass_rate_below_0.8")
    if exact_copy_rate is not None and exact_copy_rate > max_exact_copy_rate:
        warnings.append("exact_copy_train_rate_above_threshold")
    if property_fidelity is not None and property_fidelity < 0.8:
        warnings.append("conservation_property_fidelity_below_0.8")
    if aa_l1 is not None and aa_l1 > 0.2:
        warnings.append("aa_composition_l1_above_0.2")
    status = "needs_profile_or_conservation_evidence"
    if strict_pass is not None:
        status = "acceptable_initial_fidelity" if strict_pass >= 0.8 and not warnings else "low_or_uncertain_fidelity"
    elif basic_pass is not None:
        status = "uncertain_without_profile_evidence"
    return {"status": status, "warnings": warnings}


def format_markdown_report(metrics: dict[str, Any]) -> str:
    hard = metrics["hard_fidelity"]
    pattern = metrics["pattern_fidelity"]
    distribution = metrics["distribution_fidelity"]
    novelty = metrics["novelty_and_diversity_context"]
    summary = metrics["fidelity_summary"]
    completeness = metrics["evaluation_completeness"]
    lines = [
        f"# GvpA Fidelity Report: {metrics['run_id']}",
        "",
        f"- n_generated: {metrics['n_generated']}",
        f"- status: {summary['status']}",
        f"- warnings: {', '.join(summary['warnings']) if summary['warnings'] else 'none'}",
        "",
        "## Evidence Completeness",
        "",
        f"- profile_evidence_complete: {completeness['profile_evidence_complete']}",
        f"- conservation_evidence_complete: {completeness['conservation_evidence_complete']}",
        f"- embedding_evidence_complete: {completeness['embedding_evidence_complete']}",
        f"- functional_prediction_complete: {completeness['functional_prediction_complete']}",
        "",
        "## Hard Fidelity",
        "",
        metric_line("legal_char_rate", hard["legal_char_rate"]),
        metric_line("length_in_train_range_rate", hard["length_in_train_range_rate"]),
        metric_line("basic_fidelity_pass_rate", hard["basic_fidelity_pass_rate"]),
        metric_line("pf00741_hit_rate", hard["pf00741_hit_rate"]),
        metric_line("gvpa_core95_hit_rate", hard["gvpa_core95_hit_rate"]),
        metric_line("profile_required_hard_fidelity_pass_rate", hard["profile_required_hard_fidelity_pass_rate"]),
        metric_line("exact_copy_train_rate", hard["exact_copy_train_rate"]),
        "",
        "## Conservation And Distribution",
        "",
        metric_line("n_conserved_columns", pattern["n_conserved_columns"]),
        metric_line("mean_conservation_exact_fidelity", pattern["mean_conservation_exact_fidelity"]),
        metric_line("mean_conservation_property_fidelity", pattern["mean_conservation_property_fidelity"]),
        metric_line("aa_composition_l1_distance", distribution["aa_composition_l1_distance"]),
        metric_line("feature_mean_absolute_distance", distribution["feature_mean_absolute_distance"]),
        "",
        "## Novelty And Diversity Context",
        "",
        metric_line("mean_nearest_train_identity", novelty["mean_nearest_train_identity"]),
        metric_line("novel_sequence_rate", novelty["novel_sequence_rate"]),
        metric_line("unique_ratio", novelty["unique_ratio"]),
        metric_line("mean_pairwise_diversity", novelty["mean_pairwise_diversity"]),
        metric_line("max_duplicate_fraction", novelty["max_duplicate_fraction"]),
        "",
    ]
    return "\n".join(lines)


def metric_line(name: str, value: Any) -> str:
    if isinstance(value, float):
        return f"- {name}: {value:.6g}"
    return f"- {name}: {value}"


def mean_optional_bool(values: Iterable[bool | None]) -> float | None:
    items = [value for value in values if value is not None]
    if not items:
        return None
    return sum(1 for value in items if value) / len(items)


def mean_optional_number(values: Iterable[float | None]) -> float | None:
    items = [value for value in values if value is not None]
    if not items:
        return None
    return statistics.fmean(items)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
