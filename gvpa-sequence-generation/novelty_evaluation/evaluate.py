#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from novelty_evaluation.hmmer import run_hmmsearch
from novelty_evaluation.sequence_metrics import evaluate_sequence_set
from novelty_evaluation.structure_metrics import (
    compare_structure_sets,
    load_structure_directory,
)
from src.fasta import read_fasta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated GvpA novelty from sequence and optional PDB structures."
    )
    parser.add_argument("--generated", required=True, help="Generated FASTA.")
    parser.add_argument(
        "--reference",
        default="data/processed/dataset_v1/all.fasta",
        help="Natural reference FASTA.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--identity-threshold", type=float, default=0.95)
    parser.add_argument("--coverage-threshold", type=float, default=0.80)
    parser.add_argument("--kmer-size", type=int, default=3)
    parser.add_argument("--profile", default=None, help="Optional HMM profile for hmmsearch.")
    parser.add_argument("--profile-evalue", type=float, default=1e-5)
    parser.add_argument(
        "--generated-structures",
        default=None,
        help="Optional directory of <sequence_id>.pdb files.",
    )
    parser.add_argument(
        "--reference-structures",
        default=None,
        help="Optional directory of reference PDB files.",
    )
    parser.add_argument("--structure-similarity-threshold", type=float, default=0.50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = read_fasta(args.generated)
    references = read_fasta(args.reference)
    rows, sequence_summary = evaluate_sequence_set(
        generated,
        references,
        identity_threshold=args.identity_threshold,
        coverage_threshold=args.coverage_threshold,
        kmer_size=args.kmer_size,
    )

    profile_hits, profile_status = run_hmmsearch(
        args.profile, args.generated, args.profile_evalue
    )
    attach_profile_metrics(rows, profile_hits, profile_status)

    generated_structures, generated_structure_status = load_structure_directory(
        args.generated_structures
    )
    reference_structures, reference_structure_status = load_structure_directory(
        args.reference_structures
    )
    structure_results, structure_summary = compare_structure_sets(
        generated_structures,
        reference_structures,
        [row["sequence_id"] for row in rows],
        structure_similarity_threshold=args.structure_similarity_threshold,
    )
    structure_not_requested = (
        args.generated_structures is None or args.reference_structures is None
    )
    for row in rows:
        row.update(structure_results[row["sequence_id"]])
        if structure_not_requested:
            row["structure_status"] = "not_run"
        row["candidate_class"] = classify_candidate(row)

    summary = {
        "schema_version": "gvpa_novelty_v1",
        "generated_fasta": str(Path(args.generated)),
        "reference_fasta": str(Path(args.reference)),
        "sequence": sequence_summary,
        "profile": profile_summary(rows, profile_status, args.profile_evalue),
        "structure": {
            "generated_structure_status": generated_structure_status,
            "reference_structure_status": reference_structure_status,
            **structure_summary,
        },
        "candidate_classes": class_counts(rows),
    }

    write_csv(output_dir / "per_sequence_novelty.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    (output_dir / "report.md").write_text(build_report(summary, rows))


def attach_profile_metrics(
    rows: list[dict[str, Any]],
    profile_hits: dict[str, dict[str, Any]],
    profile_status: str,
) -> None:
    for row in rows:
        hit = profile_hits.get(row["sequence_id"])
        row["profile_status"] = profile_status
        row["profile_hit"] = hit["profile_hit"] if hit else False if profile_status == "ok" else None
        row["profile_evalue"] = hit["profile_evalue"] if hit else None
        row["profile_bit_score"] = hit["profile_bit_score"] if hit else None


def profile_summary(
    rows: list[dict[str, Any]], status: str, evalue_threshold: float
) -> dict[str, Any]:
    hits = [row["profile_hit"] for row in rows if row["profile_hit"] is not None]
    scores = [
        row["profile_bit_score"]
        for row in rows
        if row["profile_bit_score"] is not None
    ]
    return {
        "status": status,
        "evalue_threshold": evalue_threshold,
        "profile_hit_rate": sum(hits) / len(hits) if hits else None,
        "mean_profile_bit_score": statistics.fmean(scores) if scores else None,
    }


def classify_candidate(row: dict[str, Any]) -> str:
    if not row["valid_sequence"]:
        return "invalid_sequence"
    if row["profile_hit"] is False:
        return "family_profile_fail"
    if row["tm_score_symmetric"] is None:
        return "sequence_novel" if row["sequence_novel"] else "sequence_conservative"
    sequence_label = "sequence_novel" if row["sequence_novel"] else "sequence_conservative"
    structure_label = (
        "structure_conserved" if row["structure_conserved"] else "structure_divergent"
    )
    return f"{sequence_label}_{structure_label}"


def class_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = row["candidate_class"]
        counts[label] = counts.get(label, 0) + 1
    return counts


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    sequence = summary["sequence"]
    profile = summary["profile"]
    structure = summary["structure"]
    top_candidates = sorted(
        [
            row
            for row in rows
            if row["valid_sequence"] and row["sequence_novelty"] is not None
        ],
        key=lambda row: (
            row["candidate_class"] != "sequence_novel_structure_conserved",
            -row["sequence_novelty"],
        ),
    )[:10]

    lines = [
        "# GvpA Novelty Evaluation",
        "",
        "## Summary",
        "",
        f"- Generated sequences: {sequence['n_generated']}",
        f"- Valid sequence rate: {_format_rate(sequence['valid_sequence_rate'])}",
        f"- Exact reference copy rate: {_format_rate(sequence['exact_copy_reference_rate'])}",
        f"- Near-duplicate rate: {_format_rate(sequence['near_duplicate_rate'])}",
        f"- Novel rate among valid sequences: {_format_rate(sequence['sequence_novel_rate_valid'])}",
        f"- Mean sequence novelty among valid sequences: {_format_number(sequence['mean_sequence_novelty_valid'])}",
        f"- Unique ratio: {_format_rate(sequence['unique_ratio'])}",
        f"- Mean generated-set pairwise diversity: {_format_number(sequence['mean_pairwise_diversity'])}",
        f"- HMM profile status: {profile['status']}",
        f"- HMM profile hit rate: {_format_rate(profile['profile_hit_rate'])}",
        f"- Structure comparison rate: {_format_rate(structure['structure_comparison_rate'])}",
        f"- Mean nearest-reference TM-score: {_format_number(structure['mean_best_tm_score_symmetric'])}",
        "",
        "## Candidate Classes",
        "",
    ]
    for label, count in sorted(summary["candidate_classes"].items()):
        lines.append(f"- `{label}`: {count}")

    lines.extend(["", "## Top Sequence-Novel Candidates", ""])
    if not top_candidates:
        lines.append("No valid candidates.")
    else:
        lines.extend(
            [
                "| Sequence | Class | Novelty | Identity | Query cov. | Target cov. | TM-score |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in top_candidates:
            lines.append(
                "| {sequence_id} | {candidate_class} | {novelty} | {identity} | "
                "{query_coverage} | {target_coverage} | {tm_score} |".format(
                    sequence_id=row["sequence_id"],
                    candidate_class=row["candidate_class"],
                    novelty=_format_number(row["sequence_novelty"]),
                    identity=_format_number(row["nearest_reference_identity"]),
                    query_coverage=_format_number(
                        row["nearest_reference_query_coverage"]
                    ),
                    target_coverage=_format_number(
                        row["nearest_reference_target_coverage"]
                    ),
                    tm_score=_format_number(row["tm_score_symmetric"]),
                )
            )
    lines.extend(
        [
            "",
            "Sequence novelty is `1 - nearest full-length identity`. A near duplicate",
            "requires both coverage values to pass the configured threshold. Structure",
            "novelty is reported separately from GvpA structure conservation.",
            "",
        ]
    )
    return "\n".join(lines)


def _format_rate(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:.1f}%"


def _format_number(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
