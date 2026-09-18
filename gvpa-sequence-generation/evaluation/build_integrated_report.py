#!/usr/bin/env python3
"""Build a single fidelity--novelty--diversity comparison table.

This report deliberately does not reduce three objectives to one score.  Fidelity
is a screening condition; novelty and diversity are then compared side by side.
The novelty calculation uses every frozen cyanobacterial GvpA sequence as its
reference set, rather than only the training split.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fasta import read_fasta

CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")
SUMMARY_FIELDS = [
    "model",
    "model_type",
    "n_generated",
    "valid_candidate_rate",
    "family_profile_hit_rate",
    "aa_composition_l1_distance",
    "fidelity_screen",
    "mean_nearest_reference_full_length_identity",
    "exact_copy_any_real_rate",
    "novel_rate_identity_lt_0.90",
    "novel_rate_identity_lt_0.95",
    "novel_rate_identity_lt_0.98",
    "unique_ratio",
    "mean_pairwise_diversity",
    "max_duplicate_fraction",
    "interpretation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine GvpA fidelity, novelty and diversity metrics."
    )
    parser.add_argument("--manifest", required=True, help="JSON run manifest.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--coverage-threshold", type=float, default=0.80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    base_dir = manifest_path.parent
    reference_path = resolve_path(base_dir, manifest["reference"])
    references = read_fasta(reference_path)
    rows = [
        evaluate_run(base_dir, item, references, args.coverage_threshold)
        for item in manifest["runs"]
    ]
    write_csv(Path(args.output_csv), rows)
    Path(args.output_report).write_text(
        build_report(
            rows,
            manifest.get("reference_label", str(reference_path)),
            args.coverage_threshold,
        )
    )


def evaluate_run(
    base_dir: Path,
    item: dict[str, Any],
    references: list[Any],
    coverage_threshold: float,
) -> dict[str, Any]:
    generated = read_fasta(resolve_path(base_dir, item["generated"]))
    metrics = json.loads(resolve_path(base_dir, item["metrics"]).read_text())
    reference_sequences = {record.sequence for record in references}
    reference_lengths = [len(record.sequence) for record in references]
    minimum_length, maximum_length = min(reference_lengths), max(reference_lengths)
    valid = [
        record
        for record in generated
        if record.sequence
        and all(aa in CANONICAL_AA for aa in record.sequence)
        and minimum_length <= len(record.sequence) <= maximum_length
    ]
    aligner = ReferenceAligner(references)
    reference_metrics = [aligner.metrics_against_all(record.sequence) for record in valid]
    full_identities = [
        float(np.max(item_metrics["full_length_identity"]))
        for item_metrics in reference_metrics
    ]

    def novelty_rate(threshold: float) -> float | None:
        if not reference_metrics:
            return None
        return sum(not has_near_duplicate(item_metrics, threshold, coverage_threshold)
                   for item_metrics in reference_metrics) / len(reference_metrics)

    profile_rate = metrics.get("family_profile_hit_rate")
    valid_rate = len(valid) / len(generated) if generated else None
    row = {
        "model": item["model"],
        "model_type": item["model_type"],
        "n_generated": len(generated),
        "valid_candidate_rate": valid_rate,
        "family_profile_hit_rate": profile_rate,
        "aa_composition_l1_distance": metrics.get("aa_composition_l1_distance"),
        "fidelity_screen": fidelity_screen(valid_rate, profile_rate),
        "mean_nearest_reference_full_length_identity": mean_or_none(full_identities),
        "exact_copy_any_real_rate": sum(
            record.sequence in reference_sequences for record in generated
        ) / len(generated) if generated else None,
        "novel_rate_identity_lt_0.90": novelty_rate(0.90),
        "novel_rate_identity_lt_0.95": novelty_rate(0.95),
        "novel_rate_identity_lt_0.98": novelty_rate(0.98),
        "unique_ratio": metrics.get("unique_ratio"),
        "mean_pairwise_diversity": metrics.get("mean_pairwise_diversity"),
        "max_duplicate_fraction": metrics.get("max_duplicate_fraction"),
    }
    row["interpretation"] = interpretation(row)
    return row


def resolve_path(base_dir: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else base_dir / candidate


class ReferenceAligner:
    """Evaluate one query against all references with exact NW dynamic programs.

    The dynamic-programming states for all reference sequences are calculated in
    parallel. This keeps the same scoring and tie order as the scalar evaluator,
    while making the all-165-reference novelty calculation practical.
    """

    match_score, mismatch_score, gap_score = 2, -1, -2

    def __init__(self, references: list[Any]) -> None:
        self.lengths = np.array([len(record.sequence) for record in references], dtype=np.int32)
        self.max_length = int(self.lengths.max())
        self.residues = np.zeros((len(references), self.max_length), dtype=np.uint8)
        for index, record in enumerate(references):
            self.residues[index, : len(record.sequence)] = np.frombuffer(
                record.sequence.encode("ascii"), dtype=np.uint8
            )

    def metrics_against_all(self, query: str) -> dict[str, np.ndarray]:
        count = len(self.lengths)
        negative_infinity = -10**9
        previous_score = np.full((count, self.max_length + 1), negative_infinity, dtype=np.int32)
        previous_matches = np.zeros((count, self.max_length + 1), dtype=np.int32)
        previous_pairs = np.zeros((count, self.max_length + 1), dtype=np.int32)
        previous_score[:, 0] = 0
        for column in range(1, self.max_length + 1):
            available = self.lengths >= column
            previous_score[available, column] = column * self.gap_score

        for row, residue in enumerate(query, start=1):
            current_score = np.full((count, self.max_length + 1), negative_infinity, dtype=np.int32)
            current_matches = np.zeros((count, self.max_length + 1), dtype=np.int32)
            current_pairs = np.zeros((count, self.max_length + 1), dtype=np.int32)
            current_score[:, 0] = row * self.gap_score
            residue_code = ord(residue)
            for column in range(1, self.max_length + 1):
                available = self.lengths >= column
                match = self.residues[:, column - 1] == residue_code
                diagonal = previous_score[:, column - 1] + np.where(
                    match, self.match_score, self.mismatch_score
                )
                up = previous_score[:, column] + self.gap_score
                left = current_score[:, column - 1] + self.gap_score
                use_diagonal = (diagonal >= up) & (diagonal >= left)
                use_up = ~use_diagonal & (up >= left)
                current_score[:, column] = np.where(
                    use_diagonal, diagonal, np.where(use_up, up, left)
                )
                current_matches[:, column] = np.where(
                    use_diagonal,
                    previous_matches[:, column - 1] + match,
                    np.where(use_up, previous_matches[:, column], current_matches[:, column - 1]),
                )
                current_pairs[:, column] = np.where(
                    use_diagonal,
                    previous_pairs[:, column - 1] + 1,
                    np.where(use_up, previous_pairs[:, column], current_pairs[:, column - 1]),
                )
                current_score[~available, column] = negative_infinity
                current_matches[~available, column] = 0
                current_pairs[~available, column] = 0
            previous_score, previous_matches, previous_pairs = (
                current_score,
                current_matches,
                current_pairs,
            )

        indexes = np.arange(count)
        matches = previous_matches[indexes, self.lengths]
        pairs = previous_pairs[indexes, self.lengths]
        full_identity = matches / np.maximum(np.maximum(len(query), self.lengths), 1)
        return {
            "identity": np.divide(
                matches,
                pairs,
                out=np.zeros_like(matches, dtype=float),
                where=pairs != 0,
            ),
            "query_coverage": pairs / max(1, len(query)),
            "target_coverage": pairs / self.lengths,
            "full_length_identity": full_identity,
        }


def has_near_duplicate(
    metrics: dict[str, np.ndarray], identity_threshold: float, coverage_threshold: float
) -> bool:
    return bool(np.any(
        (metrics["identity"] >= identity_threshold)
        & (metrics["query_coverage"] >= coverage_threshold)
        & (metrics["target_coverage"] >= coverage_threshold)
    ))


def fidelity_screen(valid_rate: float | None, profile_rate: float | None) -> str:
    if valid_rate != 1.0:
        return "fail: invalid_or_out_of_range_outputs"
    if profile_rate is None:
        return "unassessed: profile_not_run"
    if profile_rate < 0.95:
        return "fail: profile_hit_rate_below_0.95"
    return "pass"


def interpretation(row: dict[str, Any]) -> str:
    if row["fidelity_screen"] != "pass":
        return "Not comparable as a final candidate until the fidelity screen passes."
    if row["aa_composition_l1_distance"] is not None and row["aa_composition_l1_distance"] > 0.12:
        return "Profile passes, but composition drift makes this an exploratory candidate."
    if row["novel_rate_identity_lt_0.95"] is not None and row["novel_rate_identity_lt_0.95"] < 0.10:
        return "High-fidelity but conservative; monitor memorization risk."
    return "Eligible for fidelity--novelty--diversity Pareto comparison."


def mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_report(rows: list[dict[str, Any]], reference_path: str, coverage_threshold: float) -> str:
    lines = [
        "# Integrated GvpA Model Evaluation",
        "",
        "## Decision rule",
        "",
        "No overall score is used. A model must first pass the fidelity screen: 100% legal and in-range outputs, plus PF00741 hit rate >= 0.95. Novelty and diversity are compared only after that screen.",
        "",
        f"Novelty is measured against all frozen cyanobacterial GvpA sequences in `{reference_path}`. A near duplicate requires aligned-residue identity >= the displayed threshold and bilateral coverage >= {coverage_threshold:.2f}. Thus short fragments cannot become novel merely because they align poorly.",
        "",
        "## Results",
        "",
        "| Model | Fidelity screen | Profile | Strict novelty (<0.95) | Pairwise diversity | AA L1 | Interpretation |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {screen} | {profile} | {novelty} | {diversity} | {composition} | {interpretation} |".format(
                model=row["model"],
                screen=row["fidelity_screen"],
                profile=fmt(row["family_profile_hit_rate"]),
                novelty=fmt(row["novel_rate_identity_lt_0.95"]),
                diversity=fmt(row["mean_pairwise_diversity"]),
                composition=fmt(row["aa_composition_l1_distance"]),
                interpretation=row["interpretation"],
            )
        )
    lines.extend([
        "",
        "## Reading the table",
        "",
        "- Higher novelty and diversity are not automatically better: they must be interpreted together with the fidelity screen and composition distance.",
        "- A missing profile value means unassessed, not failed. Such a baseline is retained only as a methodological comparison.",
        "- The CSV retains the three novelty thresholds (0.90, 0.95 and 0.98), exact-copy rate and duplicate statistics for the final report or plots.",
        "",
    ])
    return "\n".join(lines)


def fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.3f}"


if __name__ == "__main__":
    main()
