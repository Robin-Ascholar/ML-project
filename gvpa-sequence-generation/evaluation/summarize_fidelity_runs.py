#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SUMMARY_FIELDS = [
    "run_id",
    "n_generated",
    "status",
    "warnings",
    "profile_evidence_complete",
    "embedding_evidence_complete",
    "legal_char_rate",
    "length_in_train_range_rate",
    "basic_fidelity_pass_rate",
    "pf00741_hit_rate",
    "gvpa_core95_hit_rate",
    "profile_required_hard_fidelity_pass_rate",
    "exact_copy_train_rate",
    "mean_conservation_exact_fidelity",
    "mean_conservation_property_fidelity",
    "aa_composition_l1_distance",
    "feature_mean_absolute_distance",
    "mean_nearest_train_identity",
    "novel_sequence_rate",
    "unique_ratio",
    "mean_pairwise_diversity",
    "max_duplicate_fraction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a model-level fidelity comparison CSV from fidelity_metrics.json files."
    )
    parser.add_argument("--metrics", nargs="+", required=True, help="One or more fidelity_metrics.json files.")
    parser.add_argument("--output", default="runs/fidelity_model_comparison.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [flatten(json.loads(Path(path).read_text())) for path in args.metrics]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def flatten(metrics: dict[str, Any]) -> dict[str, Any]:
    hard = metrics.get("hard_fidelity", {})
    pattern = metrics.get("pattern_fidelity", {})
    distribution = metrics.get("distribution_fidelity", {})
    novelty = metrics.get("novelty_and_diversity_context", {})
    summary = metrics.get("fidelity_summary", {})
    completeness = metrics.get("evaluation_completeness", {})
    row = {
        "run_id": metrics.get("run_id"),
        "n_generated": metrics.get("n_generated"),
        "status": summary.get("status"),
        "warnings": ";".join(summary.get("warnings", [])),
        "profile_evidence_complete": completeness.get("profile_evidence_complete"),
        "embedding_evidence_complete": completeness.get("embedding_evidence_complete"),
        "legal_char_rate": hard.get("legal_char_rate"),
        "length_in_train_range_rate": hard.get("length_in_train_range_rate"),
        "basic_fidelity_pass_rate": hard.get("basic_fidelity_pass_rate"),
        "pf00741_hit_rate": hard.get("pf00741_hit_rate"),
        "gvpa_core95_hit_rate": hard.get("gvpa_core95_hit_rate"),
        "profile_required_hard_fidelity_pass_rate": hard.get("profile_required_hard_fidelity_pass_rate"),
        "exact_copy_train_rate": hard.get("exact_copy_train_rate"),
        "mean_conservation_exact_fidelity": pattern.get("mean_conservation_exact_fidelity"),
        "mean_conservation_property_fidelity": pattern.get("mean_conservation_property_fidelity"),
        "aa_composition_l1_distance": distribution.get("aa_composition_l1_distance"),
        "feature_mean_absolute_distance": distribution.get("feature_mean_absolute_distance"),
        "mean_nearest_train_identity": novelty.get("mean_nearest_train_identity"),
        "novel_sequence_rate": novelty.get("novel_sequence_rate"),
        "unique_ratio": novelty.get("unique_ratio"),
        "mean_pairwise_diversity": novelty.get("mean_pairwise_diversity"),
        "max_duplicate_fraction": novelty.get("max_duplicate_fraction"),
    }
    return {field: row.get(field) for field in SUMMARY_FIELDS}


if __name__ == "__main__":
    main()
