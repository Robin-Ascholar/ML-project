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
    "legal_char_rate",
    "length_in_train_range_rate",
    "exact_copy_train_rate",
    "mean_nearest_train_identity",
    "max_nearest_train_identity",
    "novel_sequence_rate",
    "unique_ratio",
    "mean_pairwise_diversity",
    "max_duplicate_fraction",
    "aa_composition_l1_distance",
    "family_profile_hit_rate",
    "family_profile_status",
    "faithfulness_rate",
    "faithfulness_definition",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build model_comparison.csv from evaluator metrics.json files.")
    parser.add_argument("--metrics", nargs="+", required=True, help="One or more metrics.json files.")
    parser.add_argument("--output", default="runs/model_comparison.csv")
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
    return {field: metrics.get(field) for field in SUMMARY_FIELDS}


if __name__ == "__main__":
    main()
