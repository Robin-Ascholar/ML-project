#!/usr/bin/env python3
"""Evaluate the faithfulness of generated GvpA protein sequences.

Faithfulness is the fraction of generated sequences that satisfy all three
fixed constraints: canonical amino-acid alphabet, the frozen train length
range, and a significant hit to the supplied GvpA HMM profile.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_all import faithfulness_value, run_profile_scan
from src.fasta import read_fasta

CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate faithfulness of generated GvpA sequences."
    )
    parser.add_argument("--generated", required=True, help="Generated FASTA to evaluate.")
    parser.add_argument("--train", default="data/processed/dataset_v1/train.fasta")
    parser.add_argument("--profile", required=True, help="GvpA HMMER profile.")
    parser.add_argument("--hmmsearch", default="hmmsearch")
    parser.add_argument("--profile-evalue", type=float, default=1e-3)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated = read_fasta(args.generated)
    train = read_fasta(args.train)
    if not train:
        raise ValueError("The training FASTA is empty; a length range is required.")
    train_min = min(len(record.sequence) for record in train)
    train_max = max(len(record.sequence) for record in train)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_hits, profile_status = run_profile_scan(
        generated_path=args.generated,
        generated=generated,
        output_dir=output_dir,
        profile_path=args.profile,
        hmmsearch=args.hmmsearch,
        evalue=args.profile_evalue,
    )
    if profile_status != "ok":
        raise RuntimeError(
            f"Faithfulness requires a successful HMMER scan; status={profile_status}"
        )

    rows: list[dict[str, Any]] = []
    for record in generated:
        legal = all(aa in CANONICAL_AA for aa in record.sequence)
        in_range = train_min <= len(record.sequence) <= train_max
        profile_hit = profile_hits.get(record.record_id, False)
        rows.append(
            {
                "sequence_id": record.record_id,
                "length": len(record.sequence),
                "legal_chars": legal,
                "length_in_train_range": in_range,
                "family_profile_hit": profile_hit,
                "faithful": faithfulness_value(legal, in_range, profile_hit),
            }
        )

    run_id = args.run_id or infer_run_id(args.generated, generated)
    metrics = {
        "schema_version": "gvpa_faithfulness_v1",
        "run_id": run_id,
        "n_generated": len(rows),
        "train_length_min": train_min,
        "train_length_max": train_max,
        "profile_evalue": args.profile_evalue,
        "faithfulness_definition": (
            "legal_chars AND length_in_train_range AND family_profile_hit"
        ),
        "faithfulness_rate": mean_bool(row["faithful"] for row in rows),
        "legal_char_rate": mean_bool(row["legal_chars"] for row in rows),
        "length_in_train_range_rate": mean_bool(
            row["length_in_train_range"] for row in rows
        ),
        "family_profile_hit_rate": mean_bool(row["family_profile_hit"] for row in rows),
        "family_profile_status": profile_status,
    }
    write_rows(output_dir / "faithfulness_per_sequence.csv", rows)
    (output_dir / "faithfulness_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def mean_bool(values: Any) -> float | None:
    items = list(values)
    if not items:
        return None
    return sum(1 for item in items if item) / len(items)


def infer_run_id(generated_path: str, records: list[Any]) -> str:
    if records and "run_id=" in records[0].description:
        return records[0].description.split("run_id=", 1)[1].split()[0]
    return Path(generated_path).stem


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
