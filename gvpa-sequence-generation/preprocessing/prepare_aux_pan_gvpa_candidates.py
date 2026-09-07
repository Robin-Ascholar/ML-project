#!/usr/bin/env python3
"""Prepare strict non-cyanobacterial GvpA candidates and frozen held-out targets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_fasta(path: Path, rows: list[dict[str, str]], label: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f">{row['sequence_id']} source={label}\n")
            sequence = row["sequence"]
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course-audit-metadata", type=Path, required=True)
    parser.add_argument("--frozen-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    audit = read_csv(args.course_audit_metadata)
    frozen = read_csv(args.frozen_metadata)
    candidates = [
        row
        for row in audit
        if set(filter(None, row["reasons"].split(";")))
        == {"not_cyanobacteria_by_ncbi_lineage"}
    ]
    heldout = [row for row in frozen if row["split"] in {"valid", "test"}]
    blue_train = [row for row in frozen if row["split"] == "train"]

    if len({row["sequence_id"] for row in candidates}) != len(candidates):
        raise RuntimeError("Duplicate candidate IDs")
    if len({row["sequence"] for row in candidates}) != len(candidates):
        raise RuntimeError("Exact duplicate candidate sequences")
    required_checks = {
        "explicit_gvpa_annotation": lambda row: row["explicit_gvpa_annotation"] == "True",
        "valid_amino_acids": lambda row: row["valid_amino_acids"] == "True",
        "ncbi_sequence_matches_local": lambda row: row["ncbi_sequence_matches_local"] == "True",
        "prk09371": lambda row: "PRK09371" in row["ncbi_region_names"],
        "cdd181805": lambda row: "CDD:181805" in row["ncbi_cdd_xrefs"],
        "noncyanobacteria": lambda row: row["is_cyanobacteria"] == "False",
    }
    for check_name, check in required_checks.items():
        failed = [row["sequence_id"] for row in candidates if not check(row)]
        if failed:
            raise RuntimeError(f"Candidate check failed ({check_name}): {len(failed)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(candidates[0])
    with (args.output_dir / "candidate_metadata.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidates)
    write_fasta(args.output_dir / "candidates.fasta", candidates, "strict_noncyanobacterial_gvpa")
    write_fasta(args.output_dir / "heldout_valid_test.fasta", heldout, "dataset_v1_heldout")
    write_fasta(args.output_dir / "blue_train.fasta", blue_train, "dataset_v1_train")

    kingdoms = Counter(
        row["ncbi_taxonomy"].split(";", 1)[0].strip() or "Unclassified"
        for row in candidates
    )
    summary = {
        "candidate_sequences": len(candidates),
        "heldout_sequences": len(heldout),
        "blue_train_sequences": len(blue_train),
        "candidate_taxonomic_domains": dict(kingdoms),
        "candidate_length_min": min(int(row["length"]) for row in candidates),
        "candidate_length_max": max(int(row["length"]) for row in candidates),
        "selection_rule": "only exclusion reason is not_cyanobacteria_by_ncbi_lineage",
    }
    (args.output_dir / "candidate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
