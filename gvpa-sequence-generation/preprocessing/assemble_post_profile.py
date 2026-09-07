#!/usr/bin/env python3
"""Assemble the post-profile GvpA staging set before MSA and freezing."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-metadata", type=Path, required=True)
    parser.add_argument("--rescued-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    strict = read_csv(args.strict_metadata)
    rescued = read_csv(args.rescued_metadata)
    rows: list[dict[str, str]] = []
    strict_by_sequence = {row["sequence"]: row["sequence_id"] for row in strict}
    rescued_duplicates: list[dict[str, str]] = []
    for row in strict:
        item = dict(row)
        item["admission_route"] = "strict_ncbi_metadata"
        rows.append(item)
    for row in rescued:
        item = dict(row)
        if item["sequence"] in strict_by_sequence:
            item["decision"] = "exclude"
            item["duplicate_sequence_of"] = strict_by_sequence[item["sequence"]]
            item["admission_route"] = "profile_rescue_exact_duplicate"
            rescued_duplicates.append(item)
            continue
        item["decision"] = "include"
        item["admission_route"] = "rescued_by_pf00741_and_core_profile"
        item["dataset_source"] = "ncbi_discovery_profile_rescue"
        rows.append(item)

    if len({row["sequence_id"] for row in rows}) != len(rows):
        raise RuntimeError("Duplicate sequence IDs in post-profile set")
    if len({row["sequence"] for row in rows}) != len(rows):
        raise RuntimeError("Exact duplicate sequences in post-profile set")

    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "staging_metadata.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    if rescued_duplicates:
        duplicate_fields = list(fields)
        for row in rescued_duplicates:
            for field in row:
                if field not in duplicate_fields:
                    duplicate_fields.append(field)
        with (args.output_dir / "rescued_exact_duplicates.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=duplicate_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rescued_duplicates)
    with (args.output_dir / "staging.fasta").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                f">{row['sequence_id']} taxid={row['ncbi_taxid']} "
                f"organism={row['ncbi_organism']} route={row['admission_route']}\n"
            )
            sequence = row["sequence"]
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")
    print(
        f"assembled={len(rows)} strict={len(strict)} "
        f"rescued_unique={len(rescued) - len(rescued_duplicates)} "
        f"rescued_duplicates={len(rescued_duplicates)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
