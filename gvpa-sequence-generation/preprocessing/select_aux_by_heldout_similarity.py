#!/usr/bin/env python3
"""Remove auxiliary proteins too similar to frozen validation/test sequences."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_fasta(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f">{row['sequence_id']} aux_status={row['aux_status']}\n")
            sequence = str(row["sequence"])
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-metadata", type=Path, required=True)
    parser.add_argument("--frozen-metadata", type=Path, required=True)
    parser.add_argument("--hits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--identity-threshold", type=float, default=0.80)
    parser.add_argument("--coverage-threshold", type=float, default=0.80)
    args = parser.parse_args()

    candidates = read_csv(args.candidate_metadata)
    frozen = {row["sequence_id"]: row for row in read_csv(args.frozen_metadata)}
    best_hit: dict[str, tuple[str, float, float, float]] = {}
    for line in args.hits.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        query, target, identity_text, _, qcov_text, tcov_text, _ = line.split("\t")
        identity = float(identity_text)
        if identity > 1:
            identity /= 100
        qcov, tcov = float(qcov_text), float(tcov_text)
        current = best_hit.get(query)
        candidate = (target, identity, qcov, tcov)
        if current is None or (identity, min(qcov, tcov), target) > (
            current[1],
            min(current[2], current[3]),
            current[0],
        ):
            best_hit[query] = candidate

    rows: list[dict[str, object]] = []
    for row in candidates:
        item: dict[str, object] = dict(row)
        hit = best_hit.get(row["sequence_id"])
        blocked = bool(
            hit
            and hit[1] >= args.identity_threshold
            and min(hit[2], hit[3]) >= args.coverage_threshold
        )
        item["closest_heldout_id"] = hit[0] if hit else ""
        item["closest_heldout_split"] = frozen[hit[0]]["split"] if hit else ""
        item["max_heldout_identity"] = hit[1] if hit else ""
        item["query_coverage_at_max_hit"] = hit[2] if hit else ""
        item["target_coverage_at_max_hit"] = hit[3] if hit else ""
        item["aux_status"] = "exclude_heldout_similarity" if blocked else "include"
        rows.append(item)

    safe = [row for row in rows if row["aux_status"] == "include"]
    excluded = [row for row in rows if row["aux_status"] != "include"]
    if len({str(row["sequence"]) for row in safe}) != len(safe):
        raise RuntimeError("Exact duplicate sequences in safe auxiliary set")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    for filename, selected in (
        ("leakage_audit_all.csv", rows),
        ("safe_metadata.csv", safe),
        ("excluded_heldout_similarity.csv", excluded),
    ):
        with (args.output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)
    write_fasta(args.output_dir / "safe.fasta", safe)
    write_fasta(args.output_dir / "excluded_heldout_similarity.fasta", excluded)

    reported_safe_identities = [
        float(row["max_heldout_identity"])
        for row in safe
        if row["max_heldout_identity"] != ""
    ]
    summary = {
        "input_candidates": len(candidates),
        "included_safe": len(safe),
        "excluded_heldout_similarity": len(excluded),
        "identity_threshold": args.identity_threshold,
        "coverage_threshold": args.coverage_threshold,
        "maximum_reported_identity_among_safe": (
            max(reported_safe_identities) if reported_safe_identities else None
        ),
        "safe_hits_at_or_above_threshold": len(reported_safe_identities),
        "blocked_valid_hits": sum(row["closest_heldout_split"] == "valid" for row in excluded),
        "blocked_test_hits": sum(row["closest_heldout_split"] == "test" for row in excluded),
    }
    (args.output_dir / "leakage_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
