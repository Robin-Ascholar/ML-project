#!/usr/bin/env python3
"""Adjudicate reviewed GvpA candidates against official and core-profile HMMs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_tbl(path: Path) -> dict[str, dict[str, float]]:
    hits: dict[str, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        hits[fields[0]] = {
            "full_sequence_evalue": float(fields[4]),
            "full_sequence_bitscore": float(fields[5]),
        }
    return hits


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_fasta(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                f">{row['sequence_id']} profile_review={row['profile_decision']} "
                f"taxid={row['ncbi_taxid']} organism={row['ncbi_organism']}\n"
            )
            sequence = str(row["sequence"])
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-metadata", type=Path, required=True)
    parser.add_argument("--official-review-tbl", type=Path, required=True)
    parser.add_argument("--core-review-tbl", type=Path, required=True)
    parser.add_argument("--core-strict-tbl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-length", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=100)
    args = parser.parse_args()

    with args.review_metadata.open(encoding="utf-8", newline="") as handle:
        rows: list[dict[str, object]] = list(csv.DictReader(handle))
    official = read_tbl(args.official_review_tbl)
    core_review = read_tbl(args.core_review_tbl)
    core_strict = read_tbl(args.core_strict_tbl)
    strict_score_floor = min(hit["full_sequence_bitscore"] for hit in core_strict.values())

    for row in rows:
        sequence_id = str(row["sequence_id"])
        length = int(str(row["length"]))
        official_hit = official.get(sequence_id)
        core_hit = core_review.get(sequence_id)
        definition = str(row["ncbi_definition"])
        regions = str(row["ncbi_region_names"])
        cdd = str(row["ncbi_cdd_xrefs"])
        annotation_support = "gvpa" in definition.lower() or (
            "PRK09371" in regions and "CDD:181805" in cdd
        )
        length_pass = args.min_length <= length <= args.max_length
        official_pass = official_hit is not None
        core_score = core_hit["full_sequence_bitscore"] if core_hit else float("-inf")
        core_pass = core_score >= strict_score_floor
        valid_aa = str(row["valid_amino_acids"]) == "True"
        rescue = length_pass and official_pass and core_pass and annotation_support and valid_aa

        row["pf00741_ga_pass"] = official_pass
        row["pf00741_bitscore"] = official_hit["full_sequence_bitscore"] if official_hit else ""
        row["core_profile_bitscore"] = core_score if core_hit else ""
        row["core_profile_score_floor"] = strict_score_floor
        row["profile_length_pass"] = length_pass
        row["profile_annotation_support"] = annotation_support
        row["profile_decision"] = "rescue" if rescue else "exclude"
        failed: list[str] = []
        if not length_pass:
            failed.append(f"length_outside_{args.min_length}_{args.max_length}")
        if not official_pass:
            failed.append("pf00741_ga_failed")
        if not core_pass:
            failed.append("below_core_profile_score_floor")
        if not annotation_support:
            failed.append("insufficient_annotation_or_prk09371_support")
        if not valid_aa:
            failed.append("invalid_amino_acids")
        row["profile_decision_reasons"] = (
            "passes_profile_review" if rescue else ";".join(failed)
        )

    rescued = [row for row in rows if row["profile_decision"] == "rescue"]
    excluded = [row for row in rows if row["profile_decision"] == "exclude"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    write_csv(args.output_dir / "profile_review_all.csv", rows, fields)
    write_csv(args.output_dir / "rescued_metadata.csv", rescued, fields)
    write_csv(args.output_dir / "excluded_metadata.csv", excluded, fields)
    write_fasta(args.output_dir / "rescued.fasta", rescued)
    write_fasta(args.output_dir / "excluded.fasta", excluded)

    summary = {
        "reviewed": len(rows),
        "rescued": len(rescued),
        "excluded": len(excluded),
        "strict_core_profile_score_floor": strict_score_floor,
        "length_range": [args.min_length, args.max_length],
        "decision_rule": (
            "PF00741 GA pass AND core-profile score >= strict-set minimum AND "
            "length pass AND explicit GvpA or PRK09371 support AND canonical amino acids"
        ),
    }
    (args.output_dir / "profile_review_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# 30条待复核序列的 profile 判定

- 待复核：{len(rows)}
- 救回：{len(rescued)}
- 排除：{len(excluded)}
- 157条严格核心集的本地 profile 最低 bit score：{strict_score_floor:.1f}

救回必须同时满足：官方 PF00741 gathering threshold、长度 {args.min_length}–{args.max_length} aa、本地核心 profile 得分不低于严格集最低值、标准氨基酸，以及明确 GvpA 名称或 PRK09371/CDD 支持。

PF00741 是广义 gas-vesicle protein family，因此没有将其单独作为 GvpA 身份证据。过长、明显截短或只有宽泛家族命中的记录均留在排除表中。
"""
    (args.output_dir / "profile_review_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
