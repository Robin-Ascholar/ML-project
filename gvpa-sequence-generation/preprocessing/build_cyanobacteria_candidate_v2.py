#!/usr/bin/env python3
"""Merge the course strict set with newly discovered strict cyanobacterial GvpA."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_fasta(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                f">{row['sequence_id']} taxid={row['ncbi_taxid']} "
                f"organism={row['ncbi_organism']} source={row['dataset_source']}\n"
            )
            sequence = str(row["sequence"])
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course-metadata", type=Path, required=True)
    parser.add_argument("--audit-metadata", type=Path, required=True)
    parser.add_argument("--discovery-summary", type=Path, required=True)
    parser.add_argument("--uniprot-tsv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    course = [row for row in read_csv(args.course_metadata) if row["decision"] == "include"]
    audit_all = read_csv(args.audit_metadata)
    audit_included = [row for row in audit_all if row["decision"] == "include"]
    audit_review = [row for row in audit_all if row["decision"] == "review"]
    discovery = json.loads(args.discovery_summary.read_text(encoding="utf-8"))
    uniprot = read_csv(args.uniprot_tsv, delimiter="\t")

    course_sequences = {row["sequence"] for row in course}
    additions = [row for row in audit_included if row["sequence"] not in course_sequences]
    addition_sequences = {row["sequence"] for row in additions}
    if len(additions) != len(addition_sequences):
        raise RuntimeError("Discovered additions are not sequence-unique")

    combined: list[dict[str, object]] = []
    addition_items: list[dict[str, object]] = []
    for row in course:
        item: dict[str, object] = dict(row)
        item["dataset_source"] = "course_strict_v1"
        item["provisional_source_flag"] = "uncultured" if "uncultured" in row["ncbi_definition"].lower() else ""
        combined.append(item)
    for row in additions:
        item = dict(row)
        item["dataset_source"] = "ncbi_discovery_20260907"
        text = " ".join(
            (row.get("full_header", ""), row.get("ncbi_definition", ""), row.get("ncbi_comment", ""))
        ).lower()
        item["provisional_source_flag"] = "MAG" if "mag:" in text or "metagenom" in text else ""
        combined.append(item)
        addition_items.append(item)

    if len({str(row["sequence"]) for row in combined}) != len(combined):
        raise RuntimeError("Combined candidate set contains exact sequence duplicates")

    fields: list[str] = []
    for row in combined + audit_review:
        for field in row:
            if field not in fields:
                fields.append(field)
    for field in ("dataset_source", "provisional_source_flag"):
        if field not in fields:
            fields.append(field)

    review_rows: list[dict[str, object]] = []
    for row in audit_review:
        item = dict(row)
        item["dataset_source"] = "ncbi_discovery_review"
        item["provisional_source_flag"] = ""
        review_rows.append(item)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "strict_included_metadata.csv", combined, fields)
    write_csv(args.output_dir / "newly_added_metadata.csv", addition_items, fields)
    write_csv(args.output_dir / "review_queue.csv", review_rows, fields)
    write_fasta(args.output_dir / "strict_included.fasta", combined)
    write_fasta(
        args.output_dir / "newly_added.fasta",
        addition_items,
    )

    audit_sequences = {row["sequence"] for row in audit_included}
    overlap = len(course_sequences & audit_sequences)
    uniprot_sequences = {row["Sequence"] for row in uniprot}
    uniprot_reviewed = [row for row in uniprot if row["Reviewed"] == "reviewed"]
    mag_additions = sum(
        "mag:" in row["ncbi_definition"].lower() or "metagenom" in row["ncbi_comment"].lower()
        for row in additions
    )
    coverage = {
        "ncbi_candidate_accessions": discovery["counts"]["union"],
        "ncbi_candidate_unique_sequences": len({row["sequence"] for row in audit_all}),
        "ncbi_strict_unique_sequences": len(audit_sequences),
        "course_strict_unique_sequences": len(course_sequences),
        "course_overlap_with_ncbi_strict": overlap,
        "course_coverage_of_ncbi_strict": round(overlap / len(audit_sequences), 4),
        "new_strict_sequences": len(additions),
        "combined_strict_candidate_sequences": len(combined),
        "new_mag_sequences": mag_additions,
        "review_sequences": len(audit_review),
        "uniprot_records": len(uniprot),
        "uniprot_unique_sequences": len(uniprot_sequences),
        "uniprot_reviewed_records": len(uniprot_reviewed),
        "uniprot_unique_not_covered_by_ncbi_strict": len(uniprot_sequences - audit_sequences),
    }
    (args.output_dir / "coverage_summary.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    report = f"""# 蓝藻 GvpA 公开数据库覆盖审计（2026-09-07）

## 结论

课程严格集不是完全齐全，但对当前公开数据库中可自动严格确认的非重复蓝藻 GvpA 覆盖较好。

- NCBI 两路检索合并：{coverage['ncbi_candidate_accessions']} 个 accession
- accession 去除完全相同序列后：{coverage['ncbi_candidate_unique_sequences']} 条候选序列
- 通过严格 taxonomy、GvpA、PRK09371、完整性、字符和 50–100 aa 检查：{coverage['ncbi_strict_unique_sequences']} 条
- 课程严格集覆盖其中：{coverage['course_overlap_with_ncbi_strict']} 条（{coverage['course_coverage_of_ncbi_strict']:.1%}）
- 本轮确认新增：{coverage['new_strict_sequences']} 条
- 合并后候选集：{coverage['combined_strict_candidate_sequences']} 条
- 仍待 profile/HMM 或人工复核：{coverage['review_sequences']} 条

新增序列中有 {coverage['new_mag_sequences']} 条来自 MAG，已标记为 provisional；最终冻结前应由 P2 再检查。

## UniProt 交叉检查

- 检索记录：{coverage['uniprot_records']} 条
- 完全去重后：{coverage['uniprot_unique_sequences']} 条
- reviewed 记录：{coverage['uniprot_reviewed_records']} 条
- 未被 NCBI 严格集覆盖的 UniProt 独特序列：{coverage['uniprot_unique_not_covered_by_ncbi_strict']} 条

这些未覆盖记录包括含 `X` 的序列或缺少本轮严格 profile 证据的 unreviewed 记录，因此没有自动并入严格集。

## 边界

这里的“覆盖”是截至检索日期，对 NCBI/UniProt 中能被名称、gene 或 PRK09371 找到的记录的覆盖，不代表自然界未知序列绝对齐全。`candidate_v2` 仍不是冻结数据；P2 完成本地 HMM/profile 和 MSA 检查后才能发布 `dataset_v1`。
"""
    (args.output_dir / "coverage_audit_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(coverage, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
