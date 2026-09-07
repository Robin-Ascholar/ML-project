#!/usr/bin/env python3
"""Join MMseqs2 cluster assignments to cleaned metadata and report redundancy."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_mapping(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                raise ValueError(f"Expected two TSV columns at {path}:{line_number}")
            representative, member = parts
            if member in mapping:
                raise ValueError(f"Duplicate cluster member in {path}: {member}")
            mapping[member] = representative
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--clusters-95", type=Path, required=True)
    parser.add_argument("--clusters-90", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with args.metadata.open(encoding="utf-8", newline="") as handle:
        metadata = list(csv.DictReader(handle))
    included = [row for row in metadata if row["decision"] == "include"]
    metadata_by_id = {row["sequence_id"]: row for row in included}

    cluster_95 = read_mapping(args.clusters_95)
    cluster_90 = read_mapping(args.clusters_90)
    expected = set(metadata_by_id)
    for label, mapping in (("95", cluster_95), ("90", cluster_90)):
        missing = expected - set(mapping)
        extra = set(mapping) - expected
        if missing or extra:
            raise ValueError(
                f"{label}% cluster IDs differ from included metadata: "
                f"missing={len(missing)}, extra={len(extra)}"
            )

    sizes_95 = Counter(cluster_95.values())
    sizes_90 = Counter(cluster_90.values())
    rows: list[dict[str, object]] = []
    for sequence_id, metadata_row in metadata_by_id.items():
        representative_95 = cluster_95[sequence_id]
        representative_90 = cluster_90[sequence_id]
        rows.append(
            {
                "sequence_id": sequence_id,
                "ncbi_taxid": metadata_row["ncbi_taxid"],
                "ncbi_organism": metadata_row["ncbi_organism"],
                "length": metadata_row["length"],
                "cluster_95_representative": representative_95,
                "cluster_95_size": sizes_95[representative_95],
                "cluster_90_representative": representative_90,
                "cluster_90_size": sizes_90[representative_90],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / "cluster_info.csv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    largest_95 = max(sizes_95.values())
    largest_90 = max(sizes_90.values())
    summary = {
        "input_sequences": total,
        "mmseqs_parameters": {
            "coverage": 0.8,
            "coverage_mode": 0,
            "sequence_identity_thresholds": [0.95, 0.90],
        },
        "clusters_95": len(sizes_95),
        "clusters_90": len(sizes_90),
        "largest_cluster_95": largest_95,
        "largest_cluster_90": largest_90,
        "largest_cluster_fraction_95": round(largest_95 / total, 4),
        "largest_cluster_fraction_90": round(largest_90 / total, 4),
    }
    with (args.output_dir / "clustering_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    report = f"""# GvpA 候选集冗余聚类报告

## 结果

- 输入严格清洗序列：{total}
- 95% 序列一致性：{len(sizes_95)} 个簇；最大簇 {largest_95} 条（{largest_95 / total:.1%}）
- 90% 序列一致性：{len(sizes_90)} 个簇；最大簇 {largest_90} 条（{largest_90 / total:.1%}）
- 参数：MMseqs2 `--min-seq-id 0.95/0.90 -c 0.8 --cov-mode 0`

## 怎么理解

{total} 是有效序列条数，不等于 {total} 份彼此独立的信息。按 95% 阈值折算只有 {len(sizes_95)} 个序列簇，按 90% 阈值只有 {len(sizes_90)} 个序列簇，数据冗余明显。

因此建议补充数据，但补充目标应是“新增蓝藻物种和新增序列簇”，而不是继续收集同一类近重复序列。补充数据合并后，必须重新执行相同的严格清洗和聚类，再按簇划分训练集、验证集和测试集，避免近似序列跨集合造成数据泄漏。

## 当前状态

聚类只用于描述冗余和支持后续按簇划分；本轮没有因为聚类而删除 `strict_included.fasta` 中的序列。该文件仍是当前候选集，不是最终冻结版 `dataset_v1`。
"""
    (args.output_dir / "clustering_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
