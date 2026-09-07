#!/usr/bin/env python3
"""Validate all artifacts and freeze the cyanobacterial GvpA dataset_v1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    sequence_id: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            sequence_id = line[1:].split(maxsplit=1)[0]
            if sequence_id in records:
                raise ValueError(f"Duplicate FASTA ID: {sequence_id}")
            records[sequence_id] = ""
        elif sequence_id is None:
            raise ValueError("Sequence before first FASTA header")
        else:
            records[sequence_id] += line.upper()
    return records


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def profile_hit_ids(path: Path) -> set[str]:
    return {
        line.split()[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    }


def write_fasta(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            organism = row["ncbi_organism"].replace(" ", "_")
            handle.write(
                f">{row['sequence_id']} split={row['split']} taxid={row['ncbi_taxid']} "
                f"organism={organism}\n"
            )
            sequence = row["sequence"]
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-fasta", type=Path, required=True)
    parser.add_argument("--source-msa", type=Path, required=True)
    parser.add_argument("--split-metadata", type=Path, required=True)
    parser.add_argument("--pf00741-tbl", type=Path, required=True)
    parser.add_argument("--core-profile-tbl", type=Path, required=True)
    parser.add_argument("--coverage-summary", type=Path, required=True)
    parser.add_argument("--profile-summary", type=Path, required=True)
    parser.add_argument("--msa-summary", type=Path, required=True)
    parser.add_argument("--split-summary", type=Path, required=True)
    parser.add_argument("--original-fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-file", type=Path, action="append", default=[])
    parser.add_argument("--profile-file", type=Path, action="append", default=[])
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty frozen directory: {args.output_dir}"
        )

    fasta = read_fasta(args.source_fasta)
    msa = read_fasta(args.source_msa)
    rows = read_csv(args.split_metadata)
    metadata_by_id = {row["sequence_id"]: row for row in rows}
    ids = set(metadata_by_id)
    if not (ids == set(fasta) == set(msa)):
        raise ValueError("FASTA, MSA and metadata IDs are not identical")
    if len(ids) != 165:
        raise ValueError(f"Expected 165 frozen sequences, found {len(ids)}")
    if len(set(fasta.values())) != len(fasta):
        raise ValueError("Exact sequence duplicates remain")
    for sequence_id in ids:
        if fasta[sequence_id] != metadata_by_id[sequence_id]["sequence"]:
            raise ValueError(f"Metadata sequence mismatch: {sequence_id}")
        if msa[sequence_id].replace("-", "") != fasta[sequence_id]:
            raise ValueError(f"MSA sequence mismatch: {sequence_id}")
        if set(fasta[sequence_id]) - VALID_AA:
            raise ValueError(f"Invalid amino acids: {sequence_id}")
        if not 50 <= len(fasta[sequence_id]) <= 100:
            raise ValueError(f"Length outside frozen range: {sequence_id}")
    if profile_hit_ids(args.pf00741_tbl) != ids:
        raise ValueError("Not every frozen sequence passes PF00741 GA")
    if profile_hit_ids(args.core_profile_tbl) != ids:
        raise ValueError("Not every frozen sequence hits the GvpA core profile")

    split_summary = json.loads(args.split_summary.read_text(encoding="utf-8"))
    msa_summary = json.loads(args.msa_summary.read_text(encoding="utf-8"))
    profile_summary = json.loads(args.profile_summary.read_text(encoding="utf-8"))
    coverage_summary = json.loads(args.coverage_summary.read_text(encoding="utf-8"))
    if split_summary["cross_split_pairs_at_or_above_threshold"] != 0:
        raise ValueError("Split leakage check did not pass")
    if msa_summary["outliers"] != 0:
        raise ValueError("MSA outliers remain")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = args.output_dir / "audit"
    profile_dir = args.output_dir / "profiles"
    audit_dir.mkdir()
    profile_dir.mkdir()

    final_rows: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda item: item["sequence_id"]):
        item = dict(row)
        item["dataset_version"] = "dataset_v1"
        item["frozen"] = "True"
        final_rows.append(item)
    fields = list(final_rows[0])
    with (args.output_dir / "metadata.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(final_rows)
    write_fasta(args.output_dir / "all.fasta", final_rows)
    for split in ("train", "valid", "test"):
        write_fasta(
            args.output_dir / f"{split}.fasta",
            [row for row in final_rows if row["split"] == split],
        )
    shutil.copyfile(args.source_msa, args.output_dir / "all.mafft.fasta")
    shutil.copyfile(args.pf00741_tbl, audit_dir / args.pf00741_tbl.name)
    shutil.copyfile(args.core_profile_tbl, audit_dir / args.core_profile_tbl.name)
    for source in args.audit_file:
        shutil.copyfile(source, audit_dir / source.name)
    for source in args.profile_file:
        shutil.copyfile(source, profile_dir / source.name)

    split_counts = Counter(row["split"] for row in final_rows)
    provisional = [row for row in final_rows if row.get("provisional_source_flag")]
    report = f"""# 蓝藻 GvpA dataset_v1 冻结报告

## 冻结结果

- 最终序列：{len(final_rows)} 条，全部非重复且只含20种标准氨基酸
- 长度：{min(map(len, fasta.values()))}–{max(map(len, fasta.values()))} aa
- train / valid / test：{split_counts['train']} / {split_counts['valid']} / {split_counts['test']}
- 全部通过官方 Pfam PF00741 gathering threshold
- 全部命中由高可信核心序列建立的本地 GvpA profile
- MAFFT异常序列：{msa_summary['outliers']} 条
- 跨集合达到95% identity且覆盖不低于80%的序列对：{split_summary['cross_split_pairs_at_or_above_threshold']} 对
- 跨集合最高identity：{split_summary['max_cross_split_identity_at_required_coverage']:.2%}

## 数据流

1. 课程原始数据：856条；严格清洗后144条。
2. NCBI双路检索：{coverage_summary['ncbi_candidate_accessions']}个 accession、{coverage_summary['ncbi_candidate_unique_sequences']}条非重复候选。
3. 数据库严格补充后：157条。
4. 30条复核候选经官方PF00741和本地核心profile检查，救回{profile_summary['rescued']}个 accession；其中1条与现有序列完全相同，最终净增加8条。
5. 最终冻结：165条。

## 保留标记

按项目决定保留{len(provisional)}条 provisional来源记录（2条MAG和1条uncultured），其标记保存在 `metadata.csv`，评价时可做敏感性分析。

## 使用规则

模型只能使用 `train.fasta` 训练或估计真实分布；`valid.fasta` 用于选择超参数；`test.fasta` 只用于最终评价。不得重新随机拆分，也不得把valid/test序列加入泛GvpA预训练集。
"""
    (args.output_dir / "dataset_report.md").write_text(report, encoding="utf-8")

    manifest = {
        "dataset_version": "dataset_v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "sequence_count": len(final_rows),
        "split_counts": dict(split_counts),
        "original_course_fasta": str(args.original_fasta),
        "original_course_fasta_sha256": sha256(args.original_fasta),
        "validation": {
            "all_canonical_amino_acids": True,
            "exact_sequence_duplicates": 0,
            "pf00741_ga_hits": len(profile_hit_ids(args.pf00741_tbl)),
            "core_profile_hits": len(profile_hit_ids(args.core_profile_tbl)),
            "msa_outliers": msa_summary["outliers"],
            "cross_split_leakage_pairs_95pct": split_summary[
                "cross_split_pairs_at_or_above_threshold"
            ],
        },
        "tools": {
            "MMseqs2": "18.8cc5c",
            "HMMER": "3.4",
            "MAFFT": "7.526",
            "Pfam_profile": "PF00741.24",
        },
        "immutable_note": "Create dataset_v2 for any future data change; do not edit dataset_v1 in place.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    files = sorted(
        path for path in args.output_dir.rglob("*") if path.is_file() and path.name != "checksums.sha256"
    )
    checksum_lines = [
        f"{sha256(path)}  {path.relative_to(args.output_dir)}" for path in files
    ]
    (args.output_dir / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
