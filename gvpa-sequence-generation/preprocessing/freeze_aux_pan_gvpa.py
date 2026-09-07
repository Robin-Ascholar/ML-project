#!/usr/bin/env python3
"""Freeze a leakage-filtered non-cyanobacterial GvpA auxiliary pretraining set."""

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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def read_cluster(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        representative, member = line.split("\t")
        if member in mapping:
            raise ValueError(f"Duplicate cluster member: {member}")
        mapping[member] = representative
    return mapping


def read_profile(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            fields = line.split()
            scores[fields[0]] = float(fields[5])
    return scores


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_fasta(
    path: Path,
    records: list[tuple[str, str, str]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for sequence_id, sequence, source in records:
            handle.write(f">{sequence_id} pretrain_source={source}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safe-metadata", type=Path, required=True)
    parser.add_argument("--excluded-metadata", type=Path, required=True)
    parser.add_argument("--cluster-95", type=Path, required=True)
    parser.add_argument("--cluster-90", type=Path, required=True)
    parser.add_argument("--pf00741-tbl", type=Path, required=True)
    parser.add_argument("--blue-train-fasta", type=Path, required=True)
    parser.add_argument("--blue-manifest", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--leakage-summary", type=Path, required=True)
    parser.add_argument("--profile-file", type=Path, action="append", default=[])
    parser.add_argument("--audit-file", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty frozen directory: {args.output_dir}"
        )
    rows = read_csv(args.safe_metadata)
    excluded = read_csv(args.excluded_metadata)
    cluster_95 = read_cluster(args.cluster_95)
    cluster_90 = read_cluster(args.cluster_90)
    profile_scores = read_profile(args.pf00741_tbl)
    blue_train = read_fasta(args.blue_train_fasta)
    blue_manifest = json.loads(args.blue_manifest.read_text(encoding="utf-8"))
    leakage_summary = json.loads(args.leakage_summary.read_text(encoding="utf-8"))

    ids = {row["sequence_id"] for row in rows}
    if not (ids == set(cluster_95) == set(cluster_90) == set(profile_scores)):
        raise ValueError("Safe metadata, cluster mappings and PF00741 hits differ")
    if leakage_summary["safe_hits_at_or_above_threshold"] != 0:
        raise ValueError("Auxiliary-to-heldout leakage remains")
    if len(ids) != 550 or len(excluded) != 105:
        raise ValueError(f"Unexpected auxiliary counts: safe={len(ids)}, excluded={len(excluded)}")
    if len({row["sequence"] for row in rows}) != len(rows):
        raise ValueError("Exact duplicates remain in auxiliary set")
    if any(set(row["sequence"]) - VALID_AA for row in rows):
        raise ValueError("Invalid amino acids remain in auxiliary set")

    sizes_95 = Counter(cluster_95.values())
    sizes_90 = Counter(cluster_90.values())
    final_rows: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: item["sequence_id"]):
        item: dict[str, object] = dict(row)
        rep95, rep90 = cluster_95[row["sequence_id"]], cluster_90[row["sequence_id"]]
        item["cluster_95_representative"] = rep95
        item["cluster_95_size"] = sizes_95[rep95]
        item["cluster_90_representative"] = rep90
        item["cluster_90_size"] = sizes_90[rep90]
        item["cluster_balanced_weight_95"] = round(1 / sizes_95[rep95], 8)
        item["pf00741_bitscore"] = profile_scores[row["sequence_id"]]
        item["dataset_version"] = "aux_pan_gvpa_v1"
        item["frozen"] = True
        final_rows.append(item)

    by_id = {str(row["sequence_id"]): row for row in final_rows}
    reps95 = sorted(set(cluster_95.values()))
    reps90 = sorted(set(cluster_90.values()))
    aux95_records = [
        (sequence_id, str(by_id[sequence_id]["sequence"]), "noncyanobacteria_95rep")
        for sequence_id in reps95
    ]
    aux90_records = [
        (sequence_id, str(by_id[sequence_id]["sequence"]), "noncyanobacteria_90rep")
        for sequence_id in reps90
    ]
    blue_records = [
        (sequence_id, sequence, "cyanobacteria_dataset_v1_train")
        for sequence_id, sequence in sorted(blue_train.items())
    ]
    combined_records = blue_records + aux95_records
    if len({sequence for _, sequence, _ in combined_records}) != len(combined_records):
        raise ValueError("Exact duplicate between auxiliary representatives and blue train")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = args.output_dir / "audit"
    profile_dir = args.output_dir / "profiles"
    audit_dir.mkdir()
    profile_dir.mkdir()
    fields = list(final_rows[0])
    with (args.output_dir / "metadata.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(final_rows)

    all_records = [
        (str(row["sequence_id"]), str(row["sequence"]), "noncyanobacteria_safe")
        for row in final_rows
    ]
    write_fasta(args.output_dir / "all_safe.fasta", all_records)
    write_fasta(args.output_dir / "pretrain_95rep.fasta", aux95_records)
    write_fasta(args.output_dir / "pretrain_90rep.fasta", aux90_records)
    write_fasta(args.output_dir / "blue_finetune_train.fasta", blue_records)
    write_fasta(args.output_dir / "pretrain_plus_blue_train.fasta", combined_records)

    shutil.copyfile(args.excluded_metadata, audit_dir / "excluded_heldout_similarity.csv")
    shutil.copyfile(args.cluster_95, audit_dir / args.cluster_95.name)
    shutil.copyfile(args.cluster_90, audit_dir / args.cluster_90.name)
    shutil.copyfile(args.pf00741_tbl, audit_dir / args.pf00741_tbl.name)
    for source in args.audit_file:
        shutil.copyfile(source, audit_dir / source.name)
    for source in args.profile_file:
        shutil.copyfile(source, profile_dir / source.name)

    domains = Counter(str(row["ncbi_taxonomy"]).split(";", 1)[0] for row in final_rows)
    report = f"""# aux_pan_gvpa_v1 冻结报告

## 内容

- 原始高可信非蓝藻GvpA候选：655条
- 因与蓝藻valid/test达到80% identity且双方coverage不低于80%而排除：105条
- 安全辅助池：550条
- 95%代表序列：{len(reps95)}条（默认辅助预训练集）
- 90%代表序列：{len(reps90)}条（更强去冗余备选）
- 默认预训练组合：{len(combined_records)}条 = {len(reps95)}条非蓝藻代表 + {len(blue_records)}条蓝藻train
- taxonomic domain：{dict(domains)}
- 全部550条通过官方Pfam PF00741 gathering threshold
- 与冻结valid/test达到80%阈值的保留序列：0条

## 推荐用法

1. AA-frequency和k-mer baseline只使用蓝藻 `dataset_v1/train.fasta`。
2. LSTM/VAE的迁移实验先用 `pretrain_plus_blue_train.fasta` 预训练，再用 `blue_finetune_train.fasta` 微调。
3. valid和test始终使用蓝藻 `dataset_v1` 中的固定文件，只用于调参和最终评价。
4. 必须同时报告“只用蓝藻train从头训练”和“泛GvpA预训练后微调”两组结果。
5. 如需使用全部550条而不是95%代表集，应读取 `cluster_balanced_weight_95` 做按簇加权，避免大簇主导训练。

## 边界

该辅助集包含细菌和古菌GvpA，只用于学习跨物种GvpA序列规律，不改变项目的蓝藻目标分布，也不能替代蓝藻valid/test评价。
"""
    (args.output_dir / "dataset_report.md").write_text(report, encoding="utf-8")
    recipe = """# 模型组使用说明

推荐实验只有两条，保持结果容易解释：

```text
实验A：dataset_v1/train.fasta → 从头训练 → 固定valid/test评价
实验B：pretrain_plus_blue_train.fasta → 预训练
       blue_finetune_train.fasta → 微调 → 同一固定valid/test评价
```

实验B微调时从较小学习率开始（例如预训练学习率的0.1–0.3倍），根据蓝藻valid集早停。任何情况下都不要把蓝藻valid/test加入预训练文件。
"""
    (args.output_dir / "training_recipe.md").write_text(recipe, encoding="utf-8")

    manifest = {
        "dataset_version": "aux_pan_gvpa_v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_blue_dataset_version": blue_manifest["dataset_version"],
        "source_blue_manifest_sha256": sha256(args.blue_manifest),
        "counts": {
            "safe_all": len(final_rows),
            "representatives_95": len(reps95),
            "representatives_90": len(reps90),
            "blue_train": len(blue_records),
            "default_combined_pretrain": len(combined_records),
            "excluded_heldout_similarity": len(excluded),
        },
        "leakage_rule": {
            "identity": 0.80,
            "query_and_target_coverage": 0.80,
            "safe_hits_at_or_above_threshold": 0,
        },
        "validation": {
            "exact_sequence_duplicates": 0,
            "canonical_amino_acids_only": True,
            "pf00741_ga_hits": len(profile_scores),
        },
        "tools": {
            "MMseqs2": "18.8cc5c",
            "HMMER": "3.4",
            "Pfam_profile": "PF00741.24",
        },
        "immutable_note": "Create aux_pan_gvpa_v2 for future changes; do not edit v1 in place.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    files = sorted(
        path
        for path in args.output_dir.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )
    (args.output_dir / "checksums.sha256").write_text(
        "\n".join(
            f"{sha256(path)}  {path.relative_to(args.output_dir)}" for path in files
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
