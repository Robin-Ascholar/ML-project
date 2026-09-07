#!/usr/bin/env python3
"""Create a deterministic split with no cross-split >=95% MMseqs2 hit."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--all-vs-all", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--identity-threshold", type=float, default=0.95)
    parser.add_argument("--coverage-threshold", type=float, default=0.8)
    args = parser.parse_args()

    with args.metadata.open(encoding="utf-8", newline="") as handle:
        metadata = list(csv.DictReader(handle))
    metadata_by_id = {row["sequence_id"]: row for row in metadata}
    ids = sorted(metadata_by_id)
    parent = {sequence_id: sequence_id for sequence_id in ids}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            if left_root < right_root:
                parent[right_root] = left_root
            else:
                parent[left_root] = right_root

    alignments: list[tuple[str, str, float, float, float]] = []
    for line_number, line in enumerate(
        args.all_vs_all.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) != 7:
            raise ValueError(f"Expected seven columns at line {line_number}")
        query, target = fields[0], fields[1]
        if query not in parent or target not in parent:
            raise ValueError(f"Unknown sequence ID at line {line_number}")
        identity = float(fields[2])
        if identity > 1:
            identity /= 100
        query_coverage, target_coverage = float(fields[4]), float(fields[5])
        alignments.append((query, target, identity, query_coverage, target_coverage))
        if (
            query != target
            and identity >= args.identity_threshold
            and min(query_coverage, target_coverage) >= args.coverage_threshold
        ):
            union(query, target)

    components: dict[str, list[str]] = defaultdict(list)
    for sequence_id in ids:
        components[find(sequence_id)].append(sequence_id)
    component_items = sorted(
        ((min(members), sorted(members)) for members in components.values()),
        key=lambda item: (-len(item[1]), item[0]),
    )

    target_train = round(len(ids) * 0.70)
    train_components: list[tuple[str, list[str]]] = []
    heldout_components: list[tuple[str, list[str]]] = []
    train_size = 0
    for item in component_items:
        if train_size < target_train:
            train_components.append(item)
            train_size += len(item[1])
        else:
            heldout_components.append(item)
    if len(heldout_components) < 2:
        raise RuntimeError("Too few independent components for validation and test")

    # Exact subset-sum dynamic programming balances the held-out components.
    total_heldout = sum(len(members) for _, members in heldout_components)
    possible: dict[int, tuple[int, ...]] = {0: ()}
    for index, (_, members) in enumerate(heldout_components):
        additions = {
            subtotal + len(members): chosen + (index,)
            for subtotal, chosen in list(possible.items())
        }
        for subtotal, chosen in additions.items():
            possible.setdefault(subtotal, chosen)
    valid_sum, valid_indices = min(
        (
            (subtotal, chosen)
            for subtotal, chosen in possible.items()
            if 0 < subtotal < total_heldout
        ),
        key=lambda item: (abs(item[0] - total_heldout / 2), item[1]),
    )
    valid_index_set = set(valid_indices)
    valid_components = [
        item for index, item in enumerate(heldout_components) if index in valid_index_set
    ]
    test_components = [
        item for index, item in enumerate(heldout_components) if index not in valid_index_set
    ]

    split_by_id: dict[str, str] = {}
    component_by_id: dict[str, str] = {}
    for split, items in (
        ("train", train_components),
        ("valid", valid_components),
        ("test", test_components),
    ):
        for representative, members in items:
            for sequence_id in members:
                split_by_id[sequence_id] = split
                component_by_id[sequence_id] = representative

    cross_pairs: set[tuple[str, str]] = set()
    leakage_pairs: set[tuple[str, str]] = set()
    max_cross_identity = 0.0
    for query, target, identity, query_coverage, target_coverage in alignments:
        if query == target or split_by_id[query] == split_by_id[target]:
            continue
        pair = tuple(sorted((query, target)))
        if pair in cross_pairs:
            continue
        cross_pairs.add(pair)
        if min(query_coverage, target_coverage) >= args.coverage_threshold:
            max_cross_identity = max(max_cross_identity, identity)
            if identity >= args.identity_threshold:
                leakage_pairs.add(pair)
    if leakage_pairs:
        raise RuntimeError(f"Cross-split leakage detected: {len(leakage_pairs)} pairs")

    rows: list[dict[str, object]] = []
    for sequence_id in ids:
        row: dict[str, object] = dict(metadata_by_id[sequence_id])
        row["cluster95_component"] = component_by_id[sequence_id]
        row["split"] = split_by_id[sequence_id]
        row["dataset_version"] = "dataset_v1"
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (args.output_dir / "split_assignments.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    for split in ("train", "valid", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        with (args.output_dir / f"{split}.fasta").open("w", encoding="utf-8") as handle:
            for row in split_rows:
                handle.write(f">{row['sequence_id']} split={split}\n")
                sequence = str(row["sequence"])
                for start in range(0, len(sequence), 80):
                    handle.write(sequence[start : start + 80] + "\n")

    split_counts = Counter(split_by_id.values())
    split_component_counts = Counter()
    for split, items in (
        ("train", train_components),
        ("valid", valid_components),
        ("test", test_components),
    ):
        split_component_counts[split] = len(items)
    summary = {
        "method": "connected components of all MMseqs2 hits at threshold",
        "deterministic": True,
        "identity_threshold": args.identity_threshold,
        "coverage_threshold": args.coverage_threshold,
        "total_sequences": len(ids),
        "total_components": len(component_items),
        "split_counts": dict(split_counts),
        "split_component_counts": dict(split_component_counts),
        "max_cross_split_identity_at_required_coverage": round(max_cross_identity, 6),
        "cross_split_pairs_at_or_above_threshold": len(leakage_pairs),
    }
    (args.output_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# dataset_v1 数据划分报告

- 总序列：{len(ids)}
- 95% identity / 80% coverage连通组：{len(component_items)}
- train：{split_counts['train']}条，{split_component_counts['train']}个连通组
- valid：{split_counts['valid']}条，{split_component_counts['valid']}个连通组
- test：{split_counts['test']}条，{split_component_counts['test']}个连通组
- 跨集合最高identity：{max_cross_identity:.2%}
- 跨集合达到或超过95%的序列对：{len(leakage_pairs)}

由于GvpA高度保守，最大连通组本身已有{len(train_components[0][1])}条，因此不能在不拆散近重复组的前提下实现70/15/15。本次优先防止泄漏，采用确定性的{split_counts['train']}/{split_counts['valid']}/{split_counts['test']}划分。
"""
    (args.output_dir / "split_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
