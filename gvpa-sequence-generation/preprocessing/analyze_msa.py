#!/usr/bin/env python3
"""Audit a protein MSA for record integrity and obvious sequence outliers."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path


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
                raise ValueError(f"Duplicate MSA ID: {sequence_id}")
            records[sequence_id] = ""
        elif sequence_id is None:
            raise ValueError("Sequence before first FASTA header")
        else:
            records[sequence_id] += line.upper()
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--msa", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-consensus-identity", type=float, default=0.75)
    args = parser.parse_args()

    msa = read_fasta(args.msa)
    with args.metadata.open(encoding="utf-8", newline="") as handle:
        metadata = list(csv.DictReader(handle))
    metadata_by_id = {row["sequence_id"]: row for row in metadata}
    if set(msa) != set(metadata_by_id):
        raise ValueError(
            f"MSA and metadata IDs differ: missing={len(set(metadata_by_id)-set(msa))}, "
            f"extra={len(set(msa)-set(metadata_by_id))}"
        )
    alignment_lengths = {len(sequence) for sequence in msa.values()}
    if len(alignment_lengths) != 1:
        raise ValueError(f"MSA rows have different lengths: {sorted(alignment_lengths)}")
    alignment_length = next(iter(alignment_lengths))

    consensus: list[str] = []
    column_coverage: list[float] = []
    for index in range(alignment_length):
        counts = Counter(sequence[index] for sequence in msa.values() if sequence[index] != "-")
        consensus.append(counts.most_common(1)[0][0] if counts else "-")
        column_coverage.append(sum(sequence[index] != "-" for sequence in msa.values()) / len(msa))

    metrics: list[dict[str, object]] = []
    for sequence_id, aligned in msa.items():
        non_gap_positions = [index for index, residue in enumerate(aligned) if residue != "-"]
        raw_sequence = aligned.replace("-", "")
        metadata_sequence = metadata_by_id[sequence_id]["sequence"]
        if raw_sequence != metadata_sequence:
            raise ValueError(f"MSA changed residues or order for {sequence_id}")
        matches = sum(aligned[index] == consensus[index] for index in non_gap_positions)
        identity = matches / len(non_gap_positions)
        first = non_gap_positions[0]
        last = non_gap_positions[-1]
        internal_gaps = aligned[first : last + 1].count("-")
        outlier_reasons: list[str] = []
        if identity < args.min_consensus_identity:
            outlier_reasons.append("low_consensus_identity")
        if not 50 <= len(raw_sequence) <= 100:
            outlier_reasons.append("length_outside_50_100")
        metrics.append(
            {
                "sequence_id": sequence_id,
                "length": len(raw_sequence),
                "alignment_length": alignment_length,
                "gap_fraction": round(aligned.count("-") / alignment_length, 6),
                "identity_to_consensus": round(identity, 6),
                "leading_gaps": first,
                "trailing_gaps": alignment_length - last - 1,
                "internal_gaps": internal_gaps,
                "admission_route": metadata_by_id[sequence_id].get("admission_route", ""),
                "provisional_source_flag": metadata_by_id[sequence_id].get(
                    "provisional_source_flag", ""
                ),
                "msa_outlier": bool(outlier_reasons),
                "msa_outlier_reasons": ";".join(outlier_reasons),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "msa_sequence_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)

    identities = [float(row["identity_to_consensus"]) for row in metrics]
    summary = {
        "sequences": len(msa),
        "alignment_length": alignment_length,
        "columns_with_90pct_coverage": sum(value >= 0.9 for value in column_coverage),
        "columns_with_50pct_coverage": sum(value >= 0.5 for value in column_coverage),
        "identity_to_consensus_min": min(identities),
        "identity_to_consensus_median": statistics.median(identities),
        "identity_to_consensus_max": max(identities),
        "outliers": sum(bool(row["msa_outlier"]) for row in metrics),
        "provisional_sources": sum(bool(row["provisional_source_flag"]) for row in metrics),
    }
    (args.output_dir / "msa_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# GvpA 多序列比对质量报告

- 序列：{summary['sequences']} 条
- MAFFT alignment长度：{alignment_length} 列
- 覆盖至少90%序列的核心列：{summary['columns_with_90pct_coverage']} 列
- 相对consensus一致性：最低 {summary['identity_to_consensus_min']:.1%}，中位数 {summary['identity_to_consensus_median']:.1%}，最高 {summary['identity_to_consensus_max']:.1%}
- 明显MSA异常序列：{summary['outliers']} 条
- provisional来源标记：{summary['provisional_sources']} 条

判定规则为长度50–100 aa且相对consensus一致性不低于{args.min_consensus_identity:.0%}。所有MSA序列去除gap后均与冻结前FASTA逐字符一致。
"""
    (args.output_dir / "msa_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
