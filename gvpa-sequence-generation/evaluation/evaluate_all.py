#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fasta import read_fasta

CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated GvpA FASTA files with shared v0.1 metrics.")
    parser.add_argument("--generated", required=True, help="Generated FASTA to evaluate.")
    parser.add_argument("--train", default="data/processed/dataset_v1/train.fasta")
    parser.add_argument("--valid", default="data/processed/dataset_v1/valid.fasta")
    parser.add_argument("--test", default="data/processed/dataset_v1/test.fasta")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--novelty-threshold", type=float, default=0.95)
    parser.add_argument("--profile", default=None, help="HMMER profile HMM for family validation.")
    parser.add_argument("--hmmsearch", default="hmmsearch", help="Path to the HMMER hmmsearch executable.")
    parser.add_argument("--profile-evalue", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated = read_fasta(args.generated)
    train = read_fasta(args.train)
    valid = read_fasta(args.valid)
    test = read_fasta(args.test)
    run_id = args.run_id or infer_run_id(args.generated, generated)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_hits, profile_status = run_profile_scan(
        generated_path=args.generated,
        generated=generated,
        output_dir=output_dir,
        profile_path=args.profile,
        hmmsearch=args.hmmsearch,
        evalue=args.profile_evalue,
    )

    train_sequences = [record.sequence for record in train]
    reference_sequences = train_sequences + [record.sequence for record in valid] + [record.sequence for record in test]
    train_lengths = [len(sequence) for sequence in train_sequences]
    train_comp = aa_composition(train_sequences)
    per_sequence = []
    for record in generated:
        nearest = nearest_train(record.sequence, train)
        legal = all(aa in CANONICAL_AA for aa in record.sequence)
        per_sequence.append(
            {
                "sequence_id": record.record_id,
                "length": len(record.sequence),
                "legal_chars": legal,
                "length_in_train_range": min(train_lengths) <= len(record.sequence) <= max(train_lengths),
                "exact_copy_train": record.sequence in set(train_sequences),
                "exact_copy_any_real": record.sequence in set(reference_sequences),
                "nearest_train_id": nearest["record_id"],
                "nearest_train_identity": nearest["identity"],
                "nearest_train_query_coverage": nearest["query_coverage"],
                "nearest_train_target_coverage": nearest["target_coverage"],
                "novel_at_threshold": nearest["identity"] < args.novelty_threshold,
                "family_profile_hit": profile_hits.get(record.record_id),
                "family_profile_error": profile_status if profile_status != "ok" else None,
            }
        )

    metrics = summarize(
        run_id,
        generated,
        per_sequence,
        train_lengths,
        train_comp,
        args.novelty_threshold,
        profile_status,
    )
    write_per_sequence(output_dir / "per_sequence_metrics.csv", per_sequence)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")


def infer_run_id(generated_path: str, records: list[Any]) -> str:
    if records:
        for field in records[0].description.split():
            if field.startswith("run_id="):
                return field.split("=", 1)[1]
    return Path(generated_path).stem


def nearest_train(sequence: str, train_records: list[Any]) -> dict[str, Any]:
    best = {"record_id": None, "identity": -1.0, "query_coverage": 0.0, "target_coverage": 0.0}
    for record in train_records:
        identity, query_cov, target_cov = global_identity(sequence, record.sequence)
        if identity > best["identity"]:
            best = {
                "record_id": record.record_id,
                "identity": identity,
                "query_coverage": query_cov,
                "target_coverage": target_cov,
            }
    return best


def global_identity(query: str, target: str) -> tuple[float, float, float]:
    aligned_q, aligned_t = needleman_wunsch(query, target)
    paired = [(q, t) for q, t in zip(aligned_q, aligned_t) if q != "-" and t != "-"]
    if not paired:
        return 0.0, 0.0, 0.0
    matches = sum(1 for q, t in paired if q == t)
    identity = matches / len(paired)
    query_coverage = len(paired) / max(1, len(query))
    target_coverage = len(paired) / max(1, len(target))
    return identity, query_coverage, target_coverage


def needleman_wunsch(query: str, target: str) -> tuple[str, str]:
    match_score = 1
    mismatch_score = -1
    gap_score = -1
    rows = len(query) + 1
    cols = len(target) + 1
    scores = [[0] * cols for _ in range(rows)]
    moves = [[""] * cols for _ in range(rows)]
    for i in range(1, rows):
        scores[i][0] = i * gap_score
        moves[i][0] = "up"
    for j in range(1, cols):
        scores[0][j] = j * gap_score
        moves[0][j] = "left"
    for i in range(1, rows):
        for j in range(1, cols):
            diag = scores[i - 1][j - 1] + (match_score if query[i - 1] == target[j - 1] else mismatch_score)
            up = scores[i - 1][j] + gap_score
            left = scores[i][j - 1] + gap_score
            best = max(diag, up, left)
            scores[i][j] = best
            moves[i][j] = "diag" if best == diag else "up" if best == up else "left"
    aligned_q: list[str] = []
    aligned_t: list[str] = []
    i = len(query)
    j = len(target)
    while i > 0 or j > 0:
        move = moves[i][j]
        if move == "diag":
            aligned_q.append(query[i - 1])
            aligned_t.append(target[j - 1])
            i -= 1
            j -= 1
        elif move == "up":
            aligned_q.append(query[i - 1])
            aligned_t.append("-")
            i -= 1
        else:
            aligned_q.append("-")
            aligned_t.append(target[j - 1])
            j -= 1
    return "".join(reversed(aligned_q)), "".join(reversed(aligned_t))


def summarize(
    run_id: str,
    generated: list[Any],
    per_sequence: list[dict[str, Any]],
    train_lengths: list[int],
    train_comp: dict[str, float],
    novelty_threshold: float,
    profile_status: str,
) -> dict[str, Any]:
    sequences = [record.sequence for record in generated]
    identities = [row["nearest_train_identity"] for row in per_sequence]
    pairwise_diversity = mean_pairwise_diversity(sequences)
    return {
        "schema_version": "gvpa_eval_v0.1",
        "run_id": run_id,
        "n_generated": len(sequences),
        "legal_char_rate": mean_bool(row["legal_chars"] for row in per_sequence),
        "length_in_train_range_rate": mean_bool(row["length_in_train_range"] for row in per_sequence),
        "mean_length": statistics.fmean(len(sequence) for sequence in sequences) if sequences else None,
        "train_length_min": min(train_lengths),
        "train_length_max": max(train_lengths),
        "exact_copy_train_rate": mean_bool(row["exact_copy_train"] for row in per_sequence),
        "exact_copy_any_real_rate": mean_bool(row["exact_copy_any_real"] for row in per_sequence),
        "mean_nearest_train_identity": statistics.fmean(identities) if identities else None,
        "max_nearest_train_identity": max(identities) if identities else None,
        "novelty_threshold": novelty_threshold,
        "novel_sequence_rate": mean_bool(row["novel_at_threshold"] for row in per_sequence),
        "unique_ratio": len(set(sequences)) / len(sequences) if sequences else None,
        "mean_pairwise_diversity": pairwise_diversity,
        "max_duplicate_fraction": max_duplicate_fraction(sequences),
        "aa_composition_l1_distance": aa_composition_l1(aa_composition(sequences), train_comp),
        "family_profile_hit_rate": mean_bool(
            row["family_profile_hit"] for row in per_sequence
            if row["family_profile_hit"] is not None
        ),
        "family_profile_status": profile_status,
    }


def run_profile_scan(
    generated_path: str,
    generated: list[Any],
    output_dir: Path,
    profile_path: str | None,
    hmmsearch: str,
    evalue: float,
) -> tuple[dict[str, bool], str]:
    if profile_path is None:
        return {}, "not_run"
    tblout = output_dir / "hmmsearch.tblout"
    command = [
        hmmsearch,
        "--noali",
        "--tblout",
        str(tblout),
        "-E",
        str(evalue),
        profile_path,
        generated_path,
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return {}, "hmmsearch_unavailable"
    if completed.returncode not in (0, 1):
        raise RuntimeError(
            f"hmmsearch failed with exit code {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    hits: dict[str, bool] = {record.record_id: False for record in generated}
    if tblout.exists():
        with tblout.open() as handle:
            for line in handle:
                if line.startswith("#") or not line.strip():
                    continue
                fields = line.split()
                if len(fields) >= 6 and float(fields[4]) <= evalue:
                    hits[fields[0]] = True
    return hits, "ok"


def mean_bool(values: Any) -> float | None:
    items = list(values)
    if not items:
        return None
    return sum(1 for item in items if item) / len(items)


def mean_pairwise_diversity(sequences: list[str]) -> float | None:
    if len(sequences) < 2:
        return None
    distances = []
    for i, left in enumerate(sequences):
        for right in sequences[i + 1 :]:
            identity, _, _ = global_identity(left, right)
            distances.append(1.0 - identity)
    return statistics.fmean(distances)


def max_duplicate_fraction(sequences: list[str]) -> float | None:
    if not sequences:
        return None
    return max(Counter(sequences).values()) / len(sequences)


def aa_composition(sequences: list[str]) -> dict[str, float]:
    counts = Counter("".join(sequences))
    total = sum(counts.values())
    if total == 0:
        return {aa: 0.0 for aa in sorted(CANONICAL_AA)}
    return {aa: counts[aa] / total for aa in sorted(CANONICAL_AA)}


def aa_composition_l1(left: dict[str, float], right: dict[str, float]) -> float:
    return sum(abs(left.get(aa, 0.0) - right.get(aa, 0.0)) for aa in sorted(CANONICAL_AA))


def write_per_sequence(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
