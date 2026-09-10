#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fasta import FastaRecord, read_fasta, write_fasta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GvpA candidates from train-set amino-acid frequencies.")
    parser.add_argument("--train", default="data/processed/dataset_v1/train.fasta")
    parser.add_argument("--output", default="runs/baseline/aa_frequency_seed42.fasta")
    parser.add_argument("--num-seqs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    train_records = read_fasta(args.train)
    lengths = [len(record.sequence) for record in train_records]
    counts = Counter("".join(record.sequence for record in train_records))
    alphabet = sorted(counts)
    weights = [counts[aa] for aa in alphabet]
    run_id = args.run_id or f"aa_frequency_seed{args.seed}"
    generated = []
    for idx in range(args.num_seqs):
        length = rng.choice(lengths)
        sequence = "".join(rng.choices(alphabet, weights=weights, k=length))
        generated.append(
            FastaRecord(
                record_id=f"{run_id}_{idx:04d}",
                description=f"run_id={run_id} model=aa_frequency seed={args.seed} index={idx}",
                sequence=sequence,
            )
        )
    write_fasta(generated, args.output)


if __name__ == "__main__":
    main()
