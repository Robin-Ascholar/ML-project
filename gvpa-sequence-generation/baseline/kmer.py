#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fasta import FastaRecord, read_fasta, write_fasta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GvpA candidates with a k-mer Markov baseline.")
    parser.add_argument("--train", default="data/processed/dataset_v1/train.fasta")
    parser.add_argument("--output", default="runs/baseline/kmer3_seed42.fasta")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--num-seqs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.k < 1:
        raise SystemExit("--k must be >= 1")
    rng = random.Random(args.seed)
    train_records = read_fasta(args.train)
    model = fit_markov([record.sequence for record in train_records], args.k)
    lengths = [len(record.sequence) for record in train_records]
    run_id = args.run_id or f"kmer{args.k}_seed{args.seed}"
    generated = []
    for idx in range(args.num_seqs):
        length = rng.choice(lengths)
        sequence = generate_sequence(model, length, args.k, rng)
        generated.append(
            FastaRecord(
                record_id=f"{run_id}_{idx:04d}",
                description=f"run_id={run_id} model=kmer k={args.k} seed={args.seed} index={idx}",
                sequence=sequence,
            )
        )
    write_fasta(generated, args.output)


def fit_markov(sequences: list[str], k: int) -> dict[str, Counter[str]]:
    model: dict[str, Counter[str]] = defaultdict(Counter)
    order = max(k - 1, 0)
    for sequence in sequences:
        model[""].update(sequence)
        padded = "^" * order + sequence + "$"
        for idx in range(order, len(padded)):
            context = padded[idx - order : idx] if order else ""
            model[context][padded[idx]] += 1
    return dict(model)


def generate_sequence(model: dict[str, Counter[str]], target_len: int, k: int, rng: random.Random) -> str:
    order = max(k - 1, 0)
    context = "^" * order
    chars: list[str] = []
    fallback = model.get("", Counter())
    while len(chars) < target_len:
        counter = model.get(context) or fallback
        if len(chars) < max(1, target_len - 5):
            counter = Counter({symbol: count for symbol, count in counter.items() if symbol != "$"}) or fallback
        symbols = list(counter)
        weights = [counter[symbol] for symbol in symbols]
        symbol = rng.choices(symbols, weights=weights, k=1)[0]
        if symbol == "$":
            if len(chars) >= max(1, target_len - 5):
                break
            continue
        chars.append(symbol)
        if order:
            context = (context + symbol)[-order:]
    return "".join(chars)


if __name__ == "__main__":
    main()
