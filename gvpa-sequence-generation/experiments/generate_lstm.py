#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required: install torch before running LSTM generation.") from exc

from src.data.dataset import ProteinTokenizer
from src.fasta import FastaRecord, read_fasta, write_fasta
from src.models.lstm import LSTMGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FASTA from a trained LSTM checkpoint.")
    parser.add_argument("--checkpoint", default="runs/lstm_smoke/checkpoint.pt")
    parser.add_argument("--tokenizer", default="configs/tokenizer.json")
    parser.add_argument("--output", default="runs/lstm_smoke/generated.fasta")
    parser.add_argument("--num-seqs", type=int, default=100)
    parser.add_argument("--train", default="data/processed/dataset_v1/train.fasta")
    parser.add_argument("--min-len", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    tokenizer = ProteinTokenizer.from_json(args.tokenizer)
    train_lengths = [len(record.sequence) for record in read_fasta(args.train)]
    min_len = args.min_len if args.min_len is not None else min(train_lengths)
    max_len = args.max_len if args.max_len is not None else max(train_lengths)
    if min_len < 1 or min_len > max_len:
        raise SystemExit("--min-len must be at least 1 and no greater than --max-len")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    train_args = checkpoint.get("args", {})
    model = LSTMGenerator(
        vocab_size=len(tokenizer.token_to_id),
        pad_id=tokenizer.pad_id,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        unk_id=tokenizer.unk_id,
        embedding_dim=int(train_args.get("embedding_dim", 64)),
        hidden_dim=int(train_args.get("hidden_dim", 128)),
        num_layers=int(train_args.get("num_layers", 2)),
        dropout=float(train_args.get("dropout", 0.1)),
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    token_batches = model.generate(
        args.num_seqs,
        max_len,
        min_len=min_len,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
    )
    run_id = f"lstm_temp{args.temperature:g}_seed{args.seed}"
    records = [
        FastaRecord(
            record_id=f"{run_id}_{idx:04d}",
            description=f"run_id={run_id} model=lstm temperature={args.temperature} seed={args.seed} index={idx}",
            sequence=tokenizer.decode(tokens),
        )
        for idx, tokens in enumerate(token_batches)
    ]
    write_fasta(records, args.output)


if __name__ == "__main__":
    main()
