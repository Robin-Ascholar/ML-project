#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required: install torch before running LSTM training.") from exc

from src.data.dataset import FastaSequenceDataset, ProteinTokenizer, collate_batch
from src.models.lstm import LSTMGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small LSTM generator on dataset_v1 train.fasta.")
    parser.add_argument("--train", default="data/processed/dataset_v1/train.fasta")
    parser.add_argument("--valid", default="data/processed/dataset_v1/valid.fasta")
    parser.add_argument("--tokenizer", default="configs/tokenizer.json")
    parser.add_argument("--output-dir", default="runs/lstm_smoke")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    tokenizer = ProteinTokenizer.from_json(args.tokenizer)
    train_ds = FastaSequenceDataset(args.train, tokenizer)
    valid_ds = FastaSequenceDataset(args.valid, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: collate_batch(x, tokenizer.pad_id))
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: collate_batch(x, tokenizer.pad_id))
    model = LSTMGenerator(vocab_size=len(tokenizer.token_to_id), pad_id=tokenizer.pad_id, bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, args.device, optimizer)
        valid_loss = run_epoch(model, valid_loader, criterion, args.device, None)
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
        print(json.dumps(history[-1]))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "tokenizer": args.tokenizer, "args": vars(args), "history": history}, output_dir / "checkpoint.pt")
    (output_dir / "train_log.json").write_text(json.dumps(history, indent=2) + "\n")


def run_epoch(model, loader, criterion, device, optimizer):
    model.train(optimizer is not None)
    losses = []
    for batch in loader:
        input_ids = torch.tensor(batch["input_ids"], dtype=torch.long, device=device)
        labels = torch.tensor(batch["labels"], dtype=torch.long, device=device)
        logits = model(input_ids)
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return sum(losses) / max(1, len(losses))


if __name__ == "__main__":
    main()
