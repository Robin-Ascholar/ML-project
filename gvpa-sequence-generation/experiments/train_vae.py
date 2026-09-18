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
    from torch.utils.data import DataLoader
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required: install torch before running VAE training.") from exc

from src.data.dataset import FastaSequenceDataset, ProteinTokenizer, collate_batch
from src.models.vae import SequenceVAE, vae_losses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a compact sequence VAE on dataset_v1 train.fasta.")
    parser.add_argument("--train", default="data/processed/dataset_v1/train.fasta")
    parser.add_argument("--valid", default="data/processed/dataset_v1/valid.fasta")
    parser.add_argument("--tokenizer", default="configs/tokenizer.json")
    parser.add_argument("--output-dir", default="runs/vae_smoke")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--beta-start", type=float, default=None, help="Initial KL weight; defaults to --beta.")
    parser.add_argument("--kl-anneal-epochs", type=int, default=0, help="Linear KL warm-up duration; 0 disables warm-up.")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--init-checkpoint", default=None, help="Optional compatible checkpoint for pretraining/fine-tuning.")
    parser.add_argument("--patience", type=int, default=0, help="Stop after this many non-improving validation epochs; 0 disables early stopping.")
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
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.weight_decay < 0 or args.beta < 0:
        raise SystemExit("epochs, batch-size, lr, beta, and weight-decay must be valid values")
    if args.beta_start is not None and args.beta_start < 0 or args.kl_anneal_epochs < 0:
        raise SystemExit("beta-start and kl-anneal-epochs must be non-negative")
    if args.embedding_dim < 1 or args.hidden_dim < 1 or args.latent_dim < 1:
        raise SystemExit("invalid VAE architecture arguments")
    model = SequenceVAE(
        vocab_size=len(tokenizer.token_to_id),
        pad_id=tokenizer.pad_id,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        unk_id=tokenizer.unk_id,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
    ).to(args.device)
    if args.init_checkpoint:
        init_checkpoint = torch.load(args.init_checkpoint, map_location=args.device)
        model.load_state_dict(init_checkpoint["model_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    best_valid = float("inf")
    best_epoch = 0
    stale_epochs = 0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        if args.kl_anneal_epochs:
            fraction = min(1.0, epoch / args.kl_anneal_epochs)
            beta = (args.beta_start if args.beta_start is not None else 0.0) + fraction * (args.beta - (args.beta_start if args.beta_start is not None else 0.0))
        else:
            beta = args.beta
        train_metrics = run_epoch(model, train_loader, tokenizer.pad_id, beta, args.device, optimizer)
        valid_metrics = run_epoch(model, valid_loader, tokenizer.pad_id, beta, args.device, None)
        row = {"epoch": epoch, "beta": beta, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"valid_{k}": v for k, v in valid_metrics.items()}}
        history.append(row)
        print(json.dumps(row))
        if valid_metrics["total"] < best_valid:
            best_valid = valid_metrics["total"]
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if args.patience and stale_epochs >= args.patience:
                break
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state or model.state_dict(),
            "tokenizer": args.tokenizer,
            "args": vars(args),
            "history": history,
            "best_epoch": best_epoch or len(history),
            "best_valid_total": best_valid,
        },
        output_dir / "checkpoint.pt",
    )
    (output_dir / "train_log.json").write_text(json.dumps(history, indent=2) + "\n")


def run_epoch(model, loader, pad_id, beta, device, optimizer):
    model.train(optimizer is not None)
    totals = {"total": [], "reconstruction": [], "kl": []}
    for batch in loader:
        input_ids = torch.tensor(batch["input_ids"], dtype=torch.long, device=device)
        labels = torch.tensor(batch["labels"], dtype=torch.long, device=device)
        outputs = model(input_ids)
        losses = vae_losses(outputs["logits"], labels, outputs["mu"], outputs["logvar"], pad_id, beta)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimizer.step()
        for key, value in losses.items():
            totals[key].append(float(value.detach().cpu()))
    return {key: sum(values) / max(1, len(values)) for key, values in totals.items()}


if __name__ == "__main__":
    main()
