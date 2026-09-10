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
    raise SystemExit("PyTorch is required: install torch before running VAE generation.") from exc

from src.data.dataset import ProteinTokenizer
from src.fasta import FastaRecord, write_fasta
from src.models.vae import SequenceVAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FASTA from a trained VAE checkpoint.")
    parser.add_argument("--checkpoint", default="runs/vae_smoke/checkpoint.pt")
    parser.add_argument("--tokenizer", default="configs/tokenizer.json")
    parser.add_argument("--output", default="runs/vae_smoke/generated.fasta")
    parser.add_argument("--num-seqs", type=int, default=100)
    parser.add_argument("--max-len", type=int, default=100)
    parser.add_argument("--latent-scale", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    tokenizer = ProteinTokenizer.from_json(args.tokenizer)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = SequenceVAE(vocab_size=len(tokenizer.token_to_id), pad_id=tokenizer.pad_id, bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    token_batches = model.generate(args.num_seqs, args.max_len, latent_scale=args.latent_scale, temperature=args.temperature, device=args.device)
    run_id = f"vae_scale{args.latent_scale:g}_temp{args.temperature:g}_seed{args.seed}"
    records = [
        FastaRecord(
            record_id=f"{run_id}_{idx:04d}",
            description=f"run_id={run_id} model=vae latent_scale={args.latent_scale} temperature={args.temperature} seed={args.seed} index={idx}",
            sequence=tokenizer.decode(tokens),
        )
        for idx, tokens in enumerate(token_batches)
    ]
    write_fasta(records, args.output)


if __name__ == "__main__":
    main()
