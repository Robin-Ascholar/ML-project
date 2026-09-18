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
    raise SystemExit("PyTorch is required: install torch before running GAN generation.") from exc

from src.data.dataset import ProteinTokenizer
from src.fasta import FastaRecord, write_fasta
from src.models.gan import SequenceGANGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GvpA FASTA from a trained WGAN-GP checkpoint.")
    parser.add_argument("--checkpoint", default="runs/gan_smoke/checkpoint.pt")
    parser.add_argument("--tokenizer", default="configs/tokenizer.json")
    parser.add_argument("--output", default="runs/gan_smoke/generated.fasta")
    parser.add_argument("--num-seqs", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--max-hydrophobic-run",
        type=int,
        default=None,
        help="Optional candidate filter: reject sequences with a longer run from AILMFWVY.",
    )
    parser.add_argument("--max-attempts", type=int, default=10000, help="Maximum sampled sequences when filtering candidates.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_seqs < 1 or args.temperature <= 0 or args.max_attempts < args.num_seqs:
        raise SystemExit("--num-seqs, --temperature, and --max-attempts must be valid")
    if args.max_hydrophobic_run is not None and args.max_hydrophobic_run < 1:
        raise SystemExit("--max-hydrophobic-run must be positive")
    tokenizer = ProteinTokenizer.from_json(args.tokenizer)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    alphabet = str(checkpoint["alphabet"])
    if alphabet != tokenizer.alphabet:
        raise SystemExit("checkpoint alphabet does not match tokenizer alphabet")
    train_args = checkpoint["args"]
    model = SequenceGANGenerator(
        max_len=int(checkpoint["max_len"]),
        num_amino_acids=len(alphabet),
        latent_dim=int(train_args["latent_dim"]),
        hidden_dim=int(train_args["hidden_dim"]),
    ).to(args.device)
    model.load_state_dict(checkpoint["generator_state"])

    random = torch.Generator(device=args.device).manual_seed(args.seed)
    length_values = torch.tensor(checkpoint["length_values"], dtype=torch.long, device=args.device)
    sampled_indices: list[list[int]] = []
    attempts = 0
    while len(sampled_indices) < args.num_seqs and attempts < args.max_attempts:
        batch_size = min(args.num_seqs - len(sampled_indices), args.max_attempts - attempts)
        selected = torch.randint(
            0,
            length_values.numel(),
            (batch_size,),
            generator=random,
            device=args.device,
        )
        lengths = length_values[selected]
        candidates = model.sample(lengths, temperature=args.temperature, generator=random)
        attempts += batch_size
        for candidate in candidates:
            sequence = "".join(alphabet[index] for index in candidate)
            if args.max_hydrophobic_run is None or longest_hydrophobic_run(sequence) <= args.max_hydrophobic_run:
                sampled_indices.append(candidate)
                if len(sampled_indices) == args.num_seqs:
                    break
    if len(sampled_indices) < args.num_seqs:
        raise SystemExit("candidate filter exceeded --max-attempts before enough sequences were accepted")
    aa_token_ids = [tokenizer.token_to_id[aa] for aa in alphabet]
    run_id = f"gan_temp{args.temperature:g}_seed{args.seed}"
    records = []
    for index, indices in enumerate(sampled_indices):
        token_ids = [aa_token_ids[aa_index] for aa_index in indices]
        records.append(
            FastaRecord(
                record_id=f"{run_id}_{index:04d}",
                description=(
                    f"run_id={run_id} model=conditional_wgan_gp temperature={args.temperature} "
                    f"seed={args.seed} index={index} target_length={len(indices)} "
                    f"max_hydrophobic_run={args.max_hydrophobic_run}"
                ),
                sequence=tokenizer.decode(token_ids),
            )
        )
    write_fasta(records, args.output)


def longest_hydrophobic_run(sequence: str) -> int:
    best = 0
    current = 0
    for residue in sequence:
        current = current + 1 if residue in "AILMFWVY" else 0
        best = max(best, current)
    return best


if __name__ == "__main__":
    main()
