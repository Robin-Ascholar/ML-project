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
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required: install torch before running GAN training.") from exc

from src.data.dataset import ProteinTokenizer
from src.fasta import read_fasta
from src.models.gan import SequenceGANCritic, SequenceGANGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a length-conditioned WGAN-GP on GvpA sequences.")
    parser.add_argument("--train", default="data/processed/dataset_v1/train.fasta")
    parser.add_argument("--tokenizer", default="configs/tokenizer.json")
    parser.add_argument("--output-dir", default="runs/gan_smoke")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--critic-steps", type=int, default=5)
    parser.add_argument("--gradient-penalty", type=float, default=10.0)
    parser.add_argument(
        "--composition-weight",
        type=float,
        default=5.0,
        help="Small-sample stabilizer matching batch position-wise amino-acid frequencies.",
    )
    parser.add_argument(
        "--hydrophobic-run-weight",
        type=float,
        default=0.0,
        help="Penalty for the soft probability of >= --hydrophobic-run-length consecutive hydrophobic residues.",
    )
    parser.add_argument("--hydrophobic-run-length", type=int, default=8)
    parser.add_argument("--temperature-start", type=float, default=1.0)
    parser.add_argument("--temperature-end", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = ProteinTokenizer.from_json(args.tokenizer)
    amino_acids = list(tokenizer.alphabet)
    aa_to_index = {aa: index for index, aa in enumerate(amino_acids)}
    hydrophobic_indices = [aa_to_index[aa] for aa in "AILMFWVY"]
    records = read_fasta(args.train)
    if not records:
        raise SystemExit("training FASTA is empty")
    max_len = max(len(record.sequence) for record in records)
    real_sequences, lengths = encode_real_sequences(records, aa_to_index, max_len)
    loader = DataLoader(
        TensorDataset(real_sequences, lengths),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    generator = SequenceGANGenerator(
        max_len=max_len,
        num_amino_acids=len(amino_acids),
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(args.device)
    critic = SequenceGANCritic(
        max_len=max_len,
        num_amino_acids=len(amino_acids),
        hidden_dim=args.hidden_dim,
    ).to(args.device)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0, 0.9))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr, betas=(0.0, 0.9))

    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        fraction = (epoch - 1) / max(1, args.epochs - 1)
        temperature = args.temperature_start + fraction * (args.temperature_end - args.temperature_start)
        metrics = train_epoch(
            generator,
            critic,
            loader,
            generator_optimizer,
            critic_optimizer,
            latent_dim=args.latent_dim,
            critic_steps=args.critic_steps,
            gradient_penalty_weight=args.gradient_penalty,
            composition_weight=args.composition_weight,
            hydrophobic_run_weight=args.hydrophobic_run_weight,
            hydrophobic_run_length=args.hydrophobic_run_length,
            hydrophobic_indices=hydrophobic_indices,
            temperature=temperature,
            device=args.device,
        )
        row = {"epoch": epoch, "temperature": temperature, **metrics}
        history.append(row)
        print(json.dumps(row))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "generator_state": generator.state_dict(),
        "critic_state": critic.state_dict(),
        "tokenizer": args.tokenizer,
        "alphabet": tokenizer.alphabet,
        "max_len": max_len,
        "length_values": lengths.tolist(),
        "args": vars(args),
        "history": history,
    }
    torch.save(checkpoint, output_dir / "checkpoint.pt")
    (output_dir / "train_log.json").write_text(json.dumps(history, indent=2) + "\n")


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1 or args.critic_steps < 1:
        raise SystemExit("epochs, batch-size, and critic-steps must be positive")
    if args.lr <= 0 or args.latent_dim < 1 or args.hidden_dim < 1:
        raise SystemExit("lr, latent-dim, and hidden-dim must be positive")
    if args.gradient_penalty < 0 or args.composition_weight < 0 or args.hydrophobic_run_weight < 0:
        raise SystemExit("regularization weights must be non-negative")
    if args.hydrophobic_run_length < 2:
        raise SystemExit("hydrophobic-run-length must be at least 2")
    if args.temperature_start <= 0 or args.temperature_end <= 0:
        raise SystemExit("temperatures must be positive")


def encode_real_sequences(records, aa_to_index: dict[str, int], max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = torch.zeros(len(records), max_len, len(aa_to_index), dtype=torch.float32)
    lengths = torch.empty(len(records), dtype=torch.long)
    for row, record in enumerate(records):
        sequence = record.sequence.upper()
        unknown = sorted(set(sequence) - set(aa_to_index))
        if unknown:
            raise SystemExit(f"{record.record_id} contains non-canonical residues: {''.join(unknown)}")
        indices = torch.tensor([aa_to_index[aa] for aa in sequence], dtype=torch.long)
        encoded[row, torch.arange(len(sequence)), indices] = 1.0
        lengths[row] = len(sequence)
    return encoded, lengths


def train_epoch(
    generator,
    critic,
    loader,
    generator_optimizer,
    critic_optimizer,
    latent_dim: int,
    critic_steps: int,
    gradient_penalty_weight: float,
    composition_weight: float,
    hydrophobic_run_weight: float,
    hydrophobic_run_length: int,
    hydrophobic_indices: list[int],
    temperature: float,
    device: str,
) -> dict[str, float]:
    generator.train()
    critic.train()
    totals = {"critic_loss": [], "generator_loss": [], "gradient_penalty": [], "composition_loss": [], "hydrophobic_run_loss": []}
    for real_cpu, lengths_cpu in loader:
        real = real_cpu.to(device)
        lengths = lengths_cpu.to(device)
        batch_size = real.size(0)

        for _ in range(critic_steps):
            noise = torch.randn(batch_size, latent_dim, device=device)
            with torch.no_grad():
                fake = generator(noise, lengths, temperature=temperature)
            real_score = critic(real, lengths)
            fake_score = critic(fake, lengths)
            penalty = gradient_penalty(critic, real, fake, lengths)
            critic_loss = fake_score.mean() - real_score.mean() + gradient_penalty_weight * penalty
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            critic_optimizer.step()

        noise = torch.randn(batch_size, latent_dim, device=device)
        fake = generator(noise, lengths, temperature=temperature)
        adversarial_loss = -critic(fake, lengths).mean()
        composition_loss = F.l1_loss(fake.mean(dim=0), real.mean(dim=0))
        hydro_loss = hydrophobic_run_loss(fake, lengths, hydrophobic_indices, hydrophobic_run_length)
        generator_loss = adversarial_loss + composition_weight * composition_loss + hydrophobic_run_weight * hydro_loss
        generator_optimizer.zero_grad(set_to_none=True)
        generator_loss.backward()
        generator_optimizer.step()

        totals["critic_loss"].append(float(critic_loss.detach().cpu()))
        totals["generator_loss"].append(float(generator_loss.detach().cpu()))
        totals["gradient_penalty"].append(float(penalty.detach().cpu()))
        totals["composition_loss"].append(float(composition_loss.detach().cpu()))
        totals["hydrophobic_run_loss"].append(float(hydro_loss.detach().cpu()))
    return {key: sum(values) / len(values) for key, values in totals.items()}


def gradient_penalty(critic, real: torch.Tensor, fake: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    alpha = torch.rand(real.size(0), 1, 1, device=real.device)
    interpolated = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    scores = critic(interpolated, lengths)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return ((gradients.flatten(1).norm(2, dim=1) - 1.0) ** 2).mean()


def hydrophobic_run_loss(
    fake: torch.Tensor,
    lengths: torch.Tensor,
    hydrophobic_indices: list[int],
    run_length: int,
) -> torch.Tensor:
    """Penalize soft probability of runs absent from the blue-GvpA training set."""
    if run_length > fake.size(1):
        return fake.new_zeros(())
    hydrophobic_probability = fake[:, :, hydrophobic_indices].sum(dim=-1)
    windows = hydrophobic_probability.unfold(dimension=1, size=run_length, step=1)
    run_probability = windows.prod(dim=-1)
    starts = torch.arange(run_probability.size(1), device=fake.device).unsqueeze(0)
    valid_windows = starts + run_length <= lengths.unsqueeze(1)
    return (run_probability * valid_windows).sum() / valid_windows.sum().clamp_min(1)


if __name__ == "__main__":
    main()
