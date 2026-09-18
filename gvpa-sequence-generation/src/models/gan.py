from __future__ import annotations

import torch
from torch import nn


def sequence_mask(lengths: torch.Tensor, max_len: int, dtype: torch.dtype) -> torch.Tensor:
    """Return a [batch, max_len, 1] mask for unpadded amino-acid positions."""
    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return (positions < lengths.unsqueeze(1)).unsqueeze(-1).to(dtype=dtype)


class SequenceGANGenerator(nn.Module):
    """Length-conditioned generator producing soft amino-acid distributions.

    The fixed length is supplied as a condition rather than learned through an
    EOS token.  This keeps adversarial training differentiable and prevents
    padding or special tokens from appearing in generated proteins.
    """

    def __init__(
        self,
        max_len: int,
        num_amino_acids: int = 20,
        latent_dim: int = 64,
        hidden_dim: int = 96,
    ):
        super().__init__()
        self.max_len = max_len
        self.num_amino_acids = num_amino_acids
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.project = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim * max_len),
            nn.LeakyReLU(0.2),
        )
        self.refine = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
        )
        self.output = nn.Conv1d(hidden_dim, num_amino_acids, kernel_size=1)

    def logits(self, noise: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if noise.ndim != 2 or noise.size(1) != self.latent_dim:
            raise ValueError(f"noise must have shape [batch, {self.latent_dim}]")
        if lengths.ndim != 1 or lengths.size(0) != noise.size(0):
            raise ValueError("lengths must have shape [batch]")
        if bool(((lengths < 1) | (lengths > self.max_len)).any()):
            raise ValueError(f"lengths must be between 1 and {self.max_len}")
        length_condition = lengths.to(dtype=noise.dtype).unsqueeze(1) / self.max_len
        hidden = self.project(torch.cat([noise, length_condition], dim=1))
        hidden = hidden.view(noise.size(0), self.hidden_dim, self.max_len)
        hidden = hidden + self.refine(hidden)
        return self.output(hidden).transpose(1, 2)

    def forward(
        self,
        noise: torch.Tensor,
        lengths: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        probabilities = torch.softmax(self.logits(noise, lengths) / temperature, dim=-1)
        return probabilities * sequence_mask(lengths, self.max_len, probabilities.dtype)

    @torch.no_grad()
    def sample(
        self,
        lengths: torch.Tensor,
        temperature: float = 0.8,
        generator: torch.Generator | None = None,
    ) -> list[list[int]]:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.eval()
        noise = torch.randn(
            lengths.size(0),
            self.latent_dim,
            device=lengths.device,
            generator=generator,
        )
        probabilities = torch.softmax(self.logits(noise, lengths) / temperature, dim=-1)
        sampled = torch.multinomial(
            probabilities.reshape(-1, self.num_amino_acids),
            num_samples=1,
            generator=generator,
        ).view(lengths.size(0), self.max_len)
        return [sampled[index, :length].tolist() for index, length in enumerate(lengths.tolist())]


class SequenceGANCritic(nn.Module):
    """Critic operating on real one-hot or generated soft protein sequences."""

    def __init__(
        self,
        max_len: int,
        num_amino_acids: int = 20,
        hidden_dim: int = 96,
    ):
        super().__init__()
        self.max_len = max_len
        self.num_amino_acids = num_amino_acids
        self.features = nn.Sequential(
            nn.Conv1d(num_amino_acids, hidden_dim, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
        )
        self.score = nn.Sequential(
            nn.Linear(hidden_dim * max_len + 1, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        expected_shape = (lengths.size(0), self.max_len, self.num_amino_acids)
        if tuple(sequences.shape) != expected_shape:
            raise ValueError(f"sequences must have shape {expected_shape}")
        mask = sequence_mask(lengths, self.max_len, sequences.dtype)
        features = self.features((sequences * mask).transpose(1, 2)).flatten(1)
        length_condition = lengths.to(dtype=sequences.dtype).unsqueeze(1) / self.max_len
        return self.score(torch.cat([features, length_condition], dim=1)).squeeze(1)
