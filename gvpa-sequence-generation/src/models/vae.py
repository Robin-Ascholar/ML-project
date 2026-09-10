from __future__ import annotations

import torch
from torch import nn


class SequenceVAE(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        bos_id: int,
        eos_id: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        latent_dim: int = 32,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.GRU(embedding_dim + latent_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        embedded = self.embedding(input_ids)
        _, hidden = self.encoder(embedded)
        hidden_last = hidden[-1]
        mu = self.mu(hidden_last)
        logvar = self.logvar(hidden_last)
        z = self.reparameterize(mu, logvar)
        logits = self.decode_teacher_forced(input_ids, z)
        return {"logits": logits, "mu": mu, "logvar": logvar, "z": z}

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_teacher_forced(self, input_ids: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        z_steps = z.unsqueeze(1).expand(-1, input_ids.size(1), -1)
        initial_hidden = torch.tanh(self.latent_to_hidden(z)).unsqueeze(0)
        decoded, _ = self.decoder(torch.cat([embedded, z_steps], dim=-1), initial_hidden)
        return self.output(decoded)

    @torch.no_grad()
    def generate(
        self,
        num_sequences: int,
        max_len: int,
        latent_scale: float = 1.0,
        temperature: float = 1.0,
        device: str | torch.device | None = None,
    ) -> list[list[int]]:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        device = torch.device(device or next(self.parameters()).device)
        self.eval()
        z = torch.randn(num_sequences, self.latent_dim, device=device) * latent_scale
        hidden = torch.tanh(self.latent_to_hidden(z)).unsqueeze(0)
        current = torch.full((num_sequences, 1), self.bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(num_sequences, dtype=torch.bool, device=device)
        generated: list[list[int]] = [[] for _ in range(num_sequences)]
        for _ in range(max_len):
            embedded = self.embedding(current)
            decoded, hidden = self.decoder(torch.cat([embedded, z.unsqueeze(1)], dim=-1), hidden)
            logits = self.output(decoded[:, -1, :]) / temperature
            logits[:, self.pad_id] = -torch.inf
            logits[:, self.bos_id] = -torch.inf
            next_ids = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).squeeze(1)
            next_ids = torch.where(finished, torch.full_like(next_ids, self.eos_id), next_ids)
            for idx, token_id in enumerate(next_ids.tolist()):
                if not finished[idx] and token_id != self.eos_id:
                    generated[idx].append(token_id)
            finished |= next_ids == self.eos_id
            if bool(finished.all()):
                break
            current = next_ids.unsqueeze(1)
        return generated


def vae_losses(logits: torch.Tensor, labels: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, pad_id: int, beta: float):
    recon = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=pad_id)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return {"total": recon + beta * kl, "reconstruction": recon, "kl": kl}
