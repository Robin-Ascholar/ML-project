from __future__ import annotations

import torch
from torch import nn


class LSTMGenerator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        bos_id: int,
        eos_id: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        hidden, _ = self.lstm(embedded)
        return self.output(hidden)

    @torch.no_grad()
    def generate(
        self,
        num_sequences: int,
        max_len: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        device: str | torch.device | None = None,
    ) -> list[list[int]]:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        device = torch.device(device or next(self.parameters()).device)
        self.eval()
        input_ids = torch.full((num_sequences, 1), self.bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(num_sequences, dtype=torch.bool, device=device)
        generated: list[list[int]] = [[] for _ in range(num_sequences)]
        state = None
        current = input_ids
        for _ in range(max_len):
            logits, state = self._step(current, state)
            logits = logits[:, -1, :] / temperature
            logits[:, self.pad_id] = -torch.inf
            logits[:, self.bos_id] = -torch.inf
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
                cutoff = values[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < cutoff, -torch.inf)
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

    def _step(self, input_ids: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None):
        embedded = self.embedding(input_ids)
        hidden, state = self.lstm(embedded, state)
        return self.output(hidden), state
