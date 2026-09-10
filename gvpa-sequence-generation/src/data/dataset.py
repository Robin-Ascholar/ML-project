from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.fasta import FastaRecord, read_fasta


@dataclass(frozen=True)
class ProteinTokenizer:
    token_to_id: dict[str, int]
    alphabet: str

    @classmethod
    def from_json(cls, path: str | Path) -> "ProteinTokenizer":
        payload = json.loads(Path(path).read_text())
        return cls(token_to_id=dict(payload["token_to_id"]), alphabet=str(payload["alphabet"]))

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<EOS>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<UNK>"]

    def encode(self, sequence: str, add_special_tokens: bool = True) -> list[int]:
        ids = [self.token_to_id.get(aa, self.unk_id) for aa in sequence.upper()]
        if add_special_tokens:
            return [self.bos_id, *ids, self.eos_id]
        return ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        specials = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
        chars: list[str] = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in specials:
                continue
            chars.append(id_to_token.get(token_id, "X"))
        return "".join(chars)


class FastaSequenceDataset:
    def __init__(self, fasta_path: str | Path, tokenizer: ProteinTokenizer):
        self.records = read_fasta(fasta_path)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        token_ids = self.tokenizer.encode(record.sequence)
        return {
            "id": record.record_id,
            "description": record.description,
            "sequence": record.sequence,
            "input_ids": token_ids[:-1],
            "labels": token_ids[1:],
            "length": len(record.sequence),
        }


def collate_batch(items: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    max_len = max(len(item["input_ids"]) for item in items)
    input_ids = [_pad(item["input_ids"], max_len, pad_id) for item in items]
    labels = [_pad(item["labels"], max_len, pad_id) for item in items]
    attention_mask = [[1] * len(item["input_ids"]) + [0] * (max_len - len(item["input_ids"])) for item in items]
    return {
        "ids": [item["id"] for item in items],
        "sequences": [item["sequence"] for item in items],
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def _pad(values: list[int], target_len: int, pad_id: int) -> list[int]:
    return values + [pad_id] * (target_len - len(values))


def load_records(path: str | Path) -> list[FastaRecord]:
    return read_fasta(path)
