from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap
from typing import Iterable


@dataclass(frozen=True)
class FastaRecord:
    record_id: str
    description: str
    sequence: str


def read_fasta(path: str | Path) -> list[FastaRecord]:
    records: list[FastaRecord] = []
    header: str | None = None
    chunks: list[str] = []
    with Path(path).open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append(_record_from_parts(header, chunks))
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)
    if header is not None:
        records.append(_record_from_parts(header, chunks))
    return records


def write_fasta(records: Iterable[FastaRecord], path: str | Path, line_width: int = 80) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in records:
            description = f" {record.description}" if record.description else ""
            handle.write(f">{record.record_id}{description}\n")
            for line in wrap(record.sequence, width=line_width):
                handle.write(f"{line}\n")


def _record_from_parts(header: str, chunks: list[str]) -> FastaRecord:
    fields = header.split(maxsplit=1)
    record_id = fields[0]
    description = fields[1] if len(fields) == 2 else ""
    return FastaRecord(record_id=record_id, description=description, sequence="".join(chunks).upper())
