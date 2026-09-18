from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def run_hmmsearch(
    profile: str | Path | None,
    generated_fasta: str | Path,
    evalue_threshold: float = 1e-5,
) -> tuple[dict[str, dict[str, Any]], str]:
    if profile is None:
        return {}, "not_run"
    executable = shutil.which("hmmsearch")
    if executable is None:
        return {}, "hmmsearch_unavailable"

    with tempfile.TemporaryDirectory(prefix="gvpa_novelty_hmmer_") as temp_dir:
        table_path = Path(temp_dir) / "hits.tbl"
        command = [
            executable,
            "--noali",
            "--tblout",
            str(table_path),
            "-E",
            str(evalue_threshold),
            str(profile),
            str(generated_fasta),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            message = completed.stderr.strip().splitlines()
            detail = message[-1] if message else f"exit_code={completed.returncode}"
            return {}, f"hmmsearch_failed: {detail}"
        return parse_tblout(table_path, evalue_threshold), "ok"


def parse_tblout(
    path: str | Path, evalue_threshold: float
) -> dict[str, dict[str, Any]]:
    hits: dict[str, dict[str, Any]] = {}
    with Path(path).open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split(maxsplit=18)
            if len(fields) < 18:
                continue
            sequence_id = fields[0]
            evalue = float(fields[4])
            bit_score = float(fields[5])
            current = hits.get(sequence_id)
            if current is None or bit_score > current["profile_bit_score"]:
                hits[sequence_id] = {
                    "profile_hit": evalue <= evalue_threshold,
                    "profile_evalue": evalue,
                    "profile_bit_score": bit_score,
                }
    return hits
