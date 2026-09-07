#!/usr/bin/env python3
"""Download the official Pfam PF00741 profile and metadata from InterPro."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests


ENTRY_URL = "https://www.ebi.ac.uk/interpro/api/entry/pfam/PF00741/"
HMM_URL = "https://www.ebi.ac.uk/interpro/api/entry/pfam/PF00741?annotation=hmm"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata_response = requests.get(ENTRY_URL, timeout=120)
    metadata_response.raise_for_status()
    metadata = metadata_response.json()
    (args.output_dir / "PF00741.interpro.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    hmm_response = requests.get(HMM_URL, timeout=120)
    hmm_response.raise_for_status()
    hmm_bytes = gzip.decompress(hmm_response.content)
    hmm_path = args.output_dir / "PF00741.hmm"
    hmm_path.write_bytes(hmm_bytes)
    if b"NAME  Gas_vesicle" not in hmm_bytes or b"ACC   PF00741" not in hmm_bytes:
        raise RuntimeError("Downloaded HMM does not identify itself as PF00741/Gas_vesicle")

    provenance = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "entry_url": ENTRY_URL,
        "hmm_url": HMM_URL,
        "hmm_sha256": hashlib.sha256(hmm_bytes).hexdigest(),
        "note": "PF00741 is a broad gas-vesicle protein family profile and is not used alone to label GvpA.",
    }
    (args.output_dir / "PF00741.provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
