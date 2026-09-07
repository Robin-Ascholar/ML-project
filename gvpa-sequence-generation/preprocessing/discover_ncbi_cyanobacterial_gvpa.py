#!/usr/bin/env python3
"""Build an auditable NCBI candidate pool for cyanobacterial GvpA.

Two complementary searches are combined: explicit GvpA annotation/gene name and
the GvpA-specific NCBIfam/CDD model PRK09371. Search hits are candidates only;
they must still pass strict_clean.py before use in modelling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
QUERIES = {
    "annotation": (
        '("gas vesicle structural protein GvpA"[Protein Name] OR '
        '"gas vesicle protein GvpA"[Protein Name] OR gvpA[Gene Name]) '
        "AND txid1117[Organism:exp] AND 50:150[Sequence Length]"
    ),
    "prk09371": "PRK09371[All Fields] AND txid1117[Organism:exp]",
}


def request_with_retries(
    session: requests.Session,
    endpoint: str,
    params: dict[str, str | int],
    retries: int,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(f"{EUTILS}/{endpoint}", params=params, timeout=120)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"NCBI request failed: {endpoint}") from last_error


def batches(items: list[str], size: int) -> list[list[str]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def normalize_fasta(text: str) -> str:
    """Normalize NCBI FASTA database-specific IDs to accession.version IDs."""
    normalized: list[str] = []
    for line in text.splitlines():
        if not line.startswith(">"):
            normalized.append(line)
            continue
        header = line[1:]
        token, separator, description = header.partition(" ")
        parts = token.split("|")
        if len(parts) >= 3 and parts[0] in {"sp", "tr"}:
            accession = parts[1]
        elif len(parts) >= 3 and parts[0] == "pdb":
            accession = f"{parts[1]}_{parts[2]}"
        elif len(parts) >= 3 and parts[0] == "pir":
            accession = parts[-1]
        else:
            accession = token
        normalized.append(f">{accession}{separator}{description}")
    return "\n".join(normalized)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "GVPA-course-completeness-audit/1.0"})
    query_uids: dict[str, set[str]] = {}
    query_translations: dict[str, str] = {}
    for name, query in QUERIES.items():
        response = request_with_retries(
            session,
            "esearch.fcgi",
            {
                "db": "protein",
                "term": query,
                "retmode": "json",
                "retmax": 10000,
                "tool": "gvpa_course_completeness_audit",
            },
            args.retries,
        )
        result = response.json()["esearchresult"]
        query_uids[name] = set(result["idlist"])
        query_translations[name] = result.get("querytranslation", "")
        time.sleep(0.4)

    union_uids = sorted(set().union(*query_uids.values()), key=int)
    summaries: dict[str, dict[str, object]] = {}
    for batch in batches(union_uids, args.batch_size):
        response = request_with_retries(
            session,
            "esummary.fcgi",
            {
                "db": "protein",
                "id": ",".join(batch),
                "retmode": "json",
                "tool": "gvpa_course_completeness_audit",
            },
            args.retries,
        )
        result = response.json()["result"]
        for uid in batch:
            if uid in result:
                summaries[uid] = result[uid]
        time.sleep(0.4)

    manifest_rows: list[dict[str, object]] = []
    for uid in union_uids:
        summary = summaries[uid]
        manifest_rows.append(
            {
                "uid": uid,
                "accession_version": summary.get("accessionversion", ""),
                "title": summary.get("title", ""),
                "organism": summary.get("organism", ""),
                "taxid": summary.get("taxid", ""),
                "length": summary.get("slen", ""),
                "source_database": summary.get("sourcedb", ""),
                "updated_date": summary.get("updatedate", ""),
                "found_by_annotation_query": uid in query_uids["annotation"],
                "found_by_prk09371_query": uid in query_uids["prk09371"],
            }
        )

    manifest_path = args.output_dir / "candidate_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    fasta_parts: list[str] = []
    for batch_index, batch in enumerate(batches(union_uids, args.batch_size)):
        response = request_with_retries(
            session,
            "efetch.fcgi",
            {
                "db": "protein",
                "id": ",".join(batch),
                "rettype": "fasta",
                "retmode": "text",
                "tool": "gvpa_course_completeness_audit",
            },
            args.retries,
        )
        text = normalize_fasta(response.text.strip())
        if text:
            fasta_parts.append(text)
        time.sleep(0.4)
    fasta_path = args.output_dir / "ncbi_cyanobacterial_gvpa_candidates.fasta"
    fasta_path.write_text("\n\n".join(fasta_parts) + "\n", encoding="utf-8")

    fasta_ids = {
        line[1:].split(maxsplit=1)[0]
        for line in fasta_path.read_text(encoding="utf-8").splitlines()
        if line.startswith(">")
    }
    manifest_accessions = {
        str(row["accession_version"]) for row in manifest_rows if row["accession_version"]
    }
    if fasta_ids != manifest_accessions:
        raise RuntimeError(
            "Downloaded FASTA IDs do not match manifest: "
            f"missing={len(manifest_accessions - fasta_ids)}, "
            f"extra={len(fasta_ids - manifest_accessions)}"
        )

    summary = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "ncbi_taxonomy_scope": "txid1117[Organism:exp]",
        "queries": QUERIES,
        "query_translations": query_translations,
        "counts": {
            "annotation": len(query_uids["annotation"]),
            "prk09371": len(query_uids["prk09371"]),
            "intersection": len(query_uids["annotation"] & query_uids["prk09371"]),
            "annotation_only": len(query_uids["annotation"] - query_uids["prk09371"]),
            "prk09371_only": len(query_uids["prk09371"] - query_uids["annotation"]),
            "union": len(union_uids),
        },
        "candidate_fasta_sha256": hashlib.sha256(fasta_path.read_bytes()).hexdigest(),
        "warning": "Search hits are candidates, not automatically accepted training records.",
    }
    (args.output_dir / "discovery_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
