#!/usr/bin/env python3
"""Strict, auditable cleaning for the course GvpA FASTA.

The script keeps the source FASTA unchanged, retrieves current RefSeq metadata,
and writes include/review/exclude outputs.  It deliberately does not replace a
failed metadata lookup with a guess based on the FASTA title.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests


VALID_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
CYANO_LINEAGE_TERMS = ("cyanobacteriota", "cyanobacteria", "cyanophyceae")
EXPLICIT_GVPA_RE = re.compile(
    r"\b(?:gas\s+vesicle\s+(?:structural\s+)?protein\s+gvp\s*a|"
    r"gvp\s*a\s+family\s+protein)\b",
    re.IGNORECASE,
)
OTHER_GVP_RE = re.compile(r"\bgvp\s*(?:j|m|k|s)\b", re.IGNORECASE)
AMBIGUOUS_PRODUCT_RE = re.compile(
    r"(?:synthesis[- ]like|gvpa[- ]like|hypothetical|uncharacterized)",
    re.IGNORECASE,
)
PARTIAL_RE = re.compile(r"\b(?:partial|fragment|incomplete)\b", re.IGNORECASE)


@dataclass(frozen=True)
class FastaRecord:
    index: int
    sequence_id: str
    header: str
    sequence: str


def parse_fasta(path: Path) -> list[FastaRecord]:
    records: list[FastaRecord] = []
    header: str | None = None
    sequence_parts: list[str] = []

    with path.open(encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    sequence_id = header.split(maxsplit=1)[0]
                    records.append(
                        FastaRecord(
                            index=len(records),
                            sequence_id=sequence_id,
                            header=header,
                            sequence="".join(sequence_parts).upper(),
                        )
                    )
                header = line[1:].strip()
                sequence_parts = []
                if not header:
                    raise ValueError(f"Empty FASTA header at line {line_number}")
            elif header is None:
                raise ValueError(f"Sequence text before first header at line {line_number}")
            else:
                sequence_parts.append(re.sub(r"\s+", "", line))

    if header is not None:
        records.append(
            FastaRecord(
                index=len(records),
                sequence_id=header.split(maxsplit=1)[0],
                header=header,
                sequence="".join(sequence_parts).upper(),
            )
        )
    return records


def chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def fetch_ncbi_batches(
    accessions: list[str],
    cache_dir: Path,
    batch_size: int,
    retries: int,
) -> list[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "GVPA-course-strict-clean/1.0",
            "Accept": "application/xml",
        }
    )
    cache_files: list[Path] = []

    for batch_index, batch in enumerate(chunks(accessions, batch_size)):
        digest = hashlib.sha256("\n".join(batch).encode()).hexdigest()[:12]
        cache_path = cache_dir / f"batch_{batch_index:03d}_{digest}.xml"
        cache_files.append(cache_path)
        if cache_path.exists() and cache_path.stat().st_size > 100:
            try:
                ET.parse(cache_path)
                continue
            except ET.ParseError:
                cache_path.unlink()

        params = {
            "db": "protein",
            "id": ",".join(batch),
            "rettype": "gb",
            "retmode": "xml",
            "tool": "gvpa_course_strict_clean",
        }
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                response = session.get(url, params=params, timeout=120)
                response.raise_for_status()
                ET.fromstring(response.content)
                cache_path.write_bytes(response.content)
                last_error = None
                break
            except (requests.RequestException, ET.ParseError) as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(min(2**attempt, 10))
        if last_error is not None:
            raise RuntimeError(
                f"NCBI fetch failed for batch {batch_index} after {retries} attempts"
            ) from last_error
        time.sleep(0.4)
    return cache_files


def feature_qualifiers(gbseq: ET.Element) -> dict[str, list[str]]:
    qualifiers: dict[str, list[str]] = defaultdict(list)
    for qualifier in gbseq.findall(".//GBQualifier"):
        name = qualifier.findtext("GBQualifier_name", default="").strip()
        value = qualifier.findtext("GBQualifier_value", default="").strip()
        if name and value:
            qualifiers[name].append(value)
    return dict(qualifiers)


def parse_ncbi_xml(cache_files: list[Path]) -> dict[str, dict[str, object]]:
    metadata: dict[str, dict[str, object]] = {}
    for cache_file in cache_files:
        root = ET.parse(cache_file).getroot()
        for gbseq in root.findall("GBSeq"):
            accession = gbseq.findtext("GBSeq_accession-version", default="").strip()
            if not accession:
                accession = gbseq.findtext("GBSeq_primary-accession", default="").strip()
            qualifiers = feature_qualifiers(gbseq)
            taxon_ids = [
                value.split(":", 1)[1]
                for value in qualifiers.get("db_xref", [])
                if value.startswith("taxon:")
            ]
            products = qualifiers.get("product", [])
            genes = qualifiers.get("gene", [])
            region_names = qualifiers.get("region_name", [])
            cdd_xrefs = [
                value for value in qualifiers.get("db_xref", []) if value.startswith("CDD:")
            ]
            protein_feature = next(
                (
                    feature
                    for feature in gbseq.findall(".//GBFeature")
                    if feature.findtext("GBFeature_key", default="").strip() == "Protein"
                ),
                None,
            )
            protein_location = (
                protein_feature.findtext("GBFeature_location", default="").strip()
                if protein_feature is not None
                else ""
            )
            protein_partial5 = bool(
                protein_feature is not None
                and protein_feature.find("GBFeature_partial5") is not None
            )
            protein_partial3 = bool(
                protein_feature is not None
                and protein_feature.find("GBFeature_partial3") is not None
            )
            ncbi_length = gbseq.findtext("GBSeq_length", default="").strip()
            metadata[accession] = {
                "ncbi_accession": accession,
                "ncbi_definition": gbseq.findtext("GBSeq_definition", default="").strip(),
                "ncbi_organism": gbseq.findtext("GBSeq_organism", default="").strip(),
                "ncbi_taxonomy": gbseq.findtext("GBSeq_taxonomy", default="").strip(),
                "ncbi_taxid": taxon_ids[0] if taxon_ids else "",
                "ncbi_comment": gbseq.findtext("GBSeq_comment", default="").strip(),
                "ncbi_sequence": gbseq.findtext("GBSeq_sequence", default="").strip().upper(),
                "ncbi_length": ncbi_length,
                "ncbi_products": " | ".join(dict.fromkeys(products)),
                "ncbi_genes": " | ".join(dict.fromkeys(genes)),
                "ncbi_region_names": " | ".join(dict.fromkeys(region_names)),
                "ncbi_cdd_xrefs": " | ".join(dict.fromkeys(cdd_xrefs)),
                "ncbi_update_date": gbseq.findtext("GBSeq_update-date", default="").strip(),
                "ncbi_protein_feature_location": protein_location,
                "ncbi_protein_partial5": protein_partial5,
                "ncbi_protein_partial3": protein_partial3,
                "ncbi_complete_by_feature": bool(
                    ncbi_length
                    and protein_location == f"1..{ncbi_length}"
                    and not protein_partial5
                    and not protein_partial3
                ),
            }
    return metadata


def extract_header_organism(header: str) -> str:
    match = re.search(r"\[([^\[\]]+)\]\s*$", header)
    return match.group(1).strip() if match else ""


def is_cyanobacteria(taxonomy: str, organism: str) -> bool:
    text = f"{taxonomy}; {organism}".lower()
    return any(term in text for term in CYANO_LINEAGE_TERMS)


def has_explicit_gvpa(metadata: dict[str, object]) -> bool:
    definition = str(metadata.get("ncbi_definition", ""))
    products = str(metadata.get("ncbi_products", ""))
    genes = {part.strip().lower() for part in str(metadata.get("ncbi_genes", "")).split("|")}
    text = f"{definition} | {products}"
    return bool(EXPLICIT_GVPA_RE.search(text)) or (
        "gvpa" in genes and "gas vesicle" in text.lower()
    )


def classify(
    record: FastaRecord,
    metadata: dict[str, object] | None,
    min_length: int,
    max_length: int,
) -> tuple[str, list[str]]:
    """Return the intrinsic decision before duplicate representatives are chosen."""
    exclude_reasons: list[str] = []
    review_reasons: list[str] = []

    if not record.sequence:
        exclude_reasons.append("empty_sequence")
    invalid = sorted(set(record.sequence) - VALID_AA)
    if invalid:
        exclude_reasons.append("invalid_amino_acids:" + "".join(invalid))

    if metadata is None:
        review_reasons.append("ncbi_metadata_not_retrieved")
    else:
        ncbi_sequence = str(metadata.get("ncbi_sequence", ""))
        taxonomy = str(metadata.get("ncbi_taxonomy", ""))
        organism = str(metadata.get("ncbi_organism", ""))
        definition = str(metadata.get("ncbi_definition", ""))
        products = str(metadata.get("ncbi_products", ""))
        comments = str(metadata.get("ncbi_comment", ""))
        annotation_text = f"{definition} | {products}"

        if ncbi_sequence and ncbi_sequence != record.sequence:
            review_reasons.append("local_sequence_differs_from_current_ncbi")
        if not is_cyanobacteria(taxonomy, organism):
            exclude_reasons.append("not_cyanobacteria_by_ncbi_lineage")
        if OTHER_GVP_RE.search(annotation_text):
            exclude_reasons.append("annotated_as_other_gvpA_like_protein")
        elif not has_explicit_gvpa(metadata):
            if AMBIGUOUS_PRODUCT_RE.search(annotation_text):
                exclude_reasons.append("ambiguous_or_non_gvpa_annotation")
            else:
                review_reasons.append("gvpa_identity_not_explicit")
        feature_partial = bool(metadata.get("ncbi_protein_partial5")) or bool(
            metadata.get("ncbi_protein_partial3")
        )
        if PARTIAL_RE.search(f"{record.header} | {definition} | {comments}") or feature_partial:
            exclude_reasons.append("partial_fragment_or_incomplete")
        elif (
            "completeness: full length" not in comments.lower()
            and not bool(metadata.get("ncbi_complete_by_feature"))
        ):
            review_reasons.append("full_length_not_explicitly_confirmed")
        region_names = {
            value.strip().upper()
            for value in str(metadata.get("ncbi_region_names", "")).split("|")
        }
        cdd_xrefs = {
            value.strip().upper()
            for value in str(metadata.get("ncbi_cdd_xrefs", "")).split("|")
        }
        if "PRK09371" not in region_names or "CDD:181805" not in cdd_xrefs:
            review_reasons.append("gvpa_cdd_prk09371_not_confirmed")

    if not (min_length <= len(record.sequence) <= max_length):
        review_reasons.append(f"length_outside_review_range:{min_length}-{max_length}")

    if exclude_reasons:
        return "exclude", sorted(set(exclude_reasons + review_reasons))
    if review_reasons:
        return "review", sorted(set(review_reasons))
    return "include", ["passes_strict_taxonomy_annotation_and_sequence_checks"]


def duplicate_preference(row: dict[str, object]) -> tuple[int, int, int]:
    """Prefer intrinsically valid and stable RefSeq representatives."""
    decision_rank = {"include": 0, "review": 1, "exclude": 2}[str(row["decision"])]
    accession = str(row["sequence_id"])
    source_rank = 0 if accession.startswith("WP_") else 1
    return decision_rank, source_rank, int(row["input_order"])


def mark_duplicate(row: dict[str, object], reason: str, representative: str) -> None:
    reasons = {
        value
        for value in str(row["reasons"]).split(";")
        if value and value != "passes_strict_taxonomy_annotation_and_sequence_checks"
    }
    reasons.add(reason)
    row["reasons"] = ";".join(sorted(reasons))
    row["decision"] = "exclude"
    if reason == "exact_duplicate_sequence":
        row["duplicate_sequence_of"] = representative
    else:
        row["duplicate_id_of"] = representative


def resolve_duplicates(rows: list[dict[str, object]]) -> None:
    """Keep the highest-quality representative, independent of input order."""
    by_sequence: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        sequence = str(row["sequence"])
        if sequence:
            by_sequence[sequence].append(row)
    for group in by_sequence.values():
        if len(group) < 2:
            continue
        ordered = sorted(group, key=duplicate_preference)
        representative = str(ordered[0]["sequence_id"])
        for row in ordered[1:]:
            mark_duplicate(row, "exact_duplicate_sequence", representative)

    by_id: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_id[str(row["sequence_id"])].append(row)
    for group in by_id.values():
        if len(group) < 2:
            continue
        ordered = sorted(group, key=duplicate_preference)
        representative = str(ordered[0]["sequence_id"])
        for row in ordered[1:]:
            mark_duplicate(row, "duplicate_sequence_id", representative)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_fasta(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            header = (
                f">{row['sequence_id']} decision={row['decision']} "
                f"taxid={row['ncbi_taxid'] or 'NA'} organism={row['ncbi_organism'] or 'NA'}"
            )
            handle.write(header + "\n")
            sequence = str(row["sequence"])
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_summary(
    output_dir: Path,
    rows: list[dict[str, object]],
    input_path: Path,
    min_length: int,
    max_length: int,
) -> None:
    decisions = Counter(str(row["decision"]) for row in rows)
    reason_counts: Counter[str] = Counter()
    for row in rows:
        if row["decision"] == "include":
            continue
        for reason in str(row["reasons"]).split(";"):
            if reason:
                reason_counts[reason] += 1
    included_lengths = [int(row["length"]) for row in rows if row["decision"] == "include"]
    included_taxa = Counter(
        str(row["ncbi_organism"]) for row in rows if row["decision"] == "include"
    )
    summary = {
        "cleaning_version": "strict_clean_v1",
        "input_file": str(input_path),
        "input_sha256": sha256(input_path),
        "rules": {
            "standard_amino_acids_only": True,
            "ncbi_lineage_must_be_cyanobacteria": True,
            "explicit_gvpa_annotation_required": True,
            "ncbi_cdd_prk09371_required": True,
            "ncbi_completeness_full_length_required": (
                "explicit comment or complete Protein feature coordinates"
            ),
            "partial_fragment_incomplete_excluded": True,
            "exact_duplicate_sequences_excluded": True,
            "sequence_mismatch_sent_to_review": True,
            "length_review_range": [min_length, max_length],
            "local_profile_hmm_validation": "pending_P2_not_claimed_by_this_run",
        },
        "counts": {"total": len(rows), **dict(decisions)},
        "included_length": {
            "min": min(included_lengths) if included_lengths else None,
            "max": max(included_lengths) if included_lengths else None,
            "mean": round(sum(included_lengths) / len(included_lengths), 3)
            if included_lengths
            else None,
        },
        "reason_counts": dict(reason_counts.most_common()),
        "included_organism_counts": dict(included_taxa.most_common()),
    }
    with (output_dir / "cleaning_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    lines = [
        "# GvpA 严格清洗结果（strict_clean_v1）",
        "",
        "## 结论",
        "",
        f"- 输入记录：{len(rows)}",
        f"- 严格纳入：{decisions.get('include', 0)}",
        f"- 待人工/工具复核：{decisions.get('review', 0)}",
        f"- 排除：{decisions.get('exclude', 0)}",
        "- 本轮完成 taxonomy、明确 GvpA 注释、NCBI CDD、完整性、字符、长度与 exact duplicate 检查。",
        "- 本轮确认 NCBI CDD `PRK09371 / CDD:181805`；独立的本地 Pfam/HMM 扫描仍由 P2 执行。",
        "",
        "## 决策规则",
        "",
        "只有同时满足下列条件才进入 strict_included.fasta：",
        "",
        "1. NCBI 当前 lineage 明确属于 Cyanobacteria/Cyanobacteriota；",
        "2. NCBI 当前 definition/product/gene 明确支持 GvpA；",
        "3. NCBI CDD 明确包含 `PRK09371 / CDD:181805`；",
        "4. NCBI 明确标记 `COMPLETENESS: full length`，或 Protein feature 完整覆盖全长且无 partial 标记；",
        "5. 未标记为 partial、fragment 或 incomplete；",
        "6. 序列只含 20 种标准氨基酸；",
        "7. 本地序列与当前 NCBI 序列一致；",
        "8. 不与已纳入记录完全重复；",
        f"9. 长度处于复核范围 {min_length}–{max_length} aa。",
        "",
        "## 排除/复核原因统计",
        "",
        "| 原因 | 数量 |",
        "|---|---:|",
    ]
    lines.extend(f"| `{reason}` | {count} |" for reason, count in reason_counts.most_common())
    lines.extend(
        [
            "",
            "注：同一条记录可能同时触发多个排除原因，因此上表数量不能直接相加。",
            "",
            "## 重要限制",
            "",
            "`strict_included.fasta` 是严格清洗候选集，还不是冻结的 dataset_v1。",
            "P2 完成 GvpA family/profile 验证、真实 MSA 与相似度聚类后，才能正式冻结数据。",
            "",
        ]
    )
    (output_dir / "cleaning_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--min-length", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=150)
    parser.add_argument("--offline", action="store_true", help="Use existing XML cache only")
    args = parser.parse_args()

    records = parse_fasta(args.input)
    if not records:
        raise ValueError("No FASTA records found")
    accessions = list(dict.fromkeys(record.sequence_id for record in records))
    if args.offline:
        cache_files = sorted(args.cache_dir.glob("batch_*.xml"))
        if not cache_files:
            raise FileNotFoundError("Offline mode requested but XML cache is empty")
    else:
        cache_files = fetch_ncbi_batches(
            accessions=accessions,
            cache_dir=args.cache_dir,
            batch_size=args.batch_size,
            retries=args.retries,
        )
    ncbi_metadata = parse_ncbi_xml(cache_files)

    rows: list[dict[str, object]] = []
    for record in records:
        metadata = ncbi_metadata.get(record.sequence_id)
        if metadata is None:
            metadata = ncbi_metadata.get(record.sequence_id.split(".", 1)[0])
        decision, reasons = classify(
            record,
            metadata,
            args.min_length,
            args.max_length,
        )
        row: dict[str, object] = {
            "input_order": record.index,
            "sequence_id": record.sequence_id,
            "full_header": record.header,
            "header_organism": extract_header_organism(record.header),
            "sequence": record.sequence,
            "length": len(record.sequence),
            "valid_amino_acids": not bool(set(record.sequence) - VALID_AA),
            "decision": decision,
            "reasons": ";".join(reasons),
            "duplicate_id_of": "",
            "duplicate_sequence_of": "",
        }
        if metadata:
            row.update(metadata)
        else:
            for name in (
                "ncbi_accession",
                "ncbi_definition",
                "ncbi_organism",
                "ncbi_taxonomy",
                "ncbi_taxid",
                "ncbi_comment",
                "ncbi_sequence",
                "ncbi_length",
                "ncbi_products",
                "ncbi_genes",
                "ncbi_region_names",
                "ncbi_cdd_xrefs",
                "ncbi_update_date",
                "ncbi_protein_feature_location",
                "ncbi_protein_partial5",
                "ncbi_protein_partial3",
                "ncbi_complete_by_feature",
            ):
                row[name] = ""
        row["is_cyanobacteria"] = is_cyanobacteria(
            str(row["ncbi_taxonomy"]), str(row["ncbi_organism"])
        )
        row["explicit_gvpa_annotation"] = has_explicit_gvpa(row)
        row["ncbi_sequence_matches_local"] = bool(row["ncbi_sequence"]) and (
            row["ncbi_sequence"] == row["sequence"]
        )
        rows.append(row)

    resolve_duplicates(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    write_csv(args.output_dir / "metadata_all.csv", rows, fieldnames)
    write_csv(
        args.output_dir / "strict_included_metadata.csv",
        [row for row in rows if row["decision"] == "include"],
        fieldnames,
    )
    write_csv(
        args.output_dir / "review_queue.csv",
        [row for row in rows if row["decision"] == "review"],
        fieldnames,
    )
    write_csv(
        args.output_dir / "removal_log.csv",
        [row for row in rows if row["decision"] == "exclude"],
        fieldnames,
    )
    for decision, filename in (
        ("include", "strict_included.fasta"),
        ("review", "review.fasta"),
        ("exclude", "excluded.fasta"),
    ):
        write_fasta(
            args.output_dir / filename,
            [row for row in rows if row["decision"] == decision],
        )
    write_summary(
        args.output_dir,
        rows,
        args.input,
        args.min_length,
        args.max_length,
    )
    print(json.dumps(Counter(row["decision"] for row in rows), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
