from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from .alignment import AlignmentResult, global_align

CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class SequenceRecord:
    record_id: str
    sequence: str


def evaluate_sequence_set(
    generated: Iterable[Any],
    references: Iterable[Any],
    identity_threshold: float = 0.95,
    coverage_threshold: float = 0.80,
    kmer_size: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    generated_records = [_as_sequence_record(record) for record in generated]
    reference_records = [_as_sequence_record(record) for record in references]
    if not reference_records:
        raise ValueError("At least one reference sequence is required.")

    reference_sequences = {record.sequence for record in reference_records}
    reference_lengths = [len(record.sequence) for record in reference_records if record.sequence]
    if not reference_lengths:
        raise ValueError("Reference sequences must not all be empty.")
    minimum_length = min(reference_lengths)
    maximum_length = max(reference_lengths)

    rows: list[dict[str, Any]] = []
    for record in generated_records:
        legal_chars = bool(record.sequence) and all(aa in CANONICAL_AA for aa in record.sequence)
        length_in_reference_range = minimum_length <= len(record.sequence) <= maximum_length
        valid_sequence = legal_chars and length_in_reference_range
        nearest_id, nearest_alignment = nearest_reference(record.sequence, reference_records)
        nearest_kmer_id, nearest_kmer_jaccard = nearest_kmer_reference(
            record.sequence, reference_records, kmer_size
        )
        coverage_pass = (
            nearest_alignment.query_coverage >= coverage_threshold
            and nearest_alignment.target_coverage >= coverage_threshold
        )
        near_duplicate = (
            valid_sequence
            and coverage_pass
            and nearest_alignment.identity >= identity_threshold
        )
        sequence_novel = valid_sequence and not near_duplicate
        rows.append(
            {
                "sequence_id": record.record_id,
                "length": len(record.sequence),
                "legal_chars": legal_chars,
                "length_in_reference_range": length_in_reference_range,
                "valid_sequence": valid_sequence,
                "exact_copy_reference": record.sequence in reference_sequences,
                "nearest_reference_id": nearest_id,
                "nearest_reference_identity": nearest_alignment.identity,
                "nearest_reference_full_length_identity": nearest_alignment.full_length_identity,
                "nearest_reference_query_coverage": nearest_alignment.query_coverage,
                "nearest_reference_target_coverage": nearest_alignment.target_coverage,
                "coverage_pass": coverage_pass,
                "near_duplicate": near_duplicate,
                "sequence_novel": sequence_novel,
                "sequence_novelty": 1.0 - nearest_alignment.full_length_identity
                if valid_sequence
                else None,
                "nearest_kmer_reference_id": nearest_kmer_id,
                "nearest_kmer_jaccard": nearest_kmer_jaccard,
                "kmer_novelty": 1.0 - nearest_kmer_jaccard if valid_sequence else None,
            }
        )

    pairwise_identities = pairwise_full_length_identities(
        [record.sequence for record in generated_records if record.sequence]
    )
    valid_rows = [row for row in rows if row["valid_sequence"]]
    summary = {
        "n_generated": len(rows),
        "n_reference": len(reference_records),
        "reference_length_min": minimum_length,
        "reference_length_max": maximum_length,
        "identity_threshold": identity_threshold,
        "coverage_threshold": coverage_threshold,
        "kmer_size": kmer_size,
        "legal_sequence_rate": _mean_bool(row["legal_chars"] for row in rows),
        "length_in_reference_range_rate": _mean_bool(
            row["length_in_reference_range"] for row in rows
        ),
        "valid_sequence_rate": _mean_bool(row["valid_sequence"] for row in rows),
        "exact_copy_reference_rate": _mean_bool(row["exact_copy_reference"] for row in rows),
        "near_duplicate_rate": _mean_bool(row["near_duplicate"] for row in rows),
        "sequence_novel_rate_all": _mean_bool(row["sequence_novel"] for row in rows),
        "sequence_novel_rate_valid": _mean_bool(row["sequence_novel"] for row in valid_rows),
        "mean_sequence_novelty_valid": _mean_optional(
            row["sequence_novelty"] for row in valid_rows
        ),
        "median_sequence_novelty_valid": _median_optional(
            row["sequence_novelty"] for row in valid_rows
        ),
        "mean_nearest_reference_identity_valid": _mean_optional(
            row["nearest_reference_identity"] for row in valid_rows
        ),
        "mean_nearest_kmer_jaccard_valid": _mean_optional(
            row["nearest_kmer_jaccard"] for row in valid_rows
        ),
        "unique_ratio": _unique_ratio(record.sequence for record in generated_records),
        "mean_pairwise_full_length_identity": _mean_optional(pairwise_identities),
        "mean_pairwise_diversity": (
            1.0 - statistics.fmean(pairwise_identities) if pairwise_identities else None
        ),
        "max_duplicate_fraction": _max_duplicate_fraction(
            record.sequence for record in generated_records
        ),
    }
    return rows, summary


def nearest_reference(
    sequence: str, references: list[SequenceRecord]
) -> tuple[str, AlignmentResult]:
    best_id = references[0].record_id
    best_alignment = global_align(sequence, references[0].sequence)
    best_key = _alignment_rank(best_alignment)
    for reference in references[1:]:
        alignment = global_align(sequence, reference.sequence)
        rank = _alignment_rank(alignment)
        if rank > best_key:
            best_id = reference.record_id
            best_alignment = alignment
            best_key = rank
    return best_id, best_alignment


def nearest_kmer_reference(
    sequence: str, references: list[SequenceRecord], kmer_size: int
) -> tuple[str, float]:
    if kmer_size < 1:
        raise ValueError("kmer_size must be at least 1.")
    query_kmers = kmer_set(sequence, kmer_size)
    best_id = references[0].record_id
    best_score = -1.0
    for reference in references:
        score = jaccard(query_kmers, kmer_set(reference.sequence, kmer_size))
        if score > best_score:
            best_id = reference.record_id
            best_score = score
    return best_id, max(best_score, 0.0)


def kmer_set(sequence: str, kmer_size: int) -> set[str]:
    if len(sequence) < kmer_size:
        return set()
    return {
        sequence[index : index + kmer_size]
        for index in range(len(sequence) - kmer_size + 1)
    }


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def pairwise_full_length_identities(sequences: list[str]) -> list[float]:
    identities: list[float] = []
    for index, left in enumerate(sequences):
        for right in sequences[index + 1 :]:
            identities.append(global_align(left, right).full_length_identity)
    return identities


def _alignment_rank(alignment: AlignmentResult) -> tuple[float, float, float]:
    return (
        alignment.full_length_identity,
        min(alignment.query_coverage, alignment.target_coverage),
        alignment.identity,
    )


def _as_sequence_record(record: Any) -> SequenceRecord:
    return SequenceRecord(record_id=str(record.record_id), sequence=str(record.sequence).upper())


def _mean_bool(values: Iterable[bool]) -> float | None:
    items = list(values)
    return sum(items) / len(items) if items else None


def _mean_optional(values: Iterable[float | None]) -> float | None:
    items = [float(value) for value in values if value is not None and not math.isnan(value)]
    return statistics.fmean(items) if items else None


def _median_optional(values: Iterable[float | None]) -> float | None:
    items = [float(value) for value in values if value is not None and not math.isnan(value)]
    return statistics.median(items) if items else None


def _unique_ratio(sequences: Iterable[str]) -> float | None:
    items = list(sequences)
    return len(set(items)) / len(items) if items else None


def _max_duplicate_fraction(sequences: Iterable[str]) -> float | None:
    items = list(sequences)
    return max(Counter(items).values()) / len(items) if items else None
