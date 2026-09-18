from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlignmentResult:
    aligned_query: str
    aligned_target: str
    matches: int
    aligned_pairs: int
    identity: float
    query_coverage: float
    target_coverage: float
    full_length_identity: float


def global_align(
    query: str,
    target: str,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -2,
) -> AlignmentResult:
    """Needleman-Wunsch alignment with metrics that retain coverage information."""
    rows = len(query) + 1
    cols = len(target) + 1
    scores = [[0] * cols for _ in range(rows)]
    moves = [[""] * cols for _ in range(rows)]

    for i in range(1, rows):
        scores[i][0] = i * gap_score
        moves[i][0] = "up"
    for j in range(1, cols):
        scores[0][j] = j * gap_score
        moves[0][j] = "left"

    for i in range(1, rows):
        for j in range(1, cols):
            diagonal = scores[i - 1][j - 1] + (
                match_score if query[i - 1] == target[j - 1] else mismatch_score
            )
            up = scores[i - 1][j] + gap_score
            left = scores[i][j - 1] + gap_score
            best = max(diagonal, up, left)
            scores[i][j] = best
            moves[i][j] = "diagonal" if best == diagonal else "up" if best == up else "left"

    aligned_query: list[str] = []
    aligned_target: list[str] = []
    i = len(query)
    j = len(target)
    while i > 0 or j > 0:
        move = moves[i][j]
        if move == "diagonal":
            aligned_query.append(query[i - 1])
            aligned_target.append(target[j - 1])
            i -= 1
            j -= 1
        elif move == "up":
            aligned_query.append(query[i - 1])
            aligned_target.append("-")
            i -= 1
        else:
            aligned_query.append("-")
            aligned_target.append(target[j - 1])
            j -= 1

    aligned_query_text = "".join(reversed(aligned_query))
    aligned_target_text = "".join(reversed(aligned_target))
    pairs = [
        (query_aa, target_aa)
        for query_aa, target_aa in zip(aligned_query_text, aligned_target_text)
        if query_aa != "-" and target_aa != "-"
    ]
    matches = sum(query_aa == target_aa for query_aa, target_aa in pairs)
    aligned_pairs = len(pairs)
    identity = matches / aligned_pairs if aligned_pairs else 0.0
    query_coverage = aligned_pairs / len(query) if query else 0.0
    target_coverage = aligned_pairs / len(target) if target else 0.0
    full_length_identity = matches / max(len(query), len(target), 1)

    return AlignmentResult(
        aligned_query=aligned_query_text,
        aligned_target=aligned_target_text,
        matches=matches,
        aligned_pairs=aligned_pairs,
        identity=identity,
        query_coverage=query_coverage,
        target_coverage=target_coverage,
        full_length_identity=full_length_identity,
    )


def aligned_index_pairs(alignment: AlignmentResult) -> list[tuple[int, int]]:
    """Return zero-based residue index pairs from an alignment."""
    query_index = -1
    target_index = -1
    pairs: list[tuple[int, int]] = []
    for query_aa, target_aa in zip(alignment.aligned_query, alignment.aligned_target):
        if query_aa != "-":
            query_index += 1
        if target_aa != "-":
            target_index += 1
        if query_aa != "-" and target_aa != "-":
            pairs.append((query_index, target_index))
    return pairs
