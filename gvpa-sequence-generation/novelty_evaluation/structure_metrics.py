from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .alignment import aligned_index_pairs, global_align

try:
    import numpy as np
except ImportError:  # Sequence-only evaluation remains available.
    np = None


THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "MSE": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


@dataclass(frozen=True)
class ProteinStructure:
    structure_id: str
    path: Path
    chain_id: str
    sequence: str
    coordinates: Any
    ca_confidence: Any


def load_structure_directory(path: str | Path | None) -> tuple[dict[str, ProteinStructure], str]:
    if path is None:
        return {}, "not_run"
    if np is None:
        return {}, "numpy_unavailable"
    directory = Path(path)
    if not directory.is_dir():
        return {}, "directory_not_found"

    structures: dict[str, ProteinStructure] = {}
    parse_failures = 0
    for pdb_path in sorted(directory.glob("*.pdb")):
        try:
            structure = parse_pdb(pdb_path)
        except ValueError:
            parse_failures += 1
            continue
        structures[pdb_path.stem] = structure
    if not structures:
        return {}, "no_usable_pdb_files"
    if parse_failures:
        return structures, f"ok_with_{parse_failures}_parse_failures"
    return structures, "ok"


def parse_pdb(path: str | Path, chain_id: str | None = None) -> ProteinStructure:
    if np is None:
        raise RuntimeError("NumPy is required for structure evaluation.")
    pdb_path = Path(path)
    chains: dict[str, list[tuple[tuple[str, str], str, list[float], float]]] = {}
    seen: set[tuple[str, str, str]] = set()

    with pdb_path.open(errors="replace") as handle:
        for line in handle:
            if not line.startswith("ATOM  ") or line[12:16].strip() != "CA":
                continue
            alternate_location = line[16:17]
            if alternate_location not in {" ", "A", "1"}:
                continue
            current_chain = line[21:22].strip() or "_"
            residue_number = line[22:26].strip()
            insertion_code = line[26:27].strip()
            residue_key = (current_chain, residue_number, insertion_code)
            if residue_key in seen:
                continue
            residue_name = line[17:20].strip().upper()
            amino_acid = THREE_TO_ONE.get(residue_name)
            if amino_acid is None:
                continue
            try:
                coordinate = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                confidence = float(line[60:66])
            except ValueError:
                continue
            seen.add(residue_key)
            chains.setdefault(current_chain, []).append(
                ((residue_number, insertion_code), amino_acid, coordinate, confidence)
            )

    if not chains:
        raise ValueError(f"No standard CA atoms found in {pdb_path}.")
    selected_chain = chain_id or max(chains, key=lambda key: len(chains[key]))
    if selected_chain not in chains:
        raise ValueError(f"Chain {selected_chain!r} not found in {pdb_path}.")
    residues = chains[selected_chain]
    return ProteinStructure(
        structure_id=pdb_path.stem,
        path=pdb_path,
        chain_id=selected_chain,
        sequence="".join(residue[1] for residue in residues),
        coordinates=np.asarray([residue[2] for residue in residues], dtype=float),
        ca_confidence=np.asarray([residue[3] for residue in residues], dtype=float),
    )


def compare_structure_sets(
    generated: dict[str, ProteinStructure],
    references: dict[str, ProteinStructure],
    generated_ids: list[str],
    structure_similarity_threshold: float = 0.50,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    compared_scores: list[float] = []
    for sequence_id in generated_ids:
        query = generated.get(sequence_id)
        if query is None:
            results[sequence_id] = _empty_structure_result("generated_structure_missing")
            continue
        if not references:
            results[sequence_id] = _empty_structure_result("reference_structures_missing")
            continue

        best: dict[str, Any] | None = None
        for reference in references.values():
            comparison = compare_structures(query, reference)
            if best is None or comparison["tm_score_symmetric"] > best["tm_score_symmetric"]:
                best = comparison
        assert best is not None
        best["structure_status"] = "ok"
        best["structure_conserved"] = (
            best["tm_score_symmetric"] >= structure_similarity_threshold
        )
        best["structure_novelty"] = 1.0 - best["tm_score_symmetric"]
        best["generated_mean_ca_confidence"] = float(np.mean(query.ca_confidence))
        best["generated_min_ca_confidence"] = float(np.min(query.ca_confidence))
        results[sequence_id] = best
        compared_scores.append(best["tm_score_symmetric"])

    summary = {
        "structure_similarity_threshold": structure_similarity_threshold,
        "n_generated_structures_loaded": len(generated),
        "n_reference_structures_loaded": len(references),
        "n_structures_compared": len(compared_scores),
        "structure_comparison_rate": (
            len(compared_scores) / len(generated_ids) if generated_ids else None
        ),
        "mean_best_tm_score_symmetric": (
            float(np.mean(compared_scores)) if compared_scores else None
        ),
        "median_best_tm_score_symmetric": (
            float(np.median(compared_scores)) if compared_scores else None
        ),
        "mean_structure_novelty": (
            1.0 - float(np.mean(compared_scores)) if compared_scores else None
        ),
    }
    return results, summary


def compare_structures(query: ProteinStructure, target: ProteinStructure) -> dict[str, Any]:
    if np is None:
        raise RuntimeError("NumPy is required for structure evaluation.")
    alignment = global_align(query.sequence, target.sequence)
    index_pairs = aligned_index_pairs(alignment)
    if len(index_pairs) < 3:
        return {
            "nearest_structure_id": target.structure_id,
            "aligned_ca_count": len(index_pairs),
            "structure_query_coverage": len(index_pairs) / max(len(query.sequence), 1),
            "structure_target_coverage": len(index_pairs) / max(len(target.sequence), 1),
            "ca_rmsd": None,
            "tm_score_query_norm": 0.0,
            "tm_score_target_norm": 0.0,
            "tm_score_symmetric": 0.0,
            "gdt_ts": 0.0,
        }

    query_coordinates = np.asarray(
        [query.coordinates[query_index] for query_index, _ in index_pairs], dtype=float
    )
    target_coordinates = np.asarray(
        [target.coordinates[target_index] for _, target_index in index_pairs], dtype=float
    )
    fitted_query, distances = superpose(query_coordinates, target_coordinates)
    del fitted_query
    query_tm = tm_score(distances, len(query.sequence))
    target_tm = tm_score(distances, len(target.sequence))
    return {
        "nearest_structure_id": target.structure_id,
        "aligned_ca_count": len(index_pairs),
        "structure_query_coverage": len(index_pairs) / max(len(query.sequence), 1),
        "structure_target_coverage": len(index_pairs) / max(len(target.sequence), 1),
        "ca_rmsd": float(np.sqrt(np.mean(distances**2))),
        "tm_score_query_norm": query_tm,
        "tm_score_target_norm": target_tm,
        "tm_score_symmetric": (query_tm + target_tm) / 2.0,
        "gdt_ts": gdt_ts(distances),
    }


def superpose(mobile: Any, target: Any) -> tuple[Any, Any]:
    mobile_center = np.mean(mobile, axis=0)
    target_center = np.mean(target, axis=0)
    centered_mobile = mobile - mobile_center
    centered_target = target - target_center
    covariance = centered_mobile.T @ centered_target
    left, _, right_transposed = np.linalg.svd(covariance)
    if np.linalg.det(left @ right_transposed) < 0:
        left[:, -1] *= -1
    rotation = left @ right_transposed
    fitted = centered_mobile @ rotation + target_center
    distances = np.linalg.norm(fitted - target, axis=1)
    return fitted, distances


def tm_score(distances: Any, normalization_length: int) -> float:
    if normalization_length <= 0 or len(distances) == 0:
        return 0.0
    if normalization_length > 15:
        d0 = 1.24 * math.pow(normalization_length - 15, 1.0 / 3.0) - 1.8
    else:
        d0 = 0.5
    d0 = max(d0, 0.5)
    return float(np.sum(1.0 / (1.0 + (distances / d0) ** 2)) / normalization_length)


def gdt_ts(distances: Any) -> float:
    if len(distances) == 0:
        return 0.0
    fractions = [float(np.mean(distances <= cutoff)) for cutoff in (1.0, 2.0, 4.0, 8.0)]
    return sum(fractions) / len(fractions)


def _empty_structure_result(status: str) -> dict[str, Any]:
    return {
        "structure_status": status,
        "nearest_structure_id": None,
        "aligned_ca_count": None,
        "structure_query_coverage": None,
        "structure_target_coverage": None,
        "ca_rmsd": None,
        "tm_score_query_norm": None,
        "tm_score_target_norm": None,
        "tm_score_symmetric": None,
        "gdt_ts": None,
        "structure_conserved": None,
        "structure_novelty": None,
        "generated_mean_ca_confidence": None,
        "generated_min_ca_confidence": None,
    }
