from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from novelty_evaluation.alignment import global_align
from novelty_evaluation.sequence_metrics import evaluate_sequence_set
from novelty_evaluation.structure_metrics import (
    compare_structures,
    np,
    parse_pdb,
)


class AlignmentTests(unittest.TestCase):
    def test_short_exact_fragment_keeps_low_target_coverage(self) -> None:
        result = global_align("GA", "GGGGGGGGGA")
        self.assertEqual(result.identity, 1.0)
        self.assertAlmostEqual(result.query_coverage, 1.0)
        self.assertAlmostEqual(result.target_coverage, 0.2)
        self.assertAlmostEqual(result.full_length_identity, 0.2)


class SequenceMetricTests(unittest.TestCase):
    def test_invalid_sequences_do_not_count_as_novel(self) -> None:
        references = [
            SimpleNamespace(record_id="ref1", sequence="ACDEFG"),
            SimpleNamespace(record_id="ref2", sequence="ACDEYG"),
        ]
        generated = [
            SimpleNamespace(record_id="copy", sequence="ACDEFG"),
            SimpleNamespace(record_id="novel", sequence="ACDQWG"),
            SimpleNamespace(record_id="empty", sequence=""),
            SimpleNamespace(record_id="fragment", sequence="AC"),
        ]
        rows, summary = evaluate_sequence_set(
            generated,
            references,
            identity_threshold=0.95,
            coverage_threshold=0.80,
            kmer_size=2,
        )
        by_id = {row["sequence_id"]: row for row in rows}

        self.assertTrue(by_id["copy"]["near_duplicate"])
        self.assertFalse(by_id["copy"]["sequence_novel"])
        self.assertTrue(by_id["novel"]["sequence_novel"])
        self.assertFalse(by_id["empty"]["valid_sequence"])
        self.assertFalse(by_id["empty"]["sequence_novel"])
        self.assertFalse(by_id["fragment"]["valid_sequence"])
        self.assertFalse(by_id["fragment"]["sequence_novel"])
        self.assertAlmostEqual(summary["valid_sequence_rate"], 0.5)
        self.assertAlmostEqual(summary["sequence_novel_rate_valid"], 0.5)


@unittest.skipIf(np is None, "NumPy is not installed")
class StructureMetricTests(unittest.TestCase):
    def test_identical_structures_have_unit_tm_score(self) -> None:
        pdb_text = _linear_pdb(["ALA", "CYS", "ASP", "GLU"], offset=0.0)
        shifted_text = _linear_pdb(["ALA", "CYS", "ASP", "GLU"], offset=10.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            left_path = Path(temp_dir) / "left.pdb"
            right_path = Path(temp_dir) / "right.pdb"
            left_path.write_text(pdb_text)
            right_path.write_text(shifted_text)
            result = compare_structures(parse_pdb(left_path), parse_pdb(right_path))

        self.assertAlmostEqual(result["ca_rmsd"], 0.0, places=6)
        self.assertAlmostEqual(result["tm_score_symmetric"], 1.0, places=6)
        self.assertAlmostEqual(result["gdt_ts"], 1.0, places=6)


def _linear_pdb(residue_names: list[str], offset: float) -> str:
    lines = []
    for index, residue_name in enumerate(residue_names, start=1):
        x = offset + index * 3.8
        lines.append(
            f"ATOM  {index:5d}  CA  {residue_name:>3s} A{index:4d}    "
            f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{90.0:6.2f}           C"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    unittest.main()
