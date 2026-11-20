"""Validation boundaries and fail-closed gate checks (standard library runner)."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from monitor_retrain import gate, validate


class MonitoringTests(unittest.TestCase):
    def validate_rows(self, rows, fields=("id", "text", "label")):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "batch.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(fields)
                writer.writerows(rows)
            return validate(path)

    @patch("monitor_retrain.eligible_feedback", return_value=True)
    def test_valid_batch(self, _):
        self.assertEqual(len(self.validate_rows([("a", "first message", "0"), ("b", "other message", "1")])), 2)

    @patch("monitor_retrain.eligible_feedback", return_value=True)
    def test_invalid_batches(self, _):
        cases = [[], [("a", "", "0")], [("a", "message", "3")],
                 [("a", "first", "0"), ("a", "second", "1")],
                 [("a", "same", "0"), ("b", "same", "1")], [("a", "first", "0")]]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.validate_rows(rows)

    def test_missing_column(self):
        with self.assertRaises(ValueError):
            self.validate_rows([], ("text", "label"))

    def test_duplicate_columns_and_extra_fields(self):
        with self.assertRaisesRegex(ValueError, "duplicate column"):
            self.validate_rows([], ("id", "text", "label", "label"))
        with self.assertRaisesRegex(ValueError, "too many CSV fields"):
            self.validate_rows([("a", "message", "1", "extra")])

    @patch("monitor_retrain.eligible_feedback", return_value=False)
    def test_protected_overlap(self, _):
        with self.assertRaisesRegex(ValueError, "overlaps"):
            self.validate_rows([("a", "protected message", "0")])

    @patch("monitor_retrain.quality", return_value=[{"pass": False}])
    def test_gate_rejection_preserves_champion(self, _):
        decision = gate({"benchmark": {"f1": .95}}, {"benchmark": {"f1": .90}})
        self.assertEqual(decision["decision"], "retain_champion")
        self.assertFalse(decision["champion_overwritten"])

    @patch("monitor_retrain.quality", return_value=[{"pass": True}])
    def test_gate_f1_regression_and_eligibility(self, _):
        self.assertEqual(gate({"benchmark": {"f1": .8}}, {"benchmark": {"f1": .9}})["decision"], "retain_champion")
        self.assertEqual(gate({"benchmark": {"f1": .9}}, {"benchmark": {"f1": .9}})["decision"], "eligible_for_promotion")


if __name__ == "__main__":
    unittest.main()
