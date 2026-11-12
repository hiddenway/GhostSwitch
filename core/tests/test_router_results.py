from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.router_results import metrics, read_records, wilson


class RouterResultsTests(unittest.TestCase):
    def test_errors_and_wrong_experts_remain_in_accuracy_denominator(self):
        def row(expected, chosen, route):
            return {"expected": expected, "chosen": chosen, "route": route, "seconds": 1.0}
        result = metrics([row("coding", "coding", "expert"), row("coding", "base", "base"),
                          row("base", "coding", "expert"), row("coding", None, "error")])
        self.assertEqual((result["n"], result["correct"], result["accuracy"]), (4, 1, 0.25))
        self.assertEqual((result["misses_to_base"], result["misses_to_wrong_expert"], result["errors"]), (1, 1, 1))
        self.assertEqual(result["specialist_n"], 3)

    def test_wilson_interval_is_bounded_for_perfect_and_zero_scores(self):
        for hits in (0, 10):
            low, high = wilson(hits, 10)
            self.assertGreaterEqual(low, -1e-15)
            self.assertLessEqual(high, 1 + 1e-15)
            self.assertLess(low, high)
            self.assertLessEqual(low, hits / 10 + 1e-15)
            self.assertGreaterEqual(high, hits / 10 - 1e-15)

    def test_duplicate_result_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "routes.jsonl"
            path.write_text((json.dumps({"key": "same"}) + "\n") * 2)
            with self.assertRaisesRegex(ValueError, "Duplicate question"):
                read_records(path)
