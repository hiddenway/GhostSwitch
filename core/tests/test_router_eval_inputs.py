from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from core import llm, router_v7, runtime_selection
from scripts import router_eval


class RouterEvaluationInputsTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.items = [{"key": f"sample:{i}", "source": "sample", "expected": "base",
                       "prompt": f"Question {i}", "category": None} for i in range(5)]

    def args(self, **changes):
        return argparse.Namespace(**{"limit": 5, "seed": 1, "dataset": None,
                                     "exclude_keys": [], "shard": None, "max": None, **changes})

    def test_exclusions_are_applied_before_sample_cap(self):
        excluded = self.root / "excluded.json"
        excluded.write_text(json.dumps(["sample:0", "sample:2"]))
        with patch.object(router_eval, "dataset", return_value=self.items):
            selected, keys = router_eval.evaluation_items(self.args(exclude_keys=[str(excluded)], max=2))
        self.assertEqual([item["key"] for item in selected], ["sample:1", "sample:3"])
        self.assertEqual(keys, {"sample:0", "sample:2"})

    def test_frozen_inputs_do_not_reload_external_datasets(self):
        frozen = self.root / "inputs.jsonl"
        frozen.write_text("".join(json.dumps(item) + "\n" for item in self.items))
        with patch.object(router_eval, "dataset", side_effect=AssertionError("Must use frozen data")):
            selected, _ = router_eval.evaluation_items(self.args(dataset=str(frozen)))
        self.assertEqual(selected, self.items)

    def test_duplicate_question_ids_fail_instead_of_inflating_sample(self):
        with patch.object(router_eval, "dataset", return_value=[self.items[0], self.items[0]]):
            with self.assertRaisesRegex(ValueError, "Duplicate question"):
                router_eval.evaluation_items(self.args())

    def test_empty_sample_is_rejected(self):
        with patch.object(router_eval, "dataset", return_value=[]):
            with self.assertRaisesRegex(ValueError, "No questions"):
                router_eval.evaluation_items(self.args())

    def test_shards_do_not_overlap(self):
        with patch.object(router_eval, "dataset", return_value=self.items):
            first, _ = router_eval.evaluation_items(self.args(shard="0/2"))
            second, _ = router_eval.evaluation_items(self.args(shard="1/2"))
        self.assertFalse({x["key"] for x in first} & {x["key"] for x in second})
        self.assertEqual(len(first) + len(second), len(self.items))

    def run_fake(self, fingerprints):
        args = self.args(router="v7", backbone="qwen35-9b", backend="mlx", model_path=None,
                         results=str(self.root / "results"), progress=10)
        def snapshot(value):
            return {"commit": "test", "code_sha256": {"router.py": value}}
        with patch.object(router_eval, "evaluation_items", return_value=(self.items, set())), \
             patch.object(router_eval, "AllExperts"), \
             patch.object(router_eval, "_model", return_value={"backend": "mlx", "model": "fake"}), \
             patch.object(router_eval, "_machine", return_value={}), \
             patch.object(router_eval, "_git", side_effect=[snapshot(x) for x in fingerprints]), \
             patch.object(runtime_selection, "resolve", return_value=SimpleNamespace(
                 backbone="qwen35-9b", model_path="fake", backend="mlx")), \
             patch.object(llm, "configure_backbone"), \
             patch.object(router_v7, "route", return_value={"route": "base", "expert_id": None}), \
             contextlib.redirect_stdout(io.StringIO()):
            router_eval.run(args)

    def test_resume_refuses_changed_source(self):
        self.run_fake(["version1", "version1"])
        with self.assertRaisesRegex(ValueError, "evaluated code changed"):
            self.run_fake(["version2"])

    def test_resume_refuses_changed_inputs(self):
        self.run_fake(["version1", "version1"])
        self.items[0] = {**self.items[0], "prompt": "A different question"}
        with self.assertRaisesRegex(ValueError, "input questions differ"):
            self.run_fake(["version1"])

    def test_source_change_during_run_is_not_certified_as_stable(self):
        with self.assertRaisesRegex(RuntimeError, "code changed during"):
            self.run_fake(["version1", "version2"])
        meta = json.loads((self.root / "results/run.json").read_text())
        self.assertFalse(meta["code_unchanged_during_run"])
