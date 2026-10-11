from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core import expert_packages

SOURCE = Path(__file__).resolve().parents[2] / "experts" / "qwen35-9b"


class BlockedExpertTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        for expert in ("reasoning_expert", "verilog_expert"):
            shutil.copytree(SOURCE / expert, self.root / "qwen35-9b" / expert)
        root = patch.object(expert_packages, "ROOT", self.root)
        root.start()
        self.addCleanup(root.stop)
        expert_packages._load_packages.cache_clear()
        self.addCleanup(expert_packages._load_packages.cache_clear)

    def evaluate(self, expert, value):
        (self.root / "qwen35-9b" / expert / "evaluation.json").write_text(json.dumps(value))
        expert_packages._load_packages.cache_clear()
        return expert_packages.load_packages("qwen35-9b")

    def tree(self, packages):
        raw = json.loads((Path(expert_packages.__file__).parent / "config" / "router_tree.json").read_text())
        return {node.get("expert_id") for node in expert_packages.package_router_config(raw, packages)["nodes"]}

    def test_blocked_expert_is_kept_for_direct_use_but_never_routed(self):
        packages = self.evaluate("verilog_expert", {
            "schema_version": 1, "verdict": "no_gain", "blocked": True,
            "reason": "VerilogEval pass@1 31% vs base 33% (n=50, p=0.75)"})
        self.assertTrue(expert_packages.blocked(packages["verilog_expert"]))
        self.assertIn("verilog_expert", packages)
        experts = self.tree(packages)
        self.assertIn("reasoning_expert", experts)
        self.assertNotIn("verilog_expert", experts)

    def test_evaluation_must_be_complete(self):
        for value in ({"schema_version": 1, "verdict": "worse", "blocked": False},
                      {"schema_version": 1, "verdict": "hurts", "blocked": True},
                      {"verdict": "helps", "blocked": False}):
            with self.assertRaises(ValueError, msg=value):
                self.evaluate("verilog_expert", value)
        packages = self.evaluate("verilog_expert", {"schema_version": 1, "verdict": "helps", "blocked": False})
        self.assertFalse(expert_packages.blocked(packages["verilog_expert"]))


if __name__ == "__main__":
    unittest.main()
