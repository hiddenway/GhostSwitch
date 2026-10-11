from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from scripts import expert_eval


class ScoringTests(unittest.TestCase):
    def test_each_answer_kind(self):
        self.assertEqual(expert_eval.score({"kind": "number_tolerant", "gold": "0.141"}, "Answer: 14.1%")[0], 1.0)
        self.assertEqual(expert_eval.score({"kind": "number_tolerant", "gold": "94"}, "Answer: 95")[0], 0.0)
        self.assertEqual(expert_eval.score({"kind": "yes_no", "gold": "No"}, "Hmm.\nAnswer: no")[0], 1.0)
        self.assertEqual(expert_eval.score({"kind": "cwe", "gold": "CWE-416"}, "Not CWE-20.\nCWE-416")[0], 1.0)
        f1, _ = expert_eval.score({"kind": "string_set", "gold": ["Japan", "Syria"]}, 'Found: ["Japan", "Asia"]')
        self.assertAlmostEqual(f1, 0.5)
        context = "CREATE TABLE t (x INT); INSERT INTO t VALUES (1), (2);"
        sql = {"kind": "sql", "context": context, "gold": "SELECT SUM(x) FROM t"}
        self.assertEqual(expert_eval.score(sql, "```sql\nSELECT 1 + 2;\n```")[0], 1.0)
        self.assertEqual(expert_eval.score(sql, "```sql\nSELECT nope FROM t\n```")[0], 0.0)
        code = {"kind": "python", "entry_point": "double", "test": "def check(f):\n    assert f(2) == 4\n"}
        self.assertEqual(expert_eval.score(code, "```python\ndef double(x):\n    return 2 * x\n```")[0], 1.0)
        self.assertEqual(expert_eval.score(code, "```python\ndef double(x):\n    return x\n```")[0], 0.0)

    def test_conll_spans_follow_the_tner_label_ids(self):
        tokens = ["Peter", "Blackburn", "visited", "New", "York", "and", "BMW", "."]
        tags = [3, 4, 0, 5, 8, 0, 1, 0]
        self.assertEqual(expert_eval.conll_entities(tokens, tags), ["Peter Blackburn", "New York", "BMW"])


class VerdictTests(unittest.TestCase):
    def write(self, folder: Path, expert: str, base: list[float], mine: list[float]) -> None:
        (folder / expert).mkdir(parents=True)
        (folder / "base").mkdir(exist_ok=True)
        base_file = f"base/{expert}-task.jsonl"
        (folder / expert / "settings.json").write_text(json.dumps({"task": "task", "quick": True, "base": base_file}))
        for path, scores in ((folder / base_file, base), (folder / expert / "expert.jsonl", mine)):
            path.write_text("\n".join(json.dumps({"id": str(i), "score": value}) for i, value in enumerate(scores)))

    def test_report_blocks_experts_that_do_not_beat_the_base_model(self):
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as root:
            results = Path(folder)
            self.write(results, "reasoning_expert", [0.0] * 20 + [1.0] * 20, [1.0] * 40)
            self.write(results, "verilog_expert", [1.0, 0.0] * 20, [0.0, 1.0] * 20)
            for expert in ("reasoning_expert", "verilog_expert"):
                (Path(root) / "experts" / "qwen35-9b" / expert).mkdir(parents=True)
            with patch.object(expert_eval, "ROOT", Path(root)), patch("builtins.print"):
                expert_eval.report(Namespace(results=folder, alpha=0.05, write=True))
            helps = json.loads((Path(root) / "experts/qwen35-9b/reasoning_expert/evaluation.json").read_text())
            blocked = json.loads((Path(root) / "experts/qwen35-9b/verilog_expert/evaluation.json").read_text())
        self.assertEqual((helps["verdict"], helps["blocked"]), ("helps", False))
        self.assertEqual((blocked["verdict"], blocked["blocked"]), ("no_gain", True))
        self.assertIn("Not better than the base model", blocked["reason"])
        self.assertIn("task (quick)", blocked["reason"])
        self.assertEqual(expert_eval.sign_test([1.0] * 10, [0.0] * 10)[:2], (10, 0))

    def test_quick_prompts_ask_for_the_answer_line_only(self):
        item = {"kind": "choice", "prompt": "Q?\n\nOptions:\nA. x\n\n" + expert_eval.benchmark.CHOICE_INSTRUCTION}
        self.assertNotIn("Think it through", expert_eval.quick(item)["prompt"])
        self.assertIn("Reply with only one line", expert_eval.quick(item)["prompt"])
        cwe = expert_eval.quick({"kind": "cwe", "prompt": "long", "description": "use-after-free in x"})
        self.assertIn("use-after-free in x", cwe["prompt"])
        args = Namespace(quick=True, limit=20, seed=0)
        self.assertEqual(expert_eval.plan(args, "coding_expert")["base"], expert_eval.plan(args, "coding_delta_expert")["base"])
        self.assertEqual(expert_eval.plan(args, "reasoning_expert")["thinking"], "off")


if __name__ == "__main__":
    unittest.main()
