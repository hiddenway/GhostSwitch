from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from scripts import benchmark


class AnswerTests(unittest.TestCase):
    def test_choice_takes_the_final_answer_line(self):
        self.assertEqual(benchmark.extract_choice("B looks tempting, but…\nAnswer: D", 10), "D")
        self.assertEqual(benchmark.extract_choice("So the answer is (C).", 4), "C")
        self.assertEqual(benchmark.extract_choice("Therefore \\boxed{B}", 4), "B")
        self.assertEqual(benchmark.extract_choice("Reasoning…\n**A**", 4), "A")

    def test_choice_ignores_words_and_letters_outside_the_options(self):
        self.assertIsNone(benchmark.extract_choice("The answer is a function of x.", 4))
        self.assertIsNone(benchmark.extract_choice("Answer: J", 4))
        self.assertIsNone(benchmark.extract_choice("", 4))

    def test_number_prefers_the_answer_line_and_drops_formatting(self):
        self.assertEqual(benchmark.extract_number("3 boxes, 12 each = 36\nAnswer: $1,036."), "1036")
        self.assertEqual(benchmark.extract_number("…so \\boxed{-4.5}"), "-4.5")
        self.assertEqual(benchmark.extract_number("She has 18 eggs left."), "18")
        item = {"kind": "number", "gold": "18"}
        self.assertTrue(benchmark.correct(item, "18.00"))
        self.assertFalse(benchmark.correct(item, None))


class SamplingTests(unittest.TestCase):
    items = [{"id": f"{category}{i}", "category": category} for category in "xyz" for i in range(4)]

    def test_order_is_seeded_interleaved_and_offsets_are_disjoint(self):
        first = benchmark.sample(self.items, 6, 0, seed=1)
        self.assertEqual(first, benchmark.sample(self.items, 6, 0, seed=1))
        self.assertEqual(sorted(item["category"] for item in first[:3]), ["x", "y", "z"])
        rest = benchmark.sample(self.items, None, 6, seed=1)
        self.assertEqual(len(first) + len(rest), len(self.items))
        self.assertFalse({item["id"] for item in first} & {item["id"] for item in rest})


class StatisticsTests(unittest.TestCase):
    def test_wilson_and_mcnemar(self):
        low, high = benchmark.wilson(80, 100)
        self.assertAlmostEqual(low, 0.711, places=3)
        self.assertAlmostEqual(high, 0.867, places=3)
        a = {str(i): i < 30 for i in range(40)}
        b = {str(i): i < 20 for i in range(40)}
        self.assertEqual(benchmark.mcnemar(a, b)[:2], (10, 0))
        self.assertLess(benchmark.mcnemar(a, b)[2], 0.01)
        self.assertEqual(benchmark.mcnemar(a, a)[2], 1.0)


class GhostSwitchTargetTests(unittest.TestCase):
    def test_missing_expert_falls_back_to_the_base_model(self):
        replies = [(422, {"status": "unsupported", "routing": {"missing_capability": "verilog_expert"}}),
                   (200, {"status": "success", "choices": [{"message": {"content": "Answer: B"}}],
                          "ghostswitch": {"route": "base"}})]
        args = Namespace(url="http://x", mode="auto", expert=None, verify=True, thinking="auto", max_tokens=64,
                         temperature=0.0, timeout=5, on_unsupported="base")
        with patch.object(benchmark, "_post", side_effect=replies) as post:
            answer = benchmark.ask_ghostswitch(args, "q")
        self.assertEqual((answer["text"], answer["missing_expert"], answer["route"]), ("Answer: B", "verilog_expert", "base"))
        self.assertEqual(post.call_args.args[1]["mode"], "base")
        with patch.object(benchmark, "_post", side_effect=replies[:1]):
            answer = benchmark.ask_ghostswitch(Namespace(**{**vars(args), "on_unsupported": "fail"}), "q")
        self.assertEqual((answer["text"], answer["status"]), ("", "unsupported"))


class RunTests(unittest.TestCase):
    def test_run_resumes_and_report_compares_runs(self):
        items = [{"id": str(i), "suite": "gsm8k", "kind": "number", "category": "", "gold": str(i), "prompt": f"q{i}"}
                 for i in range(5)]
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(benchmark, "load", return_value=items), \
             patch.object(benchmark, "ask_ghostswitch", side_effect=lambda args, prompt: {
                 "text": f"Answer: {prompt[1:]}", "status": "success", "expert": "reasoning_expert"}) as ask:
            args = Namespace(suite="gsm8k", name="mb", limit=3, offset=0, seed=0, target="ghostswitch",
                             concurrency=2, progress=10, results=folder)
            benchmark.run(args)
            benchmark.run(Namespace(**{**vars(args), "limit": 5}))
            self.assertEqual(ask.call_count, 5)
            rows = [json.loads(line) for line in (Path(folder) / "mb/gsm8k.jsonl").read_text().splitlines()]
            self.assertTrue(all(row["correct"] for row in rows))
            (Path(folder) / "base").mkdir()
            (Path(folder) / "base/gsm8k.jsonl").write_text("\n".join(json.dumps(
                {**row, "correct": row["id"] == "0", "expert": None}) for row in rows))
            with patch("builtins.print"):
                benchmark.report(Namespace(results=folder))
            report = (Path(folder) / "report.md").read_text()
        self.assertIn("| gsm8k | mb | 100.0% |", report)
        self.assertIn("| 5 | 100% | 0 |", report)
        self.assertIn("| gsm8k | base | mb | 0 | 4 |", report)


if __name__ == "__main__":
    unittest.main()
