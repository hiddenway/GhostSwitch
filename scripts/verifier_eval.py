"""Measure how often the text verifier accepts wrong answers and rejects right ones.

Each case in core/tests/data/verifier_cases.jsonl is a request, a candidate answer, and a
label: "wrong" answers should be retried, "correct" ones accepted. The script runs
core.verifier_v6.verify on every case and prints a confusion matrix per verifier version.
The server runs the exact checks only; this script turns the model reviewer on (review=True)
so it can still be measured:

    # The deterministic checks alone; the LLM reviewer is replaced by one that accepts everything:
    python scripts/verifier_eval.py --llm stub
    # The real reviewer (MLX). Its raw outputs are stored, so the baseline reuses them:
    python scripts/verifier_eval.py --llm live --baseline HEAD
    # The same comparison offline, from the stored reviewer outputs:
    python scripts/verifier_eval.py --llm replay --baseline HEAD

The reviewer runs at temperature 0 and its prompt does not depend on the deterministic checks,
so one stored output per prompt serves every verifier version, and LLM time is reported from
the stored timings. The clock line is fixed per case so prompts, and their stored outputs,
stay stable between runs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Callable
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import clock, llm  # noqa: E402

CASES = ROOT / "core" / "tests" / "data" / "verifier_cases.jsonl"
OUTPUTS = ROOT / "core" / "tests" / "data" / "verifier_llm_outputs.jsonl"
DEFAULT_NOW = "2026-10-07T12:00:00+03:00"
STUB_VERDICT = '{"status":"ACCEPT","issue":null,"feedback":null,"check":null}'
MAX_TOKENS = 512  # verifier_max_tokens of the v6 execution profiles


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_verifier(ref: str | None) -> ModuleType:
    """The working-tree verifier, or core/verifier_v6.py as it was at a git ref."""
    if ref is None:
        from core import verifier_v6
        return verifier_v6
    source = subprocess.run(["git", "show", f"{ref}:core/verifier_v6.py"], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout
    spec = importlib.util.spec_from_loader(f"core._verifier_{ref}", loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "core"  # its relative imports resolve to the current core package
    exec(compile(source, f"{ref}:core/verifier_v6.py", "exec"), module.__dict__)
    return module


def review_guidance(expert: str | None, backbone: str) -> str:
    if not expert:
        return ""
    policy = ROOT / "experts" / backbone / expert / "response_policy.json"
    if not policy.exists():
        return ""
    return json.loads(policy.read_text(encoding="utf-8")).get("review_guidance", "")


def fixed_clock(moment: str) -> type[datetime]:
    """A datetime whose now() is the given moment, so the prompt's clock line never changes."""
    fixed = datetime.fromisoformat(moment if "+" in moment[10:] else moment + DEFAULT_NOW[19:])

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: ANN001, ANN206
            return cls.fromisoformat(fixed.isoformat())

        def astimezone(self, tz=None):  # noqa: ANN001, ANN201
            return self

    return FixedDatetime


class Reviewer:
    """Stands in for llm.chat: a stub, the real model, or stored outputs of the real model."""

    def __init__(self, mode: str, outputs: Path, model: str) -> None:
        self.mode, self.path, self.model = mode, outputs, model
        self.stored: dict[str, dict[str, Any]] = {}
        if mode != "stub" and outputs.exists():
            for line in outputs.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    self.stored[record["key"]] = record
        self.calls = 0
        self.llm_ms = 0.0

    def key(self, messages: list[dict[str, Any]], kwargs: dict[str, Any]) -> str:
        body = {"model": self.model, "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
                "max_tokens": kwargs.get("max_tokens"), "temperature": kwargs.get("temperature"),
                "enable_thinking": kwargs.get("enable_thinking")}
        return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    def chat(self, messages: list[dict[str, Any]], *args: Any, **kwargs: Any) -> str:
        self.calls += 1
        debug = kwargs.get("debug")
        if self.mode == "stub":
            return STUB_VERDICT
        key = self.key(messages, kwargs)
        record = self.stored.get(key)
        if record is None:
            if self.mode == "replay":
                raise SystemExit(f"No stored reviewer output for this prompt ({key[:12]}); run with --llm live.")
            started = time.perf_counter()
            raw_debug: dict[str, Any] = {}
            raw = real_chat(messages, *args, **{**kwargs, "debug": raw_debug})
            record = {"key": key, "model": self.model, "raw": raw,
                      "finish_reason": raw_debug.get("finish_reason"),
                      "ms": round((time.perf_counter() - started) * 1000, 1)}
            self.stored[key] = record
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if debug is not None and record.get("finish_reason"):
            debug["finish_reason"] = record["finish_reason"]
        self.llm_ms += float(record["ms"])
        return record["raw"]


real_chat: Callable[..., str] = llm.chat


def run_case(verifier: ModuleType, reviewer: Reviewer, case: dict[str, Any], backbone: str) -> dict[str, Any]:
    calls, llm_ms = reviewer.calls, reviewer.llm_ms
    with ExitStack() as stack:
        stack.enter_context(patch.object(clock, "datetime", fixed_clock(case.get("now", DEFAULT_NOW))))
        stack.enter_context(patch.object(llm, "chat", reviewer.chat))
        started = time.perf_counter()
        # Verifiers before the reviewer was switched off have no review flag and always review.
        review = {"review": True} if "review" in inspect.signature(verifier.verify).parameters else {}
        verdict = verifier.verify(case["messages"], case["candidate"], finish_reason="stop", max_tokens=MAX_TOKENS,
                                  review_guidance=review_guidance(case.get("expert"), backbone),
                                  sources=case.get("sources"), **review)
        # verify() wall time; with a stub or stored outputs this is the server's own overhead.
        wall_ms = (time.perf_counter() - started) * 1000
    status = verdict["status"]
    accepted = status == "ACCEPT"
    wrong = case["label"] == "wrong"
    outcome = ("false_accept" if wrong and accepted else "true_retry" if wrong
               else "true_accept" if accepted else "false_retry")
    return {"id": case["id"], "category": case["category"], "label": case["label"], "status": status,
            "outcome": outcome, "feedback": verdict.get("feedback"), "override": verdict.get("override")
            or (verdict.get("check") or {}).get("override"),
            "llm_calls": reviewer.calls - calls, "llm_ms": round(reviewer.llm_ms - llm_ms, 1),
            "wall_ms": round(wall_ms, 2)}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["outcome"] for row in rows)
    wrong = counts["false_accept"] + counts["true_retry"]
    correct = counts["true_accept"] + counts["false_retry"]
    accepted = counts["false_accept"] + counts["true_accept"]
    return {
        **{name: counts[name] for name in ("false_accept", "true_retry", "true_accept", "false_retry")},
        "wrong": wrong, "correct": correct, "accepted": accepted,
        "false_accept_rate": counts["false_accept"] / wrong if wrong else 0.0,
        "false_retry_rate": counts["false_retry"] / correct if correct else 0.0,
        "wrong_among_accepted": counts["false_accept"] / accepted if accepted else 0.0,
        "fail": sum(1 for row in rows if row["status"] == "FAIL"),
        "llm_calls": sum(row["llm_calls"] for row in rows),
        "llm_seconds": round(sum(row["llm_ms"] for row in rows) / 1000, 1),
    }


def journal_summary(cases: list[dict[str, Any]]) -> dict[str, Any] | None:
    rows = [{"outcome": ("false_accept" if case["label"] == "wrong" and case["journal"]["verifier"] == "ACCEPT"
                         else "true_retry" if case["label"] == "wrong"
                         else "true_accept" if case["journal"]["verifier"] == "ACCEPT" else "false_retry"),
             "status": case["journal"]["verifier"], "llm_calls": 0, "llm_ms": 0}
            for case in cases if case.get("journal")]
    return summarize(rows) if rows else None


def percent(value: float) -> str:
    return f"{value * 100:.0f}%"


def print_report(results: dict[str, list[dict[str, Any]]], cases: list[dict[str, Any]], verbose: bool) -> None:
    names = list(results)
    summaries = {name: summarize(rows) for name, rows in results.items()}
    width = max(12, *(len(name) for name in names))
    print("\nConfusion matrix (label wrong = should be retried, label correct = should be accepted)\n")
    header = f"{'':40}" + "".join(f"{name:>{width + 2}}" for name in names)
    print(header)
    lines = [
        ("wrong answers accepted (false ACCEPT)", lambda s: f"{s['false_accept']}/{s['wrong']}"),
        ("wrong answers retried", lambda s: f"{s['true_retry']}/{s['wrong']}"),
        ("right answers accepted", lambda s: f"{s['true_accept']}/{s['correct']}"),
        ("right answers retried (false RETRY)", lambda s: f"{s['false_retry']}/{s['correct']}"),
        ("false ACCEPT rate", lambda s: percent(s["false_accept_rate"])),
        ("false RETRY rate", lambda s: percent(s["false_retry_rate"])),
        ("wrong among accepted", lambda s: f"{s['false_accept']}/{s['accepted']} ({percent(s['wrong_among_accepted'])})"),
        ("FAIL verdicts", lambda s: str(s["fail"])),
        ("reviewer LLM calls", lambda s: str(s["llm_calls"])),
        ("reviewer LLM time, s", lambda s: str(s["llm_seconds"])),
    ]
    for title, cell in lines:
        print(f"{title:40}" + "".join(f"{cell(summaries[name]):>{width + 2}}" for name in names))
    journal = journal_summary(cases)
    if journal:
        print(f"\nThe production verifier on the {journal['wrong'] + journal['correct']} journal cases "
              f"(various versions, as recorded): false ACCEPT {journal['false_accept']}/{journal['wrong']}, "
              f"false RETRY {journal['false_retry']}/{journal['correct']}.")

    print("\nBy category: false ACCEPT / wrong, false RETRY / correct\n")
    categories = sorted({row["category"] for rows in results.values() for row in rows})
    print(f"{'':16}" + "".join(f"{name:>{width + 2}}" for name in names))
    for category in categories:
        cells = []
        for name in names:
            summary = summarize([row for row in results[name] if row["category"] == category])
            cells.append(f"{summary['false_accept']}/{summary['wrong']}  {summary['false_retry']}/{summary['correct']}")
        print(f"{category:16}" + "".join(f"{cell:>{width + 2}}" for cell in cells))

    for name in names:
        errors = [row for row in results[name] if row["outcome"] in {"false_accept", "false_retry"}]
        print(f"\nErrors of {name}: {len(errors)}")
        for row in errors:
            detail = f" [{row['override']}]" if row.get("override") else ""
            feedback = f" — {row['feedback']}" if verbose and row.get("feedback") else ""
            print(f"  {row['outcome']:12} {row['id']}{detail}{feedback}")
    if len(names) == 2:
        before, after = (results[name] for name in names)
        changed = [(a, b) for a, b in zip(before, after) if a["outcome"] != b["outcome"]]
        print(f"\nChanged between {names[0]} and {names[1]}: {len(changed)}")
        for a, b in changed:
            print(f"  {a['id']}: {a['outcome']} -> {b['outcome']}"
                  + (f" [{b['override']}]" if b.get("override") else ""))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--llm", choices=("stub", "live", "replay"), default="stub",
                        help="stub: a reviewer that accepts everything; live: the MLX model; replay: stored outputs")
    parser.add_argument("--outputs", type=Path, default=OUTPUTS, help="Stored reviewer outputs (read and appended)")
    parser.add_argument("--baseline", metavar="GIT_REF",
                        help="Also evaluate core/verifier_v6.py from this git ref, for a before/after comparison")
    parser.add_argument("--backbone", default=llm.ACTIVE_BACKBONE, help="Backbone of the live reviewer")
    parser.add_argument("--model-path", help="Local MLX checkpoint of the backbone")
    parser.add_argument("--only", nargs="*", help="Case ids or categories to run")
    parser.add_argument("--json", type=Path, help="Write every verdict and the summaries to this file")
    parser.add_argument("--verbose", action="store_true", help="Show the feedback of each error")
    args = parser.parse_args(argv)

    cases = load_cases(args.cases)
    if args.only:
        cases = [case for case in cases if case["id"] in args.only or case["category"] in args.only]
    if args.llm == "live" and (args.model_path or args.backbone != llm.ACTIVE_BACKBONE):
        llm.configure_backbone(args.backbone, model_path=args.model_path)
    # Each backbone has one pinned checkpoint, so stored outputs are keyed by the backbone, not its path.
    reviewer = Reviewer(args.llm, args.outputs, args.backbone)
    versions: dict[str, ModuleType] = {}
    if args.baseline:
        versions[f"baseline {args.baseline}"] = load_verifier(args.baseline)
    versions["current"] = load_verifier(None)
    results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        for name, verifier in versions.items():
            results[name].append(run_case(verifier, reviewer, case, args.backbone))
    print(f"{len(cases)} cases ({sum(c['label'] == 'wrong' for c in cases)} wrong, "
          f"{sum(c['label'] == 'correct' for c in cases)} correct); reviewer: {args.llm}"
          + (f" ({reviewer.model})" if args.llm != "stub" else ""))
    print_report(dict(results), cases, args.verbose)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"reviewer": args.llm, "model": reviewer.model,
                                         "summary": {name: summarize(rows) for name, rows in results.items()},
                                         "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
