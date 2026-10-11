"""A mini benchmark of GhostSwitch as the model of the OpenCode coding agent.

Ten small Python tasks, each a folder with solution.py and test_solution.py: six bugs to find and
fix and four functions to write. OpenCode runs each task non-interactively in a Docker container
that sees only a fresh copy of the task folder: `opencode run` does not enforce OpenCode's
permission rules, and outside a container it treats the enclosing git repository as the project.
Afterwards the tests run again in a new container, and a task passes when they pass and
test_solution.py is unchanged.

Modes are the reasoning settings OpenCode can send through model variants:
  fast   reasoning_effort "none" on every step
  auto   nothing: GhostSwitch decides on each task's first step (its default)
  think  reasoning_effort "high" on every step

    docker build -t ghostswitch-agent-bench:1 -f scripts/agent_bench.Dockerfile scripts
    python scripts/agent_bench.py check                      # the tasks themselves are sound
    python scripts/agent_bench.py run --url http://127.0.0.1:8000 --name first
    python scripts/agent_bench.py report --name first

The server must already run with the model to measure. Every step goes to one expert (--expert,
coding_expert by default) through OpenCode model options, so routing does not vary between modes;
--expert auto leaves the choice to the router. Results go to results/agent-bench/<name>/, and an
interrupted run resumes there. Step counts and reasoning come from the server's request log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODES = ("fast", "auto", "think")
CHECK = "python3 -m unittest -q"
AGENT_PROMPT = (
    "You are a coding agent. Read relevant files before editing. Use tools to carry out the user's task, "
    "run the relevant tests, and report the result concisely. Never claim to have executed a command "
    "without its tool result. Work within the current project."
)

TASKS: dict[str, dict] = {
    "mean_empty": {
        "prompt": f"mean() в solution.py падает на пустом списке. Пусть для пустого списка возвращает 0.0. Тесты: {CHECK}",
        "solution": '''def mean(values):
    """Average of the numbers in values."""
    return sum(values) / len(values)
''',
        "reference": '''def mean(values):
    """Average of the numbers in values."""
    return sum(values) / len(values) if values else 0.0
''',
        "tests": '''from solution import mean


class Tests(unittest.TestCase):
    def test_average(self):
        self.assertEqual(mean([1, 2, 3]), 2)
        self.assertEqual(mean([2.5]), 2.5)
        self.assertEqual(mean([-1, 1]), 0)

    def test_empty(self):
        self.assertEqual(mean([]), 0.0)
''',
    },
    "last_n": {
        "prompt": f"last_n в solution.py возвращает лишний элемент. Найди и исправь ошибку. Тесты: {CHECK}",
        "solution": '''def last_n(items, n):
    """The last n items of the list, in order; all of them when n is larger than the list."""
    if n <= 0:
        return []
    return items[-n - 1:]
''',
        "reference": '''def last_n(items, n):
    """The last n items of the list, in order; all of them when n is larger than the list."""
    if n <= 0:
        return []
    return items[-n:]
''',
        "tests": '''from solution import last_n


class Tests(unittest.TestCase):
    def test_last_items(self):
        self.assertEqual(last_n([1, 2, 3, 4], 2), [3, 4])
        self.assertEqual(last_n([1, 2, 3, 4], 1), [4])

    def test_edges(self):
        self.assertEqual(last_n([1, 2], 5), [1, 2])
        self.assertEqual(last_n([1, 2, 3], 0), [])
        self.assertEqual(last_n([], 3), [])
''',
    },
    "roman": {
        "prompt": f"Реализуй roman_to_int в solution.py по докстрингу. Тесты: {CHECK}",
        "solution": '''def roman_to_int(numeral):
    """Convert a Roman numeral such as 'XIV' to an integer; raise ValueError for invalid input."""
    raise NotImplementedError
''',
        "reference": '''def roman_to_int(numeral):
    """Convert a Roman numeral such as 'XIV' to an integer; raise ValueError for invalid input."""
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    if not numeral or any(char not in values for char in numeral):
        raise ValueError(numeral)
    total = 0
    for char, following in zip(numeral, numeral[1:] + " "):
        value = values[char]
        total += -value if following in values and values[following] > value else value
    return total
''',
        "tests": '''from solution import roman_to_int


class Tests(unittest.TestCase):
    def test_numerals(self):
        for numeral, value in (("III", 3), ("IV", 4), ("IX", 9), ("LVIII", 58), ("MCMXCIV", 1994)):
            self.assertEqual(roman_to_int(numeral), value)

    def test_invalid(self):
        for numeral in ("", "ABC", "12"):
            with self.assertRaises(ValueError):
                roman_to_int(numeral)
''',
    },
    "merge_intervals": {
        "prompt": f"Напиши merge_intervals в solution.py: объединяет пересекающиеся и соприкасающиеся интервалы. Тесты: {CHECK}",
        "solution": '''def merge_intervals(intervals):
    """Merge overlapping or touching [start, end] intervals; return them sorted by start."""
    raise NotImplementedError
''',
        "reference": '''def merge_intervals(intervals):
    """Merge overlapping or touching [start, end] intervals; return them sorted by start."""
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged
''',
        "tests": '''from solution import merge_intervals


class Tests(unittest.TestCase):
    def test_merge(self):
        self.assertEqual(merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]), [[1, 6], [8, 10], [15, 18]])
        self.assertEqual(merge_intervals([[1, 10], [2, 3]]), [[1, 10]])

    def test_touching_unsorted_and_empty(self):
        self.assertEqual(merge_intervals([[1, 4], [4, 5]]), [[1, 5]])
        self.assertEqual(merge_intervals([[5, 7], [1, 2]]), [[1, 2], [5, 7]])
        self.assertEqual(merge_intervals([]), [])
''',
    },
    "binary_search": {
        "prompt": f"binary_search в solution.py иногда не находит элементы, которые есть в списке. Найди баг и почини. Тесты: {CHECK}",
        "solution": '''def binary_search(items, target):
    """Index of target in the sorted list items, or -1 when it is absent."""
    lo, hi = 0, len(items) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if items[mid] == target:
            return mid
        if items[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
''',
        "reference": '''def binary_search(items, target):
    """Index of target in the sorted list items, or -1 when it is absent."""
    lo, hi = 0, len(items) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if items[mid] == target:
            return mid
        if items[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
''',
        "tests": '''from solution import binary_search


class Tests(unittest.TestCase):
    def test_found(self):
        items = [1, 3, 5, 7, 9]
        for index, value in enumerate(items):
            self.assertEqual(binary_search(items, value), index)
        self.assertEqual(binary_search([2], 2), 0)

    def test_absent(self):
        self.assertEqual(binary_search([1, 3, 5], 4), -1)
        self.assertEqual(binary_search([], 1), -1)
''',
    },
    "lru_cache": {
        "prompt": f"Реализуй LRUCache в solution.py по докстрингу (get и put за O(1)). Тесты: {CHECK}",
        "solution": '''class LRUCache:
    """A cache that holds at most `capacity` keys and evicts the least recently used one.

    get(key) returns the value, or -1 when the key is absent; get and put both count as a use.
    """

    def __init__(self, capacity):
        raise NotImplementedError

    def get(self, key):
        raise NotImplementedError

    def put(self, key, value):
        raise NotImplementedError
''',
        "reference": '''from collections import OrderedDict


class LRUCache:
    """A cache that holds at most `capacity` keys and evicts the least recently used one.

    get(key) returns the value, or -1 when the key is absent; get and put both count as a use.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.items = OrderedDict()

    def get(self, key):
        if key not in self.items:
            return -1
        self.items.move_to_end(key)
        return self.items[key]

    def put(self, key, value):
        self.items[key] = value
        self.items.move_to_end(key)
        if len(self.items) > self.capacity:
            self.items.popitem(last=False)
''',
        "tests": '''from solution import LRUCache


class Tests(unittest.TestCase):
    def test_eviction(self):
        cache = LRUCache(2)
        cache.put(1, 1)
        cache.put(2, 2)
        self.assertEqual(cache.get(1), 1)
        cache.put(3, 3)
        self.assertEqual(cache.get(2), -1)
        cache.put(4, 4)
        self.assertEqual(cache.get(1), -1)
        self.assertEqual(cache.get(3), 3)
        self.assertEqual(cache.get(4), 4)

    def test_update_counts_as_use(self):
        cache = LRUCache(2)
        cache.put(1, 1)
        cache.put(2, 2)
        cache.put(1, 10)
        cache.put(3, 3)
        self.assertEqual(cache.get(2), -1)
        self.assertEqual(cache.get(1), 10)
''',
    },
    "parse_duration": {
        "prompt": f"Тест на parse_duration падает: '1h30m' даёт 1860 вместо 5400. Разберись и почини. Тесты: {CHECK}",
        "solution": '''import re


def parse_duration(text):
    """Seconds in a duration such as '1h30m', '45s' or '2h5m10s'."""
    total = 0
    for amount, unit in re.findall(r"(\\d+)([hms])", text):
        total += int(amount) * {"h": 60, "m": 60, "s": 1}[unit]
    return total
''',
        "reference": '''import re


def parse_duration(text):
    """Seconds in a duration such as '1h30m', '45s' or '2h5m10s'."""
    total = 0
    for amount, unit in re.findall(r"(\\d+)([hms])", text):
        total += int(amount) * {"h": 3600, "m": 60, "s": 1}[unit]
    return total
''',
        "tests": '''from solution import parse_duration


class Tests(unittest.TestCase):
    def test_durations(self):
        self.assertEqual(parse_duration("1h30m"), 5400)
        self.assertEqual(parse_duration("45s"), 45)
        self.assertEqual(parse_duration("2h5m10s"), 7510)
        self.assertEqual(parse_duration(""), 0)
''',
    },
    "monthly_payment": {
        "prompt": f"Напиши monthly_payment в solution.py: аннуитетный платёж по кредиту, с учётом ставки 0%. Тесты: {CHECK}",
        "solution": '''def monthly_payment(principal, annual_rate_percent, years):
    """Monthly annuity payment, rounded to cents; a 0% rate splits the principal evenly."""
    raise NotImplementedError
''',
        "reference": '''def monthly_payment(principal, annual_rate_percent, years):
    """Monthly annuity payment, rounded to cents; a 0% rate splits the principal evenly."""
    months = years * 12
    rate = annual_rate_percent / 100 / 12
    if rate == 0:
        return round(principal / months, 2)
    return round(principal * rate / (1 - (1 + rate) ** -months), 2)
''',
        "tests": '''from solution import monthly_payment


class Tests(unittest.TestCase):
    def test_payments(self):
        self.assertAlmostEqual(monthly_payment(1_000_000, 12, 5), 22244.45, places=2)
        self.assertAlmostEqual(monthly_payment(100_000, 6, 30), 599.55, places=2)

    def test_zero_rate(self):
        self.assertAlmostEqual(monthly_payment(120_000, 0, 10), 1000.0, places=2)
''',
    },
    "top_k_words": {
        "prompt": f"top_k_words в solution.py при равной частоте должна сортировать слова по алфавиту, а сейчас нет. Почини. Тесты: {CHECK}",
        "solution": '''from collections import Counter


def top_k_words(text, k):
    """The k most frequent words (case-insensitive); ties go alphabetically."""
    counts = Counter(text.lower().split())
    return [word for word, _ in counts.most_common(k)]
''',
        "reference": '''from collections import Counter


def top_k_words(text, k):
    """The k most frequent words (case-insensitive); ties go alphabetically."""
    counts = Counter(text.lower().split())
    return [word for word, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:k]]
''',
        "tests": '''from solution import top_k_words


class Tests(unittest.TestCase):
    def test_ties_go_alphabetically(self):
        self.assertEqual(top_k_words("b a c a b", 2), ["a", "b"])
        self.assertEqual(top_k_words("z y x", 3), ["x", "y", "z"])

    def test_counts_and_case(self):
        self.assertEqual(top_k_words("the cat the dog", 1), ["the"])
        self.assertEqual(top_k_words("Apple apple banana", 2), ["apple", "banana"])
''',
    },
    "brackets": {
        "prompt": f"is_balanced в solution.py считает '(]' и '([)]' правильными. Исправь, чтобы учитывались типы и порядок скобок. Тесты: {CHECK}",
        "solution": '''def is_balanced(text):
    """True when every bracket in text, (), [] and {}, is closed in the right order."""
    depth = 0
    for char in text:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0
''',
        "reference": '''def is_balanced(text):
    """True when every bracket in text, (), [] and {}, is closed in the right order."""
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for char in text:
        if char in "([{":
            stack.append(char)
        elif char in pairs:
            if not stack or stack.pop() != pairs[char]:
                return False
    return not stack
''',
        "tests": '''from solution import is_balanced


class Tests(unittest.TestCase):
    def test_balanced(self):
        for text in ("([]{})", "", "a(b)c", "{[()]}"):
            self.assertTrue(is_balanced(text), text)

    def test_unbalanced(self):
        for text in ("(]", "([)]", "((", ")("):
            self.assertFalse(is_balanced(text), text)
''',
    },
}


def write_task(folder: Path, task: dict, solution_key: str = "solution") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "solution.py").write_text(task[solution_key], encoding="utf-8")
    (folder / "test_solution.py").write_text("import unittest\n\n" + task["tests"]
                                             + '\n\nif __name__ == "__main__":\n    unittest.main()\n',
                                             encoding="utf-8")


def tests_pass(folder: Path, image: str | None = None) -> bool:
    """Whether the folder's tests pass; with an image, inside a container, since an agent wrote the code."""
    command = [sys.executable, "-m", "unittest", "-q"]
    if image:
        command = ["docker", "run", "--rm", "-v", f"{folder}:/task", "-w", "/task", image, "python", "-m", "unittest", "-q"]
    try:
        result = subprocess.run(command, cwd=folder, capture_output=True, timeout=120)
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


def check(_: argparse.Namespace) -> int:
    """Each task's starting code fails its tests and its reference solution passes them."""
    bad = 0
    with tempfile.TemporaryDirectory() as temp:
        for name, task in TASKS.items():
            start, reference = Path(temp) / name / "start", Path(temp) / name / "reference"
            write_task(start, task)
            write_task(reference, task, "reference")
            fails, passes = not tests_pass(start), tests_pass(reference)
            bad += not (fails and passes)
            print(f"{name:16} starting code fails: {fails}  reference passes: {passes}")
    return 1 if bad else 0


def opencode_config(url: str, expert: str) -> dict:
    """OpenCode settings for the benchmark: GhostSwitch, two variants, and narrow permissions."""
    return {
        "$schema": "https://opencode.ai/config.json",
        "provider": {"ghostswitch": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "GhostSwitch",
            "options": {"baseURL": url.rstrip("/") + "/v1", "apiKey": "local", "timeout": 900000,
                        "headers": {"X-GhostSwitch-Client": "OpenCode agent bench"}},
            "models": {"ghostswitch-v6": {
                "name": "GhostSwitch v6", "tool_call": True, "reasoning": False,
                "limit": {"context": 100000, "output": 4096},
                # OpenCode sends model options as request fields.
                **({} if expert == "auto" else {"options": {"mode": "direct", "expert_id": expert}}),
                "variants": {"think": {"reasoningEffort": "high"}, "fast": {"reasoningEffort": "none"}},
            }},
        }},
        "model": "ghostswitch/ghostswitch-v6",
        "small_model": "ghostswitch/ghostswitch-v6",
        "agent": {
            "bench": {"mode": "primary", "model": "ghostswitch/ghostswitch-v6", "temperature": 0,
                      "prompt": AGENT_PROMPT,
                      "tools": {"*": False, "read": True, "edit": True, "write": True, "bash": True,
                                "glob": True, "grep": True}},
            "title": {"disable": True},
            "summary": {"disable": True},
        },
        # The agent may read and edit its task folder and run the tests there, nothing else.
        "permission": {
            "*": "deny", "read": "allow", "glob": "allow", "grep": "allow", "edit": "allow",
            "bash": {"*": "deny", "python3 -m unittest*": "allow", "ls*": "allow"},
            "external_directory": "deny", "doom_loop": "deny",
        },
    }


def server_requests(url: str) -> list[dict]:
    with urllib.request.urlopen(url.rstrip("/") + "/v1/requests", timeout=30) as response:
        value = json.loads(response.read())
    return value if isinstance(value, list) else value.get("requests") or value.get("data") or []


def step_stats(url: str, started_ms: int, ended_ms: int) -> dict:
    """Requests the server logged for one task: steps, steps that reasoned, generation time."""
    steps = reasoned = 0
    generation_ms = 0.0
    for entry in server_requests(url):
        date = int(entry.get("date") or 0)
        if not started_ms <= date <= ended_ms or "bench" not in str(entry.get("source", "")):
            continue
        steps += 1
        info = (entry.get("response") or {}).get("ghostswitch") or {}
        reasoned += bool((info.get("thinking") or {}).get("used"))
        generation_ms += sum(float(stage.get("latency_ms") or 0) for stage in info.get("stages") or []
                             if stage.get("phase") == "generation")
    return {"steps": steps, "reasoned_steps": reasoned, "generation_s": round(generation_ms / 1000, 1)}


def run(args: argparse.Namespace) -> int:
    folder = ROOT / "results" / "agent-bench" / args.name
    folder.mkdir(parents=True, exist_ok=True)
    rows_path = folder / "runs.jsonl"
    done = set()
    if rows_path.exists():
        done = {(row["mode"], row["task"], row["repeat"]) for row in map(json.loads, rows_path.read_text().splitlines())}
    config_home = folder / "opencode-config"
    (config_home / "opencode").mkdir(parents=True, exist_ok=True)
    # The agent reaches the server on this computer from inside its container.
    agent_url = args.url.replace("127.0.0.1", "host.docker.internal").replace("localhost", "host.docker.internal")
    (config_home / "opencode" / "opencode.json").write_text(json.dumps(opencode_config(agent_url, args.expert), indent=2))
    tasks = args.tasks or list(TASKS)
    # Modes alternate inside each task, so a stopped run still compares them on the same tasks.
    for repeat in range(1, args.repeats + 1):
        for name in tasks:
            for mode in args.modes:
                if (mode, name, repeat) in done:
                    continue
                work = folder / "work" / f"{mode}-{name}-{repeat}"
                shutil.rmtree(work, ignore_errors=True)
                write_task(work, TASKS[name])
                tests_hash = hashlib.sha256((work / "test_solution.py").read_bytes()).hexdigest()
                container = f"agent-bench-{args.name}-{mode}-{name}-{repeat}"
                command = ["docker", "run", "--rm", "--name", container,
                           "-v", f"{work}:/task", "-v", f"{config_home / 'opencode' / 'opencode.json'}:/bench/opencode.json:ro",
                           "-e", "OPENCODE_CONFIG=/bench/opencode.json",
                           # OpenCode's provider packages and sessions persist between tasks.
                           "-v", "ghostswitch-agent-bench-cache:/root/.cache",
                           "-v", "ghostswitch-agent-bench-data:/root/.local/share",
                           "-w", "/task", args.image,
                           "opencode", "run", TASKS[name]["prompt"], "-m", "ghostswitch/ghostswitch-v6", "--agent", "bench"]
                if mode != "auto":
                    command += ["--variant", mode]
                started = time.time()
                timed_out = False
                with (work.parent / f"{mode}-{name}-{repeat}.log").open("w") as log:
                    try:
                        exit_code = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=log,
                                                   stderr=subprocess.STDOUT, timeout=args.timeout).returncode
                    except subprocess.TimeoutExpired:
                        subprocess.run(["docker", "kill", container], capture_output=True)
                        exit_code, timed_out = None, True
                ended = time.time()
                tests_unchanged = hashlib.sha256((work / "test_solution.py").read_bytes()).hexdigest() == tests_hash
                row = {"mode": mode, "task": name, "repeat": repeat,
                       "passed": tests_unchanged and tests_pass(work, args.image), "tests_unchanged": tests_unchanged,
                       "seconds": round(ended - started, 1), "timed_out": timed_out, "exit_code": exit_code,
                       **step_stats(args.url, int(started * 1000) - 1000, int(ended * 1000) + 1000)}
                with rows_path.open("a") as out:
                    out.write(json.dumps(row) + "\n")
                print(f"{mode:5} {name:16} {'PASS' if row['passed'] else 'fail'}  {row['seconds']:6.0f} s  "
                      f"{row['steps']} steps, {row['reasoned_steps']} reasoned", flush=True)
    return 0


def report(args: argparse.Namespace) -> int:
    rows = [json.loads(line) for line in (ROOT / "results" / "agent-bench" / args.name / "runs.jsonl").read_text().splitlines()]
    modes = [mode for mode in MODES if any(row["mode"] == mode for row in rows)]
    print("| Mode | Passed | Median time per task | Total time | Steps | Steps that reasoned |")
    print("|---|---:|---:|---:|---:|---:|")
    for mode in modes:
        mine = [row for row in rows if row["mode"] == mode]
        print(f"| {mode} | {sum(row['passed'] for row in mine)}/{len(mine)} | "
              f"{statistics.median(row['seconds'] for row in mine):.0f} s | {sum(row['seconds'] for row in mine) / 60:.1f} min | "
              f"{sum(row['steps'] for row in mine)} | {sum(row['reasoned_steps'] for row in mine)} |")
    print("\n| Task | " + " | ".join(modes) + " |")
    print("|---|" + "---|" * len(modes))
    for name in TASKS:
        cells = []
        for mode in modes:
            mine = [row for row in rows if row["task"] == name and row["mode"] == mode]
            cells.append(", ".join(f"{'pass' if row['passed'] else 'fail'} {row['seconds']:.0f} s"
                                   f"{' ⏱' if row['timed_out'] else ''}" for row in mine) or "—")
        print(f"| {name} | " + " | ".join(cells) + " |")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="check that each task fails before and passes with its reference solution")
    runner = commands.add_parser("run", help="run the tasks through OpenCode")
    runner.add_argument("--url", default="http://127.0.0.1:27246", help="GhostSwitch server")
    runner.add_argument("--name", default="agent-bench")
    runner.add_argument("--expert", default="coding_expert", help="expert for every step, or auto for the router")
    runner.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    runner.add_argument("--tasks", nargs="+", choices=list(TASKS))
    runner.add_argument("--repeats", type=int, default=1)
    runner.add_argument("--timeout", type=int, default=900, help="seconds per task")
    runner.add_argument("--image", default="ghostswitch-agent-bench:1", help="Docker image with OpenCode")
    reporter = commands.add_parser("report", help="summarize a run")
    reporter.add_argument("--name", default="agent-bench")
    args = parser.parse_args(argv)
    return {"check": check, "run": run, "report": report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
