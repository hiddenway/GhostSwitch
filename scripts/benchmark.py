"""Run MMLU-Pro, GSM8K and GPQA against GhostSwitch or any OpenAI-compatible server.

Items are drawn in a fixed seeded order, so `--offset` selects a disjoint slice: choose
adapters on one slice and report results on another. Answers are written to
results/<name>/<suite>.jsonl as they arrive, and an interrupted run resumes there.

    # GhostSwitch as shipped, and the same base model without experts or verifier:
    python scripts/benchmark.py run --suite gsm8k --limit 200 --name ghostswitch
    python scripts/benchmark.py run --suite gsm8k --limit 200 --name base-think \\
        --mode base --no-verify --thinking on
    # A LoRA loaded into llama-server with --lora-init-without-apply (adapter id 2):
    python scripts/benchmark.py run --suite gpqa_dev --limit 100 --name r3-full \\
        --target openai --url http://127.0.0.1:8080 --lora 2 --thinking on
    python scripts/benchmark.py report

MMLU-Pro and GSM8K read Parquet (pip install pyarrow). GPQA is gated: accept its terms on
huggingface.co/datasets/Idavidrein/gpqa and run `hf auth login` first.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DATASETS = {
    "mmlu_pro": ("TIGER-Lab/MMLU-Pro", "b189ec765aa7ed75c8acfea42df31fdae71f97be"),
    "gsm8k": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866"),
    "gpqa": ("Idavidrein/gpqa", "83022cefff930aea54f654c0b282e74b9eeda5c6"),
}
SUITES = ("mmlu_pro", "gsm8k", "gsm8k_dev", "gpqa", "gpqa_dev")
LETTERS = "ABCDEFGHIJ"
CHOICE_INSTRUCTION = ("Think it through, then give the final answer on the last line exactly as "
                      "'Answer: X', where X is the letter of the correct option.")
NUMBER_INSTRUCTION = ("Solve the problem. Give the final answer on the last line exactly as "
                      "'Answer: N', where N is a number.")


# --- Data ---------------------------------------------------------------------------


def _download(dataset: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    repo, revision = DATASETS[dataset]
    return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))


def _parquet(dataset: str, filename: str) -> list[dict]:
    try:
        import pyarrow.parquet as parquet
    except ImportError:
        raise SystemExit("MMLU-Pro and GSM8K need pyarrow: pip install pyarrow")
    return parquet.read_table(_download(dataset, filename)).to_pylist()


def _choice_prompt(question: str, options: list[str]) -> str:
    lines = "\n".join(f"{LETTERS[i]}. {option}" for i, option in enumerate(options))
    return f"{question.strip()}\n\nOptions:\n{lines}\n\n{CHOICE_INSTRUCTION}"


def _gpqa_rows(name: str) -> list[dict]:
    with _download("gpqa", name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _gpqa_items(rows: list[dict], suite: str, seed: int) -> list[dict]:
    items = []
    for row in rows:
        options = [row["Correct Answer"], row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        options = [option.strip() for option in options]
        # The option order is fixed per question and seed, so every system sees the same prompt.
        order = list(range(4))
        random.Random(f"{seed}:{row['Record ID']}").shuffle(order)
        shuffled = [options[i] for i in order]
        items.append({"id": row["Record ID"], "suite": suite, "kind": "choice", "choices": 4,
                      "gold": LETTERS[order.index(0)], "category": row.get("High-level domain") or "",
                      "prompt": _choice_prompt(row["Question"], shuffled)})
    return items


def load(suite: str, seed: int) -> list[dict]:
    if suite == "mmlu_pro":
        rows = _parquet("mmlu_pro", "data/test-00000-of-00001.parquet")
        return [{"id": str(row["question_id"]), "suite": suite, "kind": "choice", "choices": len(row["options"]),
                 "gold": row["answer"], "category": row["category"],
                 "prompt": _choice_prompt(row["question"], list(row["options"]))} for row in rows]
    if suite in ("gsm8k", "gsm8k_dev"):
        split = "test" if suite == "gsm8k" else "train"
        rows = _parquet("gsm8k", f"main/{split}-00000-of-00001.parquet")
        return [{"id": f"{split}-{i}", "suite": suite, "kind": "number", "category": "",
                 "gold": row["answer"].rsplit("####", 1)[1].strip().replace(",", ""),
                 "prompt": f"{row['question'].strip()}\n\n{NUMBER_INSTRUCTION}"} for i, row in enumerate(rows)]
    if suite == "gpqa":
        return _gpqa_items(_gpqa_rows("gpqa_diamond.csv"), suite, seed)
    if suite == "gpqa_dev":
        # Questions of the main set that are not in Diamond, for choosing adapters.
        diamond = {row["Record ID"] for row in _gpqa_rows("gpqa_diamond.csv")}
        return _gpqa_items([row for row in _gpqa_rows("gpqa_main.csv") if row["Record ID"] not in diamond], suite, seed)
    raise ValueError(f"Unknown suite {suite}")


def sample(items: list[dict], limit: int | None, offset: int, seed: int) -> list[dict]:
    """A seeded order, interleaved across categories so any prefix covers them evenly."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        groups[item["category"]].append(item)
    rng = random.Random(seed)
    for key in sorted(groups):
        rng.shuffle(groups[key])
    ordered, queues = [], [groups[key] for key in sorted(groups)]
    while any(queues):
        for queue in queues:
            if queue:
                ordered.append(queue.pop(0))
    return ordered[offset:offset + limit if limit else None]


# --- Answers ------------------------------------------------------------------------


def extract_choice(text: str, choices: int) -> str | None:
    valid = LETTERS[:choices]
    # Only the word "answer" ignores case: "the answer is a function" must not read as A.
    patterns = [rf"(?i:answer)\s*[:：]\s*\**\s*\(?\s*([{valid}])\b", rf"(?i:answer\s+is)\s*:?\s*\**\s*\(?([{valid}])\b",
                rf"\\boxed\{{\s*\(?([{valid}])\)?\s*\}}"]
    for pattern in patterns:
        found = re.findall(pattern, text)
        if found:
            return found[-1]
    last = text.strip().splitlines()[-1].strip() if text.strip() else ""
    match = re.fullmatch(rf"\**\(?([{valid}])\)?[.)]?\**", last)
    return match.group(1) if match else None


_NUMBER = r"-?\$?\s*\d[\d,]*(?:\.\d+)?"


def extract_number(text: str) -> str | None:
    for pattern in (rf"answer\s*[:：]\s*\**\s*({_NUMBER})", rf"\\boxed\{{\s*({_NUMBER})"):
        found = re.findall(pattern, text, re.IGNORECASE)
        if found:
            return re.sub(r"[\s$,]", "", found[-1])
    found = re.findall(_NUMBER, text)
    return re.sub(r"[\s$,]", "", found[-1]) if found else None


def correct(item: dict, prediction: str | None) -> bool:
    if prediction is None:
        return False
    if item["kind"] == "choice":
        return prediction == item["gold"]
    try:
        return math.isclose(float(prediction), float(item["gold"]), abs_tol=1e-6)
    except ValueError:
        return False


# --- Targets ------------------------------------------------------------------------


def _post(url: str, body: dict, timeout: float) -> tuple[int, dict]:
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        with error:
            try:
                return error.code, json.loads(error.read())
            except ValueError:
                return error.code, {}


def ask_ghostswitch(args: argparse.Namespace, prompt: str) -> dict:
    body = {"model": "ghostswitch-v6", "messages": [{"role": "user", "content": prompt}], "mode": args.mode,
            "max_tokens": args.max_tokens, "temperature": args.temperature}
    if args.mode == "direct":
        body["expert_id"] = args.expert
    if not args.verify:
        body["verify"] = False
    if args.thinking != "auto":
        body["reasoning_effort"] = "high" if args.thinking == "on" else "none"
    code, result = _post(args.url.rstrip("/") + "/v1/chat/completions", body, args.timeout)
    fallback = None
    if result.get("status") == "unsupported" and args.on_unsupported == "base":
        # The chat offers the base model when the chosen expert is not installed; do the same.
        fallback = (result.get("routing") or {}).get("missing_capability")
        retry = {key: value for key, value in body.items() if key != "expert_id"}
        code, result = _post(args.url.rstrip("/") + "/v1/chat/completions", {**retry, "mode": "base"}, args.timeout)
    info = result.get("ghostswitch") or {}
    choices = result.get("choices") or []
    return {"text": (choices[0]["message"].get("content") or "") if choices else "",
            "status": result.get("status") or f"http_{code}", "route": info.get("route"),
            "missing_expert": fallback,
            "expert": info.get("expert_id"), "attempts": (info.get("verification") or {}).get("attempts"),
            "thinking": (info.get("thinking") or {}).get("used"),
            "tokens": (result.get("usage") or {}).get("completion_tokens"),
            "error": (result.get("error") or {}).get("message")}


def ask_openai(args: argparse.Namespace, prompt: str) -> dict:
    body = {"model": args.model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens, "temperature": args.temperature}
    if args.thinking != "auto":
        body["chat_template_kwargs"] = {"enable_thinking": args.thinking == "on"}
    if args.lora is not None:
        # llama-server applies only the listed adapters; the others get scale 0.
        body["lora"] = [{"id": adapter, "scale": 1.0} for adapter in args.lora]
    code, result = _post(args.url.rstrip("/") + "/v1/chat/completions", body, args.timeout)
    choices = result.get("choices") or []
    message = choices[0]["message"] if choices else {}
    return {"text": message.get("content") or "", "status": "success" if code == 200 and choices else f"http_{code}",
            "thinking": bool(message.get("reasoning_content")), "tokens": (result.get("usage") or {}).get("completion_tokens"),
            "finish_reason": choices[0].get("finish_reason") if choices else None,
            "error": (result.get("error") or {}).get("message") if isinstance(result.get("error"), dict) else result.get("error")}


def run(args: argparse.Namespace) -> None:
    items = sample(load(args.suite, args.seed), args.limit, args.offset, args.seed)
    folder = Path(args.results) / args.name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{args.suite}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()}
    todo = [item for item in items if item["id"] not in done]
    settings = {key: value for key, value in vars(args).items() if key not in {"func"}}
    (folder / "settings.json").write_text(json.dumps(settings, indent=2))
    print(f"{args.name} · {args.suite}: {len(items)} items, {len(done)} already done, {len(todo)} to run", flush=True)
    ask = ask_ghostswitch if args.target == "ghostswitch" else ask_openai
    lock, right, finished = threading.Lock(), 0, 0
    started = time.monotonic()

    def one(item: dict) -> dict:
        begin = time.monotonic()
        try:
            answer = ask(args, item["prompt"])
        except Exception as exc:  # A failed request is a wrong answer, not a stopped run.
            answer = {"text": "", "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        text = answer["text"]
        prediction = extract_choice(text, item["choices"]) if item["kind"] == "choice" else extract_number(text)
        return {"id": item["id"], "suite": item["suite"], "category": item["category"], "gold": item["gold"],
                "prediction": prediction, "correct": correct(item, prediction),
                "latency_s": round(time.monotonic() - begin, 2), **answer}

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool, path.open("a", encoding="utf-8") as out:
        for future in as_completed([pool.submit(one, item) for item in todo]):
            row = future.result()
            with lock:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                finished += 1
                right += row["correct"]
                if finished % args.progress == 0 or finished == len(todo):
                    rate = (time.monotonic() - started) / finished
                    print(f"  {finished}/{len(todo)} · {right / finished:.1%} correct · "
                          f"{rate:.1f} s/item · about {rate * (len(todo) - finished) / 60:.0f} min left", flush=True)


# --- Report -------------------------------------------------------------------------


def wilson(right: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if not total:
        return 0.0, 0.0
    p = right / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return max(0.0, centre - spread), min(1.0, centre + spread)


def mcnemar(a: dict[str, bool], b: dict[str, bool]) -> tuple[int, int, float]:
    """Items only A got right, only B got right, and the exact two-sided p-value."""
    shared = a.keys() & b.keys()
    only_a = sum(a[i] and not b[i] for i in shared)
    only_b = sum(b[i] and not a[i] for i in shared)
    n, k = only_a + only_b, min(only_a, only_b)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return only_a, only_b, p


def report(args: argparse.Namespace) -> None:
    runs: dict[tuple[str, str], list[dict]] = {}
    for path in sorted(Path(args.results).glob("*/*.jsonl")):
        runs[(path.parent.name, path.stem)] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    lines = ["| Suite | Run | Accuracy | 95% CI | n | Answered | Missing expert → base | s/item |",
             "|---|---|---|---|---|---|---|---|"]
    for (name, suite), rows in sorted(runs.items(), key=lambda pair: (pair[0][1], pair[0][0])):
        right = sum(row["correct"] for row in rows)
        low, high = wilson(right, len(rows))
        answered = sum(row["prediction"] is not None for row in rows)
        latency = sum(row["latency_s"] for row in rows) / max(len(rows), 1)
        fallbacks = sum(bool(row.get("missing_expert")) for row in rows)
        lines.append(f"| {suite} | {name} | {right / max(len(rows), 1):.1%} | {low:.1%}–{high:.1%} | {len(rows)} | "
                     f"{answered / max(len(rows), 1):.0%} | {fallbacks} | {latency:.1f} |")
    lines += ["", "Paired comparison on the same items (McNemar): wins of each run and p-value.", "",
              "| Suite | A | B | A only | B only | p |", "|---|---|---|---|---|---|"]
    by_suite: dict[str, list[str]] = defaultdict(list)
    for name, suite in runs:
        by_suite[suite].append(name)
    for suite, names in sorted(by_suite.items()):
        for i, a in enumerate(sorted(names)):
            for b in sorted(names)[i + 1:]:
                only_a, only_b, p = mcnemar({row["id"]: row["correct"] for row in runs[(a, suite)]},
                                            {row["id"]: row["correct"] for row in runs[(b, suite)]})
                lines.append(f"| {suite} | {a} | {b} | {only_a} | {only_b} | {p:.3f} |")
    detail = []
    for (name, suite), rows in sorted(runs.items()):
        for field, title in (("category", "category"), ("expert", "GhostSwitch expert")):
            groups: dict[str, list[bool]] = defaultdict(list)
            for row in rows:
                if row.get(field) is not None and (field != "category" or row[field]):
                    groups[str(row[field] or "base")].append(row["correct"])
            if len(groups) > 1 or (field == "expert" and groups):
                detail.append(f"\n{name} · {suite} by {title}: " + ", ".join(
                    f"{key} {sum(values) / len(values):.0%} ({len(values)})" for key, values in sorted(groups.items())))
    text = "\n".join(lines + detail)
    print(text)
    Path(args.results, "report.md").write_text(text + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(required=True)
    runner = commands.add_parser("run", help="Ask one suite and record every answer")
    runner.add_argument("--suite", choices=SUITES, required=True)
    runner.add_argument("--name", required=True, help="Run name; results go to <results>/<name>/<suite>.jsonl")
    runner.add_argument("--limit", type=int, help="Number of items (default: all)")
    runner.add_argument("--offset", type=int, default=0, help="Skip this many items of the seeded order")
    runner.add_argument("--seed", type=int, default=0)
    runner.add_argument("--target", choices=("ghostswitch", "openai"), default="ghostswitch")
    runner.add_argument("--url", default="http://127.0.0.1:8000")
    runner.add_argument("--mode", choices=("auto", "base", "direct"), default="auto", help="GhostSwitch routing mode")
    runner.add_argument("--expert", help="Expert for --mode direct")
    runner.add_argument("--no-verify", dest="verify", action="store_false", help="GhostSwitch without its verifier")
    runner.add_argument("--thinking", choices=("auto", "on", "off"), default="auto")
    runner.add_argument("--on-unsupported", choices=("base", "fail"), default="base",
                        help="When routing picks an expert that is not installed: answer with the base model, or count it wrong")
    runner.add_argument("--model", default="default", help="Model name for --target openai")
    runner.add_argument("--lora", type=int, nargs="*", help="llama-server adapter ids to apply (--target openai)")
    runner.add_argument("--max-tokens", type=int, default=4096)
    runner.add_argument("--temperature", type=float, default=0.0)
    runner.add_argument("--concurrency", type=int, default=1)
    runner.add_argument("--timeout", type=float, default=1800)
    runner.add_argument("--progress", type=int, default=10, help="Print progress every N answers")
    runner.add_argument("--results", default="results")
    runner.set_defaults(func=run)
    reporter = commands.add_parser("report", help="Summarise every run in the results folder")
    reporter.add_argument("--results", default="results")
    reporter.set_defaults(func=report)
    args = parser.parse_args(argv)
    if getattr(args, "mode", None) == "direct" and not args.expert:
        parser.error("--mode direct needs --expert")
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
