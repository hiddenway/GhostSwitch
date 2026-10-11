"""Stage 1: does each expert beat the base model on its own domain?

For every expert, the same questions from a public test set go to GhostSwitch twice: once to
the expert (`direct` mode) and once to the base model (`base` mode). The verifier is off and
thinking is the same for both, so the only difference is the adapter and its prompt. Each
answer is scored automatically; a paired sign test on the per-question scores decides the
verdict, which can be written to the expert's evaluation.json.

    python scripts/expert_eval.py run --experts reasoning_expert finance_market_expert --limit 50
    python scripts/expert_eval.py report            # verdicts and suggested evaluation.json
    python scripts/expert_eval.py report --write    # write evaluation.json for every measured expert

Datasets are read through the Hugging Face datasets-server API and cached under
~/.cache/minibrain/expert-eval. Code and SQL answers are executed: run this on a disposable
server, not on a workstation.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import random
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import benchmark  # noqa: E402

CACHE = Path.home() / ".cache" / "minibrain" / "expert-eval"
ROWS_API = "https://datasets-server.huggingface.co/rows"


# --- Data ---------------------------------------------------------------------------


def _get(url: str, attempts: int = 8) -> dict:
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except Exception as error:
            if attempt == attempts - 1:
                raise
            # The datasets-server rate-limits bursts (HTTP 429); wait as long as it asks.
            retry_after = getattr(error, "headers", None) and error.headers.get("Retry-After")
            time.sleep(float(retry_after) if retry_after and str(retry_after).isdigit() else 5 * (attempt + 1))
    raise RuntimeError("unreachable")


def rows(dataset: str, config: str, split: str, limit: int, seed: int) -> list[dict]:
    """A seeded random sample of rows, cached on disk."""
    cache = CACHE / dataset.replace("/", "__") / f"{config}__{split}"
    cache.mkdir(parents=True, exist_ok=True)
    meta = cache / "total.json"
    if meta.exists():
        total = json.loads(meta.read_text())
    else:
        query = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": 0, "length": 1})
        total = _get(f"{ROWS_API}?{query}")["num_rows_total"]
        meta.write_text(json.dumps(total))
    indices = sorted(random.Random(f"{seed}:{dataset}:{config}").sample(range(total), min(limit, total)))
    # Fetch whole pages of 100 rows, so a sample costs a few requests instead of one per row.
    for start in sorted({index // 100 * 100 for index in indices if not (cache / f"{index}.json").exists()}):
        query = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                        "offset": start, "length": 100})
        for row in _get(f"{ROWS_API}?{query}")["rows"]:
            (cache / f"{row['row_idx']}.json").write_text(json.dumps(row["row"], ensure_ascii=False))
    return [{"_index": index, **json.loads((cache / f"{index}.json").read_text())} for index in indices]


def _choices(question: str, options: list[str]) -> str:
    return benchmark._choice_prompt(question, options)


def task_gsm8k(limit, seed):
    items = benchmark.sample(benchmark.load("gsm8k", seed), limit, 0, seed)
    return [{**item, "kind": "number"} for item in items]


def task_finqa(limit, seed):
    items = []
    for row in rows("dreamerdeo/finqa", "default", "test", limit, seed):
        table = "\n".join(" | ".join(cell for cell in line) for line in row["table"])
        context = " ".join(row["pre_text"])[-2500:] + "\n\nTable:\n" + table + "\n\n" + " ".join(row["post_text"])[:1500]
        items.append({"id": row["id"], "kind": "number_tolerant", "gold": row["answer"],
                      "prompt": f"{context}\n\nQuestion: {row['question']}\n\n{benchmark.NUMBER_INSTRUCTION}"})
    return items


def task_medqa(limit, seed):
    items = []
    for row in rows("GBaker/MedQA-USMLE-4-options", "default", "test", limit, seed):
        options = [row["options"][letter] for letter in "ABCD"]
        items.append({"id": str(row["_index"]), "kind": "choice", "choices": 4, "gold": row["answer_idx"],
                      "prompt": _choices(row["question"], options)})
    return items


CUAD_TASKS = ("cuad_audit_rights", "cuad_non-compete", "cuad_exclusivity", "cuad_cap_on_liability",
              "cuad_termination_for_convenience")


def task_cuad(limit, seed):
    items = []
    for config in CUAD_TASKS:
        clause = config.removeprefix("cuad_").replace("_", " ").replace("-", " ")
        for row in rows("nguha/legalbench", config, "test", math.ceil(limit / len(CUAD_TASKS)), seed):
            items.append({"id": f"{config}-{row['_index']}", "kind": "yes_no", "gold": row["answer"],
                          "prompt": f"Contract clause:\n{row['text']}\n\nDoes this clause concern {clause}? "
                                    "Give the final answer on the last line exactly as 'Answer: Yes' or 'Answer: No'."})
    return items[:limit]


def task_cti_rcm(limit, seed):
    return [{"id": row["URL"].rsplit("/", 1)[-1], "kind": "cwe", "gold": row["GT"], "prompt": row["Prompt"],
             "description": row["Description"]}
            for row in rows("AI4Sec/cti-bench", "cti-rcm", "test", limit, seed)]


def task_pentest(limit, seed):
    return [{"id": str(row["_index"]), "kind": "choice", "choices": len(row["choices"]),
             "gold": benchmark.LETTERS[int(row["answer"])], "prompt": _choices(row["question"], row["choices"])}
            for row in rows("preemware/pentesting-eval", "default", "train", limit, seed)]


def task_terraform(limit, seed):
    items = []
    for row in rows("rootly-ai-labs/terraform-s3-security-mcq", "default", "test", limit, seed):
        text = "\n\n".join(message["content"] for message in row["input"])
        items.append({"id": str(row["_index"]), "kind": "choice", "choices": 4, "gold": row["ideal"].strip(),
                      "prompt": text + "\n\nGive the final answer on the last line exactly as 'Answer: X'."})
    return items


def task_pii(limit, seed):
    items = []
    for row in rows("gretelai/gretel-pii-masking-en-v1", "default", "test", limit, seed):
        entities = [entity["entity"] for entity in ast.literal_eval(row["entities"])]
        items.append({"id": row["uid"], "kind": "string_set", "gold": entities,
                      "prompt": f"Text:\n{row['text']}\n\nList every piece of personal or sensitive information "
                                "in the text, exactly as written, as a JSON array of strings."})
    return items


# tner/conll2003 label ids; MISC is left out, since the prompt asks for people, organizations and places.
CONLL_BEGIN = {1: "ORG", 3: "PER", 5: "LOC"}
CONLL_INSIDE = {6: "ORG", 4: "PER", 8: "LOC"}


def conll_entities(tokens: list[str], tags: list[int]) -> list[str]:
    entities, current, kind = [], [], None
    for token, tag in zip(tokens, tags):
        if tag in CONLL_INSIDE and current and CONLL_INSIDE[tag] == kind:
            current.append(token)
            continue
        if current:
            entities.append(" ".join(current))
        current, kind = ([token], CONLL_BEGIN[tag]) if tag in CONLL_BEGIN else ([], None)
    if current:
        entities.append(" ".join(current))
    return entities


def task_conll(limit, seed):
    items = []
    for row in rows("tner/conll2003", "conll2003", "test", limit * 3, seed):
        entities = conll_entities(row["tokens"], row["tags"])
        if entities and len(items) < limit:
            sentence = " ".join(row["tokens"])
            items.append({"id": str(row["_index"]), "kind": "string_set", "gold": entities,
                          "prompt": f"Text:\n{sentence}\n\nList the names of every person, organization and "
                                    "place in the text, exactly as written, as a JSON array of strings."})
    return items


def task_sql(limit, seed):
    return [{"id": str(row["id"]), "kind": "sql", "gold": row["sql"], "context": row["sql_context"],
             "prompt": f"Database schema and data:\n{row['sql_context']}\n\nWrite one SQLite query that answers: "
                       f"{row['sql_prompt']}\nReturn only the SQL in a ```sql block."}
            for row in rows("gretelai/synthetic_text_to_sql", "default", "test", limit, seed)]


def task_humaneval(limit, seed):
    return [{"id": row["task_id"], "kind": "python", "test": row["test"], "entry_point": row["entry_point"],
             "gold": None, "prompt": "Complete the following Python function. Return the complete function, "
                                     f"with any imports it needs, in one ```python block.\n\n{row['prompt']}"}
            for row in rows("evalplus/humanevalplus", "default", "test", limit, seed)]


# Expert → (task, thinking for both systems). Experts not listed need a judge or tools
# that this script does not install (Verilog simulation, Lean, OpenSCAD, CAD kernels).
EXPERT_TASKS = {
    "reasoning_expert": (task_gsm8k, "on"),
    "finance_market_expert": (task_finqa, "off"),
    "medical_reasoning_expert": (task_medqa, "off"),
    "legal_contract_expert": (task_cuad, "off"),
    "cybersecurity_expert": (task_cti_rcm, "off"),
    "penetration_testing_expert": (task_pentest, "off"),
    "iac_security_expert": (task_terraform, "off"),
    "privacy_dlp_expert": (task_pii, "off"),
    "entity_extraction_expert": (task_conll, "off"),
    "text_to_sql_expert": (task_sql, "off"),
    "coding_expert": (task_humaneval, "off"),
    "software_engineering_expert": (task_humaneval, "off"),
    "coding_delta_expert": (task_humaneval, "off"),
}


# --- Quick runs ---------------------------------------------------------------------

# A quick run asks for the answer line alone where no working is needed, and caps each kind
# of answer, so all experts are measured in about half an hour on one RTX 3090.
QUICK_TOKENS = {"choice": 32, "yes_no": 32, "cwe": 32, "string_set": 256, "sql": 256,
                "python": 768, "number": 768, "number_tolerant": 768}
_CUAD_INSTRUCTION = "Give the final answer on the last line exactly as 'Answer: Yes' or 'Answer: No'."
_TERRAFORM_INSTRUCTION = "\n\nGive the final answer on the last line exactly as 'Answer: X'."


def quick(item: dict) -> dict:
    prompt = item["prompt"]
    if item["kind"] == "choice":
        prompt = (prompt.replace(benchmark.CHOICE_INSTRUCTION, "").replace(_TERRAFORM_INSTRUCTION, "").rstrip()
                  + "\n\nReply with only one line: 'Answer: X', where X is the letter of the correct option.")
    elif item["kind"] == "yes_no":
        prompt = prompt.replace(_CUAD_INSTRUCTION, "Reply with only one line: 'Answer: Yes' or 'Answer: No'.")
    elif item["kind"] == "cwe":
        prompt = ("Map this CVE description to the most appropriate CWE. Reply with only the CWE ID, "
                  f"such as CWE-79.\n\nCVE description: {item['description']}")
    return {**item, "prompt": prompt}


# --- Scoring ------------------------------------------------------------------------


def _number(text: str) -> float | None:
    value = benchmark.extract_number(text)
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _close(prediction: float, gold: float) -> bool:
    # FinQA mixes percentages and fractions ("14.1%" vs 0.141) and rounds freely.
    for candidate in (prediction, prediction / 100, prediction * 100):
        if math.isclose(candidate, gold, rel_tol=0.01, abs_tol=0.005):
            return True
    return False


def _strings(text: str) -> list[str]:
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            value = json.loads(match.group(0))
            found = []

            def walk(node):
                if isinstance(node, str):
                    found.append(node)
                elif isinstance(node, list):
                    for item in node:
                        walk(item)
                elif isinstance(node, dict):
                    for item in node.values():
                        walk(item)
            walk(value)
            return found
        except ValueError:
            pass
    return re.findall(r'"([^"\n]{1,200})"', text)


def _f1(predicted: list[str], gold: list[str]) -> float:
    normalize = lambda value: re.sub(r"\s+", " ", value).strip().lower()  # noqa: E731
    predicted_set, gold_set = {normalize(v) for v in predicted if v.strip()}, {normalize(v) for v in gold}
    if not predicted_set and not gold_set:
        return 1.0
    hits = len(predicted_set & gold_set)
    if not hits:
        return 0.0
    precision, recall = hits / len(predicted_set), hits / len(gold_set)
    return 2 * precision * recall / (precision + recall)


def _code(text: str, language: str) -> str:
    blocks = re.findall(rf"```(?:{language})?\s*\n([\s\S]*?)```", text, re.IGNORECASE)
    return (max(blocks, key=len) if blocks else text).strip()


def _sql_result(context: str, query: str):
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(context)
        rows = connection.execute(query).fetchall()
        return sorted(rows, key=repr)
    finally:
        connection.close()


def score(item: dict, text: str) -> tuple[float, str | None]:
    """A score in [0, 1] and what was extracted from the answer."""
    kind = item["kind"]
    if kind == "choice":
        prediction = benchmark.extract_choice(text, item["choices"])
        return float(prediction == item["gold"]), prediction
    if kind == "number":
        prediction = benchmark.extract_number(text)
        return float(benchmark.correct(item, prediction)), prediction
    if kind == "number_tolerant":
        prediction, gold = _number(text), _number("Answer: " + str(item["gold"]))
        return float(prediction is not None and gold is not None and _close(prediction, gold)), str(prediction)
    if kind == "yes_no":
        found = re.findall(r"(?i:answer)\s*[:：]\s*\**\s*(yes|no)\b", text, re.IGNORECASE)
        prediction = found[-1].capitalize() if found else None
        return float(prediction == item["gold"]), prediction
    if kind == "cwe":
        found = re.findall(r"CWE-\d+", text)
        prediction = found[-1] if found else None
        return float(prediction == item["gold"]), prediction
    if kind == "string_set":
        predicted = _strings(text)
        return _f1(predicted, item["gold"]), json.dumps(predicted[:20], ensure_ascii=False)
    if kind == "sql":
        query = _code(text, "sql").rstrip(";").split(";")[0]
        if not query.strip():
            return 0.0, None
        try:
            return float(_sql_result(item["context"], query) == _sql_result(item["context"], item["gold"])), query
        except sqlite3.Error as error:
            return 0.0, f"{query} -- {error}"
    if kind == "python":
        program = _code(text, "python") + "\n\n" + item["test"] + f"\n\ncheck({item['entry_point']})\n"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "solution.py"
            path.write_text(program)
            try:
                passed = subprocess.run([sys.executable, str(path)], cwd=folder, capture_output=True,
                                        timeout=30).returncode == 0
            except subprocess.TimeoutExpired:
                passed = False
        return float(passed), None
    raise ValueError(f"Unknown task kind {kind}")


# --- Runs ---------------------------------------------------------------------------


def ensure_expert(url: str, expert: str, timeout: float = 3600) -> None:
    """Enable the expert and wait until its adapter is prepared."""
    benchmark._post(url.rstrip("/") + f"/v1/experts/{expert}", {"action": "enable"}, 60)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with urllib.request.urlopen(url.rstrip("/") + "/v1/experts", timeout=30) as response:
            row = next((row for row in json.load(response)["data"] if row["id"] == expert), None)
        if row is None:
            raise SystemExit(f"{expert} is not in this server's catalog")
        if row.get("downloaded"):
            return
        if row.get("status") == "error":
            raise SystemExit(f"{expert} could not be prepared: {row.get('error')}")
        time.sleep(5)
    raise SystemExit(f"{expert} was not ready in time")


def run_system(args, label: str, path: Path, items: list[dict], thinking: str, expert: str | None) -> None:
    """Ask the expert, or the base model when `expert` is None, and append scored answers to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()} if path.exists() else set()
    todo = [item for item in items if item["id"] not in done]
    print(f"{label}: {len(items)} items, {len(todo)} to run", flush=True)

    def one(item):
        started = time.monotonic()
        options = argparse.Namespace(
            url=args.url, mode="direct" if expert else "base", expert=expert, verify=False, thinking=thinking,
            max_tokens=QUICK_TOKENS[item["kind"]] if args.quick else args.max_tokens,
            temperature=args.temperature, timeout=args.timeout, on_unsupported="fail")
        try:
            answer = benchmark.ask_ghostswitch(options, item["prompt"])
        except Exception as exc:  # A failed request scores zero; the run continues.
            answer = {"text": "", "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        value, extracted = score(item, answer["text"]) if answer["text"].strip() else (0.0, None)
        return {"id": item["id"], "score": value, "extracted": extracted, "gold": item.get("gold"),
                "latency_s": round(time.monotonic() - started, 2), **answer}

    failures = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool, path.open("a", encoding="utf-8") as out:
        for count, future in enumerate(as_completed([pool.submit(one, item) for item in todo]), 1):
            row = future.result()
            failures = failures + 1 if row["status"] != "success" else 0
            if failures >= 5:
                # A broken server would otherwise be recorded as an expert scoring zero.
                pool.shutdown(cancel_futures=True)
                raise SystemExit(f"Stopping: 5 requests in a row failed ({row.get('error')})")
            out.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            out.flush()
            if count % 10 == 0:
                print(f"  {count}/{len(todo)}", flush=True)


def plan(args, expert: str) -> dict:
    task, thinking = EXPERT_TASKS[expert]
    if args.quick:
        thinking = "off"
    variant = f"{task.__name__.removeprefix('task_')}-{thinking}-{'quick' if args.quick else 'full'}-{args.limit}-{args.seed}"
    return {"task": task.__name__.removeprefix("task_"), "thinking": thinking, "quick": args.quick,
            "limit": args.limit, "seed": args.seed, "base": f"base/{variant}.jsonl", "loader": task}


def run(args) -> None:
    if args.limit is None:
        args.limit = 20 if args.quick else 50
    experts = [expert for expert in (args.experts or list(EXPERT_TASKS)) if expert in EXPERT_TASKS]
    skipped = sorted(set(args.experts or []) - set(experts))
    if skipped:
        print(f"Skipping {', '.join(skipped)}: no automatic task (needs a judge or extra tools)")
    started = time.monotonic()
    # The base model's answers depend only on the task, so experts sharing a task share them.
    for expert in experts:
        settings = plan(args, expert)
        items = settings.pop("loader")(args.limit, args.seed)
        if args.quick:
            items = [quick(item) for item in items]
        folder = Path(args.results) / expert
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "settings.json").write_text(json.dumps(settings, indent=2))
        run_system(args, f"base · {settings['task']}", Path(args.results) / settings["base"], items,
                   settings["thinking"], None)
        ensure_expert(args.url, expert)
        run_system(args, f"{expert}", folder / "expert.jsonl", items, settings["thinking"], expert)
        print(f"  {(time.monotonic() - started) / 60:.1f} min so far", flush=True)


def sign_test(a: list[float], b: list[float]) -> tuple[int, int, float]:
    """Questions where A scored higher, where B did, and the exact two-sided p-value."""
    wins_a = sum(x > y for x, y in zip(a, b))
    wins_b = sum(y > x for x, y in zip(a, b))
    n, k = wins_a + wins_b, min(wins_a, wins_b)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return wins_a, wins_b, p


def verdict(expert_score: float, base_score: float, p: float, alpha: float) -> str:
    if p < alpha and expert_score > base_score:
        return "helps"
    if p < alpha and expert_score < base_score:
        return "hurts"
    return "no_gain"


def report(args) -> None:
    lines = ["| Expert | Task | n | Base | Expert | Expert better | Base better | p | Verdict |",
             "|---|---|---|---|---|---|---|---|---|"]
    for folder in sorted(Path(args.results).glob("*/")):
        expert = folder.name
        if not (folder / "settings.json").exists():
            continue
        settings = json.loads((folder / "settings.json").read_text())
        files = {"base": Path(args.results) / settings["base"], "expert": folder / "expert.jsonl"}
        if not all(path.exists() for path in files.values()):
            continue
        scores = {name: {json.loads(line)["id"]: json.loads(line)["score"]
                         for line in path.read_text().splitlines() if line.strip()} for name, path in files.items()}
        shared = sorted(scores["base"].keys() & scores["expert"].keys())
        if not shared:
            continue
        base = [scores["base"][i] for i in shared]
        mine = [scores["expert"][i] for i in shared]
        wins_expert, wins_base, p = sign_test(mine, base)
        base_mean, expert_mean = sum(base) / len(base), sum(mine) / len(mine)
        decision = verdict(expert_mean, base_mean, p, args.alpha)
        name = settings["task"] + (" (quick)" if settings.get("quick") else "")
        lines.append(f"| {expert} | {name} | {len(shared)} | {base_mean:.1%} | {expert_mean:.1%} | "
                     f"{wins_expert} | {wins_base} | {p:.3f} | {decision} |")
        evaluation = {
            "schema_version": 1, "verdict": decision, "blocked": decision != "helps",
            "measured": {"task": name, "n": len(shared), "base": round(base_mean, 4), "expert": round(expert_mean, 4),
                         "expert_better": wins_expert, "base_better": wins_base, "p_value": round(p, 4),
                         "date": time.strftime("%Y-%m-%d")},
        }
        if decision != "helps":
            evaluation["reason"] = (f"Not better than the base model on {name}: {expert_mean:.0%} vs "
                                    f"{base_mean:.0%} (n={len(shared)}, p={p:.2f})")
        target = ROOT / "experts" / "qwen35-9b" / expert / "evaluation.json"
        if args.write and target.parent.is_dir():
            target.write_text(json.dumps(evaluation, indent=2) + "\n")
    text = "\n".join(lines)
    print(text)
    Path(args.results, "report.md").write_text(text + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(required=True)
    runner = commands.add_parser("run", help="Ask each expert and the base model the expert's questions")
    runner.add_argument("--experts", nargs="*", help=f"Default: all with a task ({len(EXPERT_TASKS)})")
    runner.add_argument("--quick", action="store_true",
                        help="20 questions per expert, answer lines only where no working is needed (~30 min)")
    runner.add_argument("--limit", type=int, help="Questions per expert (default 20 with --quick, else 50)")
    runner.add_argument("--seed", type=int, default=0)
    runner.add_argument("--url", default="http://127.0.0.1:8000")
    runner.add_argument("--max-tokens", type=int, default=2048)
    runner.add_argument("--temperature", type=float, default=0.0)
    runner.add_argument("--concurrency", type=int, default=1)
    runner.add_argument("--timeout", type=float, default=1800)
    runner.add_argument("--results", default="results/experts")
    runner.set_defaults(func=run)
    reporter = commands.add_parser("report", help="Verdicts per expert; --write updates evaluation.json")
    reporter.add_argument("--results", default="results/experts")
    reporter.add_argument("--alpha", type=float, default=0.05)
    reporter.add_argument("--write", action="store_true")
    reporter.set_defaults(func=report)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
