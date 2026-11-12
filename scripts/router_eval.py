"""How often does the router pick the right expert?

Questions from public test sets, each labeled with the expert that should answer it, go
through the router only (by default the one the app uses for that backbone and backend): no answer is generated and no LoRA is loaded, so the run needs
just the base model. Every routable expert in the tree counts as installed, as if the user
had turned all of them on. Questions from fields with no expert in the tree (everyday
requests from No Robots, medicine from MedQA) should be answered by the base model.

    python scripts/router_eval.py run --limit 1000          # writes results/router/routes.jsonl
    python scripts/router_eval.py report                    # accuracy, confusions, latency
    python scripts/router_eval.py compare "v6 4-bit=results/router" "v7 5-bit=results/router-v7-5bit" \
        --out results/router-comparison.md                  # the runs side by side, on shared questions

A run resumes where it stopped: questions already routed are skipped. On a GPU with room for
several copies of the model, `--shard 0/3`, `--shard 1/3` and `--shard 2/3` split the set
between three processes; the report reads every routes*.jsonl in the results folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import expert_eval  # noqa: E402
from scripts.expert_eval import rows  # noqa: E402

BASE = "base"


def _cypher(limit, seed):
    return [{"id": row["instance_id"],
             "prompt": f"Graph schema:\n{row['schema'][:3000]}\n\nWrite a Cypher query that answers: {row['question']}"}
            for row in rows("neo4j/text2cypher-2024v1", "default", "test", limit, seed)]


def _lean4(limit, seed):
    return [{"id": row["id"],
             "prompt": "Complete the following Lean 4 theorem by replacing `sorry` with a proof.\n\n"
                       f"{row['formal_statement']}"}
            for row in rows("cat-searcher/minif2f-lean4", "default", "test", limit, seed)]


def _verilog(limit, seed):
    return [{"id": row["problem_id"], "prompt": row["prompt"].strip()}
            for row in rows("dakies/nvlabs-verilogeval-v2-spec-to-rtl", "default", "test", limit, seed)]


# No Robots categories that ask for ordinary help; Coding and Chat (persona system prompts) are left out.
_EVERYDAY = {"Open QA", "Brainstorm", "Rewrite", "Summarize", "Classify", "Closed QA", "Extract", "Generation"}


def _everyday(limit, seed):
    items = []
    for row in rows("HuggingFaceH4/no_robots", "default", "test", 500, seed):
        if row["category"] not in _EVERYDAY:
            continue
        messages = row["messages"]
        user = next((m["content"] for m in messages if m["role"] == "user"), None)
        if user:
            items.append({"id": row["prompt_id"][:16], "prompt": user, "category": row["category"]})
    return items[:limit]


# source → (loader, expected expert, share of the set). Coding questions accept either coding
# expert at the field level; the strict label is the one meant for single functions.
SOURCES = {
    "gsm8k": (expert_eval.task_gsm8k, "reasoning_expert", 0.12),
    "finqa": (expert_eval.task_finqa, "finance_market_expert", 0.10),
    "cuad": (expert_eval.task_cuad, "legal_contract_expert", 0.10),
    "pii": (expert_eval.task_pii, "privacy_dlp_expert", 0.10),
    "humaneval": (expert_eval.task_humaneval, "coding_expert", 0.164),
    "text2cypher": (_cypher, "text_to_cypher_expert", 0.08),
    "minif2f_lean4": (_lean4, "lean4_generalist_expert", 0.08),
    "verilogeval": (_verilog, "verilog_expert", 0.08),
    "no_robots": (_everyday, BASE, 0.136),
    "medqa": (expert_eval.task_medqa, BASE, 0.04),
}


def dataset(limit: int, seed: int) -> list[dict]:
    items = []
    for source, (loader, expected, share) in SOURCES.items():
        for item in loader(round(limit * share), seed):
            items.append({"key": f"{source}:{item['id']}", "source": source, "expected": expected,
                          "prompt": item["prompt"], "category": item.get("category")})
    random.Random(seed).shuffle(items)
    return items


def evaluation_items(args) -> tuple[list[dict], set[str]]:
    """Freeze inputs and exclude previously evaluated IDs before taking a sample."""
    excluded = set()
    for path in args.exclude_keys:
        keys = json.loads(Path(path).read_text())
        if not isinstance(keys, list) or any(not isinstance(key, str) for key in keys):
            raise ValueError("An exclusion file must contain a JSON list of question keys")
        excluded.update(keys)
    if args.dataset:
        items = [json.loads(line) for line in Path(args.dataset).read_text().splitlines() if line.strip()]
    else:
        items = dataset(args.limit, args.seed)
    seen = set()
    selected = []
    for item in items:
        if not all(isinstance(item.get(key), str) and item[key] for key in ("key", "source", "expected", "prompt")):
            raise ValueError("Each question needs a key, source, expected route and prompt")
        if item["key"] in seen:
            raise ValueError(f"Duplicate question key: {item['key']}")
        seen.add(item["key"])
        if item["key"] not in excluded:
            selected.append(item)
    if args.shard:
        index, count = (int(part) for part in args.shard.split("/"))
        if count < 1 or not 0 <= index < count:
            raise ValueError("Shard must be i/n, with 0 <= i < n")
        selected = selected[index::count]
    if args.max:
        selected = selected[:args.max]
    if not selected:
        raise ValueError("No questions remain after exclusions")
    return selected, excluded


class AllExperts:
    """Registry stand-in: every expert in the routing tree counts as installed and turned on."""

    def __init__(self) -> None:
        from core.expert_catalog import EXPERT_CATALOG
        from core.router_tree import ROUTER_TREE
        self._descriptions = {expert: EXPERT_CATALOG.by_id[expert].description
                              for expert in sorted(ROUTER_TREE._experts_below(ROUTER_TREE.root_id))}

    def descriptions(self) -> dict[str, str]:
        return dict(self._descriptions)


def field_of(expert: str | None) -> str | None:
    from core.router_tree import ROUTER_TREE
    if not expert or expert == BASE:
        return BASE
    path = ROUTER_TREE.path_for_expert(expert)
    return path[0] if path else None


def _machine() -> dict:
    def sysctl(name: str) -> str | None:
        try:
            return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    memory = sysctl("hw.memsize")
    return {"system": platform.platform(), "machine": platform.machine(),
            "chip": sysctl("machdep.cpu.brand_string") or platform.processor(),
            "memory_gb": round(int(memory) / 2**30) if memory else None}


def _git() -> dict:
    def git(*command: str) -> str:
        return subprocess.run(["git", *command], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    code = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()[:12]
            for name in ("core/router_v6.py", "core/router_v7.py", "core/thinker.py", "core/routing_context.py",
                         "core/response_language.py", "core/router_tree.py", "core/expert_packages.py",
                         "core/llm.py", "core/config/router_tree.json", "core_models/backbones.json",
                         "scripts/router_eval.py", "core/main.py", "core/request_contract.py") if (ROOT / name).exists()}
    for path in sorted([*ROOT.glob("core/config/execution_profiles/*.json"),
                        *ROOT.glob("experts/**/*.json"), *ROOT.glob("experts/**/prompt.txt")]):
        code[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return {"branch": git("branch", "--show-current"), "commit": git("rev-parse", "--short", "HEAD"),
            "uncommitted_changes": bool(git("status", "--porcelain")), "code_sha256": code}


def _model(llm, selection) -> dict:
    spec = llm.MODEL_SPECS["thinker"]
    info = {"backend": spec.backend, "model": spec.model_id}
    try:
        from core_models import download_source
        source = download_source(selection.backbone, spec.backend)
        info["published_source"] = f"{source['repo_id']} @ {source['revision']}"
    except ValueError:
        pass
    config = Path(spec.model_id).expanduser() / "config.json"
    if spec.backend == "mlx" and config.is_file():
        info["quantization"] = json.loads(config.read_text()).get("quantization")
    elif spec.backend == "gguf":
        info["quantization"] = Path(spec.model_id).stem.rsplit("-", 1)[-1]
    try:
        from importlib.metadata import version
        info["runtime"] = {name: version(name) for name in ("mlx", "mlx-lm", "llama-cpp-python")
                           if _installed(name)}
    except Exception:  # metadata only
        pass
    return info


def _installed(name: str) -> bool:
    from importlib.metadata import PackageNotFoundError, version
    try:
        version(name)
        return True
    except PackageNotFoundError:
        return False


def run(args) -> None:
    import importlib
    from core import llm, runtime_selection
    if args.router is None:
        # The router the app's execution profile names for this backbone and backend.
        from core_models import profile_name
        profile = ROOT / "core" / "config" / "execution_profiles" / f"{profile_name(args.backbone, args.backend)}.json"
        args.router = json.loads(profile.read_text()).get("router", "v6")
    router = importlib.import_module(f"core.router_{args.router}")
    if args.model_path:
        # Comparing quantizations means loading checkpoints the app does not ship.
        import core_models
        core_models.validate_local_model = lambda profile: None
    items, excluded = evaluation_items(args)
    selection = runtime_selection.resolve(args.backbone, args.backend, args.model_path)
    llm.configure_backbone(selection.backbone, model_path=selection.model_path, backend=selection.backend)
    registry = AllExperts()
    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    path = results / "routes.jsonl"
    if args.shard:
        index, count = (int(part) for part in args.shard.split("/"))
        path = results / f"routes.shard{index}.jsonl"
    done = {json.loads(line)["key"] for line in path.open()} if path.exists() else set()
    todo = [item for item in items if item["key"] not in done]
    print(f"{len(items)} questions, {len(done)} already routed, {len(todo)} to go", flush=True)
    meta_path = results / (f"run.shard{index}.json" if args.shard else "run.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"sessions": []}
    inputs = "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in items)
    inputs_sha256 = hashlib.sha256(inputs.encode()).hexdigest()
    snapshot = _git()
    if done:
        if meta.get("dataset", {}).get("inputs_sha256") != inputs_sha256:
            raise ValueError("Cannot resume: input questions differ from the previous session")
        if any(session.get("git", {}).get("code_sha256") != snapshot["code_sha256"] for session in meta["sessions"]):
            raise ValueError("Cannot resume: evaluated code changed; use a new results directory")
    (results / (f"inputs.shard{index}.jsonl" if args.shard else "inputs.jsonl")).write_text(inputs)
    excluded_json = json.dumps(sorted(excluded), ensure_ascii=False, indent=2) + "\n"
    (results / "excluded-keys.json").write_text(excluded_json)
    meta.update({"router": args.router, "dataset": {"seed": args.seed, "limit": args.limit, "max": args.max,
                                                    "shard": args.shard, "input_file": args.dataset,
                                                    "excluded_keys": len(excluded),
                                                    "excluded_keys_sha256": hashlib.sha256(excluded_json.encode()).hexdigest(),
                                                    "inputs_sha256": inputs_sha256},
                 "model": _model(llm, selection), "machine": _machine()})
    meta["sessions"].append({"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "questions": len(todo),
                             "git": snapshot})
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    started = time.monotonic()
    with path.open("a") as out:
        for number, item in enumerate(todo, 1):
            t0 = time.monotonic()
            try:
                decision = router.route([{"role": "user", "content": item["prompt"]}], registry)
                chosen = decision["expert_id"] if decision["route"] == "expert" else BASE
                record = {**{k: item[k] for k in ("key", "source", "expected", "category")},
                          "chosen": chosen, "route": decision["route"], "method": decision.get("method"),
                          "path": decision.get("path"), "steps": decision.get("steps"),
                          "seconds": round(time.monotonic() - t0, 3), "error": None}
            except Exception as error:  # a routing failure is a result too
                record = {**{k: item[k] for k in ("key", "source", "expected", "category")},
                          "chosen": None, "route": "error", "method": None, "path": None, "steps": None,
                          "seconds": round(time.monotonic() - t0, 3), "error": f"{type(error).__name__}: {error}"}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            if number % args.progress == 0 or number == len(todo):
                rate = (time.monotonic() - started) / number
                print(f"{number}/{len(todo)}  {rate:.2f} s/question  ~{rate * (len(todo) - number) / 60:.0f} min left",
                      flush=True)
    meta["completed"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    meta["code_unchanged_during_run"] = _git()["code_sha256"] == snapshot["code_sha256"]
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    if not meta["code_unchanged_during_run"]:
        raise RuntimeError("Evaluated code changed during the run; results must not be treated as one version")


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "—"


def report(args) -> None:
    results = Path(args.results)
    records = [json.loads(line) for path in sorted(results.glob("routes*.jsonl")) for line in path.open()]
    n = len(records)
    strict = sum(r["chosen"] == r["expected"] for r in records)
    field = sum(field_of(r["chosen"]) == field_of(r["expected"]) for r in records)
    specialist = [r for r in records if r["expected"] != BASE]
    everyday = [r for r in records if r["expected"] == BASE]
    errors = sum(r["route"] == "error" for r in records)
    lines = [f"# Router evaluation ({n} questions)", "",
             "| Measure | Result |", "|---|---|",
             f"| Exact expert | {strict}/{n} ({_pct(strict, n)}) |",
             f"| Right field | {field}/{n} ({_pct(field, n)}) |",
             f"| Specialist questions sent to an expert | {sum(bool(r['chosen']) and r['chosen'] != BASE for r in specialist)}/{len(specialist)} |",
             f"| Specialist questions sent to the right expert | {sum(r['chosen'] == r['expected'] for r in specialist)}/{len(specialist)} ({_pct(sum(r['chosen'] == r['expected'] for r in specialist), len(specialist))}) |",
             f"| Base-model questions kept on the base model | {sum(r['chosen'] == BASE for r in everyday)}/{len(everyday)} ({_pct(sum(r['chosen'] == BASE for r in everyday), len(everyday))}) |",
             f"| Router errors | {errors} |", ""]
    lines += ["| Source | Expected | n | Exact | Field | Most common wrong choices |", "|---|---|---|---|---|---|"]
    by_source = defaultdict(list)
    for r in records:
        by_source[r["source"]].append(r)
    for source, group in by_source.items():
        wrong = Counter(r["chosen"] or "error" for r in group if r["chosen"] != r["expected"]).most_common(3)
        lines.append(f"| {source} | {group[0]['expected']} | {len(group)} | "
                     f"{_pct(sum(r['chosen'] == r['expected'] for r in group), len(group))} | "
                     f"{_pct(sum(field_of(r['chosen']) == field_of(r['expected']) for r in group), len(group))} | "
                     + (", ".join(f"{k} ×{v}" for k, v in wrong) or "—") + " |")
    seconds = sorted(r["seconds"] for r in records)
    if seconds:
        lines += ["", f"Routing time per question: median {seconds[len(seconds) // 2]:.2f} s, "
                      f"90th percentile {seconds[int(len(seconds) * 0.9)]:.2f} s, mean {sum(seconds) / len(seconds):.2f} s."]
    methods = Counter(r["method"] for r in records)
    lines += ["", "Routing method: " + ", ".join(f"{k} ×{v}" for k, v in methods.most_common())]
    text = "\n".join(lines)
    (results / "report.md").write_text(text + "\n")
    print(text)


def _load(results: Path) -> dict[str, dict]:
    return {record["key"]: record for path in sorted(results.glob("routes*.jsonl"))
            for record in map(json.loads, path.open())}


def compare(args) -> None:
    """The runs side by side, counted only on the questions every run routed."""
    runs = []
    for spec in args.runs:
        label, _, folder = spec.partition("=")
        runs.append((label, Path(folder), _load(Path(folder))))
    shared = set.intersection(*(set(records) for _, _, records in runs))
    sample = runs[0][2]
    keys = [key for key in sample if key in shared]
    lines = [f"# Router comparison ({len(keys)} shared questions)", ""]
    header = "| Measure | " + " | ".join(label for label, _, _ in runs) + " |"
    lines += [header, "|---" * (len(runs) + 1) + "|"]

    def row(name, value):
        lines.append(f"| {name} | " + " | ".join(value([records[key] for key in keys]) for _, _, records in runs) + " |")

    def share(test):
        return lambda records: (lambda hits, total: f"{hits}/{total} ({_pct(hits, total)})")(
            sum(test(r) for r in records), len(records))

    def among(group, test):
        return lambda records: share(test)([r for r in records if group(r)])

    def seconds(q):
        return lambda records: f"{sorted(r['seconds'] for r in records)[int(len(records) * q)]:.2f} s"

    row("Exact expert", share(lambda r: r["chosen"] == r["expected"]))
    row("Right field", share(lambda r: field_of(r["chosen"]) == field_of(r["expected"])))
    row("Specialist questions → right expert", among(lambda r: r["expected"] != BASE, lambda r: r["chosen"] == r["expected"]))
    row("Base-model questions kept on base", among(lambda r: r["expected"] == BASE, lambda r: r["chosen"] == BASE))
    row("Routing time, median", seconds(0.5))
    row("Routing time, 90th percentile", seconds(0.9))
    row("Router errors", lambda records: str(sum(r["route"] == "error" for r in records)))
    lines += ["", "Right expert by source:", "", "| Source | Expected | n | " + " | ".join(label for label, _, _ in runs) + " |",
              "|---" * (len(runs) + 3) + "|"]
    sources = defaultdict(list)
    for key in keys:
        sources[sample[key]["source"]].append(key)
    for source, group in sorted(sources.items()):
        cells = [f"{sum(records[key]['chosen'] == records[key]['expected'] for key in group)}/{len(group)}"
                 for _, _, records in runs]
        lines.append(f"| {source} | {sample[group[0]]['expected']} | {len(group)} | " + " | ".join(cells) + " |")
    lines += ["", "Runs:", ""]
    for label, folder, records in runs:
        meta_path = folder / "run.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        model = meta.get("model", {})
        machine = meta.get("machine", {})
        git = (meta.get("sessions") or [{}])[-1].get("git", {})
        lines.append(f"- **{label}**: `{folder}`, {len(records)} questions routed; router {meta.get('router', '?')}, "
                     f"{model.get('backend', '?')} `{model.get('model', '?')}` quantization {model.get('quantization', '?')}, "
                     f"{machine.get('chip', '?')} {machine.get('memory_gb', '?')} GB, "
                     f"git {git.get('branch', '?')}@{git.get('commit', '?')}"
                     + (" (uncommitted changes)" if git.get("uncommitted_changes") else ""))
    text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--limit", type=int, default=1000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--backbone", default="qwen35-9b")
    r.add_argument("--backend", default="mlx", choices=("mlx", "gguf"))
    r.add_argument("--progress", type=int, default=25)
    r.add_argument("--shard", help="i/n: route only every n-th question, starting at i")
    r.add_argument("--router", choices=("v6", "v7"), help="default: the router the app uses for this backbone and backend")
    r.add_argument("--max", type=int, help="route only the first N questions of the shuffled set")
    r.add_argument("--model-path", help="another checkpoint of the backbone, e.g. a different quantization")
    r.add_argument("--dataset", help="frozen input JSONL containing key, source, expected, prompt and optional category")
    r.add_argument("--exclude-keys", action="append", default=[], help="JSON list of previously used question keys; repeatable")
    rep = sub.add_parser("report")
    for command in (r, rep):
        command.add_argument("--results", default=str(ROOT / "results" / "router"))
    cmp = sub.add_parser("compare")
    cmp.add_argument("runs", nargs="+", help="label=results-folder")
    cmp.add_argument("--out", help="write the table to this Markdown file")
    args = parser.parse_args(argv)
    {"run": run, "report": report, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
