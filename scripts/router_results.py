"""Recompute the public router report from its archived per-question records.

    python scripts/router_results.py benchmarks/2026-11-12-router-v7

No model, network access, or third-party package is needed. Dataset prompts and
model answers are not redistributed in the archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def read_records(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    records = {row["key"]: row for row in rows}
    if len(records) != len(rows):
        raise ValueError(f"Duplicate question IDs in {path}")
    return records


def wilson(hits: int, n: int) -> list[float]:
    if not n:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = hits / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [center - half, center + half]


def metrics(records: list[dict]) -> dict:
    n = len(records)
    hits = sum(row["chosen"] == row["expected"] for row in records)
    specialist = [row for row in records if row["expected"] != "base"]
    base = [row for row in records if row["expected"] == "base"]
    wrong = [row for row in records if row["chosen"] != row["expected"]]
    seconds = sorted(row["seconds"] for row in records)
    return {
        "n": n, "correct": hits, "accuracy": hits / n if n else 0,
        "accuracy_wilson_95": wilson(hits, n),
        "specialist_correct": sum(row["chosen"] == row["expected"] for row in specialist),
        "specialist_n": len(specialist),
        "base_correct": sum(row["chosen"] == "base" for row in base), "base_n": len(base),
        "errors": sum(row["route"] == "error" for row in records),
        "misses_to_base": sum(row["chosen"] == "base" for row in wrong),
        "misses_to_wrong_expert": sum(row["chosen"] not in (None, "base") for row in wrong),
        "median_seconds": statistics.median(seconds) if seconds else None,
        "p90_seconds": seconds[max(0, math.ceil(n * 0.9) - 1)] if seconds else None,
        "mean_seconds": statistics.mean(seconds) if seconds else None,
    }


def fraction(hits: int, n: int) -> str:
    return f"{hits}/{n} ({hits / n:.1%})" if n else "Not sampled"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args(argv)
    folder = args.archive
    # Verify archived bytes before recomputing any numbers.
    checksums = json.loads((folder / "checksums.json").read_text())
    for name, expected in checksums.items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Archive checksum mismatch: {name}")
    records = read_records(folder / "routes.jsonl")
    cohorts = json.loads((folder / "cohorts.json").read_text())
    fresh_keys = set(cohorts["fresh_unseen"])
    replay_keys = set(cohorts["reference_replay"])
    excluded = set(json.loads((folder / "historical-excluded-keys.json").read_text()))
    manifest = json.loads((folder / "inputs-manifest.json").read_text())
    if len(fresh_keys) != len(cohorts["fresh_unseen"]) or len(replay_keys) != len(cohorts["reference_replay"]):
        raise ValueError("Duplicate IDs in a cohort")
    if fresh_keys & replay_keys or fresh_keys & excluded:
        raise ValueError("Fresh sample overlaps replay or historical questions")
    if fresh_keys | replay_keys != set(records) or {row["key"] for row in manifest} != set(records):
        raise ValueError("Records, input manifest and cohort IDs do not match")
    if len(manifest) != len(records) or any(row["source"] != records[row["key"]]["source"] or
                                          row["expected"] != records[row["key"]]["expected"] for row in manifest):
        raise ValueError("Input manifest has duplicate IDs or labels differ from the records")
    meta = json.loads((folder / "run.json").read_text())
    if not meta.get("code_unchanged_during_run") or not meta.get("completed"):
        raise ValueError("The run did not complete with unchanged evaluated source files")
    selected = {
        "fresh_unseen": [records[key] for key in cohorts["fresh_unseen"]],
        "reference_replay": [records[key] for key in cohorts["reference_replay"]],
    }
    summary = {name: metrics(rows) for name, rows in selected.items()}
    old = read_records(folder / "historical-v7/routes.jsonl")
    if set(old) != replay_keys:
        raise ValueError("Historical paired comparison has different questions")
    current_wins = sum(records[key]["chosen"] == records[key]["expected"] and
                       old[key]["chosen"] != old[key]["expected"] for key in replay_keys)
    old_wins = sum(records[key]["chosen"] != records[key]["expected"] and
                   old[key]["chosen"] == old[key]["expected"] for key in replay_keys)
    summary["paired_replay"] = {"current_only_correct": current_wins, "historical_only_correct": old_wins}
    summary["historical_v7"] = metrics(list(old.values()))
    summary["historical_v6_gguf"] = metrics(list(read_records(folder / "historical-v6-gguf/routes.jsonl").values()))
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Router v7: current-code evaluation", "",
             "Measured on 2026-11-12. Fresh and replay samples are reported separately.", "",
             "| Measure | Fresh questions | Replayed questions |", "|---|---|---|"]
    def cells(name, render):
        lines.append(f"| {name} | " + " | ".join(render(summary[cohort]) for cohort in selected) + " |")
    cells("Expected expert or base", lambda m: fraction(m["correct"], m["n"]))
    cells("95% Wilson interval", lambda m: "–".join(f"{x:.1%}" for x in m["accuracy_wilson_95"]))
    cells("Specialist questions sent to the right expert", lambda m: fraction(m["specialist_correct"], m["specialist_n"]))
    cells("Base questions kept on base", lambda m: fraction(m["base_correct"], m["base_n"]))
    cells("Router exceptions", lambda m: str(m["errors"]))
    cells("Misses sent to base", lambda m: str(m["misses_to_base"]))
    cells("Misses sent to another expert", lambda m: str(m["misses_to_wrong_expert"]))
    cells("Median routing time", lambda m: f"{m['median_seconds']:.2f} s")
    cells("90th percentile routing time", lambda m: f"{m['p90_seconds']:.2f} s")
    for cohort, rows in selected.items():
        lines += ["", "## " + ("Fresh sample" if cohort == "fresh_unseen" else "Reference replay"), "",
                  "| Source | Expected | Correct | 95% Wilson interval | Common misses |", "|---|---|---|---|---|"]
        for source in sorted({row["source"] for row in rows}):
            group = [row for row in rows if row["source"] == source]
            misses = Counter(row["chosen"] or "error" for row in group if row["chosen"] != row["expected"])
            hits = sum(r["chosen"] == r["expected"] for r in group)
            interval = "–".join(f"{x:.1%}" for x in wilson(hits, len(group)))
            lines.append(f"| {source} | {group[0]['expected']} | "
                         f"{fraction(hits, len(group))} | {interval} | "
                         + (", ".join(f"{name} ×{count}" for name, count in misses.most_common()) or "None") + " |")
    historical = summary["historical_v7"]
    replay = summary["reference_replay"]
    lines += ["", "## Paired replay comparison", "",
              f"On the same {len(replay_keys)} questions: historical v7 "
              f"{fraction(historical['correct'], historical['n'])}; current v7 "
              f"{fraction(replay['correct'], replay['n'])}.", "",
              f"Current version alone correct: {current_wins}. Historical version alone correct: {old_wins}.", "",
              "This is a regression comparison on previously evaluated data, not a fresh holdout.", "",
              "## Conditions and limits", "",
              f"- Backend: {meta['model']['backend']}. Model: `{meta['model']['published_source']}`.",
              f"- Hardware: {meta['machine']['chip']}, {meta['machine']['memory_gb']} GB RAM.",
              "- Router only. No answer generation, LoRA loading, web search or answer-quality scoring.",
              "- Every tree expert counts as enabled. Labels come from the source dataset's designated task.",
              "- Long requests use the normal 600-token head/tail cap: first 450 tokens and last 150.",
              f"- The fresh sample excludes all {len(excluded)} IDs in the saved historical routing runs.",
              "- HumanEval is absent from the fresh sample because all 164 tasks had already been routed.",
              "- Novelty is relative to saved local routing runs, not to model or adapter training data.",
              "- Labels are task-based; ambiguous prompts and alternate valid experts are not judged independently.",
              "- Samples have different source mixes. Do not treat their accuracy difference as a model improvement.",
              "- Wilson intervals describe this sample under a binomial approximation; they do not predict production accuracy.",
              "- Replay ran first, including a cold initial request. Fresh questions followed with the model/system cache warm.",
              "- Medians and p90 are routing latency, not full response latency. The Mac was not isolated from other apps.",
              "- Only this model/backend/hardware combination was rerun. Other v7 profiles remain unmeasured.",
              "- See `run.json` for the source fingerprints, runtime versions and dirty-working-tree marker.",
              "- Prompts and answers are omitted. `inputs-manifest.json` contains IDs, labels and prompt SHA-256 hashes.",
              "- `checksums.json` covers the archived inputs to this report; `summary.json` and `report.md` are derived.",
              "", "Recompute this report without a model or network:", "", "```bash",
              f"python scripts/router_results.py {folder.as_posix()}", "```", ""]
    (folder / "report.md").write_text("\n".join(lines))
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()
