# GhostSwitch evaluation results

GhostSwitch is experimental. This page separates routing accuracy from answer
quality, and current-code measurements from historical results.

See the [detailed router comparison](../benchmarks/2026-11-12-router-v7/comparison.md)
for all four runs by task, or download the
[comparison graphic](../.github/assets/product-hunt-router-comparison.png).

## Current router v7

Measured on **2026-11-12**, after the current router v7 changes, on an **Apple
M1 Pro with 16 GB RAM**, using **Qwen3.5-9B, MLX, 5-bit**. The model revision,
runtime versions and source fingerprints are recorded in the
[run metadata](../benchmarks/2026-11-12-router-v7/run.json).

| Measure | Fresh questions | Reference replay |
|---|---|---|
| Expected expert or base | **269/300 (89.7%)** | 267/300 (89.0%) |
| Specialist questions sent to the right expert | 227/246 (92.3%) | 237/258 (91.9%) |
| Base questions kept on base | 42/54 (77.8%) | 30/42 (71.4%) |
| Median routing time | **1.60 s** | 1.79 s |
| Router exceptions | 0 | 0 |

The fresh sample and reference replay are different experiments:

- **Fresh sample:** 300 questions with no ID overlap with any of the 1,538
  questions in saved historical routing runs. Selection was frozen before
  running the current router. Its source mix includes nine datasets. HumanEval
  contributes no fresh questions because all 164 of its tasks had been routed
  previously. Verilog has only three fresh questions, so that row is especially
  uncertain. This establishes novelty relative to saved local routing runs,
  not to the model's or adapters' training data.
- **Reference replay:** the same 300 question IDs used by the earlier v7
  evaluation. This provides a paired regression comparison and includes 49
  HumanEval questions. It is not an unseen test set.

Both evaluate the **router alone**. They do not generate answers, load LoRA
adapters, execute generated code or score response quality. Every expert in the
routing tree counts as enabled, so this does not measure behavior with a user's
smaller installed library or multi-turn agent sessions. Each source is assigned
an expected specialist; everyday No Robots prompts and MedQA are labeled for
the base model. Long requests use v7's normal 600-token limit: the first 450
tokens and the last 150. These task-based labels have not been independently judged for
ambiguous prompts or alternative valid specialists.

The overall percentage depends on the source mix. Do not compare fresh and
replay percentages as if they measured a quality improvement. Wilson intervals
are descriptive binomial approximations, not guarantees about production use.
The replay ran first; its first request includes cold loading. The fresh sample
ran afterward with the model and routing system cache warm. No code or prompt
tuning took place between the cohorts. Timing measures routing, not time to a
full answer. The Mac was not isolated from other applications.

**These measurements cover only the 9B model on MLX.** They do not establish v7
accuracy or latency on the 4B model, GGUF or the Qwythos fine-tune.

See the [full report](../benchmarks/2026-11-12-router-v7/report.md) for each
dataset, common mistakes, latency and the paired comparison. The archive
includes [per-question routing records](../benchmarks/2026-11-12-router-v7/routes.jsonl),
[cohort membership](../benchmarks/2026-11-12-router-v7/cohorts.json),
[input IDs and prompt hashes](../benchmarks/2026-11-12-router-v7/inputs-manifest.json),
and the [historical exclusion list](../benchmarks/2026-11-12-router-v7/historical-excluded-keys.json).
Prompts, generated answers, server logs and local filesystem paths are omitted.

## Historical routing results

These are retained for context, not presented as measurements of current v7.

| Router and setup | Questions | Expected expert or base | Notes |
|---|---:|---:|---|
| v6, Qwen3.5-9B Q4_K_M, RTX 3090 | 1,000 | 876/1,000 (87.6%) | Older router and different hardware/backend |
| Earlier v7, Qwen3.5-9B 5-bit, M1 Pro | 300 | 267/300 (89.0%) | Older v7 source fingerprint; reused in the reference replay |

In the v6 run, 135/136 everyday prompts stayed on the base model. Of its 124
misses, 54 went to base and 70 to another specialist. Medical questions went to
the reasoning specialist in 36 cases, finance in 20, and Verilog went to the
general coding specialist in eight. **Most misses did not fall back to base.**
The effect of these wrong routes on answer quality was not measured.

The earlier v7 set was previously described as wholly unseen. An ID audit found
29 questions shared with the saved development set, so that description was
incorrect. Changing the sampling seed alone does not prevent overlap.

The historical v6 metadata was recorded after its run and does not contain the
source fingerprints available for the new run. Its three router processes
shared one GPU, so its latency is not a controlled speed comparison with MLX.
Archived records: [v6](../benchmarks/2026-11-12-router-v7/historical-v6-gguf/routes.jsonl)
and [earlier v7](../benchmarks/2026-11-12-router-v7/historical-v7/routes.jsonl).

## Historical specialist vs. base-model evaluation

These measurements were made on **2026-10-09**, using **Qwen3.5-9B Q4_K_M,
GGUF, RTX 3090**, with thinking and the verifier off. Each specialist was called
directly and compared with the base model on the same 60 task IDs. They do not
measure the router or prove an end-to-end improvement for current GhostSwitch.

| Specialist | Task | n | Base score | Specialist score | Unadjusted p |
|---|---|---:|---:|---:|---:|
| `coding_delta_expert` | humaneval | 60 | 76.7% | 78.3% | 1.0000 |
| `coding_expert` | humaneval | 60 | 76.7% | 88.3% | 0.0391 |
| `cybersecurity_expert` | cti_rcm | 60 | 56.7% | 51.7% | 0.5078 |
| `entity_extraction_expert` | conll (F1) | 60 | 67.4% | 67.2% | 0.7201 |
| `finance_market_expert` | finqa | 60 | 61.7% | 66.7% | 0.5488 |
| `iac_security_expert` | terraform | 60 | 53.3% | 51.7% | 1.0000 |
| `legal_contract_expert` | cuad | 60 | 76.7% | 83.3% | 0.1250 |
| `medical_reasoning_expert` | medqa | 60 | 66.7% | 65.0% | 1.0000 |
| `penetration_testing_expert` | pentest | 60 | 88.3% | 86.7% | 1.0000 |
| `privacy_dlp_expert` | pii (F1) | 60 | 75.3% | 78.3% | 0.1797 |
| `reasoning_expert` | gsm8k | 60 | 88.3% | 90.0% | 1.0000 |
| `software_engineering_expert` | humaneval | 60 | 76.7% | 81.7% | 0.5078 |
| `text_to_sql_expert` | sql | 60 | 58.3% | 58.3% | 1.0000 |

The coding specialist passed 53/60 tasks versus the base model's 46/60. It was
better on eight paired tasks and worse on one. Its exact paired sign-test
p-value is 0.0391 **before correction for multiple comparisons**. Thirteen
specialists were tested; this is an exploratory result on a small sample, not
a full HumanEval+ evaluation or proof of broad coding superiority. The observed
gain would not meet a Bonferroni-corrected 0.05 threshold across 13 comparisons.

For the other specialists, a non-significant difference means that this run did
not establish a benefit. It does not prove the adapter is equivalent or useless
on every task. Some adapters were evaluated outside the narrow domain described
by their authors. Differences include both the adapter and specialist prompt.

The [summary and settings](../benchmarks/2026-11-12-router-v7/historical-experts.json)
and [paired per-question scores](../benchmarks/2026-11-12-router-v7/historical-expert-scores.jsonl)
are archived. Duplicate logged IDs are deduplicated before counting, as in the
original report. These scores support recomputing the aggregates; independent
regrading requires rerunning the specialist benchmarks. The [earlier audit](expert-blocklist.ru.md) explains the
catalog decisions in Russian.

## Reproduce the report

Recalculate the router tables and verify archived file checksums with Python's
standard library. This does not load a model or require network access:

```bash
python scripts/router_results.py benchmarks/2026-11-12-router-v7
```

To run the fresh sample again on supported Apple Silicon hardware, install the
project and its 9B MLX model first, then run from the repository root:

```bash
.venv/bin/python scripts/router_eval.py run \
  --router v7 --backbone qwen35-9b --backend mlx \
  --seed 20261112 --limit 5000 --max 300 \
  --exclude-keys benchmarks/2026-11-12-router-v7/historical-excluded-keys.json \
  --results results/router-v7-fresh-reproduction
.venv/bin/python scripts/router_eval.py report \
  --results results/router-v7-fresh-reproduction
```

Compare the resulting IDs and prompt SHA-256 hashes with the fresh cohort's
input manifest before comparing scores. Public dataset contents can change;
dataset-server revisions are not pinned by the existing loaders (GSM8K is
pinned separately). Source repositories, splits and configurations are listed
in [datasets.json](../benchmarks/2026-11-12-router-v7/datasets.json). The local
measurement also saves full input prompts to `results/.../inputs.jsonl` for
exact local replays with `--dataset`; those prompts are not redistributed here.
The fresh selection used a candidate pool requested with `--limit 5000`, then
excluded historical IDs and took the first 300 remaining shuffled questions.
It is not a run of 5,000 questions.

Use a new results directory for changed source files or inputs. The runner
refuses to mix such versions on resume and checks its recorded source
fingerprints again at completion. The recorded Git parent may have a dirty
working tree; the fingerprints identify the files actually evaluated.

For new specialist answer-quality measurements, see the
[benchmarking guide](benchmarking.md). Generated code is executed by those
tests and belongs in a disposable environment; the router-only test above does
not execute model-generated code.
