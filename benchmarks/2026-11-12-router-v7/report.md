# Router v7: current-code evaluation

Measured on 2026-11-12. Fresh and replay samples are reported separately.

| Measure | Fresh questions | Replayed questions |
|---|---|---|
| Expected expert or base | 269/300 (89.7%) | 267/300 (89.0%) |
| 95% Wilson interval | 85.7%–92.6% | 85.0%–92.1% |
| Specialist questions sent to the right expert | 227/246 (92.3%) | 237/258 (91.9%) |
| Base questions kept on base | 42/54 (77.8%) | 30/42 (71.4%) |
| Router exceptions | 0 | 0 |
| Misses sent to base | 18 | 10 |
| Misses sent to another expert | 13 | 23 |
| Median routing time | 1.60 s | 1.79 s |
| 90th percentile routing time | 2.96 s | 3.20 s |

## Fresh sample

| Source | Expected | Correct | 95% Wilson interval | Common misses |
|---|---|---|---|---|
| cuad | legal_contract_expert | 46/46 (100.0%) | 92.3%–100.0% | None |
| finqa | finance_market_expert | 24/40 (60.0%) | 44.6%–73.7% | base ×15, reasoning_expert ×1 |
| gsm8k | reasoning_expert | 49/49 (100.0%) | 92.7%–100.0% | None |
| medqa | base | 16/20 (80.0%) | 58.4%–91.9% | reasoning_expert ×3, legal_contract_expert ×1 |
| minif2f_lean4 | lean4_generalist_expert | 10/10 (100.0%) | 72.2%–100.0% | None |
| no_robots | base | 26/34 (76.5%) | 60.0%–87.6% | roleplay_expert ×8 |
| pii | privacy_dlp_expert | 46/46 (100.0%) | 92.3%–100.0% | None |
| text2cypher | text_to_cypher_expert | 49/52 (94.2%) | 84.4%–98.0% | base ×3 |
| verilogeval | verilog_expert | 3/3 (100.0%) | 43.9%–100.0% | None |

## Reference replay

| Source | Expected | Correct | 95% Wilson interval | Common misses |
|---|---|---|---|---|
| cuad | legal_contract_expert | 35/35 (100.0%) | 90.1%–100.0% | None |
| finqa | finance_market_expert | 27/38 (71.1%) | 55.2%–83.0% | base ×10, reasoning_expert ×1 |
| gsm8k | reasoning_expert | 35/35 (100.0%) | 90.1%–100.0% | None |
| humaneval | coding_expert | 49/49 (100.0%) | 92.7%–100.0% | None |
| medqa | base | 3/10 (30.0%) | 10.8%–60.3% | reasoning_expert ×7 |
| minif2f_lean4 | lean4_generalist_expert | 27/27 (100.0%) | 87.5%–100.0% | None |
| no_robots | base | 27/32 (84.4%) | 68.2%–93.1% | roleplay_expert ×4, reasoning_expert ×1 |
| pii | privacy_dlp_expert | 29/30 (96.7%) | 83.3%–99.4% | legal_contract_expert ×1 |
| text2cypher | text_to_cypher_expert | 18/18 (100.0%) | 82.4%–100.0% | None |
| verilogeval | verilog_expert | 17/26 (65.4%) | 46.2%–80.6% | coding_expert ×9 |

## Paired replay comparison

On the same 300 questions: historical v7 267/300 (89.0%); current v7 267/300 (89.0%).

Current version alone correct: 0. Historical version alone correct: 0.

This is a regression comparison on previously evaluated data, not a fresh holdout.

## Conditions and limits

- Backend: mlx. Model: `mlx-community/Qwen3.5-9B-5bit @ 64b670725ad59d26c7b77430a3a4ae671afb776a`.
- Hardware: Apple M1 Pro, 16 GB RAM.
- Router only. No answer generation, LoRA loading, web search or answer-quality scoring.
- Every tree expert counts as enabled. Labels come from the source dataset's designated task.
- Long requests use the normal 600-token head/tail cap: first 450 tokens and last 150.
- The fresh sample excludes all 1538 IDs in the saved historical routing runs.
- HumanEval is absent from the fresh sample because all 164 tasks had already been routed.
- Novelty is relative to saved local routing runs, not to model or adapter training data.
- Labels are task-based; ambiguous prompts and alternate valid experts are not judged independently.
- Samples have different source mixes. Do not treat their accuracy difference as a model improvement.
- Wilson intervals describe this sample under a binomial approximation; they do not predict production accuracy.
- Replay ran first, including a cold initial request. Fresh questions followed with the model/system cache warm.
- Medians and p90 are routing latency, not full response latency. The Mac was not isolated from other apps.
- Only this model/backend/hardware combination was rerun. Other v7 profiles remain unmeasured.
- See `run.json` for the source fingerprints, runtime versions and dirty-working-tree marker.
- Prompts and answers are omitted. `inputs-manifest.json` contains IDs, labels and prompt SHA-256 hashes.
- `checksums.json` covers the archived inputs to this report; `summary.json` and `report.md` are derived.

Recompute this report without a model or network:

```bash
python scripts/router_results.py benchmarks/2026-11-12-router-v7
```
