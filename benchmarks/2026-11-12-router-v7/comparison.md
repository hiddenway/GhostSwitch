# GhostSwitch router evaluation comparison

Correct expert or base model selection. These are routing scores, not answer-quality scores.

| Task / metric | Current v7: fresh | Current v7: replay | Historical v7 | Historical v6 |
|---|---|---|---|---|
| Run date | 12 Nov 2026 | 12 Nov 2026 | 30 Oct 2026 | 27 Oct 2026 |
| Questions | 300 fresh | 300 reused | Same 300 as replay | 1,000 historical |
| Hardware | M1 Pro / 16 GB | M1 Pro / 16 GB | M1 Pro / 16 GB | RTX 3090 / 24 GB |
| Model / backend | 9B 5-bit / MLX | 9B 5-bit / MLX | 9B 5-bit / MLX | 9B Q4_K_M / GGUF |
| Overall routing | 89.7% (269/300) | 89.0% (267/300) | 89.0% (267/300) | 87.6% (876/1000) |
| Specialist routing | 92.3% (227/246) | 91.9% (237/258) | 91.9% (237/258) | 89.4% (737/824) |
| Base retention | 77.8% (42/54) | 71.4% (30/42) | 71.4% (30/42) | 79.0% (139/176) |
| Contracts / CUAD | 100.0% (46/46) | 100.0% (35/35) | 100.0% (35/35) | 93.0% (93/100) |
| Finance / FinQA | 60.0% (24/40) | 71.1% (27/38) | 71.1% (27/38) | 60.0% (60/100) |
| Math / GSM8K | 100.0% (49/49) | 100.0% (35/35) | 100.0% (35/35) | 98.3% (118/120) |
| Coding / HumanEval+ | Not sampled | 100.0% (49/49) | 100.0% (49/49) | 100.0% (164/164) |
| Medical: stay on base / MedQA | 80.0% (16/20) | 30.0% (3/10) | 30.0% (3/10) | 10.0% (4/40) |
| Formal proofs / miniF2F Lean 4 | 100.0% (10/10) | 100.0% (27/27) | 100.0% (27/27) | 96.2% (77/80) |
| Everyday: stay on base / No Robots | 76.5% (26/34) | 84.4% (27/32) | 84.4% (27/32) | 99.3% (135/136) |
| Privacy / PII | 100.0% (46/46) | 96.7% (29/30) | 96.7% (29/30) | 76.0% (76/100) |
| Graph queries / Text2Cypher | 94.2% (49/52) | 100.0% (18/18) | 100.0% (18/18) | 96.2% (77/80) |
| Hardware code / VerilogEval | 100.0% (3/3) | 65.4% (17/26) | 65.4% (17/26) | 90.0% (72/80) |
| Median routing time | 1.60 s | 1.79 s | 1.71 s | 1.99 s |
| 90th percentile time | 2.96 s | 3.20 s | 2.84 s | 2.59 s |
| Misses to base | 18 | 10 | 10 | 54 |
| Misses to wrong expert | 13 | 23 | 23 | 70 |
| Runtime exceptions | 0 | 0 | 0 | 0 |

## How to read this comparison

- Only current v7 replay and historical v7 use exactly the same 300 questions. Their correct/incorrect outcomes are identical.
- Fresh v7 and historical v6 have different question mixes and conditions. Columns do not establish an accuracy improvement or speedup.
- The fresh set excludes 1,538 IDs from saved local runs, not model training data. No fresh HumanEval tasks remained.
- Fresh Verilog has only three questions; 3/3 is not strong evidence of general accuracy.
- Historical v6 used three processes sharing one RTX 3090. Historical metadata was recorded after the run; no source fingerprint was captured.
- Current v7 source fingerprint: 240805643124; historical v7: 1c4dd6c80a82.
- All runs use Qwen3.5 9B. Model/backend/hardware differ across v6 and v7.
- Routing times exclude answer generation and LoRA loading. Current fresh timings use a warm model/system cache; the Mac was not isolated.
- Fresh overall 95% Wilson interval: 85.7% to 92.6%. Counts in each cell show correct / tested.
- Full methodology and uncertainty: [evaluation report](../../docs/evaluation.md).

Derived from the archived routes and cohort manifests in this directory.
