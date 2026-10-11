# Benchmarking GhostSwitch on an NVIDIA GPU

`scripts/benchmark.py` runs MMLU-Pro, GSM8K and GPQA Diamond against GhostSwitch and against
the same base model without experts and verifier. This page sets it up on a Linux server
with an NVIDIA RTX card, picks the best LoRA adapters on held-out questions, and then runs
the comparison.

Two rules keep the result honest:

- **Choose on one set of questions, report on another.** Adapters are compared on
  `gsm8k_dev` (GSM8K train), `gpqa_dev` (GPQA main without Diamond), and MMLU-Pro items
  after `--offset 140`. The reported numbers use GSM8K test, GPQA Diamond and the first 140
  MMLU-Pro items, which no choice was based on.
- **Same model, same runtime, same prompt.** Both systems run through the GhostSwitch server
  on the same GGUF file. The base model uses `--mode base --no-verify`. Web search stays off,
  because it could find the answers.

## 1. Server setup

```bash
git clone git@github.com:hiddenway/ghostswitch.git && cd ghostswitch
python3 -m venv .venv && . .venv/bin/activate
CMAKE_ARGS="-DGGML_CUDA=on" pip install -r requirements-gguf.txt     # llama.cpp with CUDA
pip install -r requirements-bench.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu      # LoRA conversion only
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp
export LLAMA_CPP_DIR=~/llama.cpp
hf auth login     # GPQA is gated: accept its terms on huggingface.co/datasets/Idavidrein/gpqa first
python -m core.model_download --backbone qwen35-9b --backend gguf      # 5.6 GB, Q4_K_M
```

Prepared adapters can also be copied from a Mac that already converted them, which saves
the conversion step:

```bash
rsync -av ~/.cache/ghostswitch/gguf-lora/ server:~/.cache/ghostswitch/gguf-lora/
```

## 2. Choose the adapters

Every candidate was converted and checked tensor by tensor against its source, including
llama.cpp's reordering of the linear-attention heads. Files are in
`~/.cache/ghostswitch/gguf-lora/candidates/` (or `qwen35-9b/` for the ones GhostSwitch ships).

| File | Source (pinned revision) | Size | Trained modules | Notes |
|---|---|---|---|---|
| `qwen35-9b/reasoning-filtered.gguf` | `cudabenchmarktest/qwen3.5-9b-qwen3.6-reasoning-distilled` @ `6a0a3fd`, `filtered_adapter/` | 116 MB | attention and MLP | Current `reasoning_expert`. The filtered copy drops the linear-attention layers, three of every four. |
| `candidates/reasoning-r3-full.gguf` | same repository, root adapter | 173 MB | all, including linear attention | The same training, complete. The author calls R3 an early round with regressions in instruction following. |
| `candidates/reasoning-gpt52-high.gguf` | `kyleliu789/qwen3.5-9b-gpt52-high-reasoning-original` @ `3082d30` | 173 MB | all | GPT-5.2 high-reasoning traces; small (about 580 examples, 3 epochs). |
| `candidates/reasoning-deepseek-v41-flash.gguf` | `kevinzhou21/Qwen3.5-9B-Distill-Deepseek-V4.1-Flash-Lora-20261003` @ `c7046a2` | 58 MB | attention and MLP | DeepSeek V4.1 Flash distillation; no model card. |
| `qwen35-9b/medical-medmcqa.gguf` | `jamezoon/qwen3-5-9b-medmcqa-lora` @ `e049a89` | 14 MB | attention only | Current `medical_reasoning_expert`; MedMCQA multiple choice. |
| `candidates/medical-clinical-think.gguf` | `Giordanopsouza/qwen3.5-9b-clinical-lora-v2-think` @ `ce7b8ac` | 87 MB | all text layers (vision layers dropped) | Clinical answers with thinking. |
| `candidates/medical-med1.gguf` | `hariharanv04/qwen35-9b-med1-bf16-adapter` @ `8fa3a21` | 87 MB | all | No model card. |
| `qwen35-9b/coding-agentic.gguf` | `blackshell69/Qwen3.5-9B-distilled-coding-agentic` (ready GGUF) | 116 MB | attention and MLP | Current `coding_expert`; MMLU-Pro computer science. |

Left out:

- `MaliDDD/gpqa_marginal_9b-sft_1.0` and `MaliDDD/mmlu_marginal_9b-sft_1.0` are trained on
  trajectories built from GPQA and MMLU questions themselves, in a tool-calling format. They
  would contaminate the test.
- The popular "Opus reasoning distilled" models (Jackrong v1/v2, `cudabenchmarktest` R7,
  Kerassy medical) are published as full 18 GB models, not adapters. A LoRA could be
  extracted from them on this server (mergekit-extract-lora, about 40 GB of downloads), but
  their published evaluations are small (R7: 38 questions).

Build `llama-server` with CUDA and load every candidate without applying it:

```bash
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DGGML_CUDA=ON && cmake --build ~/llama.cpp/build -j --target llama-server
A=~/.cache/ghostswitch/gguf-lora
~/llama.cpp/build/bin/llama-server -m base/qwen35-9b/gguf/Qwen3.5-9B-Q4_K_M.gguf --jinja -ngl 99 \
  -c 131072 -np 4 --port 8080 --lora-init-without-apply \
  --lora $A/qwen35-9b/reasoning-filtered.gguf --lora $A/candidates/reasoning-r3-full.gguf \
  --lora $A/candidates/reasoning-gpt52-high.gguf --lora $A/candidates/reasoning-deepseek-v41-flash.gguf \
  --lora $A/qwen35-9b/medical-medmcqa.gguf --lora $A/candidates/medical-clinical-think.gguf \
  --lora $A/candidates/medical-med1.gguf
curl -s localhost:8080/lora-adapters      # ids 0–6 in the order above
```

Ask each adapter, and the plain base model, the held-out questions. Thinking is on and the
temperature is 0.6, as Qwen recommends for thinking (greedy decoding can loop). Four slots
share the 131,072-token context, so each request has room for 16,384 tokens of reasoning:

```bash
B="python scripts/benchmark.py run --target openai --url http://127.0.0.1:8080 --thinking on \
   --temperature 0.6 --max-tokens 16384 --concurrency 4 --results results/screen"
$B --suite gsm8k_dev --limit 100 --name base
for id in 0 1 2 3; do $B --suite gsm8k_dev --limit 100 --name lora-$id --lora $id; done
# GPQA and MMLU-Pro include biology and health, so the medical adapters compete there too.
$B --suite gpqa_dev --limit 100 --name base
for id in 0 1 2 3 4 5 6; do $B --suite gpqa_dev --limit 100 --name lora-$id --lora $id; done
$B --suite mmlu_pro --offset 140 --limit 140 --name base
for id in 0 1 2 3 4 5 6; do $B --suite mmlu_pro --offset 140 --limit 140 --name lora-$id --lora $id; done
python scripts/benchmark.py report --results results/screen
```

The report gives accuracy with a 95% interval, accuracy per MMLU-Pro category, and a paired
McNemar test against every other run. Keep an adapter only where it beats the base model:
reasoning adapters on GSM8K, GPQA and the MMLU-Pro STEM categories, medical adapters on
health and biology. With 100 questions, a few points of difference is noise; look at the
p-value. When no candidate beats the current expert, keep it; when none beats the base
model, routing to the base model is the better choice for that domain.

## 3. Run the comparison

Point the winning adapters' `experts/qwen35-9b/<expert>/lora.json` at their source, then
start GhostSwitch on the GPU. Enable the experts the router may pick for these questions; a
question routed to an expert that is not installed falls back to the base model and is
counted in the report.

```bash
python -m core.main --serve --backbone qwen35-9b --backend gguf --port 8000 &
for e in reasoning_expert medical_reasoning_expert coding_expert; do
  curl -s -X POST localhost:8000/v1/experts/$e -H 'Content-Type: application/json' -d '{"action":"enable"}'
done
R="python scripts/benchmark.py run --temperature 0.6 --max-tokens 4096 --results results/final"
for suite in "gsm8k --limit 200" "gpqa" "mmlu_pro --limit 140"; do
  $R --suite $suite --name ghostswitch
  $R --suite $suite --name base-think --mode base --no-verify --thinking on
  $R --suite $suite --name base-nothink --mode base --no-verify --thinking off
done
python scripts/benchmark.py report --results results/final
```

GhostSwitch thinks within `thinking_max_tokens` (4,096) of its execution profile, on top of
the answer's `--max-tokens`; the `base-think` run uses the same allowance. Raise it in
`core/config/execution_profiles/gguf9b_v6.json` if GPQA answers are cut off, together with
`max_tokens_per_call` (thinking plus answer must fit in it); it applies to both runs. A run
can be stopped and started again with the same command; finished answers are kept.

## Coding agent mini benchmark

`scripts/agent_bench.py` measures GhostSwitch as the model of the [OpenCode](https://opencode.ai)
agent on ten small Python tasks with unit tests (six bugs to fix, four functions to write), in
three reasoning modes that OpenCode can send through model variants: `fast` (no reasoning),
`auto` (GhostSwitch decides on each task's first step, its default) and `think` (every step
reasons). Every step goes to `coding_expert` unless `--expert auto` leaves it to the router.

OpenCode runs in a Docker container that sees only a fresh copy of the task folder. Outside a
container `opencode run` treats the enclosing git repository as its project and did not enforce
its permission rules, so the script does not run it on the host. The final tests also run in a
container, because the agent wrote the code.

```bash
docker build -t ghostswitch-agent-bench:1 -f scripts/agent_bench.Dockerfile scripts
python scripts/agent_bench.py check          # each task fails before and passes with its reference
python -m core.main --serve --port 8000 &     # the model to measure
python scripts/agent_bench.py run --url http://127.0.0.1:8000 --name run1
python scripts/agent_bench.py report --name run1
```

Results, OpenCode logs and the task folders go to `results/agent-bench/<name>/`; a stopped run
resumes there. Step counts and the steps that reasoned come from the server's request log.

### Results on 2026-10-11

Qwen3.5-9B MLX with `coding_expert` on an M1 Pro with 16 GB, one run per task and mode
(`run2`):

| Mode | Passed | Median time per task | Total time | Steps | Steps that reasoned |
|---|---:|---:|---:|---:|---:|
| fast | 10/10 | 71 s | 13.8 min | 75 | 0 |
| auto | 9/10 | 97 s | 21.0 min | 94 | 6 |
| think | 10/10 | 125 s | 22.6 min | 56 | 17 |

- Reasoning did not solve more tasks: `fast` passed all ten. `auto` reasoned on the first step
  of six tasks; on the other four it took the same steps as `fast`.
- The Mac was swapping (11–12 GB of swap), so times are noisy: the same step with the same
  prompt and output took 15 s in one run and 42 s in another. Compare step counts and
  passes rather than single times.
- `auto` failed `lru_cache`: its linked list kept failing the tests, it wrote the same file
  twice in a row, and the server stopped the repeated test run (see the loop check in
  `core/main.py`).
- In `think` the coding adapter often wrote a tool call straight after the opened `<think>`,
  which the server then treated as unfinished reasoning and generated again without it; such
  a call is now accepted as the step's action, so these `think` times are higher than they
  would be now.

Before the prompt cache covered agent steps (`run1`, stopped after 11 runs), every step
computed its whole 4,000–5,500-token prompt again: 25–47 s per step and 180–314 s per task.
With the cache, steps took 2–8 s (16 s when writing a file), and the same tasks took 32–112 s.
