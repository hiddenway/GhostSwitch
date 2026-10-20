# GhostSwitch reference

The complete guide to installing, running and extending GhostSwitch. For an overview and a quick start, see the [README](../README.md).

> [!WARNING]
> GhostSwitch is an experiment: behavior, the API, settings and the expert package format may change between versions without notice.

- [Overview](#overview)
- [Install](#install)
- [Setup from source](#setup-from-source)
- [Run](#run): [first run](#first-run), [base models](#base-models), [web search](#web-search), [thinking](#thinking), [images and files](#images-and-files), [benchmarks](#benchmarks)
- [Expert library and community contributions](#expert-library-and-community-contributions)
- [Layout](#layout)
- [Response language](#response-language)
- [Renamed from OmniBrain and MiniBrain](#renamed-from-omnibrain-and-minibrain)

## Overview

GhostSwitch serves one inference step at a time on a shared base model: Qwen3.5-9B by default, or Qwen3.5-4B, each on MLX or GGUF. In `auto` mode it selects one available LoRA expert or the base model. In `direct` mode the caller names an expert; `base` mode answers with the base model alone. Exact checks in code accept the generated answer, request up to four retries with the same specialization, or fail: the answer must be non-empty and complete, agree with arithmetic stated in the request, cite only real web sources, and keep JSON fields to those requested and present in the input. No model reviews the answer: in tests a model reviewer added about 42% latency and no accuracy (see the [verifier audit](verifier-audit.ru.md)). Workflow, tools, memory, and calls to other experts belong to the caller.

## Install

One command installs GhostSwitch and starts it in the browser. Nothing else needs to be installed first: the installer brings its own Python and needs no administrator rights.

macOS (Apple Silicon, macOS 14 or newer) and Linux (x86_64 or ARM64), in Terminal:

```bash
curl -fsSL https://raw.githubusercontent.com/hiddenway/ghostswitch/main/install.sh | sh
```

Windows 10 or 11 (64-bit), in PowerShell:

```powershell
irm https://raw.githubusercontent.com/hiddenway/ghostswitch/main/install.ps1 | iex
```

The web app then opens at `http://127.0.0.1:27246/app/`, walks through the first run and downloads a base model (see **First run**). Afterwards, start GhostSwitch with `ghostswitch` in a new terminal window, or from the Start menu on Windows; closing that window or pressing Ctrl+C stops it. Port 27246 is unassigned and rarely used; when another program holds it, `ghostswitch` takes the next free port up to 27344 and prints the address, and `ghostswitch --port N` chooses one. Chats are saved in the browser per address, so a different port shows a separate chat list. Tools such as OpenCode reach the installed copy at `http://127.0.0.1:27246/v1`. `ghostswitch update` installs the latest release, `ghostswitch uninstall` removes GhostSwitch and asks whether to keep the downloaded models, and `ghostswitch doctor` shows what is installed and what to fix.

The installer picks the build for the computer: MLX on Apple Silicon; on Windows and Linux, llama.cpp with CUDA when an NVIDIA GPU with a CUDA 12 driver is present (Linux needs glibc 2.35 or newer, as in Ubuntu 22.04), and llama.cpp on the processor otherwise. If the NVIDIA build does not load, it installs the processor build instead. Macs with Intel processors and Windows on ARM are not supported. Everything goes into one folder: `~/.ghostswitch` on macOS and Linux, `%LOCALAPPDATA%\GhostSwitch` on Windows. Its `app` and `venv` are replaced on each update; `data` keeps the models, settings, caches and request journal (`GHOSTSWITCH_HOME` points the app there). The library versions come from `uv.lock`, so every installation runs the tested set.

Installer options are environment variables, set before the command (for example `curl -fsSL … | GHOSTSWITCH_NO_START=1 sh`, or `$env:GHOSTSWITCH_NO_START = "1"` in PowerShell): `GHOSTSWITCH_VARIANT` (`mlx`, `cpu` or `cuda`), `GHOSTSWITCH_VERSION` (a release tag), `GHOSTSWITCH_REF` (a branch, tag or commit instead of a release), `GHOSTSWITCH_SOURCE` (a local folder or archive; from a git checkout only its project files are copied, without `.venv` or models), `GHOSTSWITCH_ROOT` (the install folder), `GHOSTSWITCH_NO_START=1` and `GHOSTSWITCH_NO_MODIFY_PATH=1`. A chosen variant, ref or source is kept for `ghostswitch update`; `GHOSTSWITCH_VARIANT=auto`, `GHOSTSWITCH_REF=release` or `GHOSTSWITCH_SOURCE=release` returns to the default. To try the installer on a checkout before a release, run `GHOSTSWITCH_SOURCE=. GHOSTSWITCH_INSTALLER_URL="file://$PWD/install.sh" sh install.sh` in it; `ghostswitch update` then reinstalls from the checkout.

Releases come from tags: pushing `v6.0.0` (the version in `pyproject.toml`) runs `.github/workflows/release.yml`, which publishes `ghostswitch.tar.gz`, `ghostswitch.zip` and `SHA256SUMS`; the installers check the archive against them. `.github/workflows/installer.yml` installs, starts, updates and removes GhostSwitch on clean macOS, Ubuntu and Windows runners.

## Setup from source

With [uv](https://docs.astral.sh/uv/), `uv sync --extra mlx` (Apple Silicon), `--extra cpu` or `--extra cuda` creates `.venv` with the locked versions from `uv.lock`, and `uv run python -m core.cli` starts the server and opens the web app like the installed `ghostswitch` command; `python -m core.cli doctor` checks the setup. Without uv, the requirements files below install the same libraries without pinned versions.

For MLX, install `requirements-mlx.txt` (or `requirements.txt`). No model needs to be downloaded by hand: start the server (see **Run**) and open the web app, which walks through how GhostSwitch works and downloads a base model on the first run (see **First run**). Each base model reads its checkpoint from `base/<backbone>/model/` (for example `base/qwen35-9b/model/` and `base/qwen35-4b/model/`; a directory or a symlink); use `--model-path` for another validated checkpoint.

For GGUF, install `requirements-gguf.txt` with a platform-appropriate `llama-cpp-python` build. The GGUF file is found automatically in `base/<backbone>/gguf/`, LM Studio's models folder, or the Hugging Face cache (`Qwen3.5-9B-Q4_K_M.gguf`, `Qwen3.5-4B-Q4_K_M.gguf`); `--model-path` or the web app can name another file. The tokenizer and chat template still come from `base/<backbone>/model/`. Ready GGUF LoRAs download from the library directly. For PEFT-to-GGUF conversion, install the converter dependencies and set `LLAMA_CPP_DIR`:

```bash
pip install -r requirements-gguf-tools.txt
export LLAMA_CPP_DIR=/path/to/llama.cpp
# Optional manual preparation of selected specialties:
python scripts/prepare_gguf_adapters.py --backbone qwen35-9b --experts coding_expert reasoning_expert finance_market_expert
```

Prepared GGUF adapters use `~/.cache/ghostswitch/gguf-lora/<backbone>` by default (`GHOSTSWITCH_GGUF_LORA_DIR` overrides this). MLX adapters are prepared in the background or on direct demand and use `~/.cache/ghostswitch/lora` (`GHOSTSWITCH_LORA_CACHE` overrides this).

## Run

Start the HTTP endpoint:

```bash
python -m core.main --serve                                  # the model chosen in the web app; 9B on MLX at first
python -m core.main --serve --backbone qwen35-4b --backend gguf
python -m core.start_mlx --serve                             # always MLX
python -m core.start_gguf --model-path /absolute/path/Qwen3.5-9B.gguf --serve
```

The server listens on `127.0.0.1:8000` by default (the installed `ghostswitch` command uses port 27246). Open the web client at `http://127.0.0.1:8000/app/` for saved local chats, LoRA selection, response telemetry, and API documentation. Its source is in [`client/`](../client/README.md).

Send one step to `POST /v1/chat/completions`:

```json
{"model":"ghostswitch-v6","messages":[{"role":"user","content":"Write a Python function to normalize names"}],"mode":"auto","max_tokens":4096}
```

The default generation temperature is `0.7`; an explicit `temperature` overrides it.

For a fixed specialist, use `"mode":"direct","expert_id":"coding_expert"`. The response has OpenAI-style `choices` and `usage`, plus a `ghostswitch` field for route, adapter, verifier, thinking, backend, and latency. Unsupported capabilities return HTTP 422 with `status: "unsupported"`: `routing.missing_capability` names the catalog expert that fits but is not enabled, and the error explains how to enable it (`POST /v1/experts/{id}` with `{"action":"enable"}`) or to resend with `"mode":"base"`. The chat shows this as a card that downloads the expert and answers, or answers with the base model; inference or verification failures return HTTP 500 with `status: "failed"`. The endpoint supports text, images and files (see **Images and files**) and function tool calls, including SSE streaming. External tools are executed by the caller. See [OpenCode integration](../integrations/opencode/README.md) for configuration and the agent protocol. `GET /health` reports server health.

### First run

A server with no base model still starts and serves the web app; `GET /health` then reports `setup_required: true`, and chat requests return HTTP 503 with `model_not_installed`. The app opens a four-step introduction (what GhostSwitch is, how an answer is made, experts and web search, base model) and cannot be closed until a model runs. The last step shows the processor, memory, GPU and free disk space, and recommends a backend (MLX on Apple Silicon, otherwise GGUF) and the larger model whose estimated memory need is at most 55% of the memory that holds it (system memory, or an NVIDIA GPU's own memory for GGUF); otherwise the 4B. Each choice shows its download size and whether it runs comfortably, tightly (up to 80%), or not at all.

The download comes from pinned Hugging Face revisions recorded in `core_models/backbones.json` (`lmstudio-community/Qwen3.5-9B-MLX-4bit`, `mlx-community/Qwen3.5-4B-4bit`, and the `lmstudio-community` Q4_K_M GGUF files, with the tokenizer files GGUF needs), into `base/<backbone>/model/` or `base/<backbone>/gguf/`. It runs in a child process, so it can be cancelled; partial files are kept and the next attempt resumes them. The server checks free disk space first and validates the checkpoint afterwards; a server without a model then switches to the new one. `POST /v1/runtime/install` with `{"backbone":"qwen35-9b","backend":"mlx"}` starts a download and `{"action":"cancel"}` stops it (same origin only); `GET /v1/runtime/install` reports its progress. **Settings → Show introduction** replays the introduction, and choosing a model that is not downloaded in **Settings → Base model** opens the download step.

### Base models

Qwen3.5-9B on MLX is the default. Open **Settings → Base model** in the web app to switch to another base model or backend: the server saves the choice in `~/.config/ghostswitch/runtime.json` (`GHOSTSWITCH_RUNTIME_STATE` overrides the path), restarts its own process, and loads the new model on the first request, which also releases the previous model's memory. A switch waits for a running request. `GET /v1/runtime` lists the base models with their MLX/GGUF availability, download sizes and memory estimates, the hardware report and recommendation; `POST /v1/runtime` with `{"backbone":"qwen35-4b","backend":"mlx"}` (optionally `gguf_path`) switches from the same origin. Command-line flags win over the saved choice.

`qwythos-9b` is [Qwythos-9B v2](https://huggingface.co/empero-ai/Qwythos-9B-v2), an uncensored Qwen3.5-9B fine-tune (MLX: `ahmedandaloes/Qwythos-9B-v2-MLX-4bit`; GGUF: `empero-ai/Qwythos-9B-v2-GGUF` Q4_K_M). The app marks it **Uncensored**. Because it is a Qwen3.5-9B fine-tune, it uses the Qwen3.5-9B expert library (`"expert_library": "qwen35-9b"` in `core_models/backbones.json`) and accepts Hub LoRAs trained on either `Qwen/Qwen3.5-9B` or `empero-ai/Qwythos-9B-v2`; adapters added from the Hub are kept per base model. Expert evaluations were measured on Qwen3.5-9B. Image input works on GGUF only, because the MLX checkpoint has no vision weights.

Each base model has its own expert library in `experts/<backbone>/`; adapters are trained for one base and cannot be shared. Qwen3.5-4B currently has the migrated receipt extraction (enabled by default), trademark classification, Linux shell, English→Kannada translation, medical and finance adapters.

Memory on a 16 GB Mac is the limiting factor. The settings page shows roughly how much each choice needs and how much is free; the estimates for the current files are 7.8 GB (9B MLX), 6.7 GB (9B GGUF), 5.1 GB (4B MLX) and 4.0 GB (4B GGUF). Close other large applications before switching to a model that needs more than is free. The runtime bounds its caches: `GHOSTSWITCH_MLX_CACHE_MB` (MLX buffer cache, default 1024), `GHOSTSWITCH_PROMPT_CACHE_MB` (256), `GHOSTSWITCH_HISTORY_CACHE_MB` (1024) and `GHOSTSWITCH_ROUTER_CACHE_MB` (512), and it returns freed MLX buffers after every request. On MLX, experts that were used stay prepared in memory so that switching back is instant, within `GHOSTSWITCH_LORA_MEMORY_MB` (512): beyond it the least recently used adapters are dropped and prepared again from the disk cache when needed, which takes under a second. The expert in use is always kept, and `0` keeps only that one. The memory need therefore does not grow with the size of the expert library or with how many experts a session uses. GGUF loads an expert's adapter for each call and frees it afterwards. GGUF reserves the whole KV cache when the model loads, so its context is 32,768 tokens (`GHOSTSWITCH_GGUF_CONTEXT` overrides it), with flash attention on (`GHOSTSWITCH_GGUF_FLASH_ATTN=0` turns it off).

Every request repeats fixed instructions: the router's node prompts and, for some experts, a long expert prompt. They come before the current date and time in their system messages, so the model state after them is computed once and reused from the prompt cache: the router has its own budget, and expert prompts of at least 256 tokens share `GHOSTSWITCH_PROMPT_CACHE_MB` (one entry holds about 70 MB for 4B, mostly the state of the linear-attention layers). On GGUF the llama.cpp context state after these instructions is kept within the same budget. The search planner's short instructions are not cached. `usage.prompt_tokens_details.cached_tokens` counts prompt tokens taken from the cache, and **Details** shows them under input tokens and per stage. `usage.prompt_tokens` still counts every prompt token of every call in the request, including routing.

An agent such as OpenCode resends the whole conversation on every step, with the tool definitions ahead of it (Qwen3.5's template puts them before the system text, so they belong to the cached prefix). After each step the model state at the end of the conversation is kept, and the next step computes only the reply and tool result it added; a step's state replaces those of the earlier steps of its task, so one task holds one state, about 200 MB after 5,000 tokens. `GHOSTSWITCH_HISTORY_CACHE_MB` bounds these states on MLX and GGUF; a conversation whose state alone exceeds it is computed in full on every step. Agent prompts carry the date without the time, which would otherwise change the prompt every minute. On an M1 Pro with the 9B model this took an OpenCode step from 25–47 seconds to 2–8 seconds.

### Web search

A request can use the web with `"web_search": "auto"` or `"on"` (default `"off"`; the chat composer has a globe toggle). With `auto`, one short model call decides whether the question needs current or specific facts (news, releases, prices, schedules, specific people or products, or an explicit request to look something up) and writes a query from the conversation; requests that name news, weather, exchange rates, prices, or a web lookup always search. `on` always searches. The server reads up to five result pages in parallel and gives their visible text to generation as numbered sources to cite as `[1]`, `[2]`. The server rejects citation numbers that match no source; whether sourced facts are true or complete is not checked. Every request also carries the server's date and time, so questions about today need no search. Responses list the sources in `ghostswitch.web_search`, and the request stages show the query and pages read. Requests with tools are left to the agent harness.

Providers are DuckDuckGo (default, no key), the Brave Search API (`brave_api_key`, or `BRAVE_API_KEY`), and a self-hosted SearXNG (`searxng_url`). Choose one in **Settings → Web search** or with `POST /v1/search/settings` from the same origin; settings live in `~/.config/ghostswitch/search.json` (mode 600, `GHOSTSWITCH_SEARCH_SETTINGS` overrides the path) and the key is never returned. `POST /v1/search` with `{"query": "..."}` returns the results with page text for other clients. Page text is untrusted: it is only fetched from public http(s) addresses (local and private networks are refused, including after redirects), limited to 1.5 MB and 8 seconds per page, stripped of scripts and navigation, and presented to the model as data. Results are cached for ten minutes. A search sends the query to the chosen provider and fetches the result pages from this machine.

### Thinking

Thinking is selected per request, for the base model and every expert alike. An explicit `reasoning_effort` (`none` or `minimal` off; `low`, `medium` or `high` on) or `chat_template_kwargs.enable_thinking` decides first; the web chat sends it from its Thinking switch. Next, an expert's `response_policy.json` may fix it with `"thinking": true` or `false` (`reasoning_expert` sets true). Otherwise a short call to the base model (`core/thinking_decision.py`, about 0.7 s for 9B and 0.5 s for 4B on an M1 Pro) names the kind of request: math, logic, proof, planning, engineering (implementing or changing program logic, algorithms, refactoring, finding bugs) and comparison reason first; snippets (an obvious few-line function), mechanical edits (renames, docstrings, formatting, versions, running a command), facts, language, creative work and chat answer directly. On 122 labeled requests (chat messages, follow-ups and coding-agent tasks, 31 of them held out while the kinds were tuned) Qwen3.5-9B chose reasoning for all 61 tasks and answered all 61 plain requests directly; Qwen3.5-4B chose reasoning for 58 of 61 and answered all plain requests directly. That check runs only when the execution profile sets `"auto_thinking": true` and the model runs on a GPU (MLX, or llama.cpp with a GPU; on a processor alone reasoning takes minutes); otherwise thinking stays off unless requested. In an agent task (a request with tools), only the first step after a user message is decided this way, where the plan is made; steps after tool results answer directly, also for an expert whose policy thinks (source `agent_step`). Router and search planner calls never think.

The answer stays in `content`. The reasoning is returned separately in `choices[0].message.reasoning_content`, and with `stream: true` it arrives first in `delta.reasoning_content` chunks. The exact checks read only the answer. `ghostswitch.thinking` reports `enabled`, `source` (`request`, `agent_step`, `expert_policy`, `auto` with the request `kind`, or `default`), and whether the accepted answer `used` reasoning; the automatic check is a `thinking` stage. Reasoning has its own allowance, `thinking_max_tokens` in the execution profile, on top of `max_tokens`; if it does not finish within that allowance, the next attempt answers without thinking.

A caller can override the policy for any route: `reasoning_effort` `none` or `minimal` turns thinking off, and `low`, `medium`, or `high` turns it on (the level does not change the allowance). `chat_template_kwargs: {"enable_thinking": false}` is accepted as an equivalent. Reasoning echoed back in earlier assistant messages (`reasoning_content` or `reasoning`) is accepted and not sent to the model again.

### Images and files

A user message can carry images and files as OpenAI content parts: `{"type":"text","text":"..."}`, `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}` and `{"type":"file","file":{"filename":"report.pdf","file_data":"data:application/pdf;base64,..."}}`. Only base64 `data:` URLs are accepted; the server never downloads a URL. In the web app, attach with the paperclip, by pasting, or by dropping files on the chat.

Images (PNG, JPEG, WebP, BMP, the first frame of a GIF) go to the base model's own vision encoder. On MLX, the checkpoint's `vision_tower` weights load on the first image (about 0.4 GB for 4B, 0.9 GB for 9B) and stay loaded; the language model and its LoRA adapters are unchanged, with Qwen's multimodal positions for image tokens. On GGUF, llama.cpp needs the model's vision projector `mmproj-*.gguf`: it is found next to the model file, in `base/<backbone>/gguf/`, LM Studio's models folder or the Hugging Face cache, or downloaded from the pinned repository on the first image (`mmproj` in `core_models/backbones.json`; `GHOSTSWITCH_GGUF_MMPROJ` names another file). Each image is scaled to at most `GHOSTSWITCH_IMAGE_MAX_PIXELS` (default 1,048,576, about 1,000 prompt tokens), and the model sees the latest `GHOSTSWITCH_MAX_IMAGES` (default 4) images of the conversation; earlier ones remain as labels. `GET /health` reports `vision`; a checkpoint without vision weights rejects images with HTTP 400.

Files become text: UTF-8 or UTF-16 text and code, the text layer of a PDF, and DOCX paragraphs, each cut at `GHOSTSWITCH_MAX_FILE_CHARS` (default 100,000) characters with a visible note and wrapped as `<file name="...">…</file>`. A scanned PDF without a text layer and other binary files are rejected with a hint. The message text labels each image as `[Image 1: photo.png]`, so routing and search planning, which read text, know it is there; the model sees each image right after its label. Routing works as for text, so a selected LoRA expert also answers requests with images, although adapters were trained on text. Chat request bodies may be up to 64 MB and each attachment up to 20 MB; the request journal keeps only the type and size of attached data. Images and PDF text need `Pillow` and `pypdf` from `requirements-common.txt`.

For a single JSON request on stdin:

```bash
printf '%s\n' '{"messages":[{"role":"user","content":"Hello"}]}' | python -m core.start_mlx --request-json -
```

Python callers can use `core.main.run_request(payload)` or `core.main.run(query, mode="auto")` after configuring the backbone with `core.llm.configure_backbone(...)`.

### Benchmarks

`scripts/benchmark.py` asks MMLU-Pro, GSM8K and GPQA Diamond questions of GhostSwitch or of any OpenAI-compatible server (such as `llama-server` with LoRA adapters), records every answer, resumes interrupted runs, and reports accuracy with 95% intervals, per-category and per-expert results, and paired McNemar tests between runs. `"verify": false` in a request turns the verifier off, so `--mode base --no-verify` measures the base model through the same runtime. [Benchmarking on an NVIDIA GPU](benchmarking.md) covers setup, the candidate adapters, choosing them on held-out questions, and the final comparison.

`scripts/verifier_eval.py` measures the text verifier itself: on the labeled answers in `core/tests/data/verifier_cases.jsonl` (arithmetic, annuity, receipt extraction, translation, dates, web answers; most taken from the request journal) it counts wrong answers accepted and right answers retried. The server runs the exact checks only; the script also asks the switched-off model reviewer (`verify(..., review=True)`) so the two can be compared. `--llm stub` measures the exact checks alone, `--llm live` runs the MLX reviewer and stores its outputs in `core/tests/data/verifier_llm_outputs.jsonl`, and `--llm replay --baseline HEAD` compares the working tree with a git ref offline from those outputs.

## Expert library and community contributions

At the first startup of Qwen3.5-9B, GhostSwitch queues three basic specialties: **coding**, **reasoning**, and **finance**; Qwen3.5-4B starts with **receipt extraction**. The API opens while these download. Auto routing uses only enabled, prepared experts and can use the base model while downloads are pending. A direct request for an enabled expert waits for its preparation. Weights are attached only for inference; downloading does not load every expert into RAM.

Open **Experts** in the frontend to search all compatible packages, **Download and enable**, **Disable**, inspect the Hugging Face source, or retry a failed download. Switching an expert off excludes it from routing and direct requests, while preserving its disk cache. On later starts, GhostSwitch prepares the user's saved selection, including a deliberately empty selection. Downloads run one at a time and show queue, download, ready or error states.

Selection is stored per backbone/backend in `~/.config/ghostswitch/experts.json`; override the path with `GHOSTSWITCH_EXPERT_STATE`. It is shared by browsers connected to that server. A download completed after disabling an expert leaves it disabled. Changing defaults in Git does not reset existing preferences. Private or gated sources need an existing Hugging Face login and access; access/network/conversion errors appear on the expert card.

### Verified, official and Hugging Face experts

An expert package may carry an `evaluation.json` with the result of comparing it with the bare base model on a public benchmark (`scripts/expert_eval.py`). An expert that beats the base model is shown as **Verified**. An expert that is no better is blocked: the router never selects it and its card shows the reason instead of the enable button, while `"mode": "direct"` still reaches it so it can be measured again. Repositories the Hugging Face search must never offer, such as adapters trained on benchmark questions, are listed in `experts/<backbone>/blocklist.json`. The first measurement and the reason for each block are in [docs/expert-blocklist.ru.md](expert-blocklist.ru.md).

The **Hugging Face** tab of the expert library searches Hugging Face for other LoRA adapters made for the running base model. They have no curated description, prompt or measurement, so automatic routing never uses them: the user picks one for a chat. Adding one writes a small package pinned to the repository's current commit under the user's configuration folder.

### Add a new LoRA / expert through Git

**The contribution standard is one data-only package in `experts/<backbone>/<expert_id>/` (for example `experts/qwen35-9b/`):**

- `expert.json`: schema version, ID, name, description, capabilities, aliases, keywords, backbone, routing parent and selection default.
- `lora.json`: Hugging Face repository URL, pinned 40-character commit SHA, PEFT config/weights paths, and optional GGUF adapter source.
- `prompt.txt`: specialist instructions that inherit the shared language policy.
- `README.md`: source/model card, license, scope, examples and limitations.
- Optional `response_policy.json`: generation guidance, and whether the expert thinks before answering (`review_guidance` reaches only the switched-off model reviewer).

**Store the Hugging Face reference in Git, never the large LoRA weights.** Start from `experts/_template`, replace every placeholder, and validate with `python scripts/validate_experts.py`. Keep community entries disabled by default. New folders automatically join the semantic catalog, adapter registry and routing tree on restart; no global expert list edits are needed. Users pull the contribution, restart, and select it in the library. See the [complete package standard and examples](../experts/README.md).

### Library API

`GET /v1/experts` returns the full compatible library, including `enabled`, `downloaded`, `status`, `error`, `default_enabled`, `source_url`, `revision` and `progress`. While preparing, `progress` contains `phase`, the current `file`, `downloaded_bytes`, `total_bytes` and `percent` (null until the size is known). The percentage describes the current file, including resumed/cached bytes; conversion and saving have a separate stage with an indeterminate indicator. HTTP and Xet downloads both report progress. Xet also reports `received_bytes` and `bytes_per_second`: network transfer can advance before file reconstruction, so the UI shows received data separately and uses an activity indicator when a file percentage has not advanced. The frontend refreshes visible library progress every second.

```bash
curl -X POST http://127.0.0.1:8000/v1/experts/coding_expert \
  -H 'Content-Type: application/json' -d '{"action":"enable"}'
```

Actions are `enable` (save selection and prepare), `disable` (save deselection, keep cache), and `download` (prepare without changing selection). Long preparations return HTTP 202; poll `GET /v1/experts` for completion. Installation accepts registered expert IDs, not arbitrary URLs. Browser writes require the app and API to share an origin. `GET /health` reports HTTP availability, not completion of model downloads.

## Layout

- `core/main.py` is the v6 one-step runtime and `core/api.py` is the HTTP adapter. `core/runtime_selection.py` saves the base model and backend choice and restarts the process to switch; `core_models/backbones.json` describes each base model.
- `core/expert_registry.py` joins each discovered expert package with its prepared adapter and shared runtime policies. `core/expert_library.py` persists selections and prepares downloads.
- `core/router_v6.py` selects one specialization through the validated `core/config/router_tree.json` hierarchy; the model sees only the current node and its direct children. General requests can select the general branch. A selected specialization without an adapter returns `unsupported`. Package routing leaves are composed with the branch scaffold automatically. `core/verifier_v6.py` checks one result with exact checks in code.
- `core/llm.py` and `core/lora_runtime.py` retain backend inference and LoRA lifecycle.
- `core/config/execution_profiles/*_v6.json` set generation and retry limits.
- `docs/architecture-v6.ru.md` is the design specification.

Run the tests with `.venv/bin/python -m unittest discover -s core/tests -t .`, or only the v6 checks with `.venv/bin/python -m unittest core.tests.test_v6 -v`.

## Response language

All experts inherit the same response-language policy on MLX and GGUF. Explicit caller/user language instructions and established user preferences take priority; otherwise prose follows the latest user's language. An adapter's training language does not force its output language. Translation targets, source quotations, code, tool arguments, schema keys, and fixed labels retain their required form. Routing preserves domain and jurisdiction limits. The response language is not checked after generation, so an adapter's language bias can still surface in the answer. This runtime policy does not retrain adapter weights.

## Renamed from OmniBrain and MiniBrain

GhostSwitch was called OmniBrain and, before that, MiniBrain. Setups made under either name keep working:

- `OMNIBRAIN_*` and `MINIBRAIN_*` variables apply when the matching `GHOSTSWITCH_*` one is unset; the installers read `OMNIBRAIN_*` too.
- `~/.config/omnibrain`, `~/.cache/omnibrain` (or the `minibrain` folders) stay in use until `~/.config/ghostswitch` or `~/.cache/ghostswitch` exists.
- A copy installed in `~/.omnibrain` or `%LOCALAPPDATA%\OmniBrain` is updated where it is, with its models and settings. The update adds the `ghostswitch` command and keeps `omnibrain` working.
- The API accepts the model ids `omnibrain-v6` and `minibrain-v6` and the `X-OmniBrain-Client` and `X-MiniBrain-Client` headers. Chats saved in the browser and entries in the request journal move their telemetry to the new field on their own.

One thing changed for API clients: responses carry their telemetry in `ghostswitch` (and stream `ghostswitch_status`, `ghostswitch_attempt` and `ghostswitch_reset`) instead of `omnibrain`.
