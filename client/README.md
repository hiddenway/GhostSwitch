# GhostSwitch chat workspace

A dependency-free web workspace served by the GhostSwitch v6 API. No frontend build is needed. KaTeX and SVG icons are bundled locally.

```bash
./.venv/bin/python -m core.start_mlx --serve --port 8765
# http://127.0.0.1:8765/app/
```

For GGUF, use `python -m core.start_gguf --model-path /absolute/path/model.gguf --serve`.

## Workspace

- **Language:** all interface text, tooltips, accessibility labels, dates, download statuses, and API examples are in English. User conversations retain their original content.
- **Contributions:** **Upload expert / LoRA** opens a short GitHub contribution guide and links to the repository README.
- **Layout:** chat opens by default. The sidebar contains chat, experts and conversations; statistics, request history and API are grouped under **Tools**. Both themes use neutral surfaces with a blue accent. On narrow screens the sidebar opens from the menu button.
- **Chat:** streaming Markdown and formulas, a single auto/direct expert selector in the composer, and temperature/token limits under **Response settings**. Independent drafts, pinned conversations, search and message actions remain available. Renaming and Markdown export are in the chat header menu. Editing or regenerating creates a new conversation with earlier context. Shared response IDs are counted once in metrics.
- **Attachments:** the paperclip, paste, or drag and drop onto the chat adds up to eight images or files to the next message (images up to 20 MB, PDF, DOCX, text and code files up to 10 MB). Images larger than 2,048 px are scaled down in the browser first. The composer shows each attachment with a preview or size and a remove button; sent messages show image previews and file chips that open the original. Images are disabled with a notice when the server reports a model without vision. Editing, retrying and answering again keep the message's attachments.
- **First run:** a four-step introduction explains GhostSwitch, how an answer is made (route, search, answer, verify), and experts and web search, then shows this computer's processor, memory and free disk, recommends a base model and backend, and downloads it with progress, speed, time left and cancel. While the server has no model, the dialog cannot be closed and sending opens it. **Settings → Show introduction** replays it.
- **Missing experts:** when routing picks an expert that is not downloaded, the reply offers **Download and answer** (shows the download progress, then answers the same question with that expert) or **Answer with base model**. Either choice replaces the offer with the new answer.
- **Web search:** the globe toggle in the composer cycles Off, Auto (search when the question needs current or specific facts) and On (always search). Answers cite numbered sources listed under the reply; **Details** shows the query, provider and pages read. **Settings → Web search** chooses DuckDuckGo, Brave Search (API key) or SearXNG and can run a test search.
- **Reasoning:** the brain switch in the composer cycles Auto (GhostSwitch decides per request), On (always reason first) and Off (answer right away). When the server thinks before answering, a **Thinking…** block streams above the reply, collapses once the answer starts, and stays on the saved message as **Thought for N s**. Only the answer is sent back as conversation history.
- **Details:** whether the answer was reasoned and why (with the request kind when GhostSwitch decided), routing tree, expert/adapter, generation speed, token usage, all attempts and verifier feedback, raw API JSON and request/response export. Diagnostics start closed and open from **Details** on a reply or the header button. The interface upgrade closes the old default inspector once, then preserves the user's preference.
- **Experts:** descriptions, search and direct chats. Capabilities and adapter IDs expand under **Details** on each card. The catalog is loaded from `GET /v1/experts`; a cached list is clearly marked while offline. Cards show enabled/downloaded/queued/error states, a progress bar with the current file percentage and byte counts, and checking/conversion/saving stages and let the user download, enable, disable or retry. Selection is persisted on the server and restored at startup. New Git contributions follow the [expert package standard](../experts/README.md) and appear after a server restart.
- **Statistics:** request metrics, seven-day activity, route distribution, recent requests and HTTP server status. Speed is the mean of available final generation measurements; `≈` marks a rate estimated from stage latency.
- **Request log:** text/ID/expert search, status filters, complete traces and JSON export. The log contains saved replies from this browser, not all server traffic.
- **API playground:** the four existing endpoints, editable JSON, auto/direct/stream presets, HTTP status and response inspection, examples for cURL, JavaScript and Python. POST calls perform real inference and are added to local history. GET calls do not generate text.
- **Settings:** base model and backend (Qwen3.5-9B or 4B on MLX or GGUF, with each option's memory need, the memory free now, and a path field when no GGUF file is found), API origin, dark/light theme, inspector preference, generation defaults, system instruction, JSON history export/import and shortcut reference. The sidebar shows the running model; while the server restarts with a new one, sending is paused and the page waits for it to come back.

## Persistence and API

The existing `minibrain-client-v1` local-storage entry is migrated in place. Conversations, response telemetry, drafts, catalog cache and preferences stay in this browser. JSON import adds conversations without replacing current data or settings and skips duplicate conversation IDs. Downloads can contain complete prompts and responses. Attachment data stays in the browser's IndexedDB (`minibrain-attachments`); messages keep only names, sizes and small previews, so JSON exports and saved request JSON carry only those previews, and data no message refers to is removed. A message whose attachment data is missing (for example after import into another browser) sends a note in its place.

Chat inference is stateless: each chat request includes user messages and successful assistant replies as `messages`, plus the optional system instruction. User messages with attachments are sent as content parts (`image_url` and `file` with `data:` URLs, then the text). The app and API normally share an origin. A separately configured API origin needs CORS support on that server; the current backend does not expose CORS or authentication.

Streaming candidates are provisional until the final `status`. A retry resets the displayed candidate. The stop button aborts the client connection; it does not guarantee cancellation of server computation. The API playground permits one call at a time within this client.

## Checks

```bash
node --check client/app.js
./.venv/bin/python -m unittest core.tests.test_client_api core.tests.test_v6
```

For UI checks, use a separate static/fixture server so the model backend can stay stopped. Check the empty/offline states, migrated history, direct selection, SSE candidate resets, KaTeX, request details, draft switching and both desktop/mobile layouts.
