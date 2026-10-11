# OpenCode with GhostSwitch v6

The installed user configuration selects `ghostswitch/ghostswitch-v6` and the
`ghostswitch` agent. It uses only the local GhostSwitch API, including auxiliary
model calls. Automatic title and summary generation are disabled to avoid
extra inference queue time on the local backend; context compaction remains
available. Start GhostSwitch before starting OpenCode: `ghostswitch` for an installed copy, which
listens on port 27246, or from a source checkout:

```sh
./.venv/bin/python -m core.start_mlx --serve --port 27246
opencode --model ghostswitch/ghostswitch-v6 --agent ghostswitch
```

The example's `baseURL` uses port 27246; change it if the server runs on another port.

`opencode.json` in this directory is a portable example. Merge it into your
OpenCode user or project configuration. The agent enables read, search, edit,
write, and shell tools. Existing OpenCode permission rules still apply.
The local API currently does not authenticate the placeholder API key.

## API contract

POST `/v1/chat/completions` accepts function `tools`, `tool_choice` (`auto`,
`none`, `required`, or a named function), `parallel_tool_calls`, and
`stream_options.include_usage`. Assistant history can contain `tool_calls`;
tool results need the matching `tool_call_id`. All calls must have results
before another inference step. Arguments are validated against tool JSON
schemas; external schema references are not supported.

The router receives a bounded view of the conversation: the current user request
(up to 1,024 characters), the two most recent prior user turns (256 each), the
last prior assistant text (192), and short caller instructions. Agent
continuations retain recent tool names and short assistant text; file bodies,
tool arguments, and tool output are omitted. Large fenced code blocks retain
language and short samples. Long text keeps its beginning and end with an
explicit omission marker. The expert still receives the full
original history. Short follow-ups keep prior user goals as reference context.

GhostSwitch routes each request through its v6 hierarchy. Tool actions are
checked locally against their schemas and handed to OpenCode; final text is
checked only for truncation and emptiness. No model reviews either. Text-only
requests retain up to five generation attempts.

Agent generation is instructed to perform a small complete step (target: 1,024
output tokens), extending larger artifacts through successive tool calls. The
first generation honors the caller's output limit. A malformed or truncated
agent response gets at most one protocol repair, capped at 1,024 tokens or the
caller's smaller limit. The repair reduces the action's scope and continues from
existing observations. Partial calls are never executed or automatically closed.
GhostSwitch never executes external tools: OpenCode performs calls and returns
observations.

For requests with tools or tool history, streaming buffers the candidate until
local validation accepts it. Accepted calls are sent as standard
`delta.tool_calls` with `finish_reason: tool_calls`. Rejected candidates cannot
trigger actions or be concatenated into the final stream. Existing text-only
web chat retains provisional streaming and reset events.

Thinking needs no OpenCode setting. The first step after each user message is
decided like a chat message: a short check names the kind of task, and
algorithms, debugging, plans and calculations reason first (an expert whose
policy thinks, currently `reasoning_expert`, always does). Steps after tool
results answer directly, so carrying out the plan keeps its speed. Reasoning is sent as standard
`delta.reasoning_content`, separately from `content` and tool calls, and OpenCode
shows it as a thinking block (`opencode run --thinking` prints it). A thinking
step adds a few seconds with `coding_expert`, which reasons briefly, and 15–50
seconds with `reasoning_expert` or the base model on an M1 Pro. To force it on or off for every
step, set `"options": {"reasoningEffort": "high"}` or `"none"` on the model;
OpenCode sends it as `reasoning_effort`. Reasoning echoed back in history is
accepted and ignored.

The example model declares image input (`"attachment": true` and
`"modalities": {"input": ["text", "image"]}`), so screenshots pasted into
OpenCode reach the model as `image_url` parts; add the same keys to an existing
configuration. User messages may also carry `file` parts with text, code, PDF or
DOCX data. Only `data:` URLs are accepted, and the model sees the four most
recent images of a session (see **Images and files** in the main README).

The server accepts default sampling fields sent by compatible SDKs but rejects
non-default `top_p`, `stop`, and presence/frequency penalties rather than
silently ignoring them. Context is 100,000 tokens on MLX and 32,768 on GGUF (reserved when the model loads); configured output is 4,096. The context and the expert library follow the base model selected in the web app.
For agent responses, standard `usage` describes the accepted generation so the
client can track context correctly; `ghostswitch.pipeline_usage` includes routing
and all attempts. Long sessions must be compacted by the client. Local inference is serialized,
so concurrent agent/auxiliary requests may queue.


The in-app request log merges this server journal with browser-local chat
history. `GET /v1/requests` returns the latest 200 API requests, their payload,
source, live stage, completed stages and rejection reasons, generation progress,
and final route, attempts, timings, token usage, and result. Streaming agent calls
emit empty OpenAI deltas with `ghostswitch_progress` metadata during generation;
these contain token counts, not provisional tool arguments. Disconnected callers
cancel model work at the next generation checkpoint, including routing,
and prevent further retries.
The server stores request and response content in `runtime/request-journal.sqlite3`
and keeps the newest 200 entries across restarts. The OpenCode provider sets `X-GhostSwitch-Client: OpenCode` so entries
show their source in the UI; browser chat and playground requests are labelled
separately.
