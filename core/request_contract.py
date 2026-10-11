"""Validation for a single GhostSwitch v6 inference request."""

from __future__ import annotations

import json
from typing import Any

# Thinking is on or off; effort levels only express the caller's choice.
REASONING_EFFORTS = {"none": False, "minimal": False, "low": True, "medium": True, "high": True}


def _thinking_override(payload: dict[str, Any]) -> bool | None:
    """Return the caller's explicit thinking choice, or None for the expert policy."""
    choices = []
    effort = payload.get("reasoning_effort")
    if effort is not None:
        if effort not in REASONING_EFFORTS:
            raise ValueError("reasoning_effort must be none, minimal, low, medium, or high")
        choices.append(REASONING_EFFORTS[effort])
    kwargs = payload.get("chat_template_kwargs")
    if kwargs is not None:
        if (not isinstance(kwargs, dict) or set(kwargs) - {"enable_thinking"}
                or type(kwargs.get("enable_thinking", False)) is not bool):
            raise ValueError("chat_template_kwargs supports only a boolean enable_thinking")
        if "enable_thinking" in kwargs:
            choices.append(kwargs["enable_thinking"])
    if len(set(choices)) > 1:
        raise ValueError("reasoning_effort and chat_template_kwargs.enable_thinking disagree")
    return choices[0] if choices else None


def parse_request(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Request must be a JSON object")
    allowed = {"model", "messages", "mode", "expert_id", "max_tokens", "temperature", "metadata", "stream", "tools", "tool_choice", "parallel_tool_calls", "stream_options", "stop", "top_p", "presence_penalty", "frequency_penalty", "reasoning_effort", "chat_template_kwargs", "web_search", "verify"}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"Unknown request fields: {sorted(unknown)}")
    stream = payload.get("stream", False)
    if not isinstance(stream, bool):
        raise ValueError("stream must be a boolean")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty array")
    from .tool_protocol import validate_messages, validate_tools
    normalized = validate_messages(messages)
    if not any(item["role"] == "user" and item["content"].strip() for item in normalized):
        raise ValueError("messages must contain a nonempty user message")
    tools = validate_tools(payload.get("tools", []))
    choice = payload.get("tool_choice", "auto" if tools else "none")
    if isinstance(choice, dict):
        if choice.get("type") != "function" or choice.get("function", {}).get("name") not in {t["function"]["name"] for t in tools}:
            raise ValueError("tool_choice must name an available function")
    elif choice not in ("auto", "none", "required"):
        raise ValueError("Invalid tool_choice")
    if choice == "required" and not tools:
        raise ValueError("tool_choice=required needs tools")
    parallel = payload.get("parallel_tool_calls", True)
    if not isinstance(parallel, bool):
        raise ValueError("parallel_tool_calls must be a boolean")
    options = payload.get("stream_options")
    if options is not None and (not isinstance(options, dict) or set(options) - {"include_usage"} or type(options.get("include_usage", False)) is not bool):
        raise ValueError("Invalid stream_options")
    # Accept SDK defaults, but never silently ignore sampling settings.
    for field, default in (("top_p", 1), ("presence_penalty", 0), ("frequency_penalty", 0), ("stop", None)):
        if field in payload and payload[field] not in (None, default):
            raise ValueError(f"Non-default {field} is not supported")
    mode = payload.get("mode", "auto")
    expert_id = payload.get("expert_id")
    if mode not in {"auto", "direct", "base"}:
        raise ValueError("mode must be auto, direct, or base")
    if mode == "direct" and (not isinstance(expert_id, str) or not expert_id.strip()):
        raise ValueError("direct mode requires expert_id")
    if mode != "direct" and expert_id is not None:
        raise ValueError("expert_id is only valid in direct mode")
    max_tokens = payload.get("max_tokens", 4096)
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    temperature = payload.get("temperature", 0.7)
    if isinstance(temperature, bool) or not isinstance(temperature, (float, int)) or not 0 <= temperature <= 2:
        raise ValueError("temperature must be between 0 and 2")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    try:
        metadata = json.loads(json.dumps(metadata, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must contain JSON-serializable values") from exc
    web_search = payload.get("web_search", "off")
    if web_search not in {"off", "auto", "on"}:
        raise ValueError("web_search must be off, auto, or on")
    # Benchmarks and ablations can turn the verifier off; normal callers keep it.
    verify = payload.get("verify", True)
    if type(verify) is not bool:
        raise ValueError("verify must be a boolean")
    model = payload.get("model", "ghostswitch-v6")
    # omnibrain-v6 and minibrain-v6 are the ids from before the renames; existing client
    # configurations still send them.
    if model not in {"ghostswitch-v6", "omnibrain-v6", "minibrain-v6"}:
        raise ValueError("model must be ghostswitch-v6")
    return {"messages": normalized, "mode": mode, "expert_id": expert_id,
            "max_tokens": max_tokens, "temperature": float(temperature), "metadata": metadata,
            "stream": stream, "tools": tools, "tool_choice": choice, "parallel_tool_calls": parallel,
            "thinking": _thinking_override(payload), "web_search": web_search, "verify": verify}
