"""Validate external tools and translate Qwen tool blocks to chat completions."""
from __future__ import annotations

import json
import re
import uuid
from typing import Any

from jsonschema import Draft202012Validator, ValidationError, SchemaError

from . import attachments


def validate_tools(tools: Any) -> list[dict]:
    if not isinstance(tools, list):
        raise ValueError("tools must be an array")
    names = set()
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ValueError("Only function tools are supported")
        function = tool.get("function", {})
        name = function.get("name") if isinstance(function, dict) else None
        if not isinstance(name, str) or not re.fullmatch(r"[\w-]{1,64}", name) or name in names:
            raise ValueError("Tool names must be unique valid function names")
        schema = function.get("parameters", {"type": "object"})
        if not isinstance(schema, dict):
            raise ValueError("Tool parameters must be a JSON schema object")
        # Remote schema resolution must never initiate network requests.
        def check_refs(value):
            if isinstance(value, dict):
                if "$ref" in value and not str(value["$ref"]).startswith("#"):
                    raise ValueError("External tool schema references are unsupported")
                for child in value.values():
                    check_refs(child)
            elif isinstance(value, list):
                for child in value:
                    check_refs(child)
        check_refs(schema)
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as exc:
            raise ValueError("Invalid tool parameter schema") from exc
        names.add(name)
    return tools


def parse_candidate(text: str, tools: list[dict], choice: Any, parallel: bool = True) -> dict:
    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S)
    remainder = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.S).strip()
    if "<tool_call>" in remainder or "</tool_call>" in remainder:
        raise ValueError("Incomplete tool call block")
    definitions = {tool["function"]["name"]: tool["function"] for tool in tools}
    calls = []
    for block in blocks:
        try:
            if block.lstrip().startswith("<function="):
                function = re.fullmatch(r"\s*<function=([\w-]+)>\s*(.*?)\s*</function>\s*", block, re.S)
                if not function:
                    raise ValueError("Malformed Qwen function block")
                name, body = function.groups()
                if name not in definitions:
                    raise ValueError("Unknown tool")
                parameters = re.findall(r"<parameter=([\w-]+)>\n?(.*?)</parameter>", body, re.S)
                if re.sub(r"<parameter=[\w-]+>.*?</parameter>", "", body, flags=re.S).strip():
                    raise ValueError("Malformed Qwen parameter block")
                arguments = {}
                properties = definitions[name].get("parameters", {}).get("properties", {})
                for key, raw in parameters:
                    if key in arguments:
                        raise ValueError("Duplicate parameter")
                    raw = raw.removesuffix("\n")
                    kind = properties.get(key, {}).get("type")
                    if kind == "string" or isinstance(kind, list) and "string" in kind:
                        arguments[key] = raw
                    else:
                        try:
                            arguments[key] = json.loads(raw)
                        except ValueError:
                            arguments[key] = raw
            else:
                value = json.loads(block)
                name, arguments = value["name"], value["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if name not in definitions or not isinstance(arguments, dict):
                raise ValueError("Unknown tool or non-object arguments")
            Draft202012Validator(definitions[name].get("parameters", {})).validate(arguments)
        except (ValueError, KeyError, TypeError, ValidationError) as exc:
            raise ValueError(f"Invalid tool call: {exc}") from exc
        calls.append({"id": "call_" + uuid.uuid4().hex, "type": "function",
                      "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}})
    if calls and choice == "none":
        raise ValueError("Tool calls are forbidden by tool_choice=none")
    if not calls and (choice == "required" or isinstance(choice, dict)):
        raise ValueError("A tool call is required")
    if isinstance(choice, dict) and any(call["function"]["name"] != choice["function"]["name"] for call in calls):
        raise ValueError("Call only the tool selected by tool_choice")
    if not parallel and len(calls) > 1:
        raise ValueError("Return only one tool call")
    result = {"role": "assistant", "content": remainder or None}
    if calls:
        result["tool_calls"] = calls
    return result


REASONING_FIELDS = {"reasoning_content", "reasoning"}


def validate_messages(messages: list) -> list[dict]:
    normalized = []
    pending: set[str] = set()
    seen: set[str] = set()
    for message in messages:
        if (not isinstance(message, dict)
                or set(message) - {"role", "content", "tool_calls", "tool_call_id", "name", *REASONING_FIELDS}):
            raise ValueError("Invalid message fields")
        role, content = message.get("role"), message.get("content")
        if role not in {"system", "developer", "user", "assistant", "tool"}:
            raise ValueError("Invalid message role")
        # Clients may echo earlier reasoning; it is accepted but never re-sent to the model.
        if REASONING_FIELDS & set(message) and (
                role != "assistant"
                or any(not isinstance(message[field], (str, type(None)))
                       for field in REASONING_FIELDS & set(message))):
            raise ValueError("Reasoning fields must be strings on assistant messages")
        calls = message.get("tool_calls")
        if pending and role != "tool":
            raise ValueError("Tool results must follow assistant tool calls")
        if not isinstance(content, (str, list)) and not (role == "assistant" and calls and content is None):
            raise ValueError("Each message needs text content or assistant tool calls")
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in pending:
                raise ValueError("Tool result has no matching pending call")
            pending.remove(call_id)
        elif "tool_call_id" in message:
            raise ValueError("tool_call_id is only valid for tool results")
        if calls is not None:
            if role != "assistant" or not isinstance(calls, list) or not calls:
                raise ValueError("tool_calls must be a nonempty assistant array")
            for call in calls:
                if not isinstance(call, dict) or call.get("type") != "function":
                    raise ValueError("Invalid assistant tool call")
                call_id, function = call.get("id"), call.get("function")
                if not isinstance(call_id, str) or not call_id or call_id in seen or not isinstance(function, dict):
                    raise ValueError("Invalid or duplicate tool call id")
                if not isinstance(function.get("name"), str) or not isinstance(function.get("arguments"), str):
                    raise ValueError("Tool call needs function name and JSON arguments")
                try:
                    args = json.loads(function["arguments"])
                except ValueError as exc:
                    raise ValueError("Tool arguments must be JSON") from exc
                if not isinstance(args, dict):
                    raise ValueError("Tool arguments must be an object")
                pending.add(call_id)
                seen.add(call_id)
        item = {key: value for key, value in message.items() if key not in REASONING_FIELDS}
        if isinstance(content, list):
            # Content parts become text plus decoded images; finalize() writes the text below.
            item["content"] = attachments.parse_parts(content, role)
        normalized.append(item)
    if pending:
        raise ValueError("Missing tool results for assistant calls")
    attachments.finalize(normalized)
    return normalized
