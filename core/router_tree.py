"""Configurable, model-routed hierarchy for selecting specialist experts."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import llm


MAX_ROUTING_DEPTH = 6
MAX_ROUTER_BRANCHING_FACTOR = 15
ENABLE_THINKER_ROUTING_FALLBACK = False
FUNCTIONGEMMA_MAX_TOKENS = 48
TREE_PATH = Path(__file__).with_name("config") / "router_tree.json"


@dataclass(frozen=True)
class RouterNode:
    id: str
    description: str
    aliases: tuple[str, ...] = ()
    children: tuple[str, ...] = ()
    expert_id: str | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> "RouterNode":
        node_id = str(value.get("id", "")).strip()
        description = str(value.get("description", "")).strip()
        aliases = value.get("aliases", [])
        children = value.get("children", [])
        if not node_id or not description:
            raise ValueError("Every router node needs a non-empty id and description")
        if not isinstance(aliases, list) or any(not isinstance(item, str) for item in aliases):
            raise ValueError(f"Router node {node_id!r} aliases must be a list of strings")
        if not isinstance(children, list) or any(not isinstance(item, str) for item in children):
            raise ValueError(f"Router node {node_id!r} children must be a list of ids")
        expert_id = value.get("expert_id")
        if expert_id is not None and (not isinstance(expert_id, str) or not expert_id.strip()):
            raise ValueError(f"Router node {node_id!r} has an invalid expert_id")
        return cls(
            id=node_id,
            description=description,
            aliases=tuple(item.strip() for item in aliases if item.strip()),
            children=tuple(item.strip() for item in children),
            expert_id=expert_id.strip() if isinstance(expert_id, str) else None,
        )


class RouterTree:
    """Validated tree whose leaves map to expert/provider ids."""

    def __init__(
        self,
        root_id: str,
        nodes: list[RouterNode | dict[str, object]],
        *,
        max_depth: int = MAX_ROUTING_DEPTH,
        known_experts: set[str] | None = None,
    ) -> None:
        self.root_id = root_id
        self.max_depth = max_depth
        converted = [node if isinstance(node, RouterNode) else RouterNode.from_mapping(node) for node in nodes]
        self.nodes: dict[str, RouterNode] = {}
        for node in converted:
            if node.id in self.nodes:
                raise ValueError(f"Duplicate router node id: {node.id}")
            self.nodes[node.id] = node
        self._validate(known_experts)
        self._descendant_cache: dict[str, frozenset[str]] = {}

    @classmethod
    def load(
        cls,
        path: Path = TREE_PATH,
        *,
        known_experts: set[str] | None = None,
    ) -> "RouterTree":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("nodes"), list):
            raise ValueError("Router tree config must contain a nodes array")
        if path == TREE_PATH:
            from .expert_packages import load_packages, package_router_config
            raw = package_router_config(raw, load_packages())
        return cls(
            str(raw.get("root_id", "")),
            raw["nodes"],
            max_depth=int(raw.get("max_depth", MAX_ROUTING_DEPTH)),
            known_experts=known_experts,
        )

    def _validate(self, known_experts: set[str] | None) -> None:
        if self.root_id not in self.nodes:
            raise ValueError(f"Router tree root does not exist: {self.root_id}")
        if not 1 <= self.max_depth <= MAX_ROUTING_DEPTH:
            raise ValueError(f"Router tree max_depth must be between 1 and {MAX_ROUTING_DEPTH}")
        for node in self.nodes.values():
            if len(node.children) != len(set(node.children)):
                raise ValueError(f"Duplicate child id in router node {node.id!r}")
            if len(node.children) > MAX_ROUTER_BRANCHING_FACTOR:
                raise ValueError(
                    f"Router node {node.id!r} has {len(node.children)} children; "
                    f"maximum is {MAX_ROUTER_BRANCHING_FACTOR}. Add an intermediate level."
                )
            for child_id in node.children:
                if child_id not in self.nodes:
                    raise ValueError(f"Router node {node.id!r} references missing child {child_id!r}")
            if node.children and node.expert_id is not None:
                raise ValueError(f"Non-leaf router node {node.id!r} cannot also map to an expert")
            # A backbone without expert packages keeps only the root.
            if not node.children and node.expert_id is None and node.id != self.root_id:
                raise ValueError(f"Dead-end router leaf {node.id!r} has no expert_id")
            if node.expert_id is not None and known_experts is not None and node.expert_id not in known_experts:
                raise ValueError(f"Router leaf {node.id!r} references unknown expert {node.expert_id!r}")

        visiting: set[str] = set()
        visited: set[str] = set()

        def walk(node_id: str, depth: int) -> None:
            if depth > self.max_depth:
                raise ValueError(f"Router tree exceeds MAX_ROUTING_DEPTH at {node_id!r}")
            if node_id in visiting:
                raise ValueError(f"Router tree contains a cycle through {node_id!r}")
            if node_id in visited:
                return
            visiting.add(node_id)
            for child_id in self.nodes[node_id].children:
                walk(child_id, depth + 1)
            visiting.remove(node_id)
            visited.add(node_id)

        walk(self.root_id, 0)
        unreachable = set(self.nodes) - visited
        if unreachable:
            raise ValueError(f"Router tree contains unreachable nodes: {sorted(unreachable)[:5]}")

    def _experts_below(self, node_id: str) -> frozenset[str]:
        cached = self._descendant_cache.get(node_id)
        if cached is not None:
            return cached
        node = self.nodes[node_id]
        if node.expert_id is not None:
            result = frozenset({node.expert_id})
        else:
            result = frozenset().union(*(self._experts_below(child) for child in node.children))
        self._descendant_cache[node_id] = result
        return result

    def available_children(self, node_id: str, available_experts: set[str]) -> list[RouterNode]:
        node = self.nodes[node_id]
        return [
            self.nodes[child_id]
            for child_id in node.children
            if self._experts_below(child_id) & available_experts
        ]

    def path_for_expert(self, expert_id: str) -> list[str] | None:
        def find(node_id: str, path: list[str]) -> list[str] | None:
            node = self.nodes[node_id]
            current = [*path, node_id]
            if node.expert_id == expert_id:
                return current
            for child_id in node.children:
                result = find(child_id, current)
                if result:
                    return result
            return None

        full = find(self.root_id, [])
        return full[1:] if full else None

    def route(
        self,
        original_request: str,
        available_experts: list[str],
        *,
        thinker_fallback: bool = ENABLE_THINKER_ROUTING_FALLBACK,
        max_tokens: int = FUNCTIONGEMMA_MAX_TOKENS,
    ) -> dict[str, object]:
        started = time.perf_counter_ns()
        available = set(available_experts) & set(self._experts_below(self.root_id))
        levels: list[dict[str, object]] = []
        path: list[str] = []
        methods: list[str] = []
        functiongemma_calls = 0
        thinker_calls = 0

        def finish(
            expert_id: str | None = None,
            error: str | None = None,
            failure_level: int | None = None,
            failure_reason: str | None = None,
        ) -> dict[str, object]:
            telemetry = {
                "levels": levels,
                "routing_depth": len(path),
                "functiongemma_calls": functiongemma_calls,
                "thinker_fallback_calls": thinker_calls,
                "thinker_fallback_used": thinker_calls > 0,
                "total_latency_ms": round((time.perf_counter_ns() - started) / 1_000_000, 2),
                "errors_by_level": {
                    str(level["level"]): str(level.get("parse_failure_reason") or level.get("failure_reason"))
                    for level in levels
                    if level.get("parse_failure_reason") or level.get("failure_reason")
                },
            }
            result: dict[str, object] = {
                "method": "router_tree",
                "expert_id": expert_id,
                "path": path,
                "methods": methods,
                "routing_telemetry": telemetry,
            }
            if expert_id is not None:
                result["expert_id"] = expert_id
            if error is not None:
                result["error"] = error
            if failure_level is not None:
                result["failure_level"] = failure_level
            if failure_reason is not None:
                result["failure_reason"] = failure_reason
            return result

        if not available:
            return finish(error="No available expert is represented in the router tree.")
        if len(available) == 1:
            expert_id = next(iter(available))
            expert_path = self.path_for_expert(expert_id)
            if expert_path is None:
                return finish(error=f"No route path exists for expert {expert_id!r}.")
            path.extend(expert_path)
            methods.extend(["availability"] * max(0, len(expert_path) - 1) + ["leaf"])
            return finish(expert_id)

        current_id = self.root_id
        for level_number in range(1, self.max_depth + 1):
            current = self.nodes[current_id]
            if current.expert_id is not None:
                if current.expert_id not in available:
                    return finish(error=f"Expert {current.expert_id!r} is unavailable.")
                return finish(current.expert_id)
            options = self.available_children(current_id, available)
            option_ids = [node.id for node in options]
            if not options:
                return finish(error=f"Router node {current_id!r} has no available child routes.")
            if len(options) == 1:
                selected = options[0]
                selected_by = "availability"
                selected_id = selected.id
                level_record: dict[str, object] = {
                    "level": level_number,
                    "node": current_id,
                    "available_routes": option_ids,
                    "selected": selected.id,
                    "router": "availability",
                    "raw_output": None,
                    "parse_success": True,
                    "latency_ms": 0.0,
                    "failure_reason": None,
                }
            else:
                route_tools = build_route_tools(options)
                messages = [
                    {
                        "role": "developer",
                        "content": (
                            "You are a model that can do function calling with the following functions. "
                            "Choose exactly one function that best matches the user's requested outcome. "
                            "Do not answer or solve the request."
                        ),
                    },
                    {"role": "user", "content": original_request},
                ]
                call_started = time.perf_counter_ns()
                functiongemma_calls += 1
                call_debug: dict[str, Any] = {}
                try:
                    raw = llm.chat(
                        messages, max_tokens=max_tokens, role="router", tools=route_tools, temperature=0, debug=call_debug
                    )
                    if call_debug.get("finish_reason") == "length":
                        parsed = {
                            "selected": None,
                            "parse_success": False,
                            "status": "router_truncated_output",
                            "failure_reason": "truncated_output",
                        }
                    else:
                        parsed = parse_function_route_output(raw, {route_tool_name(node.id): node.id for node in options})
                except Exception as exc:
                    raw = ""
                    parsed = {
                        "selected": None,
                        "parse_success": False,
                        "status": "router_call_error",
                        "failure_reason": "router_runtime_error",
                        "runtime_exception": type(exc).__name__,
                    }
                fg_ms = round((time.perf_counter_ns() - call_started) / 1_000_000, 2)
                selected_id = parsed.get("selected")
                selected_by = "functiongemma"
                failure_reason = parsed.get("failure_reason")
                level_record = {
                    "level": level_number,
                    "node": current_id,
                    "available_routes": option_ids,
                    "selected": selected_id,
                    "router": "functiongemma",
                    "raw_output": raw,
                    "debug_trace": call_debug,
                    "parse_success": bool(parsed.get("parse_success")),
                    "parse_failure_reason": failure_reason,
                    "runtime_exception": parsed.get("runtime_exception"),
                    "finish_reason": call_debug.get("finish_reason"),
                    "max_tokens": max_tokens,
                    "latency_ms": fg_ms,
                }
                if not parsed.get("parse_success") and thinker_fallback:
                    fallback_started = time.perf_counter_ns()
                    thinker_calls += 1
                    fallback_messages = [
                        {
                            "role": "developer",
                            "content": (
                                "You are a model that can do function calling with the following functions. "
                                "Choose exactly one function that best matches the user's requested outcome. "
                                "Do not answer or solve the request."
                            ),
                        },
                        {"role": "user", "content": original_request},
                    ]
                    try:
                        fallback_raw = llm.chat(
                            fallback_messages, role="thinker", max_tokens=64, temperature=0, tools=route_tools
                        )
                        fallback = parse_function_route_output(
                            fallback_raw, {route_tool_name(node.id): node.id for node in options}
                        )
                    except Exception as exc:
                        fallback_raw = ""
                        fallback = {
                            "selected": None,
                            "parse_success": False,
                            "failure_reason": f"thinker_call_error:{type(exc).__name__}",
                        }
                    level_record["thinker_raw_output"] = fallback_raw
                    level_record["thinker_latency_ms"] = round(
                        (time.perf_counter_ns() - fallback_started) / 1_000_000, 2
                    )
                    level_record["fallback_parse_success"] = bool(fallback.get("parse_success"))
                    if fallback.get("parse_success"):
                        selected_id = fallback.get("selected")
                        selected_by = "thinker"
                        level_record["selected"] = selected_id
                        level_record["router"] = "thinker"
                    else:
                        level_record["failure_reason"] = fallback.get("failure_reason", failure_reason)
                        levels.append(level_record)
                        return finish(
                            error="functiongemma_routing_failure",
                            failure_level=level_number,
                            failure_reason=str(level_record["failure_reason"]),
                        )
                elif not parsed.get("parse_success"):
                    level_record["failure_reason"] = failure_reason
                    levels.append(level_record)
                    return finish(
                        error="functiongemma_routing_failure",
                        failure_level=level_number,
                        failure_reason=str(failure_reason or "malformed_tool_call"),
                    )
                levels.append(level_record)

            if not isinstance(selected_id, str) or selected_id not in option_ids:
                return finish(
                    error="functiongemma_routing_failure",
                    failure_level=level_number,
                    failure_reason="route_function_not_allowed",
                )
            if selected_id in path:
                return finish(error="Routing path repeated a node; cycle guard stopped traversal.")
            path.append(selected_id)
            selected_node = self.nodes[selected_id]
            methods.append("leaf" if selected_node.expert_id is not None else selected_by)
            if selected_node.expert_id is not None:
                return finish(selected_node.expert_id)
            current_id = selected_id

        return finish(error=f"Maximum routing depth {self.max_depth} reached before an expert leaf.")


def route_tool_name(node_id: str) -> str:
    """Map a router child id to its native function declaration name."""
    return "route_" + re.sub(r"\W+", "_", node_id)


def build_route_tools(options: list[RouterNode]) -> list[dict[str, object]]:
    """Expose only immediate children as no-argument native function choices."""
    names = [route_tool_name(node.id) for node in options]
    if len(names) != len(set(names)):
        raise ValueError("Sibling route ids collide after conversion to function names")
    return [
        {
            "type": "function",
            "function": {
                "name": route_tool_name(node.id),
                "description": node.description,
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        }
        for node in options
    ]


def parse_function_route_output(raw: str, allowed_functions: dict[str, str]) -> dict[str, object]:
    """Parse one native route_<child>() call and reject malformed or unavailable choices."""
    matches = list(re.finditer(
        r"(?:<start_function_call>)?call:(?P<name>[A-Za-z_][\w.-]*)\s*\{(?P<args>[^{}]*)\}(?:<end_function_call>)?",
        raw,
    ))
    if not matches:
        partial = bool(re.search(r"(?:call:|<start_function_call>|<tool_call>|\broute_[\w.-]*\s*\()", raw))
        reason = "malformed_tool_call" if partial else "no_tool_call"
        return {"selected": None, "function": None, "parse_success": False, "status": "router_parse_error", "failure_reason": reason}
    if any(match.group("args").strip() for match in matches):
        return {"selected": None, "function": None, "parse_success": False, "status": "router_parse_error", "failure_reason": "malformed_tool_call"}
    names = [match.group("name") for match in matches]
    unique_names = list(dict.fromkeys(names))
    if len(unique_names) > 1:
        return {"selected": None, "function": None, "parse_success": False, "status": "router_parse_error", "failure_reason": "multiple_inconsistent_calls"}
    if len(matches) > 1:
        return {"selected": None, "function": unique_names[0], "parse_success": False, "status": "router_parse_error", "failure_reason": "multiple_function_calls"}
    function_name = unique_names[0]
    selected = allowed_functions.get(function_name)
    if selected is None:
        return {"selected": None, "function": function_name, "parse_success": False, "status": "router_invalid_route", "failure_reason": "route_function_not_allowed"}
    return {"selected": selected, "function": function_name, "parse_success": True, "status": "selected", "failure_reason": None}


from .expert_catalog import EXPERT_CATALOG

ROUTER_TREE = RouterTree.load(known_experts=set(EXPERT_CATALOG.by_id))
