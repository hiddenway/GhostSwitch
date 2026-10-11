"""Bound the routing view without changing the caller's inference history."""
from __future__ import annotations

import re
from typing import Any

CURRENT_REQUEST_CHARS = 1024
PRIOR_USER_CHARS = 256
ASSISTANT_CHARS = 192
INSTRUCTION_CHARS = 128
CODE_CHARS = 128

# A bare request to keep going names no new task; it continues the previous user turn.
_CONTINUE = re.compile(
    r'(?:продолж\w*|дальше|давай\s+дальше|continue|go\s+on|keep\s+going|carry\s+on|proceed|resume)'
    r'(?:[\s,]+(?:пожалуйста|please))?[\s.!…]*',
    re.IGNORECASE,
)


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = ' [omitted] '
    remaining = limit - len(marker)
    head = remaining // 2
    return text[:head] + marker + text[-(remaining - head):]


def _text(content: Any, limit: int) -> str:
    if not isinstance(content, str):
        return ''
    # Preserve the language and both ends of a large code example; the router
    # needs its domain, while the expert receives the unmodified example.
    def code(match: re.Match) -> str:
        body = match.group(3)
        if len(body) <= CODE_CHARS:
            return match.group(0)
        return f' [code {match.group(2).strip()}: {_clip(body, CODE_CHARS)}] '
    compact = re.sub(r'^(```|~~~)([^\n]*)\n(.*?)\n\1[ \t]*$', code, content, flags=re.S | re.M)
    return _clip(re.sub(r'\s+', ' ', compact).strip(), limit)


def routing_turn(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    """The history that ends with the user turn defining the task, and whether this request continues it.

    Agent steps after that turn (assistant tool calls and tool results) and a bare
    "continue" from the user add no new task, so they are routed like the turn itself.
    """
    users = [i for i, m in enumerate(messages) if m['role'] == 'user']
    position = len(users) - 1
    continuing = users[position] < len(messages) - 1
    while position > 0 and isinstance(content := messages[users[position]].get('content'), str) \
            and _CONTINUE.fullmatch(content.strip()):
        position -= 1
        continuing = True
    return messages[:users[position] + 1], continuing


def routing_request(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Select bounded semantic context, without word-based routing rules.

    Only the history up to the last user message is read: tool arguments and observations
    are often entire files and would make every node prompt grow with the agent's progress.
    """
    current_index = next(i for i in range(len(messages) - 1, -1, -1)
                         if messages[i]['role'] == 'user')
    prior = messages[:current_index]
    # Two prior user turns retain the goal behind references such as "continue"
    # and "do the same". The last assistant text can explain their referent.
    user_indexes = [i for i, m in enumerate(prior) if m['role'] == 'user'][-2:]
    assistant_indexes = [i for i, m in enumerate(prior)
                         if m['role'] == 'assistant' and m.get('content')][-1:]
    selected = sorted(user_indexes + assistant_indexes)
    context = [{'role': prior[i]['role'], 'content': _text(
        prior[i].get('content'), PRIOR_USER_CHARS if prior[i]['role'] == 'user' else ASSISTANT_CHARS,
    )} for i in selected]
    state = {
        'current_user_request': _text(messages[current_index]['content'], CURRENT_REQUEST_CHARS),
        'conversation_context': context,
    }
    instructions = [m for m in messages[:current_index] if m['role'] in {'system', 'developer'}][-2:]
    if instructions:
        state['caller_instructions'] = [
            {'role': m['role'], 'content': _text(m.get('content'), INSTRUCTION_CHARS)}
            for m in instructions
        ]
    return state
