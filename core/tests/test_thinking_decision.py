"""The automatic thinking decision: the kind of request, and when the check runs at all."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from core import llm, main, thinking_decision
from core.tests.test_v6 import PROFILE, Registry, ThinkingRegistry

AUTO = {**PROFILE, "auto_thinking": True}
ACCEPT = {"status": "ACCEPT", "feedback": None}
BASE = {"route": "base", "expert_id": None, "confidence": .9, "missing_capability": None}
REASONED = "Worked it out.</think>\n\n403"
TOOLS = [{"type": "function", "function": {"name": "read", "description": "Read a file",
                                           "parameters": {"type": "object", "properties": {}}}}]


class DecisionTests(unittest.TestCase):
    def test_kinds_that_need_working_out_think_and_others_answer_directly(self):
        for raw, expected in (('{"kind":"math"}', (True, "math")), ('{"kind":"logic"}', (True, "logic")),
                              ('{"kind":"planning"}', (True, "planning")), ('{"kind":"engineering"}', (True, "engineering")),
                              ('{"kind":"edit"}', (False, "edit")), ('{"kind":"snippet"}', (False, "snippet")),
                              ('{"kind":"chat"}', (False, "chat")), ('{"kind":"facts"}', (False, "facts")),
                              ("not json", (False, None))):
            with self.subTest(raw=raw), patch.object(thinking_decision.llm, "chat", return_value=raw):
                self.assertEqual(thinking_decision.decide([{"role": "user", "content": "q"}]), expected)

    def test_the_check_starts_the_answer_and_keeps_its_instructions_cached(self):
        with patch.object(thinking_decision.llm, "chat", return_value='{"kind":"math"}') as chat:
            thinking_decision.decide([{"role": "user", "content": "2+2?"}])
        kwargs = chat.call_args.kwargs
        self.assertEqual(kwargs["prefill"], '{"kind":"')
        self.assertFalse(kwargs["enable_thinking"])
        self.assertTrue(kwargs["cache_namespace"].startswith("capability-v2:"))

    def test_a_failed_check_answers_without_reasoning_but_cancellation_stops(self):
        with patch.object(thinking_decision.llm, "chat", side_effect=RuntimeError("model error")):
            self.assertEqual(thinking_decision.decide([{"role": "user", "content": "q"}]), (False, None))
        with patch.object(thinking_decision.llm, "chat", side_effect=llm.GenerationCancelled()):
            with self.assertRaises(llm.GenerationCancelled):
                thinking_decision.decide([{"role": "user", "content": "q"}])


class PipelineTests(unittest.TestCase):
    messages = [{"role": "user", "content": "Сколько будет 17 * 23 + 144 / 12?"}]

    def run_request(self, payload, outputs, registry=None, gpu=True, profile=AUTO):
        with patch.object(main, "_profile", return_value=profile), \
             patch.object(main.llm, "runs_on_gpu", return_value=gpu), \
             patch.object(main.router_v6, "route", return_value=BASE), \
             patch.object(main.verifier_v6, "verify", return_value=ACCEPT), \
             patch.object(main.llm, "chat", side_effect=outputs) as chat:
            result = main.run_request({"messages": self.messages, **payload}, registry=registry or Registry())
        return result, chat

    def test_an_agent_task_decides_on_its_first_step(self):
        result, chat = self.run_request({"tools": TOOLS}, ['{"kind":"engineering"}', REASONED])
        self.assertEqual(result["status"], "success")
        self.assertTrue(chat.call_args.kwargs["enable_thinking"])
        self.assertEqual(result["ghostswitch"]["thinking"]["source"], "auto")

    def test_agent_steps_after_a_tool_result_answer_directly_even_for_a_thinking_expert(self):
        history = [*self.messages,
                   {"role": "assistant", "content": None, "tool_calls": [
                       {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
                   {"role": "tool", "tool_call_id": "call_1", "content": "def mean(xs): return sum(xs) / len(xs)"}]
        for registry in (Registry(), ThinkingRegistry()):
            with self.subTest(registry=type(registry).__name__):
                result, chat = self.run_request({"messages": history, "tools": TOOLS, "mode": "direct",
                                                 "expert_id": "coding_expert"}, ["Done."], registry=registry)
                self.assertEqual(result["status"], "success")
                self.assertEqual(chat.call_count, 1)
                self.assertFalse(chat.call_args.kwargs["enable_thinking"])
                self.assertEqual(result["ghostswitch"]["thinking"]["source"], "agent_step")

    def test_a_calculation_is_answered_with_reasoning(self):
        result, chat = self.run_request({}, ['{"kind":"math"}', REASONED])
        self.assertEqual(result["status"], "success")
        self.assertTrue(chat.call_args.kwargs["enable_thinking"])
        self.assertEqual(result["ghostswitch"]["thinking"],
                         {"enabled": True, "source": "auto", "kind": "math", "used": True})
        stage = next(stage for stage in result["ghostswitch"]["stages"] if stage["phase"] == "thinking")
        self.assertEqual((stage["status"], stage["kind"]), ("think", "math"))

    def test_conversation_is_answered_directly(self):
        result, chat = self.run_request({}, ['{"kind":"chat"}', "Привет!"])
        self.assertFalse(chat.call_args.kwargs["enable_thinking"])
        self.assertEqual(result["ghostswitch"]["thinking"],
                         {"enabled": False, "source": "auto", "kind": "chat", "used": False})

    def test_no_check_when_the_request_expert_profile_or_processor_decides(self):
        cases = {
            "the request turns it off": ({"reasoning_effort": "none"}, {}, False),
            "the expert policy turns it on": ({"mode": "direct", "expert_id": "coding_expert"},
                                              {"registry": ThinkingRegistry()}, True),
            "the profile has no auto_thinking": ({}, {"profile": PROFILE}, False),
            "the model runs on the processor": ({}, {"gpu": False}, False),
        }
        for name, (payload, options, thinks) in cases.items():
            with self.subTest(name):
                result, chat = self.run_request(payload, [REASONED], **options)
                self.assertEqual(result["status"], "success")
                self.assertEqual(chat.call_count, 1)
                self.assertEqual(chat.call_args.kwargs["enable_thinking"], thinks)
                self.assertNotIn("thinking", [stage["phase"] for stage in result["ghostswitch"]["stages"]])


if __name__ == "__main__":
    unittest.main()
