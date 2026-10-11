from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core import main, router_v6, verifier_v6
from core.request_contract import parse_request
from core.router_tree import RouterNode, RouterTree


PROFILE = {"backend": "mlx", "max_tokens_per_call": 10240,
           "router_max_tokens": 32, "verifier_max_tokens": 32, "max_retries": 1}


class Registry:
    def descriptions(self):
        return {"coding_expert": "Writes and debugs code"}

    def is_available(self, expert_id, backend):
        return expert_id == "coding_expert" and backend == "mlx"

    def check_backbone_compatibility(self, expert_id):
        return expert_id == "coding_expert"

    def get(self, expert_id):
        return SimpleNamespace(adapter_id="coding-adapter", prompt="Write code")

    def review_guidance(self, expert_id):
        return ""


class ThinkingRegistry(Registry):
    """An expert whose response policy enables thinking."""

    def get(self, expert_id):
        return SimpleNamespace(adapter_id="reasoning-adapter", prompt="Reason carefully", thinking=True)


class V6Tests(unittest.TestCase):
    def setUp(self):
        self.profile = patch.object(main, "_profile", return_value=PROFILE)
        self.profile.start()
        self.addCleanup(self.profile.stop)
        self.registry = Registry()
        self.messages = [{"role": "user", "content": "Implement this function"}]
        router_v6._STICKY_ROUTES.clear()
        self.addCleanup(router_v6._STICKY_ROUTES.clear)

    def test_direct_accept_returns_exact_expert_output_without_routing(self):
        with patch.object(main.router_v6, "route") as router, \
             patch.object(main.verifier_v6, "verify", return_value={"status": "ACCEPT", "feedback": None}), \
             patch.object(main.llm, "chat", return_value="def f():\n    return 1\n") as chat:
            result = main.run_request({"messages": self.messages, "mode": "direct", "expert_id": "coding_expert"}, registry=self.registry)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["choices"][0]["message"]["content"], "def f():\n    return 1\n")
        self.assertEqual(result["ghostswitch"]["verification"]["attempts"], 1)
        self.assertEqual([stage["phase"] for stage in result["ghostswitch"]["stages"]],
                         ["selection", "adapter_check", "generation", "verification"])
        self.assertEqual(chat.call_args.kwargs["adapter"], "coding_expert")
        router.assert_not_called()

    def test_valid_agent_tool_action_skips_model_verifier(self):
        tools = [{"type": "function", "function": {
            "name": "read_file", "description": "Read a project file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                           "required": ["path"], "additionalProperties": False},
        }}]
        candidate = (
            "<tool_call>\n<function=read_file>\n"
            "<parameter=path>README.md</parameter>\n</function>\n</tool_call>"
        )
        with patch.object(main.verifier_v6, "verify_agent") as verify_agent, \
             patch.object(main.llm, "chat", return_value=candidate):
            result = main.run_request({"messages": self.messages, "mode": "direct",
                                       "expert_id": "coding_expert", "tools": tools},
                                      registry=self.registry)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(result["ghostswitch"]["verification"]["status"], "SKIPPED")
        verification_stage = result["ghostswitch"]["stages"][-1]
        self.assertEqual(verification_stage["status"], "skipped")
        self.assertEqual(verification_stage["reason"], "structurally_validated_tool_action")
        self.assertEqual(verification_stage["usage"]["total_tokens"], 0)
        verify_agent.assert_not_called()

    def test_agent_final_text_still_uses_model_verifier(self):
        tools = [{"type": "function", "function": {
            "name": "read_file", "parameters": {"type": "object"},
        }}]
        with patch.object(main.verifier_v6, "verify_agent",
                          return_value={"status": "ACCEPT", "feedback": None}) as verify_agent, \
             patch.object(main.llm, "chat", return_value="Implemented and tested."):
            result = main.run_request({"messages": self.messages, "mode": "direct",
                                       "expert_id": "coding_expert", "tools": tools},
                                      registry=self.registry)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["ghostswitch"]["verification"]["status"], "ACCEPT")
        verify_agent.assert_called_once()

    def test_retry_keeps_same_expert_and_passes_feedback(self):
        decisions = [{"status": "RETRY_WITH_FEEDBACK", "feedback": "Incomplete"},
                     {"status": "ACCEPT", "feedback": None}]
        with patch.object(main.verifier_v6, "verify", side_effect=decisions), \
             patch.object(main.llm, "chat", side_effect=["partial", "complete"]) as chat:
            result = main.run_request({"messages": self.messages, "mode": "direct", "expert_id": "coding_expert"}, registry=self.registry)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["ghostswitch"]["verification"]["attempts"], 2)
        self.assertEqual([call.kwargs["adapter"] for call in chat.call_args_list], ["coding_expert", "coding_expert"])
        retry_messages = chat.call_args.args[0]
        self.assertIn("Incomplete", retry_messages[0]["content"])
        self.assertIn("return only the revised answer", retry_messages[0]["content"])
        self.assertEqual([message for message in retry_messages if message["role"] == "user"], self.messages)
        self.assertNotIn("partial", "\n".join(message["content"] for message in retry_messages))

    def test_auto_base_does_not_attach_adapter(self):
        decision = {"route": "base", "expert_id": None, "confidence": .9, "missing_capability": None}
        with patch.object(main.router_v6, "route", return_value=decision), \
             patch.object(main.verifier_v6, "verify", return_value={"status": "ACCEPT", "feedback": None}), \
             patch.object(main.llm, "chat", return_value="hello") as chat:
            result = main.run_request({"messages": self.messages}, registry=self.registry)
        self.assertEqual(result["status"], "success")
        self.assertIsNone(chat.call_args.kwargs["adapter"])
        self.assertEqual(result["ghostswitch"]["route"], "base")

    def test_unsupported_does_not_generate(self):
        decision = {"route": "unsupported", "expert_id": None,
                    "confidence": .7, "missing_capability": "radiology"}
        with patch.object(main.router_v6, "route", return_value=decision), \
             patch.object(main.llm, "chat") as chat:
            result = main.run_request({"messages": self.messages}, registry=self.registry)
        self.assertEqual(result["status"], "unsupported")
        self.assertEqual(result["routing"]["missing_capability"], "radiology")
        self.assertEqual(result["error"]["type"], "expert_not_enabled")
        self.assertIn('"mode": "base"', result["error"]["message"])
        chat.assert_not_called()

    def test_base_mode_answers_with_the_backbone_without_routing(self):
        with patch.object(main.router_v6, "route") as router, \
             patch.object(main.verifier_v6, "verify", return_value={"status": "ACCEPT", "feedback": None}), \
             patch.object(main.llm, "chat", return_value="Ответ.") as chat:
            result = main.run_request({"messages": self.messages, "mode": "base"}, registry=self.registry)
        self.assertEqual((result["status"], result["ghostswitch"]["route"]), ("success", "base"))
        self.assertIsNone(chat.call_args.kwargs["adapter"])
        self.assertEqual(result["ghostswitch"]["stages"][0], {"phase": "selection", "status": "base", "expert_id": None})
        router.assert_not_called()
        with self.assertRaisesRegex(ValueError, "expert_id"):
            main.run_request({"messages": self.messages, "mode": "base", "expert_id": "coding_expert"},
                             registry=self.registry)

    def test_verifier_can_be_turned_off_for_benchmarks(self):
        with patch.object(main.verifier_v6, "verify") as verify, \
             patch.object(main.llm, "chat", return_value="42") as chat:
            result = main.run_request({"messages": self.messages, "mode": "base", "verify": False},
                                      registry=self.registry)
        verify.assert_not_called()
        self.assertEqual(chat.call_count, 1)
        self.assertEqual(result["choices"][0]["message"]["content"], "42")
        stage = result["ghostswitch"]["stages"][-1]
        self.assertEqual((stage["status"], stage["reason"]), ("skipped", "disabled_by_request"))
        with self.assertRaisesRegex(ValueError, "verify"):
            main.run_request({"messages": self.messages, "verify": "no"}, registry=self.registry)

    def test_direct_missing_expert_is_unsupported(self):
        with patch.object(main.router_v6, "route") as router, \
             patch.object(main.llm, "chat") as chat:
            result = main.run_request({"messages": self.messages, "mode": "direct",
                                       "expert_id": "missing_expert"}, registry=self.registry)
        self.assertEqual(result["status"], "unsupported")
        self.assertEqual(result["routing"]["missing_capability"], "missing_expert")
        router.assert_not_called()
        chat.assert_not_called()

    def test_failed_verification_returns_no_answer(self):
        with patch.object(main.verifier_v6, "verify", return_value={"status": "FAIL", "feedback": "Bad output"}), \
             patch.object(main.llm, "chat", return_value="bad"):
            result = main.run_request({"messages": self.messages, "mode": "direct", "expert_id": "coding_expert"}, registry=self.registry)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["choices"], [])
        self.assertEqual(result["error"]["type"], "verification_failed")

    def test_retry_exhaustion_returns_final_fail_status(self):
        retry = {"status": "RETRY_WITH_FEEDBACK", "feedback": "Fix the contradiction."}
        with patch.object(main.verifier_v6, "verify", return_value=retry), \
             patch.object(main.llm, "chat", side_effect=["first", "still wrong"]):
            result = main.run_request({"messages": self.messages, "mode": "direct",
                                       "expert_id": "coding_expert"}, registry=self.registry)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["ghostswitch"]["verification"]["status"], "FAIL")
        self.assertEqual(result["ghostswitch"]["verification"]["attempts"], 2)

    def test_five_attempt_limit_retries_rejected_candidates(self):
        profile = {**PROFILE, "max_retries": 4}
        decisions = ([{"status": "RETRY_WITH_FEEDBACK", "feedback": "Fix the code."}] * 4
                     + [{"status": "ACCEPT", "feedback": None}])
        with patch.object(main, "_profile", return_value=profile), \
             patch.object(main.verifier_v6, "verify", side_effect=decisions), \
             patch.object(main.llm, "chat", side_effect=["broken"] * 4 + ["working"]) as chat:
            result = main.run_request({"messages": self.messages, "mode": "direct",
                                       "expert_id": "coding_expert"}, registry=self.registry)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["choices"][0]["message"]["content"], "working")
        self.assertEqual(result["ghostswitch"]["verification"]["attempts"], 5)
        self.assertEqual(chat.call_count, 5)

    def test_request_rejects_v5_orchestration_fields(self):
        with self.assertRaisesRegex(ValueError, "Unknown request fields"):
            parse_request({"messages": self.messages, "jobs": []})
        with self.assertRaisesRegex(ValueError, "requires expert_id"):
            parse_request({"messages": self.messages, "mode": "direct"})

    def test_router_retries_malformed_json_once(self):
        raw = '{"route":"expert","expert_id":"coding_expert","confidence":0.9}'
        with patch.object(router_v6.thinker, "route_hierarchical",
                          return_value={"expert_id": None, "error": "No suitable expert"}), \
             patch.object(router_v6.llm, "chat", side_effect=["broken", raw]) as chat:
            result = router_v6.route(self.messages, self.registry)
        self.assertEqual(result["expert_id"], "coding_expert")
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["method"], "flat_fallback")
        self.assertEqual(chat.call_count, 2)

    def test_router_uses_hierarchical_path(self):
        with patch.object(router_v6.thinker, "route_hierarchical", return_value={
            "expert_id": "coding_expert", "route_path": ["coding", "coding_agentic"],
            "routing_telemetry": {"routing_calls": 2}, "error": None,
        }) as hierarchy, patch.object(router_v6.llm, "chat") as flat:
            result = router_v6.route(self.messages, self.registry)
        self.assertEqual(result["expert_id"], "coding_expert")
        self.assertEqual(result["method"], "hierarchical")
        self.assertEqual(result["path"], ["coding", "coding_agentic"])
        self.assertEqual(result["steps"], [])
        hierarchy.assert_called_once()
        flat.assert_not_called()

    def test_router_marks_follow_up_as_current_request(self):
        messages = [
            {"role": "user", "content": "Рассчитай сложный процент"},
            {"role": "assistant", "content": "За год получилось 110 рублей."},
            {"role": "user", "content": "Зачем он нужен?"},
        ]
        with patch.object(router_v6.thinker, "route_hierarchical", return_value={
            "expert_id": router_v6._GENERAL_ROUTE_ID,
            "route_path": [router_v6._GENERAL_ROUTE_ID],
            "routing_steps": [],
        }) as hierarchy:
            result = router_v6.route(messages, self.registry)
        request = json.loads(hierarchy.call_args.args[0])
        self.assertEqual(request["current_user_request"], "Зачем он нужен?")
        self.assertEqual(request["conversation_context"], messages[:-1])
        self.assertEqual(result["route"], "base")
        self.assertEqual(result["method"], "hierarchical_general")

    def test_router_exposes_only_compact_decision_steps(self):
        raw = {"routing_steps": [{"node": "root", "choice": "coding", "latency_ms": 12,
                                   "call_metrics": [{"prompt": "private"}],
                                   "expert_fit": {"expert_id": "coding_expert", "match": True,
                                                  "latency_ms": 4, "call_metrics": [{"prompt": "private"}]}}]}
        self.assertEqual(router_v6._public_steps(raw), [
            {"node": "root", "choice": "coding", "latency_ms": 12,
             "expert_fit": {"expert_id": "coding_expert", "match": True, "latency_ms": 4}},
        ])

    def test_unavailable_tree_expert_reports_missing_capability(self):
        with patch.object(router_v6.thinker, "route_hierarchical", return_value={
            "expert_id": "verilog_expert", "route_path": ["engineering_design", "verilog"],
            "routing_telemetry": {"routing_calls": 2},
        }) as hierarchy, patch.object(router_v6.llm, "chat", return_value='{"conversation":false}') as chat:
            result = router_v6.route(self.messages, self.registry)
        self.assertEqual(result["route"], "unsupported")
        self.assertEqual(result["missing_capability"], "verilog_expert")
        self.assertEqual(result["attempts"], 3)
        self.assertIn("verilog_expert", hierarchy.call_args.args[1])
        # Only the conversation check runs; the flat router never sees the request.
        chat.assert_called_once()
        self.assertIn("not installed", chat.call_args.args[0][0]["content"])

    def test_conversation_routed_to_missing_specialist_is_answered_by_base(self):
        messages = [{"role": "user", "content": "Проверь договор аренды на риски"},
                    {"role": "assistant", "content": "Юридический эксперт не установлен."},
                    {"role": "user", "content": "что за бред?"}]
        with patch.object(router_v6.thinker, "route_hierarchical", return_value={
            "expert_id": "legal_contract_expert", "route_path": ["legal_contract", "legal_contract_expert"],
            "routing_telemetry": {"routing_calls": 3},
        }), patch.object(router_v6.llm, "chat", return_value='{"conversation":true}') as chat:
            result = router_v6.route(messages, self.registry)
        self.assertEqual((result["route"], result["expert_id"], result["missing_capability"]),
                         ("base", None, None))
        self.assertEqual(result["method"], "hierarchical_conversation")
        request = json.loads(chat.call_args.args[0][1]["content"])
        self.assertEqual(request["current_user_request"], "что за бред?")

    def test_domain_request_for_missing_specialist_stays_unsupported(self):
        legal = [{"role": "user", "content": "Можешь проверить мой договор аренды на риски?"}]
        for raw in ('{"conversation":false}', "not json", '{"conversation":"yes"}'):
            with patch.object(router_v6.thinker, "route_hierarchical", return_value={
                "expert_id": "legal_contract_expert", "route_path": ["legal_contract"],
            }), patch.object(router_v6.llm, "chat", return_value=raw):
                result = router_v6.route(legal, self.registry)
            self.assertEqual((result["route"], result["missing_capability"]),
                             ("unsupported", "legal_contract_expert"), raw)

    def test_agent_steps_keep_the_turn_route_without_router_calls(self):
        decision = {"expert_id": "coding_expert", "route_path": ["coding", "coding_agentic"],
                    "routing_telemetry": {"routing_calls": 4}, "routing_steps": [{"node": "root"}]}
        messages = [{"role": "system", "content": "You are a coding agent."},
                    {"role": "user", "content": "создай сайт для смм агенства"}]
        with patch.object(router_v6.thinker, "route_hierarchical", return_value=decision) as hierarchy, \
             patch.object(router_v6.llm, "chat") as chat:
            first = router_v6.route(messages, self.registry)
            for step in range(5):
                messages = messages + [
                    {"role": "assistant", "content": None, "tool_calls": [
                        {"id": f"call_{step}", "type": "function",
                         "function": {"name": "write", "arguments": json.dumps({"content": "x" * 50000})}}]},
                    {"role": "tool", "tool_call_id": f"call_{step}", "content": "Wrote file successfully."},
                ]
                result = router_v6.route(messages, self.registry)
                self.assertEqual((result["route"], result["expert_id"], result["path"]),
                                 ("expert", "coding_expert", ["coding", "coding_agentic"]))
                self.assertEqual((result["method"], result["steps"], result["attempts"]),
                                 ("sticky_continuation", [], 0))
            # A bare "continue" after the agent answered keeps the same specialist too.
            messages = messages + [{"role": "assistant", "content": "Сайт готов."},
                                   {"role": "user", "content": "продолжи"}]
            self.assertEqual(router_v6.route(messages, self.registry)["method"], "sticky_continuation")
        self.assertEqual(first["method"], "hierarchical")
        hierarchy.assert_called_once()
        chat.assert_not_called()

    def test_new_user_turn_is_routed_again(self):
        decisions = [{"expert_id": "coding_expert", "route_path": ["coding"]},
                     {"expert_id": router_v6._GENERAL_ROUTE_ID, "route_path": []}]
        messages = [{"role": "user", "content": "Implement a parser"},
                    {"role": "assistant", "content": "Done."},
                    {"role": "user", "content": "What is DNS?"}]
        with patch.object(router_v6.thinker, "route_hierarchical", side_effect=decisions) as hierarchy:
            router_v6.route(messages[:1], self.registry)
            result = router_v6.route(messages, self.registry)
        self.assertEqual((result["route"], result["method"]), ("base", "hierarchical_general"))
        self.assertEqual(hierarchy.call_count, 2)

    def test_cold_continuation_routes_only_the_user_turn(self):
        messages = [{"role": "user", "content": "Inspect the parser"},
                    {"role": "assistant", "content": "Reading", "tool_calls": [
                        {"id": "read_1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": "read_1", "content": "SECRET_OBSERVATION" * 1000}]
        with patch.object(router_v6.thinker, "route_hierarchical", return_value={
            "expert_id": "coding_expert", "route_path": ["coding"],
        }) as hierarchy:
            result = router_v6.route(messages, self.registry)
            again = router_v6.route(messages, self.registry)
        self.assertEqual((result["method"], again["method"]), ("hierarchical", "sticky_continuation"))
        request = json.loads(hierarchy.call_args.args[0])
        self.assertEqual(request, {"current_user_request": "Inspect the parser", "conversation_context": []})
        hierarchy.assert_called_once()

    def test_sticky_route_needs_the_expert_to_stay_available(self):
        class NoExperts(Registry):
            def descriptions(self):
                return {}

        continuation = self.messages + [{"role": "assistant", "content": "Working on it"},
                                        {"role": "user", "content": "continue"}]
        with patch.object(router_v6.thinker, "route_hierarchical", return_value={
            "expert_id": "coding_expert", "route_path": ["coding"],
        }):
            router_v6.route(self.messages, self.registry)
        result = router_v6.route(continuation, NoExperts())
        self.assertEqual((result["route"], result["method"]), ("base", "no_available_experts"))

    def test_unsupported_turn_is_not_sticky(self):
        continuation = self.messages + [{"role": "assistant", "content": "Working on it"},
                                        {"role": "user", "content": "продолжай"}]
        with patch.object(router_v6.thinker, "route_hierarchical", return_value={
            "expert_id": "verilog_expert", "route_path": ["verilog"],
        }) as hierarchy, patch.object(router_v6.llm, "chat", return_value='{"conversation":false}'):
            first = router_v6.route(self.messages, self.registry)
            second = router_v6.route(continuation, self.registry)
        self.assertEqual((first["route"], second["route"]), ("unsupported", "unsupported"))
        self.assertEqual(hierarchy.call_count, 2)

    def test_agent_step_through_pipeline_records_sticky_routing(self):
        tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]
        first = [{"role": "user", "content": "Inspect the parser"}]
        candidate = '<tool_call>{"name":"read","arguments":{}}</tool_call>'
        with patch.object(router_v6.thinker, "route_hierarchical", return_value={
            "expert_id": "coding_expert", "route_path": ["coding"],
        }) as hierarchy, patch.object(main.llm, "chat", return_value=candidate):
            step = main.run_request({"messages": first, "tools": tools}, registry=self.registry)
            message = step["choices"][0]["message"]
            history = first + [message, {"role": "tool", "tool_call_id": message["tool_calls"][0]["id"],
                                         "content": "def parse(): pass"}]
            result = main.run_request({"messages": history, "tools": tools}, registry=self.registry)
        routing = result["ghostswitch"]["stages"][0]
        self.assertEqual((routing["phase"], routing["status"], routing["method"]),
                         ("routing", "expert", "sticky_continuation"))
        self.assertEqual(routing["usage"]["total_tokens"], 0)
        self.assertEqual(result["ghostswitch"]["expert_id"], "coding_expert")
        hierarchy.assert_called_once()

    def test_rejected_expert_is_excluded_from_fallback(self):
        tree = RouterTree("root", [
            RouterNode("root", "Select a specialist", children=("first", "second")),
            RouterNode("first", "First specialty", expert_id="first"),
            RouterNode("second", "Second specialty", expert_id="second"),
        ])

        class TwoExpertRegistry:
            def descriptions(self):
                return {"first": "First specialty", "second": "Second specialty"}

        hierarchy_result = {"expert_id": None, "error": "First specialty does not fit",
                            "routing_steps": [{"expert_fit": {"expert_id": "first", "match": False}}]}
        with patch.object(router_v6.thinker, "route_hierarchical", return_value=hierarchy_result), \
             patch.object(router_v6.llm, "chat", return_value='{"route":"expert","expert_id":"second"}') as flat:
            result = router_v6.route(self.messages, TwoExpertRegistry(), tree=tree)
        self.assertEqual(result["expert_id"], "second")
        self.assertNotIn("First specialty", flat.call_args.args[0][0]["content"])

    def test_general_tree_choice_resolves_to_base(self):
        with patch.object(router_v6.thinker, "route_hierarchical", return_value={
            "expert_id": router_v6._GENERAL_ROUTE_ID, "route_path": [router_v6._GENERAL_ROUTE_ID],
        }) as hierarchy, patch.object(router_v6.llm, "chat", return_value='{"route":"base","expert_id":null}'):
            result = router_v6.route([{"role": "user", "content": "Hello"}], self.registry)
        self.assertEqual(result["route"], "base")
        self.assertEqual(result["method"], "hierarchical_general")
        hierarchy.assert_called_once()

    def test_general_tree_choice_never_invents_a_missing_specialty(self):
        with patch.object(router_v6.thinker, "route_hierarchical", return_value={
            "expert_id": router_v6._GENERAL_ROUTE_ID, "route_path": [router_v6._GENERAL_ROUTE_ID],
        }), patch.object(router_v6.llm, "chat") as chat:
            result = router_v6.route([{"role": "user", "content": "расскажи стишок на каннада про слона"}],
                                     self.registry)
        self.assertEqual(result["route"], "base")
        self.assertIsNone(result["missing_capability"])
        chat.assert_not_called()

    def test_flat_fallback_turns_an_unlisted_specialty_into_base(self):
        with patch.object(router_v6.thinker, "route_hierarchical",
                          return_value={"expert_id": None, "error": "No suitable expert"}), \
             patch.object(router_v6.llm, "chat", return_value=(
                 '{"route":"unsupported","expert_id":null,"missing_capability":"Kannada poetry"}')):
            result = router_v6.route(self.messages, self.registry)
        self.assertEqual((result["route"], result["missing_capability"]), ("base", None))
        self.assertEqual(result["method"], "flat_fallback")

    def test_hierarchical_router_narrows_100_virtual_profiles(self):
        nodes = [RouterNode("root", "Choose a specialty group", children=tuple(f"group_{n}" for n in range(10)))]
        for group in range(10):
            nodes.append(RouterNode(f"group_{group}", f"Specialty group {group}",
                                    children=tuple(f"virtual_{group * 10 + n}" for n in range(10))))
            for item in range(10):
                number = group * 10 + item
                nodes.append(RouterNode(f"virtual_{number}", f"Virtual specialty {number}",
                                        expert_id=f"virtual_{number}"))
        tree = RouterTree("root", nodes, known_experts={f"virtual_{n}" for n in range(100)})

        class VirtualRegistry:
            def descriptions(self):
                return {f"virtual_{n}": f"Virtual specialty {n}" for n in range(100)}

        prompts = []

        def choose(messages, **kwargs):
            prompts.append(messages[0]["content"])
            system = messages[0]["content"]
            if '"id":"root"' in system:
                return '{"choice":"__v6_specialists__"}'
            if '"id":"__v6_specialists__"' in system:
                return '{"choice":"group_3"}'
            return '{"choice":"virtual_37"}'

        with patch.object(router_v6.llm, "chat", side_effect=choose):
            result = router_v6.route(self.messages, VirtualRegistry(), tree=tree)
        self.assertEqual(result["expert_id"], "virtual_37")
        self.assertEqual(result["path"], ["group_3", "virtual_37"])
        self.assertEqual(len(prompts), 3)
        self.assertNotIn("virtual_99", prompts[2])

    def test_router_normalizes_expert_id_in_route_field(self):
        route = router_v6._validate({"route": "coding_expert", "expert_id": None,
                                     "confidence": 0.9}, {"coding_expert"})
        self.assertEqual(route["route"], "expert")
        self.assertEqual(route["expert_id"], "coding_expert")

    def test_verifier_requires_complete_output(self):
        with patch.object(verifier_v6.llm, "chat") as chat:
            result = verifier_v6.verify(self.messages, "partial", finish_reason="length", review=True)
        self.assertEqual(result["status"], "RETRY_WITH_FEEDBACK")
        chat.assert_not_called()

    def test_verifier_runs_exact_checks_without_the_model_by_default(self):
        with patch.object(verifier_v6.llm, "chat") as chat:
            accepted = verifier_v6.verify(self.messages, "candidate", finish_reason="stop")
            wrong = verifier_v6.verify(self.ARITHMETIC, "553", finish_reason="stop")
            final = verifier_v6.verify_agent(self.ARITHMETIC, {"role": "assistant", "content": "403"}, [],
                                             finish_reason="stop")
            empty = verifier_v6.verify_agent(self.ARITHMETIC, {"role": "assistant", "content": ""}, [],
                                             finish_reason="stop")
        self.assertEqual(accepted, {"status": "ACCEPT", "feedback": None})
        self.assertEqual(wrong["status"], "RETRY_WITH_FEEDBACK")
        self.assertEqual(final, {"status": "ACCEPT", "feedback": None})
        self.assertEqual(empty["status"], "RETRY_WITH_FEEDBACK")
        chat.assert_not_called()

    def test_verifier_retries_when_its_own_json_is_truncated(self):
        def truncated(*args, **kwargs):
            kwargs["debug"]["finish_reason"] = "length"
            return '{"status":"RETRY_WITH_FEEDBACK","feedback":"cut off'
        with patch.object(verifier_v6.llm, "chat", side_effect=truncated):
            result = verifier_v6.verify(self.messages, "candidate", finish_reason="stop", review=True)
        self.assertEqual(result["status"], "RETRY_WITH_FEEDBACK")
        self.assertIn("truncated", result["feedback"])

    def test_verifier_rejects_meta_commentary_instead_of_an_answer(self):
        messages = [{"role": "user", "content": "Объясни дисконтирование подробнее"}]
        with patch.object(verifier_v6.llm, "chat") as chat:
            result = verifier_v6.verify(
                messages,
                "Объяснение слишком краткое. Нужно добавить пример расчета.",
                finish_reason="stop",
                review=True,
            )
        self.assertEqual(result["status"], "RETRY_WITH_FEEDBACK")
        self.assertIn("directly", result["feedback"])
        chat.assert_not_called()

    def test_verifier_allows_requested_answer_review(self):
        messages = [{"role": "user", "content": "Оцени этот ответ"}]
        with patch.object(verifier_v6.llm, "chat", return_value='{"status":"ACCEPT","feedback":null}') as chat:
            result = verifier_v6.verify(
                messages,
                "Ответ слишком краткий и не содержит примера.",
                finish_reason="stop",
                review=True,
            )
        self.assertEqual(result["status"], "ACCEPT")
        chat.assert_called_once()

    def test_verifier_retries_legacy_fail_verdict_for_fixable_code(self):
        raw = '{"status":"FAIL","feedback":"The Python code has syntax errors."}'
        with patch.object(verifier_v6.llm, "chat", return_value=raw):
            result = verifier_v6.verify(self.messages, "def broken(", finish_reason="stop", review=True)
        self.assertEqual(result, {"status": "RETRY_WITH_FEEDBACK",
                                  "feedback": "The Python code has syntax errors."})

    ARITHMETIC = [{"role": "user", "content": "Сколько будет 17 * 23 + 144 / 12? Дай только число."}]

    def review(self, candidate, verdict, messages=None):
        with patch.object(verifier_v6.llm, "chat", return_value=json.dumps(verdict, ensure_ascii=False)):
            return verifier_v6.verify(messages or self.ARITHMETIC, candidate, finish_reason="stop", review=True)

    def test_verifier_check_confirms_a_result_it_called_wrong(self):
        result = self.review("403", {"status": "RETRY_WITH_FEEDBACK", "issue": "result",
                                     "feedback": "Результат неверен. Правильный ответ: 403.",
                                     "check": "17*23+144/12"})
        self.assertEqual(result["status"], "ACCEPT")
        self.assertIsNone(result["feedback"])
        self.assertEqual(result["check"]["override"], "confirmed")

    def test_verifier_check_keeps_format_rejections(self):
        result = self.review("**Calculation:** 391 + 12 = 403. **Answer: 403**",
                             {"status": "RETRY_WITH_FEEDBACK", "issue": "format",
                              "feedback": "Return only the number 403.", "check": "17*23+144/12"})
        self.assertEqual(result["status"], "RETRY_WITH_FEEDBACK")
        self.assertEqual(result["feedback"], "Return only the number 403.")

    def test_verifier_check_rejects_accepted_wrong_arithmetic(self):
        result = self.review("553", {"status": "ACCEPT", "issue": None, "feedback": None,
                                     "check": "17*23+144/12"})
        self.assertEqual(result["status"], "RETRY_WITH_FEEDBACK")
        self.assertNotIn("403", result["feedback"])
        self.assertEqual(result["check"]["override"], "contradicted")

    def test_verifier_trusts_only_direct_checks_against_an_accept(self):
        loan = [{"role": "user", "content": "Кредит 1 000 000 рублей на 5 лет под 12% годовых. Ежемесячный платёж?"}]
        result = self.review("≈ 22 100 рублей", {"status": "ACCEPT", "issue": None, "feedback": None,
                                                  "check": "1000000*0.01*1.01**60/(1.01**60-1)"}, loan)
        self.assertEqual(result["status"], "ACCEPT")
        self.assertFalse(result["check"]["agrees"])

    def test_verifier_withholds_its_own_values_from_result_feedback(self):
        loan = [{"role": "user", "content": "Кредит 1 000 000 рублей на 5 лет под 12% годовых. Ежемесячный платёж?"}]
        result = self.review("Платёж ≈ 18 968,54 рубля.",
                             {"status": "RETRY_WITH_FEEDBACK", "issue": "result",
                              "feedback": "Ставка за месяц 1%. Правильный платёж ≈ 23 115,54 рубля."}, loan)
        self.assertEqual(result["status"], "RETRY_WITH_FEEDBACK")
        self.assertNotIn("23 115,54", result["feedback"])
        self.assertIn("…", result["feedback"])

    def test_verifier_drops_language_claims_without_cjk_text(self):
        claim = {"status": "RETRY_WITH_FEEDBACK", "issue": "language",
                 "feedback": "Use English instead of Chinese symbols (×, −)."}
        messages = [{"role": "user", "content": "What is 48 * 37 - 296?"}]
        result = self.review("48 × 37 = 1776; 1776 − 296 = 1480", claim, messages)
        self.assertEqual(result["status"], "ACCEPT")
        self.assertEqual(result["override"], "unsupported_language_claim")
        # Real Chinese text and requests about Chinese keep the reviewer's verdict.
        self.assertEqual(self.review("Температура 他便血", claim, messages)["status"], "RETRY_WITH_FEEDBACK")
        to_chinese = [{"role": "user", "content": "Translate 'good morning' into Chinese."}]
        self.assertEqual(self.review("Good morning", claim, to_chinese)["status"], "RETRY_WITH_FEEDBACK")

    def test_verifier_drops_demands_for_the_language_already_used(self):
        news = [{"role": "user", "content": "последние новости"}]
        claim = {"status": "RETRY_WITH_FEEDBACK", "issue": "language",
                 "feedback": "Отвечайте на русском языке, как в запросе."}
        result = self.review("Главные новости на сегодня: в Москве открылась выставка `ai-expo` [1].", claim, news)
        self.assertEqual(result, {"status": "ACCEPT", "feedback": None, "override": "language_already_used"})
        # A wrong language, Ukrainian text, or another kind of problem keeps the verdict.
        self.assertEqual(self.review("Here are today's headlines and the main events.", claim, news)["status"],
                         "RETRY_WITH_FEEDBACK")
        self.assertEqual(self.review("Ось головні новини сьогодні.", claim, news)["status"], "RETRY_WITH_FEEDBACK")
        incomplete = {**claim, "issue": "completeness", "feedback": "Отвечайте на русском и назовите источники."}
        result = self.review("Главные новости на сегодня.", incomplete, news)
        self.assertEqual((result["status"], result["issue"]), ("RETRY_WITH_FEEDBACK", "completeness"))

    def test_verifier_ignores_unsafe_check_expressions(self):
        pencils = [{"role": "user", "content": "У Маши было 3 коробки по 12 карандашей, она отдала 7. Сколько осталось?"}]
        result = self.review("31", {"status": "ACCEPT", "feedback": None,
                                    "check": "__import__('os').system('true')"}, pencils)
        self.assertEqual(result, {"status": "ACCEPT", "feedback": None})

    def test_verifier_repeats_review_after_invalid_json(self):
        with patch.object(verifier_v6.llm, "chat",
                          side_effect=["not json", '{"status":"ACCEPT","feedback":null}']) as chat:
            result = verifier_v6.verify(self.ARITHMETIC, "403", finish_reason="stop", review=True)
        self.assertEqual(result["status"], "ACCEPT")
        self.assertIn("validation_error", chat.call_args.args[0][1]["content"])
        with patch.object(verifier_v6.llm, "chat", side_effect=["oops", "still not json"]):
            result = verifier_v6.verify(self.ARITHMETIC, "403", finish_reason="stop", review=True)
        self.assertEqual(result["status"], "FAIL")

    def test_agent_verifier_repeats_review_after_invalid_json(self):
        candidate = {"role": "assistant", "content": "403"}
        with patch.object(verifier_v6.llm, "chat",
                          side_effect=["oops", '{"status":"ACCEPT","feedback":null}']):
            result = verifier_v6.verify_agent(self.ARITHMETIC, candidate, [], finish_reason="stop", review=True)
        self.assertEqual(result, {"status": "ACCEPT", "feedback": None})

    def test_rejected_candidate_is_kept_in_verification_stage(self):
        decisions = [{"status": "RETRY_WITH_FEEDBACK", "feedback": "Incomplete"},
                     {"status": "ACCEPT", "feedback": None}]
        with patch.object(main.verifier_v6, "verify", side_effect=decisions), \
             patch.object(main.llm, "chat", side_effect=["partial", "complete"]):
            result = main.run_request(self.direct(), registry=self.registry)
        rejected, accepted = [s for s in result["ghostswitch"]["stages"] if s["phase"] == "verification"]
        self.assertEqual(rejected["candidate_preview"], "partial")
        self.assertNotIn("candidate_preview", accepted)

    def direct(self, **extra):
        return {"messages": self.messages, "mode": "direct", "expert_id": "coding_expert", **extra}

    def test_thinking_expert_returns_reasoning_separately_from_answer(self):
        accept = {"status": "ACCEPT", "feedback": None}
        with patch.object(main.verifier_v6, "verify", return_value=accept) as verify, \
             patch.object(main.llm, "chat", return_value="17 * 23 = 391; 144 / 12 = 12.\n</think>\n\n403") as chat:
            result = main.run_request(self.direct(), registry=ThinkingRegistry())
        self.assertEqual(result["status"], "success")
        message = result["choices"][0]["message"]
        self.assertEqual(message["content"], "403")
        self.assertEqual(message["reasoning_content"], "17 * 23 = 391; 144 / 12 = 12.")
        self.assertTrue(chat.call_args.kwargs["enable_thinking"])
        # Reasoning gets its own allowance on top of the answer budget.
        self.assertEqual(chat.call_args.kwargs["max_tokens"], 4096 + 4096)
        self.assertEqual(verify.call_args.args[1], "403")
        self.assertEqual(result["ghostswitch"]["thinking"],
                         {"enabled": True, "source": "expert_policy", "used": True})
        generation = next(stage for stage in result["ghostswitch"]["stages"] if stage["phase"] == "generation")
        self.assertTrue(generation["thinking"])

    def test_expert_without_thinking_policy_answers_directly(self):
        with patch.object(main.verifier_v6, "verify", return_value={"status": "ACCEPT", "feedback": None}), \
             patch.object(main.llm, "chat", return_value="def f(): pass") as chat:
            result = main.run_request(self.direct(), registry=self.registry)
        self.assertFalse(chat.call_args.kwargs["enable_thinking"])
        self.assertNotIn("reasoning_content", result["choices"][0]["message"])
        self.assertEqual(result["ghostswitch"]["thinking"],
                         {"enabled": False, "source": "default", "used": False})

    def test_thinking_stream_splits_reasoning_and_answer_events(self):
        def chat(messages, **kwargs):
            for piece in ("Plan", " step</th", "ink>\n\n4", "03"):
                kwargs["on_token"](piece)
            return "Plan step</think>\n\n403"

        events = []
        with patch.object(main.verifier_v6, "verify", return_value={"status": "ACCEPT", "feedback": None}), \
             patch.object(main.llm, "chat", side_effect=chat):
            main.run_request(self.direct(), registry=ThinkingRegistry(), on_event=events.append)
        self.assertEqual("".join(e["delta"] for e in events if e["type"] == "reasoning"), "Plan step")
        self.assertEqual("".join(e["delta"] for e in events if e["type"] == "token"), "403")
        statuses = [e["status"] for e in events if e["type"] == "status"]
        self.assertLess(statuses.index("Thinking"), statuses.index("Generating response"))

    def test_unfinished_thinking_retries_without_thinking(self):
        def chat(messages, **kwargs):
            if kwargs["enable_thinking"]:
                kwargs["debug"]["finish_reason"] = "length"
                return "Still reasoning about the problem"
            return "403"

        with patch.object(main.verifier_v6, "verify", return_value={"status": "ACCEPT", "feedback": None}) as verify, \
             patch.object(main.llm, "chat", side_effect=chat) as chat_mock:
            result = main.run_request(self.direct(), registry=ThinkingRegistry())
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["choices"][0]["message"]["content"], "403")
        self.assertNotIn("reasoning_content", result["choices"][0]["message"])
        self.assertEqual([call.kwargs["enable_thinking"] for call in chat_mock.call_args_list], [True, False])
        self.assertEqual(result["ghostswitch"]["stages"][2]["status"], "thinking_overflow")
        self.assertEqual(result["ghostswitch"]["stages"][3]["reason"], "thinking_overflow")
        self.assertFalse(result["ghostswitch"]["thinking"]["used"])
        verify.assert_called_once()

    def test_caller_can_force_thinking_on_or_off(self):
        accept = {"status": "ACCEPT", "feedback": None}
        decision = {"route": "base", "expert_id": None, "confidence": .9, "missing_capability": None}
        with patch.object(main.router_v6, "route", return_value=decision), \
             patch.object(main.verifier_v6, "verify", return_value=accept), \
             patch.object(main.llm, "chat", return_value="Reasoned.</think>\n\nAnswer") as chat:
            result = main.run_request({"messages": self.messages, "reasoning_effort": "high"},
                                      registry=self.registry)
        self.assertTrue(chat.call_args.kwargs["enable_thinking"])
        self.assertEqual(result["ghostswitch"]["thinking"]["source"], "request")
        with patch.object(main.verifier_v6, "verify", return_value=accept), \
             patch.object(main.llm, "chat", return_value="403") as chat:
            main.run_request(self.direct(chat_template_kwargs={"enable_thinking": False}),
                             registry=ThinkingRegistry())
        self.assertFalse(chat.call_args.kwargs["enable_thinking"])

    def test_request_validates_thinking_controls(self):
        self.assertIsNone(parse_request({"messages": self.messages})["thinking"])
        self.assertFalse(parse_request({"messages": self.messages, "reasoning_effort": "none"})["thinking"])
        with self.assertRaisesRegex(ValueError, "reasoning_effort must be"):
            parse_request({"messages": self.messages, "reasoning_effort": "max"})
        with self.assertRaisesRegex(ValueError, "disagree"):
            parse_request({"messages": self.messages, "reasoning_effort": "high",
                           "chat_template_kwargs": {"enable_thinking": False}})
        with self.assertRaisesRegex(ValueError, "only a boolean enable_thinking"):
            parse_request({"messages": self.messages, "chat_template_kwargs": {"top_k": 20}})

    def test_echoed_assistant_reasoning_is_accepted_but_not_reused(self):
        history = [{"role": "user", "content": "2+2?"},
                   {"role": "assistant", "content": "4", "reasoning_content": "Add the numbers."},
                   {"role": "user", "content": "And 3+3?"}]
        normalized = parse_request({"messages": history})["messages"]
        self.assertEqual(normalized[1], {"role": "assistant", "content": "4"})
        with self.assertRaisesRegex(ValueError, "Reasoning fields"):
            parse_request({"messages": [{"role": "user", "content": "Hi", "reasoning_content": "x"}]})


if __name__ == "__main__":
    unittest.main()
