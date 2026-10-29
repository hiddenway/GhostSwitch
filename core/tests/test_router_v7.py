from __future__ import annotations

import json
import re
import unittest
from unittest.mock import patch

from core import llm, main, router_v6, router_v7
from core.router_tree import RouterNode, RouterTree


TREE = RouterTree("root", [
    RouterNode("root", "Choose a field", children=("coding", "hardware")),
    RouterNode("coding", "Software", children=("coding_agentic", "coding_swegym")),
    RouterNode("coding_agentic", "Writes code", expert_id="coding_expert"),
    RouterNode("coding_swegym", "Repository work", expert_id="software_engineering_expert"),
    RouterNode("hardware", "Hardware design", children=("verilog",)),
    RouterNode("verilog", "Verilog and RTL", expert_id="verilog_expert"),
])
STEP = re.compile(r"ROUTING STEP at node (\S+)\. Options: (.*)\. Reply with one letter\.")


class Registry:
    def __init__(self, installed=("coding_expert", "software_engineering_expert")):
        self.installed = installed

    def descriptions(self):
        return {expert: expert for expert in self.installed}


class FakeScorer:
    """Answers each step with the option a script names for its node."""

    def __init__(self, script):
        self.script, self.steps, self.recorded = script, [], None

    def choose(self, step, count):
        node, listed = STEP.fullmatch(step).groups()
        options = re.findall(r"[A-Z]\) (\S+)", listed)
        self.steps.append((node, options))
        assert len(options) == count
        index = options.index(self.script[node])
        return index, [0.9 if i == index else 0.1 / max(1, count - 1) for i in range(count)]

    def record(self, steps):
        self.recorded = steps


class RouterV7Tests(unittest.TestCase):
    def setUp(self):
        router_v6._STICKY_ROUTES.clear()
        self.addCleanup(router_v6._STICKY_ROUTES.clear)
        self.messages = [{"role": "user", "content": "Implement a parser"}]

    def route(self, script, messages=None, registry=None, conversation=False):
        scorer = FakeScorer(script)
        with patch.object(router_v7, "_scorer", return_value=scorer) as factory, \
             patch.object(router_v7, "_is_conversation", return_value=conversation):
            result = router_v7.route(messages or self.messages, registry or Registry(), 128, tree=TREE)
        return result, scorer, factory

    def test_installed_expert_is_chosen_in_two_lettered_steps(self):
        result, scorer, _ = self.route({"root": "coding", "coding": "coding_agentic"})
        self.assertEqual((result["route"], result["expert_id"], result["method"]),
                         ("expert", "coding_expert", "v7"))
        self.assertEqual(result["path"], ["coding", "coding_agentic"])
        self.assertEqual([step["node"] for step in result["steps"]], ["root", "coding"])
        self.assertEqual(result["confidence"], 0.9)
        self.assertEqual(result["attempts"], 2)
        # The root offers the fields and general, without none; below it none comes last.
        self.assertEqual(scorer.steps[0][1], ["coding", "hardware", "general"])
        self.assertEqual(scorer.steps[1][1], ["coding_agentic", "coding_swegym", "none"])
        self.assertEqual(scorer.recorded, 2)

    def test_general_and_none_go_to_the_base_model(self):
        result, _, _ = self.route({"root": "general"})
        self.assertEqual((result["route"], result["method"], result["path"]), ("base", "v7_general", []))
        result, _, _ = self.route({"root": "coding", "coding": "none"})
        self.assertEqual((result["route"], result["method"], result["expert_id"]), ("base", "v7_none", None))

    def test_whole_tree_is_offered_and_a_missing_expert_is_unsupported(self):
        """Like v6: a catalog expert that is not turned on is offered for download."""
        result, scorer, _ = self.route({"root": "hardware", "hardware": "verilog"})
        self.assertEqual((result["route"], result["missing_capability"], result["method"]),
                         ("unsupported", "verilog_expert", "v7_unavailable"))
        self.assertEqual(result["path"], ["hardware", "verilog"])
        self.assertIn("hardware", scorer.steps[0][1])

    def test_conversation_about_a_missing_specialty_is_answered_by_base(self):
        result, _, _ = self.route({"root": "hardware", "hardware": "verilog"}, conversation=True)
        self.assertEqual((result["route"], result["method"]), ("base", "v7_conversation"))

    def test_without_installed_experts_nothing_is_scored(self):
        result, _, factory = self.route({}, registry=Registry(installed=()))
        self.assertEqual((result["route"], result["method"]), ("base", "no_available_experts"))
        factory.assert_not_called()

    def test_installed_expert_outside_the_tree_is_a_routing_error(self):
        with self.assertRaises(router_v6.RoutingError):
            self.route({}, registry=Registry(installed=("coding_expert", "lean4_expert")))

    def test_agent_steps_keep_the_turn_route_without_scoring(self):
        messages = [{"role": "system", "content": "You are a coding agent."},
                    {"role": "user", "content": "Build a landing page"}]
        first, _, _ = self.route({"root": "coding", "coding": "coding_agentic"}, messages)
        for step in range(3):
            messages = messages + [
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": f"call_{step}", "type": "function",
                     "function": {"name": "write", "arguments": json.dumps({"content": "x" * 5000})}}]},
                {"role": "tool", "tool_call_id": f"call_{step}", "content": "ok"},
            ]
            result, _, factory = self.route({}, messages)
            self.assertEqual((result["method"], result["expert_id"], result["steps"]),
                             ("sticky_continuation", "coding_expert", []))
            factory.assert_not_called()
        messages = messages + [{"role": "assistant", "content": "Done."}, {"role": "user", "content": "continue"}]
        self.assertEqual(self.route({}, messages)[0]["method"], "sticky_continuation")
        self.assertEqual(first["method"], "v7")

    def test_new_user_turn_is_routed_again_and_only_the_turn_is_read(self):
        messages = [{"role": "user", "content": "Inspect the parser"},
                    {"role": "assistant", "content": "Reading", "tool_calls": [
                        {"id": "r1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": "r1", "content": "SECRET_OBSERVATION" * 100}]
        with patch.object(router_v7, "_scorer", return_value=FakeScorer({"root": "general"})) as factory:
            router_v7.route(messages, Registry(), tree=TREE)
        request = factory.call_args.args[1]
        self.assertIn("Inspect the parser", request)
        self.assertNotIn("SECRET_OBSERVATION", request)
        later = messages + [{"role": "assistant", "content": "Done."}, {"role": "user", "content": "What is DNS?"}]
        result, _, factory = self.route({"root": "general"}, later)
        factory.assert_called_once()
        self.assertEqual(result["method"], "v7_general")

    def test_failures_are_routing_errors_and_cancellation_passes_through(self):
        class Broken(FakeScorer):
            def __init__(self, error):
                super().__init__({})
                self.error = error

            def choose(self, step, count):
                raise self.error

        for error, expected in ((ValueError("bad logits"), router_v6.RoutingError),
                                (llm.GenerationCancelled("Client disconnected"), llm.GenerationCancelled)):
            scorer = Broken(error)
            with patch.object(router_v7, "_scorer", return_value=scorer), self.assertRaises(expected):
                router_v7.route(self.messages, Registry(), tree=TREE)
            self.assertEqual(scorer.recorded, 0)

    def test_chat_scorer_rejects_a_reply_that_is_not_an_option(self):
        scorer = router_v7._ChatScorer("system", "request")
        with patch.object(router_v7.llm, "chat", return_value="Z") as chat, \
             self.assertRaises(router_v6.RoutingError):
            scorer.choose("ROUTING STEP at node root. Options: A) coding B) general. Reply with one letter.", 2)
        self.assertEqual((chat.call_args.kwargs["max_tokens"], chat.call_args.kwargs["temperature"]), (2, 0))
        for empty in ("", " ", "\n"):
            with patch.object(router_v7.llm, "chat", return_value=empty), self.assertRaises(router_v6.RoutingError):
                scorer.choose("ROUTING STEP at node root. Options: A) x B) y. Reply with one letter.", 2)
        with patch.object(router_v7.llm, "chat", return_value=" b"):
            self.assertEqual(scorer.choose("ROUTING STEP at node root. Options: A) x B) y. Reply with one letter.", 2),
                             (1, []))

    def test_long_requests_keep_their_head_and_tail(self):
        class Tokens:
            def encode(self, text):
                return list(text)

            def decode(self, ids):
                return "".join(ids)

        short = "x" * router_v7.MAX_REQUEST_TOKENS
        self.assertEqual(router_v7._truncate(Tokens(), short), short)
        long = "H" * 500 + "M" * 1000 + "T" * 200
        cut = router_v7._truncate(Tokens(), long)
        self.assertEqual(cut, "H" * 450 + "\n[...]\n" + "T" * 150)

    def test_catalog_lists_every_node_once_with_general_at_the_root(self):
        tree = router_v7._one_step_root(TREE)
        choosable = set(TREE._experts_below(TREE.root_id)) | {router_v6._GENERAL_ROUTE_ID}
        catalog = router_v7._catalog(tree, choosable)
        self.assertEqual(catalog.count("\nNODE "), 3)
        self.assertIn("- general:", catalog)
        self.assertIn("- verilog: Verilog and RTL", catalog)
        self.assertEqual(catalog, router_v7._catalog(tree, choosable))

    def test_mlx_scorer_telemetry_counts_reused_and_computed_tokens(self):
        scorer = object.__new__(router_v7._MLXScorer)
        scorer.loaded = type("Loaded", (), {"spec": type("Spec", (), {"model_id": "m", "total_params": 9})()})()
        # Instructions S=1500 from saved state, request R=100 computed once, two steps of 20 tokens:
        # each step's prompt is S+R+20, and only R and the step lines were computed.
        scorer.prompt_tokens, scorer.computed = 2 * (1500 + 100 + 20), 100 + 2 * 20
        scorer.system_cache_hit, scorer.elapsed_ns = True, 2_000_000
        with llm.capture_telemetry() as telemetry:
            scorer.record(2)
        call = telemetry.calls[0]
        self.assertEqual((call["role"], call["trace_stage"], call["output_tokens"]), ("thinker", "routing", 0))
        self.assertEqual(main._usage(telemetry.calls),
                         {"prompt_tokens": 3240, "completion_tokens": 0, "total_tokens": 3240,
                          "prompt_tokens_details": {"cached_tokens": 3100}})
        self.assertEqual(telemetry.task_metrics()["thinker_latency_ms"], 2.0)


class ProfileDispatchTests(unittest.TestCase):
    PROFILE = {"backend": "mlx", "max_tokens_per_call": 10240, "router_max_tokens": 32,
               "verifier_max_tokens": 32, "max_retries": 1}

    def run_with(self, profile):
        decision = {"route": "base", "expert_id": None, "confidence": None, "missing_capability": None,
                    "method": "x", "path": [], "steps": []}
        with patch.object(main, "_profile", return_value=profile), \
             patch.object(main.router_v6, "route", return_value=decision) as v6, \
             patch.object(main.router_v7, "route", return_value=decision) as v7, \
             patch.object(main.verifier_v6, "verify", return_value={"status": "ACCEPT", "feedback": None}), \
             patch.object(main.llm, "chat", return_value="hello"):
            result = main.run_request({"messages": [{"role": "user", "content": "hi"}]}, registry=Registry())
        self.assertEqual(result["status"], "success")
        return v6, v7

    def test_the_profile_names_the_router(self):
        v6, v7 = self.run_with(self.PROFILE)
        v6.assert_called_once()
        v7.assert_not_called()
        v6, v7 = self.run_with({**self.PROFILE, "router": "v7"})
        v7.assert_called_once()
        v6.assert_not_called()
        self.assertEqual(v7.call_args.args[2], 32)

    def test_only_the_9b_mlx_profile_uses_v7(self):
        profiles = main._PROFILE_ROOT
        routers = {path.stem: json.loads(path.read_text()).get("router", "v6") for path in profiles.glob("*_v6.json")}
        self.assertEqual({name for name, router in routers.items() if router == "v7"}, {"mlx9b_v6"})
        self.assertTrue(set(routers.values()) <= set(main.ROUTERS))


if __name__ == "__main__":
    unittest.main()
