from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from core import llm, main, verifier_v6
from core.tests.test_v6 import PROFILE, Registry

TOOLS = [{"type": "function", "function": {"name": "read", "description": "Read a file",
                                           "parameters": {"type": "object", "properties": {}}}}]


class LongPromptRegistry(Registry):
    def get(self, expert_id):
        return SimpleNamespace(adapter_id="coding-adapter", prompt="Write careful code. " * 40)


class CharTokenizer:
    """One token per character, with Qwen3.5's layout: tool definitions ahead of the system text."""

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, enable_thinking=False,
                            tokenize=False, continue_final_message=False):
        text = ""
        for message in messages:
            if message["role"] == "system":
                text += "<s>" + (json.dumps(tools) if tools else "") + message["content"] + "</s>"
            elif message["role"] == "assistant":
                text += "<a>" + json.dumps(message.get("tool_calls") or message.get("content")) + "</a>"
            else:
                text += f"<{message['role'][0]}>" + message["content"] + f"</{message['role'][0]}>"
        return text + ("<a>" if add_generation_prompt else "")

    def encode(self, text):
        return [ord(character) for character in text]


class CountingModel:
    """Moves the cache one position per token and counts the tokens it computed."""

    def __init__(self):
        self.computed = 0

    def __call__(self, inputs, cache):
        length = inputs.shape[1]
        self.computed += length
        for entry in cache:
            entry.update_and_fetch(llm.mx.zeros((1, 1, length, 1)), llm.mx.zeros((1, 1, length, 1)))
        return llm.mx.zeros((1, length, 1))


class PromptCacheTests(unittest.TestCase):
    def test_verifier_caches_its_fixed_rules_ahead_of_the_clock(self):
        with patch.object(verifier_v6.llm, "chat", return_value='{"status":"ACCEPT","feedback":null}') as chat:
            verifier_v6.verify([{"role": "user", "content": "2+2?"}], "4", finish_reason="stop", review=True)
        system = chat.call_args.args[0][0]["content"]
        self.assertEqual(chat.call_args.kwargs["cache_prefix"], verifier_v6.SYSTEM)
        self.assertTrue(system.startswith(verifier_v6.SYSTEM + "\n\nCurrent date and time:"))

    def test_agent_verifier_caches_the_instructions_before_the_clock(self):
        with patch.object(verifier_v6.llm, "chat", return_value='{"status":"ACCEPT","feedback":null}') as chat:
            verifier_v6.verify_agent([{"role": "user", "content": "Fix the bug"}], {"content": "Fixed."}, [],
                                     finish_reason="stop", review=True)
        prefix = chat.call_args.kwargs["cache_prefix"]
        self.assertIn("FINAL TEXT", prefix)
        self.assertTrue(chat.call_args.args[0][0]["content"].startswith(prefix + " Current date and time:"))

    def test_generation_puts_the_expert_prompt_before_the_clock_and_caches_it(self):
        with patch.object(main, "_profile", return_value=PROFILE), \
             patch.object(main.verifier_v6, "verify", return_value={"status": "ACCEPT", "feedback": None}), \
             patch.object(main.llm, "chat", return_value="done") as chat:
            main.run_request({"messages": [{"role": "user", "content": "Implement it"}], "mode": "direct",
                              "expert_id": "coding_expert"}, registry=LongPromptRegistry())
        prefix = chat.call_args.kwargs["cache_prefix"]
        self.assertEqual(prefix, main._BASE_SYSTEM_PROMPT + "\n\n" + "Write careful code. " * 40)
        system = chat.call_args.args[0][0]["content"]
        self.assertTrue(system.startswith(prefix + "\n\nCurrent date and time:"))
        self.assertFalse(chat.call_args.kwargs["cache_history"])

    def test_an_agent_step_keeps_its_conversation_and_dates_its_prompt_by_the_day(self):
        with patch.object(main, "_profile", return_value=PROFILE), \
             patch.object(main.verifier_v6, "verify", return_value={"status": "ACCEPT", "feedback": None}), \
             patch.object(main.llm, "chat", return_value="done") as chat:
            main.run_request({"messages": [{"role": "user", "content": "Fix the bug"}], "tools": TOOLS,
                              "mode": "direct", "expert_id": "coding_expert"}, registry=Registry())
        self.assertTrue(chat.call_args.kwargs["cache_history"])
        system = chat.call_args.args[0][0]["content"]
        self.assertIn("\n\nCurrent date: ", system)
        self.assertNotIn("Current date and time", system)

    def test_usage_reports_cached_prompt_tokens(self):
        usage = main._usage([{"input_tokens": 700, "output_tokens": 20, "cached_prompt_tokens": 610},
                             {"input_tokens": 300, "output_tokens": 50}])
        self.assertEqual(usage["prompt_tokens"], 1000)
        self.assertEqual(usage["prompt_tokens_details"], {"cached_tokens": 610})

    def test_common_prefix(self):
        self.assertEqual(llm._common_prefix([1, 2, 3, 9], [1, 2, 3, 4, 5]), 3)
        self.assertEqual(llm._common_prefix([], [1]), 0)


class HistoryStoreTests(unittest.TestCase):
    def test_a_step_replaces_the_earlier_steps_of_its_conversation(self):
        store = llm._HistoryStore(budget=100)
        store.keep("model", [1, 2], 10, lambda: "first step")
        store.keep("other model", [1, 2], 10, lambda: "other model")
        store.keep("model", [5, 6], 10, lambda: "another task")
        store.keep("model", [1, 2, 3], 10, lambda: "second step")
        self.assertEqual(store.find("model", [1, 2, 3, 4]), ([1, 2, 3], "second step"))
        self.assertIsNone(store.find("model", [1, 2]))
        self.assertEqual(store.find("other model", [1, 2, 9]), ([1, 2], "other model"))
        self.assertEqual(store.find("model", [5, 6]), ([5, 6], "another task"))
        self.assertIsNone(store.find("model", [7]))
        self.assertEqual(store.nbytes, 30)

    def test_the_budget_evicts_the_least_recently_used_and_skips_a_state_that_cannot_fit(self):
        store = llm._HistoryStore(budget=25)
        store.keep("model", [1], 10, lambda: "a")
        store.keep("model", [2], 10, lambda: "b")
        store.find("model", [1, 5])
        store.keep("model", [3], 10, lambda: "c")
        self.assertIsNone(store.find("model", [2]))
        self.assertEqual(store.find("model", [1]), ([1], "a"))
        snapshot = Mock()
        store.keep("model", [4], 30, snapshot)
        snapshot.assert_not_called()
        self.assertEqual(store.nbytes, 20)


SECOND_STEP = [{"role": "assistant", "content": None, "tool_calls": [
                   {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": {}}}]},
               {"role": "tool", "tool_call_id": "call_1", "content": "def mean(xs): return sum(xs) / len(xs)"}]


class AgentHistoryCases:
    """An agent's steps on one backend; ``model.computed`` counts the prompt tokens computed."""

    PREFIX = "Fixed instructions. " * 20
    SYSTEM = {"role": "system", "content": PREFIX + "Current date: today."}

    def step(self, messages):
        self.model.computed = 0
        with llm.capture_telemetry() as telemetry:
            llm.chat(messages, tools=TOOLS, cache_prefix=self.PREFIX, cache_history=True)
        return telemetry.calls[-1]

    @staticmethod
    def history(messages):
        return len(CharTokenizer().apply_chat_template(messages, tools=TOOLS))

    def test_a_step_computes_only_the_exchange_the_previous_step_did_not_see(self):
        first = [self.SYSTEM, {"role": "user", "content": "Fix the bug"}]
        self.assertEqual(self.step(first)["cached_prompt_tokens"], 0)
        self.assertEqual(self.model.computed, self.history(first) + len("<a>"))
        self.after_reply()
        call = self.step([*first, *SECOND_STEP])
        self.assertEqual(call["cached_prompt_tokens"], self.history(first))
        self.assertEqual(self.model.computed, self.history([*first, *SECOND_STEP]) - self.history(first) + len("<a>"))
        self.assertEqual(len(llm._HISTORY.entries), 1)

    def test_the_tool_definitions_belong_to_the_cached_prefix(self):
        self.step([self.SYSTEM, {"role": "user", "content": "First task"}])
        self.after_reply()
        call = self.step([self.SYSTEM, {"role": "user", "content": "Second task"}])
        self.assertEqual(call["cached_prompt_tokens"], len("<s>" + json.dumps(TOOLS) + self.PREFIX))

    def after_reply(self):
        """What generating the reply leaves behind."""


@unittest.skipUnless(llm.mx is not None, "the model state cache runs on MLX")
class AgentHistoryTests(AgentHistoryCases, unittest.TestCase):
    def setUp(self):
        from mlx_lm.models.cache import KVCache, LRUPromptCache

        self.model = CountingModel()
        loaded = llm.LoadedModel("thinker", llm.ModelSpec("fake-model", 1), self.model, CharTokenizer())

        def generate(model, tokenizer, prompt, max_tokens, sampler, prompt_cache=None, **kwargs):
            self.model.computed += len(prompt)
            yield SimpleNamespace(text="ok", token=1, finish_reason="stop", prompt_tokens=len(prompt),
                                  prompt_tps=1000.0, generation_tokens=1, generation_tps=10.0)

        for patcher in (patch.object(llm, "get_model", return_value=loaded),
                        patch.object(llm, "make_prompt_cache", side_effect=lambda model: [KVCache()]),
                        patch.object(llm, "stream_generate", side_effect=generate),
                        patch.object(llm, "_PROMPT_CACHE", LRUPromptCache(max_size=8)),
                        patch.object(llm, "_HISTORY", llm._HistoryStore(1 << 30))):
            patcher.start()
            self.addCleanup(patcher.stop)


class FakeLlama:
    """The parts of llama_cpp.Llama the GGUF path uses; a completion computes only the
    tokens after those already in the context, as llama-cpp-python does."""

    def __init__(self):
        self.n_tokens = 0
        self.input_ids = [0] * 4096
        self.computed = 0

    def tokenize(self, text, add_bos=False, special=True):
        return [ord(character) for character in text.decode("utf-8")]

    def n_ctx(self):
        return len(self.input_ids)

    def reset(self):
        self.n_tokens = 0

    def eval(self, tokens):
        self.input_ids[self.n_tokens:self.n_tokens + len(tokens)] = tokens
        self.n_tokens += len(tokens)
        self.computed += len(tokens)

    def create_completion(self, prompt, **kwargs):
        kept = llm._common_prefix(self.input_ids[:self.n_tokens], prompt[:-1])
        self.n_tokens = kept
        self.eval(prompt[kept:])
        yield {"choices": [{"text": "ok", "finish_reason": "stop"}]}


class GgufAgentHistoryTests(AgentHistoryCases, unittest.TestCase):
    def setUp(self):
        self.model = FakeLlama()
        loaded = llm.LoadedModel("thinker", llm.ModelSpec("fake-model", 1, backend="gguf"), self.model,
                                 CharTokenizer())

        def restore(llama, state, tokens):
            self.assertEqual(state, tuple(tokens))
            llama.input_ids[:len(tokens)] = tokens
            llama.n_tokens = len(tokens)

        for patcher in (patch.object(llm, "get_model", return_value=loaded),
                        patch.object(llm, "_gguf_state_size", return_value=10),
                        patch.object(llm, "_gguf_save_state",
                                     side_effect=lambda llama: tuple(llama.input_ids[:llama.n_tokens])),
                        patch.object(llm, "_gguf_restore_state", side_effect=restore),
                        patch.object(llm, "_GGUF_PREFIX_STATES", llm.OrderedDict()),
                        patch.object(llm, "_HISTORY", llm._HistoryStore(1 << 30))):
            patcher.start()
            self.addCleanup(patcher.stop)

    def after_reply(self):
        # The context holds the reply, which the next prompt renders differently.
        self.model.eval([ord(character) for character in "ok"])
