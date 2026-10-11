import json
import unittest
from unittest.mock import patch

from core import main, router_v6, verifier_v6
from core.request_contract import parse_request
from core.tool_protocol import parse_candidate
from core.tests.test_v6 import PROFILE, Registry

TOOLS = [{"type": "function", "function": {"name": "read", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"],
    "additionalProperties": False}}}]
USER = {"role": "user", "content": "Inspect this file"}

class ToolProtocolTests(unittest.TestCase):
    def test_valid_calls_and_history_round_trip(self):
        message = parse_candidate('<tool_call>{"name":"read","arguments":{"path":"a.py"}}</tool_call>', TOOLS, "auto")
        call_id = message["tool_calls"][0]["id"]
        history = [USER, message, {"role": "tool", "tool_call_id": call_id, "content": "return 1"}]
        parsed = parse_request({"messages": history, "tools": TOOLS})
        self.assertEqual(parsed["messages"], history)
        self.assertEqual(router_v6._routing_request(history),
                         {"current_user_request": "Inspect this file", "conversation_context": []})

    def test_native_qwen_xml_preserves_multiline_strings(self):
        text = '<tool_call>\n<function=read>\n<parameter=path>\na.py\n</parameter>\n</function>\n</tool_call>'
        result = parse_candidate(text, TOOLS, 'auto')
        self.assertEqual(json.loads(result['tool_calls'][0]['function']['arguments']), {'path': 'a.py'})

    def test_caller_systems_merge_and_tool_arguments_render_as_objects(self):
        message = parse_candidate('<tool_call>{"name":"read","arguments":{"path":"a.py"}}</tool_call>', TOOLS, 'auto')
        history = [{"role": "system", "content": "Caller instruction"}, USER, message,
                   {"role": "tool", "tool_call_id": message['tool_calls'][0]['id'], "content": "observed"}]
        with patch.object(main, '_profile', return_value=PROFILE), \
             patch.object(main.llm, 'chat', return_value='Done') as chat, \
             patch.object(main.verifier_v6, 'verify_agent', return_value={"status": "ACCEPT", "feedback": None}):
            main.run_request({"messages": history, "tools": TOOLS, "mode": "direct", "expert_id": "coding_expert"}, registry=Registry())
        rendered = chat.call_args.args[0]
        self.assertEqual(sum(m['role'] == 'system' for m in rendered), 1)
        self.assertIn('Caller instruction', rendered[0]['content'])
        self.assertEqual(rendered[2]['tool_calls'][0]['function']['arguments'], {'path': 'a.py'})
        self.assertIsInstance(message['tool_calls'][0]['function']['arguments'], str)

    def test_action_verifier_reviews_only_next_step(self):
        message = parse_candidate('<tool_call>{"name":"read","arguments":{"path":"a.py"}}</tool_call>', TOOLS, 'auto')
        with patch.object(verifier_v6.llm, 'chat', return_value='{"status":"ACCEPT","feedback":null}') as chat:
            result = verifier_v6.verify_agent([USER], message, TOOLS, finish_reason='stop', review=True)
        self.assertEqual(result['status'], 'ACCEPT')
        prompt = chat.call_args.args[0]
        data = json.loads(prompt[1]['content'])
        self.assertEqual(data['proposed_next_action'], message['tool_calls'])
        self.assertNotIn('final_response', data)

    def test_reject_invalid_calls(self):
        for text, choice in [
            ('<tool_call>{"name":"read","arguments":{}}</tool_call>', "auto"),
            ('<tool_call>{"name":"unknown","arguments":{}}</tool_call>', "auto"),
            ('<tool_call>{"name":"read","arguments":{"path":2}}</tool_call>', "auto"),
            ('<tool_call>{"name":"read","arguments":{"path":"x"}}', "auto"),
            ('<tool_call>{"name":"read","arguments":{"path":"x"}}</tool_call>', "none"),
            ('Done', "required"),
        ]:
            with self.subTest(text=text, choice=choice), self.assertRaises(ValueError):
                parse_candidate(text, TOOLS, choice)

    def test_reject_orphan_results_and_pending_calls(self):
        with self.assertRaises(ValueError):
            parse_request({"messages": [USER, {"role": "tool", "content": "ok", "tool_call_id": "missing"}]})
        call = parse_candidate('<tool_call>{"name":"read","arguments":{"path":"x"}}</tool_call>', TOOLS, "auto")
        with self.assertRaises(ValueError):
            parse_request({"messages": [USER, call]})

    def test_tool_action_is_handed_to_harness_without_verifier_retry(self):
        events = []
        output = '<tool_call>{"name":"read","arguments":{"path":"bad"}}</tool_call>'
        with patch.object(main, '_profile', return_value=PROFILE), \
             patch.object(main.llm, 'chat', return_value=output), \
             patch.object(main.verifier_v6, 'verify_agent') as verify_agent:
            result = main.run_request({"messages": [USER], "mode": "direct", "expert_id": "coding_expert", "tools": TOOLS}, registry=Registry(), on_event=events.append)
        accepted = [event for event in events if event["type"] == "accepted_message"]
        self.assertEqual(len(accepted), 1)
        self.assertIn('bad', accepted[0]["message"]["tool_calls"][0]["function"]["arguments"])
        self.assertFalse(any(event["type"] in ('token', 'reset') for event in events))
        self.assertEqual(result["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(result["ghostswitch"]["verification"]["status"], "SKIPPED")
        self.assertEqual(result["ghostswitch"]["verification"]["attempts"], 1)
        verify_agent.assert_not_called()

    def test_repeated_completed_tool_action_gets_one_corrective_retry(self):
        previous = parse_candidate(
            '<tool_call>{"name":"read","arguments":{"path":"a.py"}}</tool_call>',
            TOOLS, "auto",
        )
        history = [USER, previous, {
            "role": "tool", "tool_call_id": previous["tool_calls"][0]["id"],
            "content": "File contents: def f(): return 1",
        }]
        repeated = '<tool_call>{"name":"read","arguments":{"path":"a.py"}}</tool_call>'
        next_action = '<tool_call>{"name":"read","arguments":{"path":"b.py"}}</tool_call>'
        with patch.object(main, "_profile", return_value=PROFILE), \
             patch.object(main.llm, "chat", side_effect=[repeated, next_action]) as chat, \
             patch.object(main.verifier_v6, "verify_agent") as verify_agent:
            result = main.run_request({"messages": history, "tools": TOOLS,
                                       "mode": "direct", "expert_id": "coding_expert"},
                                      registry=Registry())
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["ghostswitch"]["verification"]["attempts"], 2)
        self.assertEqual(result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
                         '{"path": "b.py"}')
        retry_system = chat.call_args_list[1].args[0][0]["content"]
        self.assertIn("already has a result", retry_system)
        verify_agent.assert_not_called()

    def test_actions_repeated_in_turn_without_new_results_are_a_loop(self):
        def proposal(index, name, arguments):
            return {"role": "assistant", "content": None, "tool_calls": [
                {"id": f"call_{index}", "type": "function",
                 "function": {"name": name, "arguments": json.dumps(arguments)}}]}

        def step(index, name, arguments, result):
            return [proposal(index, name, arguments), {"role": "tool", "tool_call_id": f"call_{index}", "content": result}]

        read = ("read", {"path": "solution.py"}, "def roman_to_int(numeral): ...")
        search = ("grep", {"pattern": "^import"}, "Found 2 matches")
        circling = [USER, *step(1, *read), *step(2, *search), *step(3, *read)]
        self.assertIn("nothing new", main._repeated_completed_tool_action(circling, proposal(9, *search[:2])))
        # A file read again after an edit, or tests run again after another change, may show something new.
        edited = [USER, *step(1, *read), *step(2, "edit", {"path": "solution.py", "new": "a"}, "Edited")]
        self.assertIsNone(main._repeated_completed_tool_action(edited, proposal(9, *read[:2])))
        tests = ("bash", {"command": "python3 -m unittest -q"}, "FAILED (failures=1)")
        retried = [USER, *step(1, *tests), *step(2, "edit", {"new": "a"}, "Edited"),
                   *step(3, *tests), *step(4, "edit", {"new": "b"}, "Edited")]
        self.assertIsNone(main._repeated_completed_tool_action(retried, proposal(9, *tests[:2])))

    def test_a_tool_call_without_closed_reasoning_is_the_action(self):
        call = '<tool_call>{"name":"read","arguments":{"path":"a.py"}}</tool_call>'

        def run(output, finish_reason):
            def chat(messages, **kwargs):
                kwargs["debug"]["finish_reason"] = finish_reason if kwargs["enable_thinking"] else "stop"
                return output if kwargs["enable_thinking"] else call

            with patch.object(main, "_profile", return_value=PROFILE), \
                 patch.object(main.llm, "chat", side_effect=chat) as chat_mock:
                result = main.run_request({"messages": [USER], "tools": TOOLS, "reasoning_effort": "high",
                                           "mode": "direct", "expert_id": "coding_expert"}, registry=Registry())
            return result, chat_mock.call_count

        for output, reasoning in ((call, None), ("I need the file first.\n" + call, "I need the file first.")):
            with self.subTest(output=output):
                result, calls = run(output, "stop")
                message = result["choices"][0]["message"]
                self.assertEqual((result["status"], calls), ("success", 1))
                self.assertEqual(json.loads(message["tool_calls"][0]["function"]["arguments"]), {"path": "a.py"})
                self.assertEqual(message.get("reasoning_content"), reasoning)
                self.assertEqual(result["ghostswitch"]["thinking"]["used"], reasoning is not None)
        # The final message of a task, written the same way, is the answer.
        result, calls = run("Done: the tests pass.", "stop")
        self.assertEqual((result["status"], calls), ("success", 1))
        self.assertEqual(result["choices"][0]["message"]["content"], "Done: the tests pass.")
        self.assertNotIn("reasoning_content", result["choices"][0]["message"])
        # Cut off at the output limit, the reasoning did overflow: the step is generated again without it.
        result, calls = run("Long reasoning " + call, "length")
        self.assertEqual((result["status"], calls), ("success", 2))
        self.assertEqual(result["ghostswitch"]["stages"][2]["status"], "thinking_overflow")

    def test_persistent_duplicate_tool_action_stops_after_one_retry(self):
        previous = parse_candidate(
            '<tool_call>{"name":"read","arguments":{"path":"a.py"}}</tool_call>',
            TOOLS, "auto",
        )
        history = [USER, previous, {
            "role": "tool", "tool_call_id": previous["tool_calls"][0]["id"], "content": "Already read",
        }]
        repeated = '<tool_call>{"name":"read","arguments":{"path":"a.py"}}</tool_call>'
        with patch.object(main, "_profile", return_value=PROFILE), \
             patch.object(main.llm, "chat", return_value=repeated) as chat:
            result = main.run_request({"messages": history, "tools": TOOLS,
                                       "mode": "direct", "expert_id": "coding_expert"},
                                      registry=Registry())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["ghostswitch"]["verification"]["status"], "FAIL")
        self.assertEqual(chat.call_count, 2)

class AgentRecoveryTests(unittest.TestCase):
    def run_agent(self, chat, **kwargs):
        with patch.object(main, '_profile', return_value={**PROFILE, 'max_retries': 4}), \
             patch.object(main.llm, 'chat', side_effect=chat), \
             patch.object(main.verifier_v6, 'verify_agent') as verifier:
            result = main.run_request({'messages': [USER], 'tools': TOOLS,
                                       'mode': 'direct', 'expert_id': 'coding_expert',
                                       'max_tokens': 4096}, registry=Registry(), **kwargs)
        verifier.assert_not_called()
        return result

    def test_truncated_action_repair_is_smaller_and_continues_history(self):
        calls, events = [], []
        def chat(messages, **kwargs):
            calls.append((messages, kwargs))
            kwargs['debug']['finish_reason'] = 'length' if len(calls) == 1 else 'stop'
            return ('<tool_call><function=read><parameter=path>oversized' if len(calls) == 1 else
                    '<tool_call>{"name":"read","arguments":{"path":"a.py"}}</tool_call>')
        result = self.run_agent(chat, on_event=events.append)
        self.assertEqual(result['status'], 'success')
        self.assertEqual([call[1]['max_tokens'] for call in calls], [4096, 1024])
        correction = calls[1][0][0]['content']
        self.assertIn('nothing was executed', correction)
        self.assertIn("Do not restart the user's entire task", correction)
        self.assertNotIn("Answer the user's original request again", correction)
        self.assertEqual(calls[1][0][-1], USER)
        stages = [e['stage'] for e in events if e['type'] == 'stage']
        self.assertEqual(stages, result['ghostswitch']['stages'])
        self.assertTrue(any(s.get('reason') == 'tool_call_truncated' for s in stages))
        self.assertEqual(len([e for e in events if e['type'] == 'accepted_message']), 1)

    def test_persistent_truncation_never_executes_partial_action(self):
        calls, events = [], []
        def chat(messages, **kwargs):
            calls.append(kwargs)
            kwargs['debug']['finish_reason'] = 'length'
            return '<tool_call>{"name":"read","arguments":{"path":"x"}}'
        result = self.run_agent(chat, on_event=events.append)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['choices'], [])
        self.assertEqual(result['ghostswitch']['verification']['status'], 'FAIL')
        self.assertFalse(any(e['type'] == 'accepted_message' for e in events))

    def test_invalid_schema_also_has_only_one_protocol_repair(self):
        calls = []
        def chat(messages, **kwargs):
            calls.append(kwargs)
            return '<tool_call>{"name":"read","arguments":{"path":5}}</tool_call>'
        result = self.run_agent(chat)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['ghostswitch']['stages'][-1]['reason'], 'invalid_tool_call')

    def test_disconnect_cancels_without_retrying_or_verifying(self):
        calls = []
        disconnected = False
        def chat(messages, **kwargs):
            nonlocal disconnected
            calls.append(kwargs)
            disconnected = True
            main.llm.check_cancelled()
        result = self.run_agent(chat, is_cancelled=lambda: disconnected)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result['status'], 'canceled')
        self.assertEqual(result['error']['type'], 'client_disconnected')
        self.assertEqual(result['choices'], [])
        # Cancellation context must not leak to the next independent request.
        main.llm.check_cancelled()

    def test_routing_cancellation_preserves_canceled_status(self):
        with patch.object(main, '_profile', return_value={**PROFILE, 'max_retries': 4}), \
             patch.object(router_v6.thinker, 'route_hierarchical', side_effect=main.llm.GenerationCancelled('Disconnected')), \
             patch.object(main.llm, 'chat') as chat:
            result = main.run_request({'messages': [USER], 'tools': TOOLS}, registry=Registry())
        self.assertEqual(result['status'], 'canceled')
        self.assertEqual(result['error']['type'], 'client_disconnected')
        chat.assert_not_called()

    def test_already_disconnected_request_never_loads_model(self):
        def chat(*args, **kwargs):
            self.fail('Disconnected requests must not invoke a model')
        result = self.run_agent(chat, is_cancelled=lambda: True)
        self.assertEqual(result['status'], 'canceled')
        self.assertEqual(result['ghostswitch']['verification']['attempts'], 0)


if __name__ == "__main__":
    unittest.main()
