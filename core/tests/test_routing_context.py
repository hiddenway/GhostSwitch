import copy
import json
import unittest
from unittest.mock import patch
from core import main
from core.tests.test_v6 import PROFILE, Registry
from core.tests.test_tool_protocol import TOOLS
from core.routing_context import routing_request, routing_turn


class RoutingContextTests(unittest.TestCase):
    def test_large_tool_arguments_and_outputs_do_not_reach_router(self):
        messages = [
            {'role': 'user', 'content': 'Build a Python parser'},
            {'role': 'assistant', 'content': 'Writing the parser', 'tool_calls': [
                {'function': {'name': 'write', 'arguments': json.dumps({'content': 'SECRET_FILE_BODY' * 10000})}}]},
            {'role': 'tool', 'content': 'SECRET_OUTPUT' * 10000},
        ]
        original = copy.deepcopy(messages)
        turn, continuing = routing_turn(messages)
        self.assertEqual((turn, continuing), (messages[:1], True))
        for state in (routing_request(messages), routing_request(turn)):
            self.assertEqual(state, {'current_user_request': 'Build a Python parser', 'conversation_context': []})
        self.assertEqual(messages, original)

    def test_agent_progress_does_not_grow_the_routing_view(self):
        messages = [{'role': 'system', 'content': 'You are a coding agent.'},
                    {'role': 'user', 'content': 'создай сайт для смм агенства'}]
        views = []
        for step in range(30):
            turn, continuing = routing_turn(messages)
            views.append(json.dumps(routing_request(turn), ensure_ascii=False))
            self.assertEqual((turn, continuing), (messages[:2], step > 0))
            messages = messages + [
                {'role': 'assistant', 'content': f'Step {step}', 'tool_calls': [
                    {'function': {'name': 'write', 'arguments': json.dumps({'content': '<div>' * 2000})}}]},
                {'role': 'tool', 'content': 'Wrote file successfully. ' * 200},
            ]
        self.assertEqual(len(set(views)), 1)

    def test_bare_continue_routes_the_turn_it_continues(self):
        task = {'role': 'user', 'content': 'Создай одностраничный сайт'}
        for reply in ('продолжи', 'Продолжай.', 'давай дальше!', 'Continue, please', 'go on'):
            messages = [{'role': 'system', 'content': 'You are a coding agent.'}, task,
                        {'role': 'assistant', 'content': 'Создал index.html'},
                        {'role': 'user', 'content': reply},
                        {'role': 'user', 'content': 'продолжи'}]
            self.assertEqual(routing_turn(messages), (messages[:2], True), reply)

    def test_continue_with_a_new_instruction_is_a_new_turn(self):
        messages = [{'role': 'user', 'content': 'Создай сайт'},
                    {'role': 'assistant', 'content': 'Готово'},
                    {'role': 'user', 'content': 'продолжи, но перепиши на TypeScript'}]
        self.assertEqual(routing_turn(messages), (messages, False))
        first = [{'role': 'user', 'content': 'continue'}]
        self.assertEqual(routing_turn(first), (first, False))

    def test_references_retain_recent_goals_but_not_unbounded_history(self):
        messages = [{'role': 'user', 'content': 'old' * 1000}] * 100
        messages += [{'role': 'user', 'content': 'Build a website'},
                     {'role': 'assistant', 'content': 'The page is ready'},
                     {'role': 'user', 'content': 'Add a contact form'},
                     {'role': 'assistant', 'content': 'I will edit index.html'},
                     {'role': 'user', 'content': 'Continue'}]
        state = routing_request(messages)
        self.assertEqual(state['current_user_request'], 'Continue')
        self.assertEqual(state['conversation_context'], [
            {'role': 'user', 'content': 'Build a website'},
            {'role': 'user', 'content': 'Add a contact form'},
            {'role': 'assistant', 'content': 'I will edit index.html'},
        ])
        self.assertLess(len(json.dumps(state)), 1000)

    def test_new_topic_remains_current_despite_prior_specialist_task(self):
        state = routing_request([{'role': 'user', 'content': 'Calculate compound interest'},
                                 {'role': 'assistant', 'content': 'The balance is 110'},
                                 {'role': 'user', 'content': 'What is DNS?'}])
        self.assertEqual(state['current_user_request'], 'What is DNS?')
        self.assertEqual(state['conversation_context'][0]['content'], 'Calculate compound interest')

    def test_code_language_and_instructions_after_large_code_survive(self):
        text = 'Inspect this hardware:\n```verilog\nmodule demo;\n' + 'wire x;\n' * 10000 + 'endmodule\n```\nFix the reset logic.'
        state = routing_request([{'role': 'user', 'content': text}])
        result = state['current_user_request']
        self.assertIn('verilog', result)
        self.assertIn('endmodule', result)
        self.assertIn('Fix the reset logic.', result)
        self.assertLess(len(result), 1024)

    def test_instruction_defined_task_is_preserved(self):
        state = routing_request([{'role': 'system', 'content': 'Translate incoming text into Kannada.'},
                                 {'role': 'user', 'content': 'Good morning'}])
        self.assertIn('Kannada', state['caller_instructions'][0]['content'])

    def test_adversarial_size_is_bounded_even_with_many_actions(self):
        messages = [{'role': 'system', 'content': 'rules' * 10000}]
        messages += [{'role': role, 'content': 'content' * 10000} for role in ('user', 'assistant', 'user', 'assistant', 'user')]
        messages += [{'role': 'assistant', 'content': 'text' * 10000, 'tool_calls': [
            {'function': {'name': f'tool_{i}', 'arguments': 'content' * 10000}} for i in range(100)]},
                     {'role': 'tool', 'content': 'output' * 10000}]
        state = routing_request(messages)
        self.assertLess(len(json.dumps(state, ensure_ascii=False)), 2600)
        self.assertNotIn('output', json.dumps(state))
        self.assertNotIn('tool_', json.dumps(state))

    def test_generation_still_receives_entire_observation(self):
        observation = 'file contents\n' * 10000
        messages = [{'role': 'user', 'content': 'Inspect the parser'},
                    {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'read_1', 'type': 'function',
                     'function': {'name': 'read', 'arguments': '{"path":"a.py"}'}}]},
                    {'role': 'tool', 'tool_call_id': 'read_1', 'content': observation}]
        candidate = '<tool_call>{"name":"read","arguments":{"path":"b.py"}}</tool_call>'
        with patch.object(main, '_profile', return_value=PROFILE), \
             patch.object(main.router_v6, 'route', return_value={'route': 'expert', 'expert_id': 'coding_expert',
                       'confidence': None, 'missing_capability': None}), \
             patch.object(main.llm, 'chat', return_value=candidate) as chat:
            result = main.run_request({'messages': messages, 'tools': TOOLS}, registry=Registry())
        self.assertEqual(result['status'], 'success')
        self.assertEqual(chat.call_args.args[0][-1]['content'], observation)
        self.assertNotIn('file contents', json.dumps(routing_request(messages)))
