from __future__ import annotations

import json
import threading
import tempfile
import socket
import time
from pathlib import Path
import unittest
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from core import api


class ClientApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from core.request_journal import RequestJournal
        cls.journal_dir = tempfile.TemporaryDirectory()
        cls.journal_patch = patch.object(api, 'JOURNAL', RequestJournal(Path(cls.journal_dir.name) / 'journal.sqlite3'))
        cls.journal_patch.start()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), api.Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)
        cls.journal_patch.stop()
        cls.journal_dir.cleanup()

    def fetch(self, path: str, *, body: dict | None = None):
        request = Request(self.base + path,
                          data=json.dumps(body).encode() if body is not None else None,
                          headers={"Content-Type": "application/json"} if body is not None else {})
        try:
            with urlopen(request) as response:
                return response.status, response.headers, response.read()
        except HTTPError as error:
            return error.code, error.headers, error.read()

    def test_client_assets_and_model_list(self):
        status, headers, body = self.fetch("/app/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["Content-Type"])
        self.assertIn(b"GhostSwitch", body)
        for path in ("/app/app.js", "/app/styles.css"):
            status, _, body = self.fetch(path)
            self.assertEqual(status, 200)
            self.assertTrue(body)
        status, _, body = self.fetch("/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["data"][0]["id"], "ghostswitch-v6")

    def test_experts_endpoint_uses_current_registry(self):
        profile = SimpleNamespace(name="Coding", description="Writes code", capabilities=("Python",))
        adapter = SimpleNamespace(repo_id="org/coding", revision="a" * 40)
        definition = SimpleNamespace(expert_id="coding_expert", adapter_id="coding-agentic",
                                     profile=profile, adapter=adapter)
        registry = SimpleNamespace(get=lambda expert_id: definition,
                                   check_backbone_compatibility=lambda expert_id: True)
        library = SimpleNamespace(status=lambda expert_id: {"enabled": True, "downloaded": True})
        packages = {"coding_expert": {"profile": {"default_enabled": True}}}
        with patch.object(api, "EXPERT_REGISTRY", registry), \
             patch.object(api, "load_packages", return_value=packages), \
             patch.object(api, "ADAPTERS", {"coding_expert": adapter}), \
             patch.object(api, "LIBRARY", library):
            status, _, body = self.fetch("/v1/experts")
        self.assertEqual(status, 200)
        expert = json.loads(body)["data"][0]
        self.assertEqual(expert["id"], "coding_expert")
        self.assertEqual(expert["source_url"], "https://huggingface.co/org/coding")
        self.assertTrue(expert["enabled"])

    def test_chat_completions_passes_request_to_runtime(self):
        response = {"status": "success", "choices": [{"message": {"role": "assistant", "content": "Hello"}}]}
        payload = {"model": "ghostswitch-v6", "messages": [{"role": "user", "content": "Hi"}]}
        with patch.object(api, "run_request", return_value=response) as run:
            status, _, body = self.fetch("/v1/chat/completions", body=payload)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "Hello")
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], payload)
        self.assertTrue(run.call_args.kwargs["completion_id"].startswith("chatcmpl-"))
        self.assertTrue(callable(run.call_args.kwargs["on_event"]))

    def test_chat_completions_streams_openai_chunks_and_final_metadata(self):
        def stream_request(payload, *, on_event, completion_id, is_cancelled, parsed=None):
            on_event({"type": "status", "status": "Генерация ответа", "attempt": 1})
            on_event({"type": "token", "delta": "Привет", "attempt": 1})
            return {
                "id": completion_id, "object": "chat.completion", "created": 1,
                "model": "ghostswitch-v6", "status": "success",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Привет"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                "ghostswitch": {"verification": {"status": "ACCEPT"}},
            }

        payload = {"model": "ghostswitch-v6", "messages": [{"role": "user", "content": "Привет"}], "stream": True}
        with patch.object(api, "run_request", side_effect=stream_request):
            status, headers, body = self.fetch("/v1/chat/completions", body=payload)
        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", headers["Content-Type"])
        events = [line[6:] for line in body.decode().splitlines() if line.startswith("data: ")]
        chunks = [json.loads(event) for event in events if event != "[DONE]"]
        self.assertIn({"content": "Привет"}, [chunk["choices"][0]["delta"] for chunk in chunks])
        final = next(chunk for chunk in chunks if chunk.get("status") == "success")
        self.assertEqual(final["usage"]["completion_tokens"], 1)
        self.assertEqual(final["ghostswitch"]["verification"]["status"], "ACCEPT")
        self.assertEqual(events[-1], "[DONE]")

    def test_reasoning_streams_as_reasoning_content_delta(self):
        def stream_request(payload, *, on_event, completion_id, is_cancelled, parsed=None):
            on_event({"type": "reasoning", "delta": "17 * 23 = 391", "attempt": 1})
            on_event({"type": "token", "delta": "403", "attempt": 1})
            return {"status": "success", "choices": [{"message": {
                "role": "assistant", "content": "403", "reasoning_content": "17 * 23 = 391"},
                "finish_reason": "stop"}]}

        payload = {"messages": [{"role": "user", "content": "17 * 23"}], "stream": True}
        with patch.object(api, "run_request", side_effect=stream_request):
            _, _, body = self.fetch("/v1/chat/completions", body=payload)
        deltas = [json.loads(line[6:])["choices"][0]["delta"]
                  for line in body.decode().splitlines() if line.startswith("data: {")]
        self.assertLess(deltas.index({"reasoning_content": "17 * 23 = 391"}), deltas.index({"content": "403"}))

    def test_accepted_action_streams_standard_tool_delta(self):
        from core.tool_protocol import parse_candidate
        from core.tests.test_tool_protocol import TOOLS, USER
        message = parse_candidate('<tool_call>{"name":"read","arguments":{"path":"a.py"}}</tool_call>', TOOLS, 'auto')
        def run(payload, *, on_event, completion_id, is_cancelled, parsed=None):
            on_event({"type": "accepted_message", "message": message})
            return {"status": "success", "choices": [{"message": message, "finish_reason": "tool_calls"}]}
        with patch.object(api, 'run_request', side_effect=run):
            status, _, body = self.fetch('/v1/chat/completions', body={"messages": [USER], "tools": TOOLS, "stream": True})
        chunks = [json.loads(line[6:]) for line in body.decode().splitlines() if line.startswith('data: {')]
        call = next(chunk['choices'][0]['delta']['tool_calls'][0] for chunk in chunks if 'tool_calls' in chunk['choices'][0]['delta'])
        self.assertEqual(call['index'], 0)
        self.assertEqual(call['function']['name'], 'read')
        self.assertEqual(chunks[-1]['choices'][0]['finish_reason'], 'tool_calls')
        self.assertNotIn('error', chunks[-1])

    def test_agent_progress_is_metadata_without_provisional_arguments(self):
        from core.tests.test_tool_protocol import TOOLS, USER
        def run(payload, *, on_event, completion_id, is_cancelled, parsed=None):
            on_event({"type": "status", "status": "Генерация ответа", "attempt": 1})
            on_event({"type": "progress", "attempt": 1, "generated_tokens": 64, "max_tokens": 4096})
            stage = {"phase": "verification", "status": "FAIL", "reason": "tool_call_truncated", "attempt": 1}
            on_event({"type": "stage", "stage": stage})
            live = next(e for e in api.JOURNAL.list() if e['id'] == completion_id)
            self.assertEqual(live['response']['ghostswitch']['stages'], [stage])
            self.assertEqual(live['response']['ghostswitch']['progress']['generated_tokens'], 64)
            return {"status": "failed", "choices": [], "error": {"type": "verification_failed"}}
        with patch.object(api, 'run_request', side_effect=run):
            _, _, body = self.fetch('/v1/chat/completions', body={"messages": [USER], "tools": TOOLS, "stream": True})
        chunks = [json.loads(line[6:]) for line in body.decode().splitlines() if line.startswith('data: {')]
        progress = next(c for c in chunks if 'ghostswitch_progress' in c)
        self.assertEqual(progress['ghostswitch_progress']['generated_tokens'], 64)
        self.assertEqual(progress['choices'][0]['delta'], {})
        self.assertFalse(any('tool_calls' in c['choices'][0]['delta'] for c in chunks))

    def test_closed_stream_is_detected_without_waiting_for_write_failure(self):
        completed = threading.Event()
        errors = []
        def run(payload, *, on_event, completion_id, is_cancelled, parsed=None):
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and not is_cancelled():
                time.sleep(0.01)
            if not is_cancelled():
                errors.append('Socket closure was not detected')
            completed.set()
            return {"status": "canceled", "choices": [], "error": {"type": "client_disconnected"}}
        payload = json.dumps({"messages": [{"role": "user", "content": "Hi"}], "stream": True}).encode()
        with patch.object(api, 'run_request', side_effect=run):
            connection = socket.create_connection(self.server.server_address, timeout=3)
            connection.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: '
                               + str(len(payload)).encode() + b'\r\n\r\n' + payload)
            connection.recv(4096)
            connection.close()
            self.assertTrue(completed.wait(3))
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
