from __future__ import annotations

import json
import os
import stat
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from core import api, main, verifier_v6, web_search
from core.request_contract import parse_request
from core.tests.test_v6 import PROFILE, Registry

DUCKDUCKGO_PAGE = """
<div class="result results_links result--ad"><a class="result__a" href="https://ads.example/x">Ad</a></div>
<div class="result results_links web-result">
  <a rel="nofollow" class="result__a" href="https://qwen.ai/blog?id=qwen3.5">Qwen3.5: <b>Native</b> Agents</a>
  <a class="result__snippet" href="https://qwen.ai/blog?id=qwen3.5">We announce the <b>release</b> of Qwen3.5.</a>
</div>
<div class="result results_links web-result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fpage&amp;rut=1">Example</a>
  <a class="result__snippet">Second snippet.</a>
</div>
"""


def search_result() -> web_search.SearchResult:
    return web_search.SearchResult("qwen3.5 release", "duckduckgo", [
        web_search.Source(1, "Qwen blog", "https://qwen.ai/blog", "Released in February 2026.",
                          text="Qwen3.5 was released in February 2026."),
        web_search.Source(2, "Wikipedia", "https://en.wikipedia.org/wiki/Qwen", "Qwen is a model family."),
    ])


class ParsingTests(unittest.TestCase):
    def test_duckduckgo_results_skip_ads_and_unwrap_redirects(self):
        parser = web_search._DuckDuckGoParser()
        parser.feed(DUCKDUCKGO_PAGE)
        self.assertEqual([(r["title"], r["url"]) for r in parser.results], [
            ("Qwen3.5: Native Agents", "https://qwen.ai/blog?id=qwen3.5"),
            ("Example", "https://example.org/page"),
        ])
        self.assertEqual(parser.results[0]["snippet"], "We announce the release of Qwen3.5.")

    def test_page_text_drops_scripts_and_navigation(self):
        parser = web_search._TextExtractor()
        parser.feed("<html><title>Release notes</title><nav>Home About Blog</nav>"
                    "<script>steal()</script><p>Qwen3.5 was released with open weights in February.</p>"
                    "<footer>Copyright</footer></html>")
        self.assertEqual(parser.text(), "Qwen3.5 was released with open weights in February.")
        self.assertEqual(parser.title, "Release notes")

    def test_local_and_non_http_addresses_are_refused(self):
        for url in ("http://127.0.0.1:8000/health", "http://localhost/", "http://10.0.0.1/", "file:///etc/passwd"):
            with self.assertRaises(ValueError, msg=url):
                web_search.read_page(url, 100)

    def test_sources_are_numbered_and_marked_untrusted(self):
        prompt = web_search.sources_prompt(search_result())
        self.assertIn("untrusted data", prompt)
        self.assertIn("[1] Qwen blog\nURL: https://qwen.ai/blog\nQwen3.5 was released", prompt)
        self.assertIn("[2] Wikipedia", prompt)
        self.assertIn("returned no results", web_search.sources_prompt(web_search.SearchResult("x", "duckduckgo")))


class SettingsTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "search.json"
        environment = patch.dict(os.environ, {"GHOSTSWITCH_SEARCH_SETTINGS": str(self.path)})
        environment.start()
        self.addCleanup(environment.stop)

    def test_api_key_is_stored_privately_and_never_returned(self):
        with patch.dict(os.environ, {"BRAVE_API_KEY": ""}):
            self.assertEqual(web_search.public_settings()["provider"], "duckduckgo")
            saved = web_search.save_settings({"provider": "brave", "brave_api_key": "secret"})
        self.assertNotIn("secret", json.dumps(saved))
        self.assertTrue(saved["brave_api_key_set"])
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        with self.assertRaises(ValueError):
            web_search.save_settings({"provider": "google"})
        with self.assertRaises(ValueError):
            web_search.save_settings({"searxng_url": "javascript:alert(1)"})


class PipelineTests(unittest.TestCase):
    def setUp(self):
        profile = patch.object(main, "_profile", return_value=PROFILE)
        profile.start()
        self.addCleanup(profile.stop)

    def run_with(self, mode, planner, answer="Qwen3.5 вышла в феврале 2026 года [1]."):
        accept = {"status": "ACCEPT", "feedback": None}
        with patch.object(main.verifier_v6, "verify", return_value=accept) as verify, \
             patch.object(main.web_search, "search", return_value=search_result()) as search, \
             patch.object(main.llm, "chat", side_effect=[json.dumps(planner), answer]) as chat:
            result = main.run_request({"messages": [{"role": "user", "content": "Когда вышла Qwen3.5?"}],
                                       "mode": "direct", "expert_id": "coding_expert", "web_search": mode},
                                      registry=Registry())
        return result, verify, search, chat

    def test_search_results_reach_generation_verifier_and_response(self):
        result, verify, search, chat = self.run_with("on", {"search": True, "query": "Qwen3.5 release date"})
        search.assert_called_once_with("Qwen3.5 release date")
        system = chat.call_args_list[1].args[0][0]["content"]
        self.assertIn("WEB SEARCH RESULTS", system)
        self.assertIn("Qwen3.5 was released in February 2026.", system)
        self.assertEqual([source["index"] for source in verify.call_args.kwargs["sources"]], [1, 2])
        self.assertEqual(result["ghostswitch"]["web_search"]["sources"][0]["url"], "https://qwen.ai/blog")
        stage = next(s for s in result["ghostswitch"]["stages"] if s["phase"] == "web_search")
        self.assertEqual((stage["status"], stage["provider"]), ("completed", "duckduckgo"))

    def test_auto_mode_skips_search_when_not_needed(self):
        result, verify, search, chat = self.run_with("auto", {"search": False, "query": "hello"}, "Привет!")
        search.assert_not_called()
        self.assertIsNone(verify.call_args.kwargs["sources"])
        self.assertIsNone(result["ghostswitch"]["web_search"])
        stage = next(s for s in result["ghostswitch"]["stages"] if s["phase"] == "web_search")
        self.assertEqual(stage["status"], "not_needed")

    def test_auto_mode_always_searches_for_news(self):
        planner = {"search": False, "query": "последние новости"}
        with patch.object(main.verifier_v6, "verify", return_value={"status": "ACCEPT", "feedback": None}), \
             patch.object(main.web_search, "search", return_value=search_result()) as search, \
             patch.object(main.llm, "chat", side_effect=[json.dumps(planner, ensure_ascii=False), "Новости [1]."]):
            main.run_request({"messages": [{"role": "user", "content": "последние новости"}], "mode": "direct",
                              "expert_id": "coding_expert", "web_search": "auto"}, registry=Registry())
        search.assert_called_once_with("последние новости")
        for request in ("какая сейчас дата", "составь расписание тренировок", "напиши стих про осень"):
            self.assertIsNone(main._FRESH_FACTS.search(request), request)

    def test_search_is_off_by_default(self):
        with patch.object(main.verifier_v6, "verify", return_value={"status": "ACCEPT", "feedback": None}), \
             patch.object(main.web_search, "search") as search, \
             patch.object(main.llm, "chat", return_value="ok") as chat:
            main.run_request({"messages": [{"role": "user", "content": "Hi"}], "mode": "direct",
                              "expert_id": "coding_expert"}, registry=Registry())
        search.assert_not_called()
        self.assertEqual(chat.call_count, 1)
        with self.assertRaisesRegex(ValueError, "web_search"):
            parse_request({"messages": [{"role": "user", "content": "Hi"}], "web_search": "always"})


class CitationTests(unittest.TestCase):
    SOURCES = [{"index": 1, "title": "A", "url": "https://a.example", "text": "a"},
               {"index": 2, "title": "B", "url": "https://b.example", "text": "b"}]

    def test_invented_citation_is_rejected_without_a_model_call(self):
        with patch.object(verifier_v6.llm, "chat") as chat:
            result = verifier_v6.verify([{"role": "user", "content": "Q"}], "Fact [3].",
                                        finish_reason="stop", sources=self.SOURCES, review=True)
        self.assertEqual(result["status"], "RETRY_WITH_FEEDBACK")
        self.assertIn("[3]", result["feedback"])
        chat.assert_not_called()

    def test_code_indexes_are_not_citations_and_sources_reach_the_reviewer(self):
        with patch.object(verifier_v6.llm, "chat", return_value='{"status":"ACCEPT","feedback":null}') as chat:
            result = verifier_v6.verify([{"role": "user", "content": "Q"}], "Use items[5] as in [2].",
                                        finish_reason="stop", sources=self.SOURCES, review=True)
        self.assertEqual(result["status"], "ACCEPT")
        payload = json.loads(chat.call_args.args[0][1]["content"])
        self.assertEqual([s["index"] for s in payload["web_sources"]], [1, 2])

    def test_fact_disputes_on_sourced_answers_are_ignored(self):
        dispute = '{"status":"RETRY_WITH_FEEDBACK","issue":"result","feedback":"Уберите книгу весом 210 кг."}'
        with patch.object(verifier_v6.llm, "chat", return_value=dispute):
            sourced = verifier_v6.verify([{"role": "user", "content": "новости"}], "Украли книгу весом 210 кг [2].",
                                         finish_reason="stop", sources=self.SOURCES, review=True)
            unsourced = verifier_v6.verify([{"role": "user", "content": "новости"}], "Украли книгу весом 210 кг.",
                                           finish_reason="stop", review=True)
        self.assertEqual(sourced["override"], "sourced_answer_dispute")
        self.assertEqual(unsourced["status"], "RETRY_WITH_FEEDBACK")
        more = '{"status":"RETRY_WITH_FEEDBACK","issue":"completeness","feedback":"Добавьте даты на ноябрь."}'
        with patch.object(verifier_v6.llm, "chat", return_value=more):
            result = verifier_v6.verify([{"role": "user", "content": "даты разблокировок?"}],
                                        "В источниках есть только октябрь [1].", finish_reason="stop",
                                        sources=self.SOURCES, review=True)
        self.assertEqual(result["status"], "ACCEPT")


class SearchApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), api.Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def fetch(self, path, body=None, headers=None):
        request = Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                          headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urlopen(request) as response:
                return response.status, json.loads(response.read())
        except HTTPError as error:
            with error:
                return error.code, json.loads(error.read())

    def test_search_endpoint_returns_sources_with_text(self):
        with patch.object(api.web_search, "search", return_value=search_result()):
            status, body = self.fetch("/v1/search", {"query": "qwen3.5 release"})
        self.assertEqual(status, 200)
        self.assertEqual(body["sources"][0]["text"], "Qwen3.5 was released in February 2026.")
        status, _ = self.fetch("/v1/search", {"query": " "})
        self.assertEqual(status, 400)

    def test_settings_changes_must_come_from_the_app(self):
        status, _ = self.fetch("/v1/search/settings", {"provider": "brave"}, {"Origin": "https://evil.example"})
        self.assertEqual(status, 403)


if __name__ == "__main__":
    unittest.main()


class ClockTests(unittest.TestCase):
    def test_generation_knows_the_current_date_and_the_verifier_asks_no_model(self):
        with patch.object(main, "_profile", return_value=PROFILE), \
             patch.object(main.clock, "now_line", return_value="Current date and time: Tuesday, 06 October 2026."), \
             patch.object(main.llm, "chat", side_effect=["Сегодня 6 октября 2026 года."]) as chat:
            result = main.run_request({"messages": [{"role": "user", "content": "какая сейчас дата"}],
                                       "mode": "direct", "expert_id": "coding_expert"}, registry=Registry())
        self.assertEqual(result["status"], "success")
        (generation,) = chat.call_args_list
        self.assertIn("06 October 2026", generation.args[0][0]["content"])

    def test_reviewer_knows_the_current_date(self):
        with patch.object(verifier_v6.clock, "now_line", return_value="Current date and time: Tuesday, 06 October 2026."), \
             patch.object(verifier_v6.llm, "chat", return_value='{"status":"ACCEPT","feedback":null}') as chat:
            verifier_v6.verify([{"role": "user", "content": "какая сейчас дата"}], "Сегодня 6 октября 2026 года.",
                               finish_reason="stop", review=True)
        self.assertIn("06 October 2026", chat.call_args.args[0][0]["content"])
