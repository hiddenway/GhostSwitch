from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import core_models
from core import api, main, runtime_selection
from core.expert_packages import blocked, load_packages, package_router_config

ROOT = Path(__file__).resolve().parents[2]


class BackboneRegistryTests(unittest.TestCase):
    def test_each_backbone_has_packages_profiles_and_a_composed_tree(self):
        for backbone in core_models.backbone_ids():
            for backend in ("mlx", "gguf"):
                profile = ROOT / "core/config/execution_profiles" / f"{core_models.profile_name(backbone, backend)}.json"
                self.assertEqual(json.loads(profile.read_text())["backbone_id"], backbone)
            raw = json.loads((ROOT / "core/config/router_tree.json").read_text())
            tree = package_router_config(raw, load_packages(backbone))
            leaves = {node["expert_id"] for node in tree["nodes"] if node.get("expert_id")}
            # Every package except those an evaluation blocked is reachable.
            self.assertEqual(leaves, {expert for expert, package in load_packages(backbone).items()
                                      if not blocked(package)})
            self.assertTrue(all(node.get("children") or node.get("expert_id") or node["id"] == "root"
                                for node in tree["nodes"]))

    def test_qwythos_is_an_uncensored_backbone_with_its_own_lineage(self):
        self.assertTrue(core_models.uncensored("qwythos-9b"))
        self.assertFalse(core_models.uncensored("qwen35-9b"))
        self.assertEqual(core_models.expert_library("qwythos-9b"), "qwen35-9b")
        self.assertEqual(core_models.expert_library("qwen35-9b"), "qwen35-9b")
        # A Qwen3.5-9B fine-tune: every curated 9B expert loads on it, and Qwythos LoRAs are accepted too.
        self.assertEqual(set(load_packages("qwythos-9b")), set(load_packages("qwen35-9b")))
        self.assertTrue(core_models.adapter_base_names("qwen35-9b") < core_models.adapter_base_names("qwythos-9b"))
        self.assertIn("empero-ai/Qwythos-9B-v2", core_models.adapter_base_names("qwythos-9b"))
        self.assertEqual(core_models.download_source("qwythos-9b", "gguf")["filename"], "Qwythos-9B-v2-Q4_K_M.gguf")

    def test_receipts_belong_to_the_4b_library_only(self):
        self.assertIn("receipt_extraction_expert", load_packages("qwen35-4b"))
        self.assertNotIn("receipt_extraction_expert", load_packages("qwen35-9b"))
        self.assertTrue(load_packages("qwen35-4b")["receipt_extraction_expert"]["profile"]["default_enabled"])

    def test_active_backbone_comes_from_the_environment(self):
        with patch.dict(os.environ, {core_models.ACTIVE_BACKBONE_ENV: "qwen35-4b"}):
            self.assertEqual(core_models.active_backbone(), "qwen35-4b")
        with patch.dict(os.environ, {core_models.ACTIVE_BACKBONE_ENV: "qwen99"}):
            with self.assertRaises(ValueError):
                core_models.active_backbone()

    def test_gguf_file_is_found_in_lm_studio_models(self):
        with tempfile.TemporaryDirectory() as home:
            model = Path(home) / ".lmstudio/models/lmstudio-community/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"GGUF")
            with patch.object(Path, "home", return_value=Path(home)), \
                 patch.dict(os.environ, {"HF_HUB_CACHE": str(Path(home) / "hub")}):
                self.assertEqual(core_models.find_gguf("qwen35-4b"), model)
                self.assertIsNone(core_models.find_gguf("qwen35-9b"))


class MemoryBudgetTests(unittest.TestCase):
    def test_caches_are_bounded_by_default_and_configurable(self):
        from core import llm
        mib = 1024 * 1024
        self.assertEqual(llm.MLX_CACHE_LIMIT_BYTES, 1024 * mib)
        self.assertEqual(llm.PROMPT_CACHE_BYTES, 256 * mib)
        self.assertEqual(llm._ROUTER_CACHE_BYTES, 512 * mib)
        with patch.dict(os.environ, {"GHOSTSWITCH_MLX_CACHE_MB": "2048", "GHOSTSWITCH_PROMPT_CACHE_MB": "zero"}):
            self.assertEqual(llm._env_bytes("GHOSTSWITCH_MLX_CACHE_MB", 1024), 2048 * mib)
            self.assertEqual(llm._env_bytes("GHOSTSWITCH_PROMPT_CACHE_MB", 256), 256 * mib)

    def test_each_request_returns_freed_memory(self):
        from core.tests.test_v6 import PROFILE, Registry
        with patch.object(main, "_profile", return_value=PROFILE), \
             patch.object(main.verifier_v6, "verify", return_value={"status": "ACCEPT", "feedback": None}), \
             patch.object(main.llm, "chat", return_value="ok"), \
             patch.object(main.llm, "release_memory") as release:
            main.run_request({"messages": [{"role": "user", "content": "Hi"}], "mode": "direct",
                              "expert_id": "coding_expert"}, registry=Registry())
        release.assert_called_once()

    def test_gguf_context_is_bounded_and_overridable(self):
        for backbone in core_models.backbone_ids():
            self.assertEqual(core_models.gguf_context(backbone), 32768)
        with patch.dict(os.environ, {"GHOSTSWITCH_GGUF_CONTEXT": "8192"}):
            self.assertEqual(core_models.gguf_context("qwen35-9b"), 8192)

    def test_memory_estimate_adds_context_and_caches_to_weights(self):
        gib = 1024 ** 3
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "model.gguf"
            with model.open("wb") as handle:
                handle.truncate(2 * gib)
            # 2 GiB weights + 32768 tokens x 32 KiB KV + 0.5 GiB buffers.
            self.assertEqual(runtime_selection.memory_estimate_gb("qwen35-4b", "gguf", str(model)), 3.5)
            weights = Path(folder) / "model.safetensors"
            with weights.open("wb") as handle:
                handle.truncate(3 * gib)
            with patch.object(runtime_selection, "local_model_dir", return_value=Path(folder)):
                # 3 GiB weights + 1 GiB MLX cache + 0.25 prompt + 0.5 router + 0.5 working KV.
                self.assertEqual(runtime_selection.memory_estimate_gb("qwen35-4b", "mlx"), 5.2)
        # A file that is not downloaded yet is estimated from its published size.
        self.assertEqual(runtime_selection.memory_estimate_gb("qwen35-4b", "gguf", "/missing.gguf"), 4.0)
        self.assertGreater(runtime_selection.system_memory()["total_gb"], 0)


class RuntimeSelectionTests(unittest.TestCase):
    def setUp(self):
        self.state = tempfile.TemporaryDirectory()
        self.addCleanup(self.state.cleanup)
        environment = patch.dict(os.environ, {"GHOSTSWITCH_RUNTIME_STATE": str(Path(self.state.name) / "runtime.json")})
        environment.start()
        self.addCleanup(environment.stop)

    def test_command_line_wins_over_saved_choice_which_wins_over_default(self):
        self.assertEqual(runtime_selection.resolve(), runtime_selection.Selection("qwen35-9b", "mlx"))
        runtime_selection.save(runtime_selection.Selection("qwen35-4b", "mlx"))
        self.assertEqual(runtime_selection.resolve().backbone, "qwen35-4b")
        self.assertEqual(runtime_selection.resolve("qwen35-9b").backbone, "qwen35-9b")

    def test_gguf_needs_a_file(self):
        with patch.object(runtime_selection, "find_gguf", return_value=None):
            with self.assertRaisesRegex(ValueError, "No GGUF file"):
                runtime_selection.resolve("qwen35-4b", "gguf")
        selection = runtime_selection.resolve("qwen35-4b", "gguf", "/models/q.gguf")
        self.assertEqual(selection.model_path, "/models/q.gguf")

    def test_restart_command_keeps_server_options_and_replaces_the_runtime(self):
        argv = ["--backend", "mlx", "--serve", "--port", "8765", "--backbone=qwen35-9b", "--model-path", "/old"]
        selection = runtime_selection.Selection("qwen35-4b", "gguf", "/m/q.gguf")
        self.assertEqual(runtime_selection.command(argv, selection),
                         ["--serve", "--port", "8765", "--backbone", "qwen35-4b", "--backend", "gguf",
                          "--model-path", "/m/q.gguf"])

    def test_main_restarts_when_another_backbone_is_requested(self):
        with patch.object(runtime_selection, "restart", side_effect=SystemExit("restarted")) as restart:
            with self.assertRaises(SystemExit):
                main.main(["--backbone", "qwen35-4b", "--serve"])
        argv, selection = restart.call_args.args
        self.assertEqual(argv, ["--backbone", "qwen35-4b", "--serve"])
        self.assertEqual(selection.backbone, "qwen35-4b")


class RuntimeApiTests(unittest.TestCase):
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

    def setUp(self):
        self.state = tempfile.TemporaryDirectory()
        self.addCleanup(self.state.cleanup)
        environment = patch.dict(os.environ, {"GHOSTSWITCH_RUNTIME_STATE": str(Path(self.state.name) / "runtime.json")})
        environment.start()
        self.addCleanup(environment.stop)

    def fetch(self, path, body=None, headers=None):
        request = Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                          headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urlopen(request) as response:
                return response.status, json.loads(response.read())
        except HTTPError as error:
            with error:
                return error.code, json.loads(error.read())

    def test_runtime_lists_every_backbone_and_backend(self):
        status, body = self.fetch("/v1/runtime")
        self.assertEqual(status, 200)
        self.assertEqual(body["current"]["backbone"], "qwen35-9b")
        self.assertEqual([row["id"] for row in body["options"]], list(core_models.backbone_ids()))
        self.assertEqual(set(body["options"][0]["backends"]), {"mlx", "gguf"})
        self.assertIn("memory_gb", body["options"][0]["backends"]["mlx"])
        self.assertGreater(body["memory"]["total_gb"], 0)
        self.assertEqual(body["memory"]["releasable_gb"], 0)
        status, health = self.fetch("/health")
        self.assertEqual(health["backbone_name"], "Qwen3.5-9B")

    def test_switch_validates_and_restarts(self):
        status, body = self.fetch("/v1/runtime", {"backbone": "qwen99", "backend": "mlx"})
        self.assertEqual(status, 400)
        status, body = self.fetch("/v1/runtime", {"backbone": "qwen35-4b", "backend": "mlx"},
                                  {"Origin": "https://example.com"})
        self.assertEqual(status, 403)
        with patch.object(api, "get_backbone"), patch.object(api, "_restart") as restart:
            status, body = self.fetch("/v1/runtime", {"backbone": "qwen35-9b", "backend": "mlx"})
            self.assertEqual((status, body["status"]), (200, "current"))
            status, body = self.fetch("/v1/runtime", {"backbone": "qwen35-4b", "backend": "mlx"})
            self.assertEqual((status, body["status"], body["name"]), (202, "restarting", "Qwen3.5-4B"))
            for _ in range(50):
                if restart.called:
                    break
                threading.Event().wait(0.02)
        restart.assert_called_once()
        self.assertEqual(restart.call_args.args[0].backbone, "qwen35-4b")
        self.assertEqual(runtime_selection.saved()["backbone"], "qwen35-4b")

    def test_switch_waits_for_idle_runtime(self):
        main._RUN_LOCK.acquire()
        try:
            with patch.object(api, "get_backbone"):
                status, body = self.fetch("/v1/runtime", {"backbone": "qwen35-4b", "backend": "mlx"})
        finally:
            main._RUN_LOCK.release()
        self.assertEqual(status, 409)


if __name__ == "__main__":
    unittest.main()
