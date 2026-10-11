from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from core import api, main, model_download, runtime_selection

GIB = 1024 ** 3


def machine(total_gb, *, apple=True, mlx=True, gguf=True, gpu_gb=None):
    gpu = ({"name": "Apple GPU", "memory_gb": None} if apple else
           {"name": "RTX", "memory_gb": gpu_gb} if gpu_gb else None)
    return {"apple_silicon": apple, "gpu": gpu, "memory": {"total_gb": total_gb, "available_gb": 4},
            "disk_free_gb": 50, "backends_installed": {"mlx": mlx, "gguf": gguf}}


class RecommendationTests(unittest.TestCase):
    def test_largest_model_that_fits_comfortably_on_the_native_backend(self):
        self.assertEqual(runtime_selection.recommend(machine(16))["backbone"], "qwen35-9b")
        self.assertEqual(runtime_selection.recommend(machine(16))["backend"], "mlx")
        self.assertEqual(runtime_selection.recommend(machine(8))["backbone"], "qwen35-4b")
        self.assertIn("too much", runtime_selection.recommend(machine(8))["reason"])
        # Without Apple Silicon, llama.cpp runs the model and the GPU's own memory decides.
        pc = runtime_selection.recommend(machine(64, apple=False, mlx=False, gpu_gb=8))
        self.assertEqual((pc["backend"], pc["backbone"]), ("gguf", "qwen35-4b"))
        self.assertEqual(runtime_selection.recommend(machine(16, apple=False, gpu_gb=24))["backbone"], "qwen35-9b")

    def test_fit_leaves_room_for_the_system(self):
        self.assertEqual(runtime_selection.fit(7.8, 16), "comfortable")
        self.assertEqual(runtime_selection.fit(5.1, 8), "tight")
        self.assertEqual(runtime_selection.fit(7.8, 8), "too_large")
        self.assertIsNone(runtime_selection.fit(None, 16))

    def test_hardware_reports_memory_disk_and_backends(self):
        report = runtime_selection.hardware()
        self.assertGreater(report["memory"]["total_gb"], 0)
        self.assertGreaterEqual(report["disk_free_gb"], 0)
        self.assertEqual(set(report["backends_installed"]), {"mlx", "gguf"})


class InstalledTests(unittest.TestCase):
    def test_only_a_complete_local_checkpoint_counts(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch("core_models._PROJECT_ROOT", Path(folder)):
            # No local folder: the 4B falls back to a Hugging Face name, which is not installed.
            self.assertFalse(runtime_selection.installed(runtime_selection.Selection("qwen35-4b", "mlx")))
            (Path(folder) / "base/qwen35-4b/model").mkdir(parents=True)
            (Path(folder) / "base/qwen35-4b/model/tokenizer.json").write_text("{}")
            self.assertFalse(runtime_selection.installed(runtime_selection.Selection("qwen35-4b", "mlx")))
        self.assertFalse(runtime_selection.installed(runtime_selection.Selection("qwen35-4b", "gguf")))

    def test_memory_estimate_uses_the_published_size_before_download(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(runtime_selection, "local_model_dir", return_value=Path(folder)):
            self.assertEqual(runtime_selection.memory_estimate_gb("qwen35-4b", "mlx"), 5.1)
        self.assertEqual(runtime_selection.memory_estimate_gb("qwen35-9b", "gguf", None), 6.7)


class SetupModeTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        environment = patch.dict(os.environ, {"GHOSTSWITCH_RUNTIME_STATE": str(Path(folder.name) / "runtime.json")})
        environment.start()
        self.addCleanup(environment.stop)
        reason = patch.object(api, "SETUP_REASON", None)
        reason.start()
        self.addCleanup(reason.stop)

    def test_server_starts_without_a_model_and_does_not_load_one(self):
        with patch.object(runtime_selection, "installed", return_value=False), \
             patch.object(main.llm, "configure_backbone") as configure, \
             patch.object(api, "serve") as serve:
            main.main(["--serve", "--port", "8123"])
        configure.assert_not_called()
        serve.assert_called_once_with("127.0.0.1", 8123)
        self.assertIn("not downloaded yet", api.SETUP_REASON)

    def test_single_requests_still_need_a_model(self):
        with patch.object(runtime_selection, "installed", return_value=False), \
             patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            main.main(["--request-json", "-"])


class SetupApiTests(unittest.TestCase):
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
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        environment = patch.dict(os.environ, {"GHOSTSWITCH_RUNTIME_STATE": str(Path(folder.name) / "runtime.json")})
        environment.start()
        self.addCleanup(environment.stop)
        reason = patch.object(api, "SETUP_REASON", "Qwen3.5-9B for MLX is not downloaded yet")
        reason.start()
        self.addCleanup(reason.stop)

    def fetch(self, path, body=None, headers=None):
        request = Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                          headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urlopen(request) as response:
                return response.status, json.loads(response.read())
        except HTTPError as error:
            with error:
                return error.code, json.loads(error.read())

    def test_health_runtime_and_chat_report_the_missing_model(self):
        status, health = self.fetch("/health")
        self.assertEqual((status, health["setup_required"]), (200, True))
        status, body = self.fetch("/v1/runtime")
        self.assertTrue(body["setup_required"])
        self.assertFalse(body["current"]["ready"])
        self.assertIn(body["recommended"]["backbone"], {"qwen35-9b", "qwen35-4b"})
        option = body["options"][0]["backends"]["mlx"]
        self.assertTrue({"installed", "download_gb", "fit", "enough_disk"} <= set(option))
        status, body = self.fetch("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi"}]})
        self.assertEqual((status, body["error"]["type"]), (503, "model_not_installed"))

    def test_choosing_the_default_model_after_setup_restarts_the_server(self):
        with patch.object(api, "get_backbone"), patch.object(api, "_restart") as restart:
            status, body = self.fetch("/v1/runtime", {"backbone": "qwen35-9b", "backend": "mlx"})
            for _ in range(50):
                if restart.called:
                    break
                threading.Event().wait(0.02)
        self.assertEqual((status, body["status"]), (202, "restarting"))
        restart.assert_called_once()

    def test_install_starts_cancels_and_refuses_other_origins(self):
        job = {"backbone": "qwen35-4b", "backend": "mlx", "status": "starting"}
        with patch.object(runtime_selection, "installed", return_value=False), \
             patch.object(api.INSTALLER, "start", return_value=job) as start, \
             patch.object(api.INSTALLER, "cancel", return_value={**job, "status": "canceled"}) as cancel:
            status, body = self.fetch("/v1/runtime/install", {"backbone": "qwen35-4b", "backend": "mlx"})
            self.assertEqual((status, body["install"]["status"]), (202, "starting"))
            start.assert_called_once_with("qwen35-4b", "mlx")
            status, body = self.fetch("/v1/runtime/install", {"action": "cancel"})
            self.assertEqual(body["install"]["status"], "canceled")
            cancel.assert_called_once()
            status, _ = self.fetch("/v1/runtime/install", {"backbone": "qwen35-4b", "backend": "mlx"},
                                   {"Origin": "https://evil.example"})
            self.assertEqual(status, 403)
            status, _ = self.fetch("/v1/runtime/install", {"backbone": "qwen99", "backend": "mlx"})
            self.assertEqual(status, 400)
        with patch.object(runtime_selection, "installed", return_value=True):
            status, body = self.fetch("/v1/runtime/install", {"backbone": "qwen35-4b", "backend": "mlx"})
        self.assertEqual((status, body["install"]["status"]), (200, "done"))


class DownloadTests(unittest.TestCase):
    def info(self, *files):
        return SimpleNamespace(siblings=[SimpleNamespace(rfilename=name, size=size) for name, size in files])

    def test_plan_skips_complete_files_and_downloads_the_largest_last(self):
        mlx = self.info((".gitattributes", 1), ("README.md", 1), ("config.json", 3), ("tokenizer.json", 20),
                        ("tokenizer_config.json", 9), ("chat_template.jinja", 7), ("model.safetensors", 3000))
        gguf = self.info(("Qwen3.5-4B-Q4_K_M.gguf", 2700), ("Qwen3.5-4B-Q8_0.gguf", 4400))
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(model_download, "_PROJECT_ROOT", Path(folder)), \
             patch.object(model_download, "local_model_dir", return_value=Path(folder) / "model"), \
             patch("huggingface_hub.HfApi.model_info", side_effect=lambda repo, **_: gguf if "GGUF" in repo else mlx):
            (Path(folder) / "model").mkdir()
            (Path(folder) / "model/config.json").write_text("abc")
            files = model_download.plan("qwen35-4b", "mlx")
            self.assertEqual([item["filename"] for item in files],
                             ["chat_template.jinja", "tokenizer_config.json", "tokenizer.json", "model.safetensors"])
            files = model_download.plan("qwen35-4b", "gguf")
        self.assertEqual([item["filename"] for item in files],
                         ["chat_template.jinja", "tokenizer_config.json", "tokenizer.json", "Qwen3.5-4B-Q4_K_M.gguf"])
        self.assertTrue(files[-1]["local_dir"].endswith("base/qwen35-4b/gguf"))

    def test_install_reports_progress_and_checks_free_space(self):
        files = [{"repo_id": "r", "revision": "v", "filename": "model.safetensors", "size": 100,
                  "local_dir": "/tmp/ghostswitch-test-model"}]

        def download(repo_id, filename, *, revision, local_dir, progress):
            progress({"downloaded_bytes": 50, "bytes_per_second": 10})
            progress({"downloaded_bytes": 100, "bytes_per_second": 10})

        events = []
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(model_download, "_PROJECT_ROOT", Path(folder)), \
             patch.object(model_download, "plan", return_value=files), \
             patch("core.download_progress.download_file", side_effect=download), \
             patch("core_models.get_backbone") as validate:
            files[0]["local_dir"] = folder
            self.assertIsNone(model_download.install("qwen35-4b", "mlx", events.append))
            validate.assert_called_once()
            self.assertEqual(events[0], {"event": "plan", "total_bytes": 100, "files": 1})
            self.assertEqual(events[-2]["downloaded_bytes"], 100)
            self.assertEqual(events[-1], {"event": "verifying"})
            with patch("shutil.disk_usage", return_value=SimpleNamespace(free=10)), \
                 self.assertRaisesRegex(OSError, "Not enough disk space"):
                model_download.install("qwen35-4b", "mlx", events.append)

    def test_installer_follows_the_child_process(self):
        lines = [json.dumps({"event": "plan", "total_bytes": 100, "files": 1}),
                 json.dumps({"event": "progress", "file": "m", "downloaded_bytes": 40, "total_bytes": 100,
                             "bytes_per_second": 5, "elapsed_s": 8}),
                 json.dumps({"event": "verifying"}), json.dumps({"event": "done", "model_path": None})]

        class Child:
            def __init__(self, *args, **kwargs):
                self.stdout = io.StringIO("\n".join(lines) + "\n")
                self.returncode = None

            def poll(self):
                return self.returncode

            def wait(self):
                self.returncode = 0

        installer = model_download.Installer()
        with patch("subprocess.Popen", Child):
            installer.start("qwen35-4b", "mlx")
            for _ in range(100):
                if installer.status()["status"] == "done":
                    break
                threading.Event().wait(0.01)
        self.assertEqual(installer.status()["status"], "done")
        self.assertEqual(installer.status()["downloaded_bytes"], 40)


if __name__ == "__main__":
    unittest.main()
