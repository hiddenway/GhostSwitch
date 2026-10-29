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
        # The published sizes, not whatever checkpoint this computer has.
        empty = tempfile.TemporaryDirectory()
        self.addCleanup(empty.cleanup)
        local = patch.object(runtime_selection, "local_model_dir", return_value=Path(empty.name))
        local.start()
        self.addCleanup(local.stop)
        # The 5-bit 9B sits exactly at the 'comfortable' limit of a 16 GB Mac (0.55 x 16 GB):
        # any growth of the estimate would recommend the 4B to every 16 GB Mac.
        self.assertEqual(runtime_selection.memory_estimate_gb("qwen35-9b", "mlx"), 8.8)
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
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(runtime_selection, "installed", return_value=False), \
             patch.object(runtime_selection, "local_model_dir", return_value=Path(folder)), \
             patch.object(main.llm, "configure_backbone") as configure, \
             patch.object(api, "serve") as serve:
            main.main(["--serve", "--port", "8123"])
        configure.assert_not_called()
        serve.assert_called_once_with("127.0.0.1", 8123)
        self.assertIn("not downloaded yet", api.SETUP_REASON)

    def test_an_older_quantization_on_disk_asks_for_an_update(self):
        """The 9B moved from the 4-bit to the 5-bit checkpoint; the old one is not 'missing'."""
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "config.json").write_text(json.dumps(
                {"model_type": "qwen3_5", "quantization_config": {"bits": 4, "group_size": 64}}))
            (Path(folder) / "model-00001-of-00002.safetensors").write_bytes(b"w" * 1000)
            with patch.object(runtime_selection, "local_model_dir", return_value=Path(folder)):
                self.assertEqual(runtime_selection.outdated_checkpoint("qwen35-9b"), 4)
                # The old shards do not stand in for the checkpoint that replaces them.
                self.assertEqual(runtime_selection.memory_estimate_gb("qwen35-9b", "mlx"), 8.8)
                reason = runtime_selection.setup_reason(runtime_selection.Selection("qwen35-9b", "mlx"))
                self.assertIn("needs an update", reason)
                self.assertIn("5-bit", reason)
                self.assertIsNone(runtime_selection.outdated_checkpoint("qwen35-4b"))
                self.assertIn("not downloaded yet",
                              runtime_selection.setup_reason(runtime_selection.Selection("qwen35-9b", "gguf")))
            (Path(folder) / "config.json").write_text(json.dumps(
                {"model_type": "qwen3_5", "quantization_config": {"bits": 5, "group_size": 64}}))
            with patch.object(runtime_selection, "local_model_dir", return_value=Path(folder)):
                self.assertIsNone(runtime_selection.outdated_checkpoint("qwen35-9b"))

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
    LARGE = {"model.safetensors", "Qwen3.5-4B-Q4_K_M.gguf", "Qwen3.5-4B-Q8_0.gguf"}

    def info(self, *files):
        return SimpleNamespace(siblings=[
            SimpleNamespace(rfilename=name, size=size,
                            lfs=SimpleNamespace(size=size) if name in self.LARGE else None)
            for name, size in files])

    @staticmethod
    def record(folder: Path, name: str, revision: str) -> None:
        """What hf_hub_download leaves next to a file it downloaded into a local folder."""
        metadata = folder / ".cache" / "huggingface" / "download" / f"{name}.metadata"
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(f"{revision}\netag\n1.0\n")

    def test_plan_skips_complete_files_and_downloads_the_largest_last(self):
        mlx = self.info((".gitattributes", 1), ("README.md", 1), ("config.json", 3), ("tokenizer.json", 20),
                        ("tokenizer_config.json", 9), ("chat_template.jinja", 7), ("model.safetensors", 3000))
        gguf = self.info(("Qwen3.5-4B-Q4_K_M.gguf", 2700), ("Qwen3.5-4B-Q8_0.gguf", 4400))
        revision = model_download.download_source("qwen35-4b", "mlx")["revision"]
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(model_download, "_PROJECT_ROOT", Path(folder)), \
             patch.object(model_download, "local_model_dir", return_value=Path(folder) / "model"), \
             patch("huggingface_hub.HfApi.model_info", side_effect=lambda repo, **_: gguf if "GGUF" in repo else mlx):
            model = Path(folder) / "model"
            model.mkdir()
            (model / "config.json").write_text("abc")
            self.record(model, "config.json", revision)
            files = model_download.plan("qwen35-4b", "mlx")
            self.assertEqual([item["filename"] for item in files],
                             ["chat_template.jinja", "tokenizer_config.json", "tokenizer.json", "model.safetensors"])
            files = model_download.plan("qwen35-4b", "gguf")
        self.assertEqual([item["filename"] for item in files],
                         ["chat_template.jinja", "tokenizer_config.json", "tokenizer.json", "Qwen3.5-4B-Q4_K_M.gguf"])
        self.assertTrue(files[-1]["local_dir"].endswith("base/qwen35-4b/gguf"))

    def test_plan_replaces_same_size_files_from_another_checkpoint(self):
        """A new quantization of the same model has a config.json and index of the same size."""
        mlx = self.info(("config.json", 3), ("model.safetensors.index.json", 5), ("model.safetensors", 3000))
        revision = model_download.download_source("qwen35-4b", "mlx")["revision"]
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(model_download, "local_model_dir", return_value=Path(folder)), \
             patch("huggingface_hub.HfApi.model_info", return_value=mlx):
            model = Path(folder)
            (model / "config.json").write_text("abc")
            (model / "model.safetensors.index.json").write_text("abcde")
            (model / "model.safetensors").write_bytes(b"x" * 3000)
            self.record(model, "config.json", "0" * 40)
            # Another revision's config, an index with no record, and an LFS shard with no record.
            self.assertEqual([item["filename"] for item in model_download.plan("qwen35-4b", "mlx")],
                             ["config.json", "model.safetensors.index.json"])
            self.record(model, "config.json", revision)
            self.record(model, "model.safetensors.index.json", revision)
            self.record(model, "model.safetensors", "0" * 40)
            self.assertEqual([item["filename"] for item in model_download.plan("qwen35-4b", "mlx")],
                             ["model.safetensors"])

    @classmethod
    def downloaded(cls, local_dir, filename, revision, size):
        """What hf_hub_download leaves in a local folder: the file and its revision record."""
        (Path(local_dir) / filename).write_bytes(b"n" * size)
        cls.record(Path(local_dir), filename, revision)

    def test_install_reports_progress_and_checks_free_space(self):
        files = [{"repo_id": "r", "revision": "v", "filename": "model.safetensors", "size": 100, "lfs": True,
                  "local_dir": "/tmp/ghostswitch-test-model"}]

        def download(repo_id, filename, *, revision, local_dir, progress):
            progress({"downloaded_bytes": 50, "bytes_per_second": 10})
            progress({"downloaded_bytes": 100, "bytes_per_second": 10})
            self.downloaded(local_dir, filename, revision, 100)

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

    def test_an_update_replaces_old_files_first_and_counts_their_space(self):
        """4-bit -> 5-bit: same file names, so the old files go before any new one arrives."""
        names = ["config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        with tempfile.TemporaryDirectory() as folder:
            files = [{"repo_id": "new", "revision": "n" * 40, "filename": name, "size": size, "lfs": lfs,
                      "local_dir": folder} for name, size, lfs in
                     zip(names, (3, 600, 170), (False, True, True))]
            for name, size in zip(names, (3, 550, 60)):
                self.downloaded(folder, name, "o" * 40, size)
            seen = []

            def download(repo_id, filename, *, revision, local_dir, progress):
                # Nothing of the old checkpoint is left once the first new file arrives.
                seen.append(sorted(path.name for path in Path(local_dir).glob("*") if path.is_file()))
                self.downloaded(local_dir, filename, revision, next(i["size"] for i in files if i["filename"] == filename))

            with patch.object(model_download, "plan", return_value=files), \
                 patch("core.download_progress.download_file", side_effect=download), \
                 patch("core_models.get_backbone"), \
                 patch("shutil.disk_usage", return_value=SimpleNamespace(free=200)):
                # 773 bytes to download, 200 free, 613 of old files replaced: it fits (margin aside).
                with patch.object(model_download, "_DISK_MARGIN", 0):
                    model_download.install("qwen35-9b", "mlx", lambda event: None)
            self.assertEqual(seen[0], [])

    def test_a_gguf_install_never_removes_the_mlx_checkpoints_files(self):
        """Its tokenizer files live in the MLX folder; a failed download must not take them away."""
        mlx = self.info(("tokenizer.json", 20), ("tokenizer_config.json", 9), ("chat_template.jinja", 7))
        gguf = self.info(("Qwen3.5-4B-Q4_K_M.gguf", 2700))
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "model"
            model.mkdir()
            for name, size in (("tokenizer.json", 20), ("tokenizer_config.json", 9), ("chat_template.jinja", 7)):
                (model / name).write_bytes(b"t" * size)   # hand-placed: same size, no revision record
            with patch.object(model_download, "_PROJECT_ROOT", Path(folder)), \
                 patch.object(model_download, "local_model_dir", return_value=model), \
                 patch.object(model_download, "gguf_dir", return_value=Path(folder) / "gguf"), \
                 patch("huggingface_hub.HfApi.model_info", side_effect=lambda repo, **_: gguf if "GGUF" in repo else mlx), \
                 patch("core.download_progress.download_file", side_effect=OSError("network down")), \
                 self.assertRaises(OSError):
                self.assertEqual([item["filename"] for item in model_download.plan("qwen35-4b", "gguf")],
                                 ["Qwen3.5-4B-Q4_K_M.gguf"])
                model_download.install("qwen35-4b", "gguf", lambda event: None)
            self.assertEqual(sorted(path.name for path in model.iterdir()),
                             ["chat_template.jinja", "tokenizer.json", "tokenizer_config.json"])

    def test_an_outdated_symlinked_checkpoint_is_left_alone(self):
        """The update downloads into a folder of its own instead of overwriting the link's target."""
        with tempfile.TemporaryDirectory() as folder:
            elsewhere = Path(folder) / "lmstudio" / "Qwen3.5-9B-MLX-4bit"
            elsewhere.mkdir(parents=True)
            (elsewhere / "config.json").write_text("old")
            link = Path(folder) / "base" / "qwen35-9b" / "model"
            link.parent.mkdir(parents=True)
            link.symlink_to(elsewhere)
            with patch.object(model_download, "local_model_link", return_value=link), \
                 patch.object(model_download, "local_model_dir", side_effect=lambda backbone: link.resolve()), \
                 patch.object(model_download, "plan", return_value=[]), \
                 patch("core_models.get_backbone"):
                model_download.install("qwen35-9b", "mlx", lambda event: None)
            self.assertFalse(link.is_symlink())
            self.assertTrue(link.is_dir())
            self.assertEqual((elsewhere / "config.json").read_text(), "old")

    def test_install_fails_when_a_file_is_still_the_old_one(self):
        """A failed request to the Hub can leave the old file in place; that is not a success."""
        with tempfile.TemporaryDirectory() as folder:
            files = [{"repo_id": "new", "revision": "n" * 40, "filename": "model.safetensors", "size": 10,
                      "lfs": True, "local_dir": folder}]

            def download(repo_id, filename, *, revision, local_dir, progress):
                self.downloaded(local_dir, filename, "o" * 40, 10)

            with patch.object(model_download, "plan", return_value=files), \
                 patch("core.download_progress.download_file", side_effect=download), \
                 patch("core_models.get_backbone") as validate, \
                 self.assertRaisesRegex(OSError, "did not complete"):
                model_download.install("qwen35-9b", "mlx", lambda event: None)
            validate.assert_not_called()

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
