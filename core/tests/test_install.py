"""The installed layout: the data folder, the ghostswitch command, and the cross-process file lock."""

from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import core_models
from core import cli, model_download

ROOT = Path(__file__).resolve().parents[2]


class DataFolderTests(unittest.TestCase):
    def test_a_checkout_keeps_its_data_in_the_project_folder(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GHOSTSWITCH_HOME", None)
            self.assertEqual(core_models.data_root(), ROOT)
            self.assertEqual(core_models.local_model_dir("qwen35-9b"), ROOT / "base" / "qwen35-9b" / "model")
            self.assertEqual(core_models.user_dir(Path.home() / ".config").parent, Path.home() / ".config")

    def test_an_installed_copy_keeps_models_settings_and_caches_in_ghostswitch_home(self):
        with tempfile.TemporaryDirectory() as home, patch.dict(os.environ, {"GHOSTSWITCH_HOME": home}):
            home = Path(home).resolve()
            self.assertEqual(core_models.data_root(), home)
            self.assertEqual(core_models.local_model_dir("qwen35-4b"), home / "base" / "qwen35-4b" / "model")
            self.assertEqual(model_download.gguf_dir("qwen35-9b"), home / "base" / "qwen35-9b" / "gguf")
            self.assertEqual(core_models.user_dir(Path.home() / ".config"), home / "config")
            self.assertEqual(core_models.user_dir(Path.home() / ".cache"), home / "cache")


class _Health(BaseHTTPRequestHandler):
    body = b"{}"

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *args):
        pass


class CommandTests(unittest.TestCase):
    def serve(self, body: dict) -> int:
        handler = type("Handler", (_Health,), {"body": json.dumps(body).encode()})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server.server_address[1]

    def test_health_recognizes_only_a_ghostswitch_server(self):
        self.assertEqual(cli.health(self.serve({"status": "ok", "model": "ghostswitch-v6"}))["status"], "ok")
        # A copy from before the rename answers with its old model id.
        self.assertEqual(cli.health(self.serve({"status": "ok", "model": "omnibrain-v6"}))["status"], "ok")
        self.assertIsNone(cli.health(self.serve({"status": "ok", "model": "something-else"})))

    def test_a_busy_port_is_skipped_and_a_requested_busy_port_is_refused(self):
        with socket.socket() as taken:
            taken.bind(("127.0.0.1", 0))
            taken.listen()
            port = taken.getsockname()[1]
            self.assertFalse(cli.port_free(port))
            self.assertIsNone(cli.choose_port(port))
            with patch.object(cli, "_PORTS", range(port, port + 50)):
                chosen = cli.choose_port(None)
            self.assertNotEqual(chosen, port)
            self.assertTrue(cli.port_free(chosen))

    def test_the_ports_stay_in_the_unassigned_block_below_outgoing_connection_ports(self):
        # IANA leaves 27018-27344 unassigned; systems hand out 32768 and up to outgoing connections.
        self.assertEqual(cli.DEFAULT_PORT, 27246)
        self.assertGreaterEqual(cli._PORTS[0], 27018)
        self.assertLessEqual(cli._PORTS[-1], 27344)

    def test_start_is_the_default_command(self):
        with patch.object(cli, "start", return_value=0) as start:
            self.assertEqual(cli.main([]), 0)
            start.assert_called_with(None, browser=True, verbose=False)
            cli.main(["--port", "9100", "--no-browser"])
            start.assert_called_with(9100, browser=False, verbose=False)

    def test_update_outside_an_installed_copy_explains_the_installer(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(cli.main(["update"]), 1)
        self.assertIn("install.sh", out.getvalue())
        self.assertIn("install.ps1", out.getvalue())

    def test_version_comes_from_pyproject(self):
        self.assertRegex(cli.version(), r"^\d+\.\d+\.\d+$")


class FileLockTests(unittest.TestCase):
    def test_a_second_process_waits_for_the_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            lock = Path(folder) / "adapter.lock"
            holder = subprocess.Popen(
                [sys.executable, "-c",
                 "import sys, time; from pathlib import Path; from core.file_lock import exclusive\n"
                 "with exclusive(Path(sys.argv[1])):\n    print('held', flush=True)\n    time.sleep(1.0)",
                 str(lock)],
                cwd=ROOT, stdout=subprocess.PIPE, text=True)
            self.addCleanup(holder.wait)
            self.addCleanup(holder.stdout.close)
            self.assertEqual(holder.stdout.readline().strip(), "held")
            from core.file_lock import exclusive

            started = time.monotonic()
            with exclusive(lock):
                waited = time.monotonic() - started
            self.assertGreater(waited, 0.5)


if __name__ == "__main__":
    unittest.main()
