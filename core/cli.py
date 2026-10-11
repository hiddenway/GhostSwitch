"""The ghostswitch command: start the app and open it in the browser, or check this installation.

`ghostswitch` runs the server in a child process, waits until it answers, opens the web app and stops
the server on Ctrl+C. The installer's launcher handles `ghostswitch update` and `ghostswitch uninstall`
itself, because both replace the folder this module runs from.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import time
import tomllib
import urllib.request
import webbrowser
from pathlib import Path

from core_models import data_root

APP_ROOT = Path(__file__).resolve().parents[1]
# "BRAIN" on a phone keypad, from the MiniBrain days; kept so existing setups find the app.
# 8000 and 8080 are taken by many dev servers; 27018-27344 is unassigned by IANA and lies below
# the ports operating systems hand out to outgoing connections (32768 up).
DEFAULT_PORT = 27246
_PORTS = range(DEFAULT_PORT, 27345)
# The server answers only after it has prepared the base model and the expert library.
_START_TIMEOUT_S = 600
# Switching the base model in the app restarts the server; on Windows that is a new process.
_RESTART_GRACE_S = 120
_LOG_LIMIT_BYTES = 10 * 1024 * 1024
_INSTALL_COMMANDS = (
    "macOS and Linux:  curl -fsSL https://raw.githubusercontent.com/hiddenway/ghostswitch/main/install.sh | sh\n"
    "Windows:          irm https://raw.githubusercontent.com/hiddenway/ghostswitch/main/install.ps1 | iex"
)


def version() -> str:
    try:
        with (APP_ROOT / "pyproject.toml").open("rb") as handle:
            return str(tomllib.load(handle)["project"]["version"])
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return "unknown"


def _state_path() -> Path:
    return data_root() / "runtime" / "server.json"


def _log_path() -> Path:
    return data_root() / "runtime" / "logs" / "server.log"


def app_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/app/"


def health(port: int, timeout: float = 1.0) -> dict | None:
    """The server's /health reply on this port, or None when no GhostSwitch answers there."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as response:
            body = json.loads(response.read())
    except (OSError, ValueError):
        return None
    if isinstance(body, dict) and str(body.get("model", "")).startswith(("ghostswitch", "omnibrain", "minibrain")):
        return body
    return None


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if sys.platform != "win32":
            # The server reuses a port in TIME_WAIT, so the probe does too; on Windows this
            # option would let it bind a port another program is using.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def running_port() -> int | None:
    """The port of a GhostSwitch server already running on this computer."""
    try:
        recorded = int(json.loads(_state_path().read_text(encoding="utf-8"))["port"])
    except (OSError, ValueError, KeyError, TypeError):
        recorded = None
    for port in ([recorded] if recorded else []) + [DEFAULT_PORT]:
        if health(port):
            return port
    return None


def choose_port(requested: int | None) -> int | None:
    """The requested port, or the first free one from DEFAULT_PORT on."""
    if requested is not None:
        return requested if port_free(requested) else None
    return next((port for port in _PORTS if port_free(port)), None)


def _open_log():
    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > _LOG_LIMIT_BYTES:
        path.replace(path.with_suffix(".log.1"))
    return path.open("a", encoding="utf-8")


def _log_tail(lines: int = 30) -> str:
    try:
        return "\n".join(_log_path().read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def open_browser(url: str) -> None:
    if os.environ.get("GHOSTSWITCH_NO_BROWSER"):
        return
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return
    try:
        webbrowser.open(url)
    except webbrowser.Error:
        pass


def _wait_until_ready(child: subprocess.Popen, port: int) -> bool:
    deadline = time.monotonic() + _START_TIMEOUT_S
    while time.monotonic() < deadline:
        if health(port):
            return True
        # Exit code 0 before answering is a restart into another process (Windows); keep waiting.
        if child.poll() not in (None, 0):
            return False
        time.sleep(0.5)
    return False


def _follow(child: subprocess.Popen, port: int) -> int:
    """Wait while the server runs; return its exit code, or 0 after a restart ended normally."""
    while child.poll() is None:
        time.sleep(1)
    if child.returncode != 0:
        return child.returncode
    # The process ended cleanly but a restart may have replaced it: follow the port instead.
    while True:
        deadline = time.monotonic() + _RESTART_GRACE_S
        while not health(port, timeout=2):
            if time.monotonic() > deadline:
                return 0
            time.sleep(1)
        while health(port, timeout=5):
            time.sleep(2)


def _stop(child: subprocess.Popen) -> None:
    # Ctrl+C reached the server as well; give it a moment to close, then end it.
    for signal_sent in (False, True):
        try:
            child.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            if signal_sent:
                child.kill()
            else:
                child.terminate()


def _interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def start(port: int | None = None, browser: bool = True, verbose: bool = False) -> int:
    # Ending this command (closing the terminal, kill) stops the server too instead of orphaning it.
    for name in ("SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _interrupt)
    running = running_port()
    if running and (port is None or port == running):
        print(f"GhostSwitch is already running: {app_url(running)}")
        if browser:
            open_browser(app_url(running))
        return 0
    chosen = choose_port(port)
    if chosen is None:
        print(f"Port {port} is in use by another program; choose another with --port." if port
              else f"No free port between {_PORTS[0]} and {_PORTS[-1]}.", file=sys.stderr)
        return 1
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(APP_ROOT), env.get("PYTHONPATH"))))
    # Files and prompts are UTF-8; Windows would otherwise read them in its legacy code page.
    env.setdefault("PYTHONUTF8", "1")
    env["PYTHONUNBUFFERED"] = "1"
    log = None if verbose else _open_log()
    command = [sys.executable, "-m", "core.main", "--serve", "--host", "127.0.0.1", "--port", str(chosen)]
    print(f"Starting GhostSwitch {version()}...", flush=True)
    child = subprocess.Popen(command, cwd=APP_ROOT, env=env, stdout=log,
                             stderr=subprocess.STDOUT if log else None)
    try:
        if not _wait_until_ready(child, chosen):
            if child.poll() is None:
                child.terminate()
            print("GhostSwitch could not start.", file=sys.stderr)
            if log is not None:
                print(_log_tail(), file=sys.stderr)
                print(f"\nFull log: {_log_path()}", file=sys.stderr)
            print("Run `ghostswitch doctor` to check this installation.", file=sys.stderr)
            return 1
        state = _state_path()
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text(json.dumps({"pid": child.pid, "port": chosen}), encoding="utf-8")
        url = app_url(chosen)
        print(f"GhostSwitch is running: {url}")
        print("Keep this window open while you use it. Press Ctrl+C to stop.", flush=True)
        if browser:
            open_browser(url)
        code = _follow(child, chosen)
        if code != 0:
            print(f"GhostSwitch stopped unexpectedly (exit code {code}).", file=sys.stderr)
            if log is not None:
                print(_log_tail(), file=sys.stderr)
            return code
        print("GhostSwitch stopped.")
        return 0
    except KeyboardInterrupt:
        print("\nStopping GhostSwitch...", flush=True)
        try:
            _stop(child)
        except KeyboardInterrupt:
            child.kill()
        return 0
    finally:
        _state_path().unlink(missing_ok=True)
        if log is not None:
            log.close()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _llama_cpp_gpu() -> str:
    """Whether llama.cpp loads, and whether it can use a GPU."""
    from . import cuda_libraries

    try:
        cuda_libraries.prepare()
        import llama_cpp

        return "GPU offload available" if llama_cpp.llama_supports_gpu_offload() else "processor only"
    except Exception as error:  # A broken native library raises OSError or worse; report it.
        return f"fails to load: {error}"


def doctor() -> int:
    """Print what this installation has and what is missing."""
    from . import runtime_selection

    problems: list[str] = []
    print(f"GhostSwitch {version()}")

    def row(name: str, value: object) -> None:
        print(f"  {name:<16} {value}")

    if os.environ.get("GHOSTSWITCH_ROOT"):
        row("Installed in", os.environ["GHOSTSWITCH_ROOT"])
    row("App", APP_ROOT)
    row("Data", data_root())
    row("Python", f"{platform.python_version()} ({sys.executable})")
    row("System", f"{platform.system()} {platform.release()} ({platform.machine()})")
    machine = runtime_selection.hardware()
    memory = machine["memory"]
    row("Memory", f"{memory['total_gb']} GB" + (f", {memory['available_gb']} GB free now"
                                                 if memory.get("available_gb") is not None else ""))
    row("Disk", f"{machine['disk_free_gb']} GB free")
    gpu = machine.get("gpu")
    row("GPU", f"{gpu['name']}" + (f", {gpu['memory_gb']} GB" if gpu.get("memory_gb") else "") if gpu else "none found")
    mlx = _package_version("mlx-lm")
    llama = _package_version("llama-cpp-python")
    row("MLX", f"mlx-lm {mlx}" if mlx else "not installed")
    if llama:
        status = _llama_cpp_gpu()
        row("llama.cpp", f"llama-cpp-python {llama}, {status}")
        if status.startswith("fails"):
            problems.append("llama.cpp does not load; run the installer again to repair it.")
        elif gpu and "NVIDIA" in str(gpu.get("name")) and status == "processor only":
            problems.append("An NVIDIA GPU is present but this llama.cpp build runs on the processor; "
                            "run the installer again with GHOSTSWITCH_VARIANT=cuda.")
    else:
        row("llama.cpp", "not installed")
    if not mlx and not llama:
        problems.append("No inference backend is installed; run the installer again.")
    try:
        selection = runtime_selection.resolve()
        ready = runtime_selection.installed(selection)
        row("Base model", f"{selection.backbone} on {selection.backend.upper()}, "
                          f"{'downloaded' if ready else 'not downloaded yet (the app downloads it)'}")
    except ValueError as error:
        row("Base model", f"not downloaded yet ({error})")
    port = running_port()
    row("Server", f"running at {app_url(port)}" if port else "not running")
    row("Log", _log_path())
    if problems:
        print("\nTo fix:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nNo problems found.")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or (argv[0].startswith("-") and argv[0] not in {"-h", "--help", "--version"}):
        argv.insert(0, "start")
    parser = argparse.ArgumentParser(prog="ghostswitch", description="Run GhostSwitch and open it in your browser.")
    parser.add_argument("--version", action="version", version=f"GhostSwitch {version()}")
    commands = parser.add_subparsers(dest="command", metavar="command")
    starter = commands.add_parser("start", help="start GhostSwitch and open it in the browser (the default)")
    starter.add_argument("--port", type=int,
                         help=f"port for the web app (default: {DEFAULT_PORT}, or the next free one)")
    starter.add_argument("--no-browser", action="store_true", help="do not open the browser")
    starter.add_argument("--verbose", action="store_true", help="show the server log in this window")
    commands.add_parser("doctor", help="check this installation and show what to fix")
    commands.add_parser("update", help="install the latest version")
    commands.add_parser("uninstall", help="remove GhostSwitch from this computer")
    args = parser.parse_args(argv)
    if args.command == "start":
        return start(args.port, browser=not args.no_browser, verbose=args.verbose)
    if args.command == "doctor":
        return doctor()
    # Reached only outside an installed copy, whose launcher handles these two itself.
    print(f"`ghostswitch {args.command}` works in a copy installed by the installer. "
          f"To install GhostSwitch, run:\n{_INSTALL_COMMANDS}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
