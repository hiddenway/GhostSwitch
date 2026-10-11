"""Download a base model from Hugging Face into base/<backbone>/ of the data folder.

The server runs this module in a child process (python -m core.model_download), which
prints progress as JSON lines. Ending the process cancels a download; Hugging Face keeps
the partial file, so the next attempt resumes it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from core_models import backbone_ids, data_root, display_name, download_source, local_model_dir

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
# GGUF inference still reads the tokenizer and chat template from the MLX checkpoint folder.
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
_SKIPPED = {".gitattributes", "README.md"}
_DISK_MARGIN = 1024 ** 3


def gguf_dir(backbone: str) -> Path:
    return data_root() / "base" / backbone / "gguf"


def plan(backbone: str, backend: str) -> list[dict]:
    """The files still missing for a backbone and backend, with their sizes."""
    from huggingface_hub import HfApi

    api = HfApi()
    mlx = download_source(backbone, "mlx")
    mlx_files = {item.rfilename: item.size for item in api.model_info(
        mlx["repo_id"], revision=mlx["revision"], files_metadata=True).siblings}
    wanted = []
    if backend == "mlx":
        wanted += [(mlx, name, size, local_model_dir(backbone)) for name, size in mlx_files.items()
                   if name not in _SKIPPED]
    else:
        gguf = download_source(backbone, "gguf")
        gguf_files = {item.rfilename: item.size for item in api.model_info(
            gguf["repo_id"], revision=gguf["revision"], files_metadata=True).siblings}
        if gguf["filename"] not in gguf_files:
            raise ValueError(f"{gguf['filename']} is missing from {gguf['repo_id']}")
        wanted += [(mlx, name, mlx_files[name], local_model_dir(backbone)) for name in TOKENIZER_FILES]
        wanted.append((gguf, gguf["filename"], gguf_files[gguf["filename"]], gguf_dir(backbone)))
    files = []
    for source, name, size, folder in wanted:
        path = folder / name
        if path.is_file() and size is not None and path.stat().st_size == size:
            continue
        files.append({"repo_id": source["repo_id"], "revision": source["revision"], "filename": name,
                      "size": size or 0, "local_dir": str(folder)})
    # Small files first, so the long download is the last step.
    return sorted(files, key=lambda item: item["size"])


def install(backbone: str, backend: str, emit: Callable[[dict], None]) -> str | None:
    """Download what is missing, check the result, and return the GGUF path for GGUF."""
    from core_models import get_backbone

    from .download_progress import download_file

    files = plan(backbone, backend)
    total = sum(item["size"] for item in files)
    target = data_root() / "base" / backbone
    target.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(target).free
    if total + _DISK_MARGIN > free:
        raise OSError(f"Not enough disk space: the download needs {total / 1024 ** 3:.1f} GB "
                      f"and {free / 1024 ** 3:.1f} GB is free")
    emit({"event": "plan", "total_bytes": total, "files": len(files)})
    done, started, last = 0, time.monotonic(), 0.0
    for item in files:
        Path(item["local_dir"]).mkdir(parents=True, exist_ok=True)

        def progress(update: dict, item: dict = item) -> None:
            nonlocal last
            now = time.monotonic()
            current = int(update.get("downloaded_bytes") or 0)
            if now - last < 0.25 and current < item["size"]:
                return
            last = now
            emit({"event": "progress", "file": item["filename"], "downloaded_bytes": done + min(current, item["size"]),
                  "total_bytes": total, "bytes_per_second": update.get("bytes_per_second") or 0,
                  "elapsed_s": round(now - started, 1)})

        download_file(item["repo_id"], item["filename"], revision=item["revision"],
                      local_dir=item["local_dir"], progress=progress)
        done += item["size"]
    emit({"event": "verifying"})
    model_path = None
    if backend == "gguf":
        model_path = str(gguf_dir(backbone) / download_source(backbone, "gguf")["filename"])
    # Raises when the checkpoint or GGUF file is incomplete or the wrong model.
    get_backbone(backbone, model_path=model_path, backend=backend, validate_checkpoint=True)
    return model_path


class Installer:
    """One base-model download at a time, run in a child process the server can stop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._job: dict | None = None

    def status(self) -> dict | None:
        with self._lock:
            return dict(self._job) if self._job else None

    def start(self, backbone: str, backend: str) -> dict:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                if (self._job["backbone"], self._job["backend"]) == (backbone, backend):
                    return dict(self._job)
                raise RuntimeError(f"{display_name(self._job['backbone'])} is already downloading")
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(_PROJECT_ROOT), env.get("PYTHONPATH"))))
            self._process = subprocess.Popen(
                [sys.executable, "-m", "core.model_download", "--backbone", backbone, "--backend", backend],
                cwd=_PROJECT_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            self._job = {"backbone": backbone, "name": display_name(backbone), "backend": backend,
                         "status": "starting", "file": None, "downloaded_bytes": 0, "total_bytes": None,
                         "bytes_per_second": 0, "error": None, "model_path": None}
            threading.Thread(target=self._follow, args=(self._process,), daemon=True).start()
            return dict(self._job)

    def cancel(self) -> dict | None:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                self._job["status"] = "canceled"
                self._process.terminate()
            return dict(self._job) if self._job else None

    def _follow(self, process: subprocess.Popen) -> None:
        for line in process.stdout:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            with self._lock:
                if self._process is not process or self._job["status"] == "canceled":
                    continue
                kind = event.pop("event", None)
                if kind == "plan":
                    self._job.update(status="downloading", total_bytes=event["total_bytes"])
                elif kind == "progress":
                    self._job.update(status="downloading", **event)
                elif kind == "verifying":
                    self._job["status"] = "verifying"
                elif kind == "done":
                    self._job.update(status="done", model_path=event.get("model_path"))
                elif kind == "error":
                    self._job.update(status="error", error=event.get("message"))
        process.wait()
        with self._lock:
            if self._process is process and self._job["status"] not in {"done", "error", "canceled"}:
                self._job.update(status="error", error=f"The download stopped (exit code {process.returncode})")


INSTALLER = Installer()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download a GhostSwitch base model")
    parser.add_argument("--backbone", choices=backbone_ids(), required=True)
    parser.add_argument("--backend", choices=("mlx", "gguf"), required=True)
    args = parser.parse_args(argv)

    def emit(event: dict) -> None:
        print(json.dumps(event), flush=True)

    try:
        model_path = install(args.backbone, args.backend, emit)
    except Exception as exc:  # Reported to the app; the process exit code says it failed.
        emit({"event": "error", "message": str(exc) or type(exc).__name__})
        raise SystemExit(1)
    emit({"event": "done", "model_path": model_path})


if __name__ == "__main__":
    main()
