"""Persistent user selection and background preparation of expert data files."""
from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from core_models import user_dir

from . import llm
from .expert_packages import blocked, load_packages
from .lora_runtime import ADAPTERS, ensure_native_adapter, native_adapter_path, gguf_adapter_path


class ExpertLibrary:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="expert-download")
        self._scope = None
        self._selected: set[str] = set()
        self._jobs: dict[str, dict] = {}
        self._futures = {}

    @property
    def path(self) -> Path:
        return Path(os.environ.get("GHOSTSWITCH_EXPERT_STATE", user_dir(Path.home() / ".config") / "experts.json")).expanduser()

    def _initialize(self) -> None:
        scope = f"{llm.ACTIVE_BACKBONE}:{llm.MODEL_SPECS['thinker'].backend}"
        if self._scope == scope:
            return
        state = json.loads(self.path.read_text()) if self.path.exists() else {"schema_version": 1, "selections": {}}
        if not isinstance(state, dict) or state.get("schema_version") != 1 or not isinstance(state.get("selections"), dict):
            raise ValueError(f"Invalid expert selection state: {self.path}")
        selected = state["selections"].get(scope)
        if selected is None:
            selected = [eid for eid, p in load_packages(llm.ACTIVE_BACKBONE).items()
                        if p["profile"].get("default_enabled") and eid in ADAPTERS and not blocked(p)]
        if not isinstance(selected, list) or any(not isinstance(eid, str) for eid in selected):
            raise ValueError(f"Invalid expert selection for {scope}")
        self._selected = set(selected)
        self._scope = scope
        self._jobs = {}
        self._futures = {}
        self._save()

    def _save(self) -> None:
        state = json.loads(self.path.read_text()) if self.path.exists() else {"schema_version": 1, "selections": {}}
        state["selections"][self._scope] = sorted(self._selected)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".partial")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        os.replace(temporary, self.path)

    def enabled(self, expert_id: str) -> bool:
        with self._lock:
            self._initialize()
            return expert_id in self._selected

    def downloaded(self, expert_id: str) -> bool:
        if llm.MODEL_SPECS["thinker"].backend == "gguf":
            path = gguf_adapter_path(expert_id)
            try:
                metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text())
                row = load_packages(llm.ACTIVE_BACKBONE)[expert_id]["adapter"]
                from scripts.prepare_gguf_adapters import adapter_metadata
                return path.is_file() and path.stat().st_size > 0 and metadata == adapter_metadata(row)
            except (OSError, ValueError, KeyError):
                return False
        path = native_adapter_path(expert_id)
        try:
            config = json.loads((path / "adapter_config.json").read_text())
            spec = ADAPTERS[expert_id]
            if not isinstance(config, dict):
                return False
            return ((path / "adapters.safetensors").is_file()
                    and (path / "adapters.safetensors").stat().st_size > 0
                    and (path / "source_manifest.json").is_file()
                    and config.get("runtime_format_version") == 2
                    and config.get("source_repo") == spec.repo_id
                    and config.get("source_revision") == spec.revision
                    and config.get("source_config_filename", "adapter_config.json") == spec.source_config_filename
                    and config.get("source_filename", "adapter_model.safetensors") == spec.source_filename)
        except (OSError, ValueError, KeyError):
            return False

    def status(self, expert_id: str) -> dict:
        with self._lock:
            self._initialize()
            downloaded = self.downloaded(expert_id)
            job = self._jobs.get(expert_id, {})
            return {"enabled": expert_id in self._selected, "downloaded": downloaded,
                    "status": job.get("status") if job.get("status") in {"queued", "downloading", "error"}
                              else ("ready" if downloaded else "not_downloaded"),
                    "error": job.get("error"), "progress": dict(job["progress"]) if job.get("progress") else None}

    def startup(self) -> None:
        with self._lock:
            self._initialize()
            packages = load_packages(llm.ACTIVE_BACKBONE)
            for expert_id in sorted(self._selected & ADAPTERS.keys()):
                if not blocked(packages.get(expert_id, {})):
                    self._queue(expert_id)

    def manage(self, expert_id: str, action: str) -> dict:
        with self._lock:
            self._initialize()
            if expert_id not in ADAPTERS:
                raise KeyError(expert_id)
            if action not in {"enable", "disable", "download"}:
                raise ValueError("action must be enable, disable, or download")
            package = load_packages(llm.ACTIVE_BACKBONE).get(expert_id)
            if action == "enable" and package and blocked(package):
                raise ValueError(f"{expert_id} is blocked: {package['evaluation']['reason']}")
            if action == "disable":
                self._selected.discard(expert_id)
            elif action == "enable":
                self._selected.add(expert_id)
            self._save()
            if action != "disable":
                self._queue(expert_id)
            return self.status(expert_id)

    def _queue(self, expert_id: str) -> None:
        if self.downloaded(expert_id) or self._jobs.get(expert_id, {}).get("status") in {"queued", "downloading"}:
            return
        self._jobs[expert_id] = {"status": "queued", "error": None}
        self._futures[expert_id] = self._executor.submit(self._prepare, expert_id)

    def prepare_for_request(self, expert_id: str) -> None:
        with self._lock:
            self._initialize()
            if expert_id not in ADAPTERS:
                raise KeyError(expert_id)
            self._queue(expert_id)
            future = self._futures.get(expert_id)
        if future is not None:
            future.result()
        if not self.downloaded(expert_id):
            raise RuntimeError(self.status(expert_id).get("error") or "Expert preparation failed")

    def _update_progress(self, expert_id: str, update: dict) -> None:
        with self._lock:
            job = self._jobs[expert_id]
            progress = job.setdefault("progress", {})
            progress.update(update)
            total = progress.get("total_bytes")
            progress["percent"] = round(min(100, 100 * progress.get("downloaded_bytes", 0) / total), 1) if total else None

    def _prepare(self, expert_id: str) -> None:
        with self._lock:
            self._jobs[expert_id] = {"status": "downloading", "error": None}
        try:
            if llm.MODEL_SPECS["thinker"].backend == "mlx":
                ensure_native_adapter(expert_id, progress=lambda update: self._update_progress(expert_id, update))
            else:
                from scripts.prepare_gguf_adapters import prepare_expert
                prepare_expert(expert_id, gguf_adapter_path(expert_id).parent,
                               progress=lambda update: self._update_progress(expert_id, update))
        except Exception as exc:
            with self._lock:
                self._jobs[expert_id] = {"status": "error", "error": str(exc)}
        else:
            with self._lock:
                self._jobs[expert_id] = {"status": "ready", "error": None}


LIBRARY = ExpertLibrary()
