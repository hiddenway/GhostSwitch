"""The persisted choice of backbone and backend; switching restarts the server process.

A process serves one backbone: expert catalogs, the routing tree and the adapter
registry are built for it at import time. A switch therefore replaces the process
with a new one for the selected backbone, which also releases the previous model.
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from core_models import (
    ACTIVE_BACKBONE_ENV, DEFAULT_BACKBONE, backbone_ids, data_root, display_name, download_source, find_gguf,
    get_backbone, gguf_context, kv_bytes_per_token, local_model_dir, user_dir,
)

BACKENDS = ("mlx", "gguf")
_GIB = 1024 ** 3
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_RUNTIME_FLAGS = {"--backbone", "--backend", "--model-path"}
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def state_path() -> Path:
    return Path(os.environ.get(
        "GHOSTSWITCH_RUNTIME_STATE", user_dir(Path.home() / ".config") / "runtime.json",
    )).expanduser()


@dataclass(frozen=True)
class Selection:
    backbone: str
    backend: str
    model_path: str | None = None


def saved() -> dict:
    try:
        value = json.loads(state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def save(selection: Selection) -> None:
    state = saved()
    state.update(backbone=selection.backbone, backend=selection.backend)
    if selection.backend == "gguf" and selection.model_path:
        state.setdefault("gguf_paths", {})[selection.backbone] = selection.model_path
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _gguf_path(backbone: str, state: dict) -> str | None:
    remembered = (state.get("gguf_paths") or {}).get(backbone)
    if isinstance(remembered, str) and Path(remembered).expanduser().is_file():
        return remembered
    found = find_gguf(backbone)
    return str(found) if found else None


def resolve(backbone: str | None = None, backend: str | None = None,
            model_path: str | None = None) -> Selection:
    """Command-line values win over the saved choice, which wins over 9B on MLX."""
    state = saved()
    if backbone is None:
        backbone = state.get("backbone") if state.get("backbone") in backbone_ids() else DEFAULT_BACKBONE
    if backend is None:
        backend = state.get("backend") if state.get("backend") in BACKENDS else "mlx"
    if backend == "gguf" and model_path is None:
        model_path = _gguf_path(backbone, state)
        if model_path is None:
            raise ValueError(
                f"No GGUF file found for {display_name(backbone)}; pass --model-path or place "
                f"it under {data_root() / 'base' / backbone / 'gguf'}"
            )
    return Selection(backbone, backend, model_path)


def installed(selection: Selection) -> bool:
    """Whether the selected base model is on this computer and complete."""
    if selection.backend == "gguf":
        # Loading checks the GGUF metadata; reading it here would take seconds per file.
        path = Path(selection.model_path).expanduser() if selection.model_path else None
        tokenizer = local_model_dir(selection.backbone)
        return (path is not None and path.suffix.lower() == ".gguf" and path.is_file()
                and all((tokenizer / name).is_file() for name in _TOKENIZER_FILES))
    try:
        model_id = get_backbone(selection.backbone, validate_checkpoint=True).model_id
    except (OSError, ValueError):
        return False
    # A Hugging Face repository name would be downloaded on the first request instead.
    return Path(model_id).is_dir()


def _weights_bytes(backbone: str, backend: str) -> int:
    try:
        return int(download_source(backbone, backend).get("weights_bytes") or 0)
    except ValueError:
        return 0


def memory_estimate_gb(backbone: str, backend: str, gguf_path: str | None = None) -> float | None:
    """Approximate memory the server needs with this backbone loaded."""
    if backend == "gguf":
        path = Path(gguf_path).expanduser() if gguf_path else None
        # Before a download, the published size of the pinned file stands in for it.
        weights = path.stat().st_size if path is not None and path.is_file() else _weights_bytes(backbone, "gguf")
        if not weights:
            return None
        # The KV cache for the whole context is reserved at load, plus compute buffers.
        extra = gguf_context(backbone) * kv_bytes_per_token(backbone) + _GIB // 2
    else:
        weights = (sum(file.stat().st_size for file in local_model_dir(backbone).glob("*.safetensors"))
                   or _weights_bytes(backbone, "mlx"))
        if not weights:
            return None
        from . import llm
        # Bounded MLX buffer, prompt and router caches, plus a working KV cache.
        extra = llm.MLX_CACHE_LIMIT_BYTES + llm.PROMPT_CACHE_BYTES + llm._ROUTER_CACHE_BYTES + _GIB // 2
    return round((weights + extra) / _GIB, 1)


def _windows_memory() -> tuple[int, int]:
    """Total and available physical memory from GlobalMemoryStatusEx (Windows has no sysconf)."""
    import ctypes

    class Status(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    *((name, ctypes.c_ulonglong) for name in (
                        "ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                        "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual"))]

    status = Status(dwLength=ctypes.sizeof(Status))
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return int(status.ullTotalPhys), int(status.ullAvailPhys)


def system_memory() -> dict:
    """Total RAM and memory other programs could give up now, in GiB."""
    if sys.platform == "win32":
        total, available = _windows_memory()
        return {"total_gb": round(total / _GIB, 1), "available_gb": round(available / _GIB, 1)}
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    available = None
    try:
        if sys.platform == "darwin":
            report = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
            page = int(re.search(r"page size of (\d+) bytes", report).group(1))
            pages = dict(re.findall(r"^Pages (free|inactive|speculative|purgeable):\s+(\d+)\.", report, re.M))
            available = page * sum(int(value) for value in pages.values())
        elif Path("/proc/meminfo").is_file():
            match = re.search(r"MemAvailable:\s+(\d+) kB", Path("/proc/meminfo").read_text())
            available = int(match.group(1)) * 1024 if match else None
    except (OSError, ValueError, AttributeError, subprocess.SubprocessError):
        available = None
    return {"total_gb": round(total / _GIB, 1),
            "available_gb": round(available / _GIB, 1) if available is not None else None}


def _command_output(*command: str) -> str:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def hardware() -> dict:
    """The processor, memory, GPU, free disk space, and installed backends of this computer."""
    apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"
    chip = (_command_output("sysctl", "-n", "machdep.cpu.brand_string") if sys.platform == "darwin" else "")
    if not chip and Path("/proc/cpuinfo").is_file():
        match = re.search(r"^model name\s*:\s*(.+)$", Path("/proc/cpuinfo").read_text(errors="replace"), re.M)
        chip = match.group(1).strip() if match else ""
    gpu = None
    if apple_silicon:
        gpu = {"name": "Apple GPU (shares system memory)", "memory_gb": None}
    elif shutil.which("nvidia-smi"):
        row = _command_output("nvidia-smi", "--query-gpu=name,memory.total",
                              "--format=csv,noheader,nounits").splitlines()
        if row and "," in row[0]:
            name, memory = (part.strip() for part in row[0].split(",", 1))
            gpu = {"name": name, "memory_gb": round(int(memory) / 1024, 1) if memory.isdigit() else None}
    target = data_root() / "base"
    while not target.exists() and target.parent != target:
        # A fresh install has no data folder yet; measure the disk it will be created on.
        target = target.parent
    disk = shutil.disk_usage(target)
    return {"platform": {"darwin": "macOS", "win32": "Windows"}.get(sys.platform, sys.platform.capitalize()),
            "machine": platform.machine(), "chip": chip or platform.processor() or None,
            "apple_silicon": apple_silicon, "gpu": gpu, "memory": system_memory(),
            "disk_free_gb": round(disk.free / _GIB, 1),
            "backends_installed": {"mlx": importlib.util.find_spec("mlx_lm") is not None,
                                   "gguf": importlib.util.find_spec("llama_cpp") is not None}}


def fit(need_gb: float | None, budget_gb: float | None) -> str | None:
    """How a model's memory need compares with the memory that can hold it."""
    if not need_gb or not budget_gb:
        return None
    # Leave room for the system and the browser; models above 80% push the computer into swap.
    return "comfortable" if need_gb <= 0.55 * budget_gb else "tight" if need_gb <= 0.8 * budget_gb else "too_large"


def memory_budget_gb(machine: dict, backend: str) -> float | None:
    """Memory that holds the model: system memory, or the NVIDIA GPU's own for GGUF."""
    gpu = machine.get("gpu") or {}
    if backend == "gguf" and gpu.get("memory_gb"):
        return gpu["memory_gb"]
    return machine["memory"].get("total_gb")


def recommend(machine: dict) -> dict:
    """The backend and the largest base model this computer runs comfortably."""
    installed_backends = machine["backends_installed"]
    if machine["apple_silicon"] and (installed_backends["mlx"] or not installed_backends["gguf"]):
        backend, why = "mlx", "MLX runs natively on the Apple GPU"
    else:
        backend, why = "gguf", "llama.cpp (GGUF) runs on this processor and GPU"
    budget = memory_budget_gb(machine, backend)
    large = DEFAULT_BACKBONE
    need = memory_estimate_gb(large, backend)
    backbone = large if fit(need, budget) == "comfortable" else "qwen35-4b"
    memory = f"{budget:g} GB of {'GPU ' if backend == 'gguf' and (machine.get('gpu') or {}).get('memory_gb') else ''}memory"
    size = (f"{display_name(backbone)} needs about {memory_estimate_gb(backbone, backend)} GB"
            if backbone == large else f"{display_name(large)} needs about {need} GB, too much to run "
            f"comfortably, so {display_name(backbone)} is the better fit")
    return {"backbone": backbone, "backend": backend,
            "reason": f"{why}. With {memory}, {size}." if budget else f"{why}."}


def options() -> list[dict]:
    """Every backbone and backend, with whether this machine can run it now."""
    state = saved()
    mlx_installed = importlib.util.find_spec("mlx_lm") is not None
    gguf_installed = importlib.util.find_spec("llama_cpp") is not None
    rows = []
    for backbone in backbone_ids():
        local = local_model_dir(backbone)
        mlx_local = False
        try:
            model_id = get_backbone(backbone, validate_checkpoint=True).model_id
            mlx_local = Path(model_id).is_dir()
            mlx_ready = True
            mlx_detail = "Local checkpoint" if mlx_local else f"Downloads {model_id} on first start"
        except (OSError, ValueError):
            mlx_ready = False
            mlx_detail = f"MLX checkpoint missing in {local}"
        if not mlx_installed:
            mlx_ready, mlx_detail = False, "mlx-lm is not installed"
        gguf_file = _gguf_path(backbone, state)
        tokenizer_ready = all((local / name).is_file() for name in _TOKENIZER_FILES)
        if not gguf_installed:
            gguf_ready, gguf_detail = False, "llama-cpp-python is not installed"
        elif gguf_file is None:
            gguf_ready, gguf_detail = False, "GGUF file not found; enter its path"
        elif not tokenizer_ready:
            gguf_ready, gguf_detail = False, f"Tokenizer files missing in {local}"
        else:
            gguf_ready, gguf_detail = True, gguf_file
        rows.append({
            "id": backbone, "name": display_name(backbone), "default": backbone == DEFAULT_BACKBONE,
            "backends": {
                "mlx": {"available": mlx_ready, "detail": mlx_detail,
                        "installed": mlx_installed and mlx_local,
                        "download_gb": round(_weights_bytes(backbone, "mlx") / _GIB, 1),
                        "memory_gb": memory_estimate_gb(backbone, "mlx")},
                "gguf": {"available": gguf_ready, "detail": gguf_detail, "path": gguf_file,
                         # Reading a multi-gigabyte file's metadata is left to the switch itself.
                         "installed": gguf_ready,
                         "download_gb": round(_weights_bytes(backbone, "gguf") / _GIB, 1),
                         "memory_gb": memory_estimate_gb(backbone, "gguf", gguf_file)},
            },
        })
    return rows


def overview() -> dict:
    """Options with how well each fits this computer, the hardware, and a recommendation."""
    machine = hardware()
    rows = options()
    for row in rows:
        for backend, status in row["backends"].items():
            status["fit"] = fit(status["memory_gb"], memory_budget_gb(machine, backend))
            status["enough_disk"] = status["installed"] or machine["disk_free_gb"] >= status["download_gb"] + 1
    return {"hardware": machine, "recommended": recommend(machine), "options": rows}


def command(argv: list[str], selection: Selection) -> list[str]:
    """Arguments for core.main with this process's options and the new selection."""
    kept, skip = [], False
    for arg in argv:
        if skip:
            skip = False
            continue
        name = arg.split("=", 1)[0]
        if name in _RUNTIME_FLAGS:
            skip = "=" not in arg
            continue
        kept.append(arg)
    kept += ["--backbone", selection.backbone, "--backend", selection.backend]
    if selection.backend == "gguf" and selection.model_path:
        kept += ["--model-path", selection.model_path]
    return kept


def restart(argv: list[str], selection: Selection) -> None:
    """Replace this process with a server for the selected backbone."""
    env = dict(os.environ)
    env[ACTIVE_BACKBONE_ENV] = selection.backbone
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(_PROJECT_ROOT), env.get("PYTHONPATH"))))
    sys.stdout.flush()
    sys.stderr.flush()
    os.execve(sys.executable, [sys.executable, "-m", "core.main", *command(argv, selection)], env)
