"""Hot-swappable PEFT LoRA adapters on one resident MLX backbone.

The backbone is never reloaded for an adapter switch.  Prepared LoRA wrappers
retain references to its original quantized layers, so detach restores the
exact same base modules.  Generation holds the manager lock while an adapter is
active because MLX model modules are process-global mutable state.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import os
import re
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .download_progress import download_file, ProgressCallback
from .file_lock import exclusive
try:
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    from mlx_lm.tuner.lora import LoRALinear
    from mlx_lm.tuner.utils import load_adapters, remove_lora_layers
    from safetensors import safe_open
    _MLX_IMPORT_ERROR: ImportError | None = None
except ImportError as error:  # GGUF runs on Linux without the Apple MLX stack.
    mx = None  # type: ignore[assignment]
    tree_unflatten = None  # type: ignore[assignment]
    LoRALinear = Any  # type: ignore[misc,assignment]
    load_adapters = None  # type: ignore[assignment]
    remove_lora_layers = None  # type: ignore[assignment]
    safe_open = None  # type: ignore[assignment]
    _MLX_IMPORT_ERROR = error
from core_models import active_backbone, adapter_base_names, expert_library, get_backbone, user_dir


@dataclass(frozen=True)
class AdapterSpec:
    repo_id: str
    revision: str
    base_model_id: str
    accepted_base_names: tuple[str, ...] = ("Qwen/Qwen3.5-9B",)
    gguf_file: str | None = None
    gguf_repo_id: str | None = None
    gguf_revision: str | None = None
    source_subdir: str | None = None
    source_config_filename: str = "adapter_config.json"
    source_filename: str = "adapter_model.safetensors"


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_BASE_ROOT = _PROJECT_ROOT / "base"


def _resolve_backbone_path(path: str) -> str:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = _PROJECT_ROOT / value
    return str(value.resolve())


def _load_adapter_manifest(
    backbone_id: str, model_id: str, *, validate_checkpoint_path: bool = True,
) -> tuple[dict[str, AdapterSpec], dict[str, str]]:
    path = _BASE_ROOT / backbone_id / "adapters.json"
    if not path.is_file():
        raise FileNotFoundError(f"No LoRA manifest for backbone {backbone_id!r}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("expert_packages"):
        from .expert_packages import load_packages
        value["adapters"] = {key: item["adapter"] for key, item in load_packages(backbone_id).items()}
    entries = value.get("adapters")
    if not isinstance(entries, dict):
        raise ValueError(f"Invalid LoRA manifest: {path}")
    expected_model_id = value.get("backbone_model_id")
    accepted_model_ids = {expected_model_id} if isinstance(expected_model_id, str) else set()
    local_model_path = value.get("local_backbone_model_path")
    if isinstance(local_model_path, str):
        accepted_model_ids.add(_resolve_backbone_path(local_model_path))
    if validate_checkpoint_path and entries and accepted_model_ids and model_id not in accepted_model_ids:
        raise ValueError(f"LoRA manifest {path} targets a different backbone checkpoint")
    specs: dict[str, AdapterSpec] = {}
    display_ids: dict[str, str] = {}
    allowed_lineages = adapter_base_names(backbone_id)
    for expert_id, row in entries.items():
        if (
            not isinstance(expert_id, str)
            or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", expert_id) is None
            or not isinstance(row, dict)
        ):
            raise ValueError(f"Invalid LoRA entry in {path}")
        repo_id = row.get("repo_id")
        revision = row.get("revision")
        accepted = row.get("accepted_base_names")
        display_id = row.get("display_id")
        gguf_file = row.get("gguf_file")
        gguf_repo_id = row.get("gguf_repo_id")
        gguf_revision = row.get("gguf_revision")
        source_subdir = row.get("source_subdir")
        if (
            not isinstance(repo_id, str) or not repo_id
            or not isinstance(revision, str) or re.fullmatch(r"[a-f0-9]{40}", revision) is None
            or not isinstance(accepted, list) or not accepted
            or any(not isinstance(name, str) or not name for name in accepted)
            or not isinstance(display_id, str) or not display_id
        ):
            raise ValueError(f"Invalid LoRA metadata for {expert_id!r} in {path}")
        if not set(accepted).issubset(allowed_lineages):
            raise ValueError(f"LoRA {expert_id!r} declares an incompatible source backbone for {backbone_id}")
        if gguf_file is not None and (not isinstance(gguf_file, str) or Path(gguf_file).name != gguf_file):
            raise ValueError(f"Invalid GGUF adapter filename for {expert_id!r} in {path}")
        if gguf_repo_id is not None and (not isinstance(gguf_repo_id, str) or not gguf_repo_id):
            raise ValueError(f"Invalid GGUF adapter repository for {expert_id!r} in {path}")
        if gguf_revision is not None and (
            not isinstance(gguf_revision, str) or re.fullmatch(r"[a-f0-9]{40}", gguf_revision) is None
        ):
            raise ValueError(f"Invalid GGUF adapter revision for {expert_id!r} in {path}")
        if source_subdir is not None and (
            not isinstance(source_subdir, str) or Path(source_subdir).is_absolute()
            or ".." in Path(source_subdir).parts
        ):
            raise ValueError(f"Invalid adapter source directory for {expert_id!r} in {path}")
        specs[expert_id] = AdapterSpec(
            repo_id, revision, model_id, tuple(accepted), gguf_file,
            gguf_repo_id, gguf_revision, source_subdir,
            row.get("source_config_filename", f"{source_subdir}/adapter_config.json" if source_subdir else "adapter_config.json"),
            row.get("source_filename", f"{source_subdir}/adapter_model.safetensors" if source_subdir else "adapter_model.safetensors"),
        )
        display_ids[expert_id] = display_id
    return specs, display_ids


ADAPTERS: dict[str, AdapterSpec] = {}
ADAPTER_DISPLAY_IDS: dict[str, str] = {}
_GGUF_BACKBONE_ID: str | None = None
_GGUF_ADAPTER_LOCK = threading.RLock()


def select_adapter_base(backbone_id: str, model_id: str) -> None:
    """Switch registry before model loading; no cross-backbone adapter reuse."""
    global _GGUF_BACKBONE_ID
    if LORA_MANAGER.active_adapter or LORA_MANAGER.cached_adapters:
        raise RuntimeError("Cannot change backbone after preparing LoRA adapters")
    specs, display_ids = _load_adapter_manifest(backbone_id, model_id)
    ADAPTERS.clear()
    ADAPTERS.update(specs)
    ADAPTER_DISPLAY_IDS.clear()
    ADAPTER_DISPLAY_IDS.update(display_ids)
    _GGUF_BACKBONE_ID = None


def refresh_package_adapters(backbone_id: str, model_id: str) -> None:
    """Add or drop adapters after the user added or removed one, without a restart."""
    specs, display_ids = _load_adapter_manifest(backbone_id, model_id, validate_checkpoint_path=False)
    if _GGUF_BACKBONE_ID is not None:
        specs = {name: spec for name, spec in specs.items() if spec.gguf_file is not None}
    for name in [name for name in ADAPTERS if name not in specs]:
        ADAPTERS.pop(name, None)
        ADAPTER_DISPLAY_IDS.pop(name, None)
    ADAPTERS.update(specs)
    ADAPTER_DISPLAY_IDS.update({name: value for name, value in display_ids.items() if name in specs})


def select_gguf_adapter_base(backbone_id: str) -> None:
    """Load only the pinned GGUF adapter registry for a GGUF backbone."""
    global _GGUF_BACKBONE_ID
    if LORA_MANAGER.active_adapter or LORA_MANAGER.cached_adapters:
        raise RuntimeError("Cannot change backbone after preparing LoRA adapters")
    specs, display_ids = _load_adapter_manifest(backbone_id, "", validate_checkpoint_path=False)
    specs = {name: spec for name, spec in specs.items() if spec.gguf_file is not None}
    display_ids = {name: value for name, value in display_ids.items() if name in specs}
    ADAPTERS.clear()
    ADAPTERS.update(specs)
    ADAPTER_DISPLAY_IDS.clear()
    ADAPTER_DISPLAY_IDS.update(display_ids)
    _GGUF_BACKBONE_ID = backbone_id


def gguf_adapter_path(name: str) -> Path:
    """Resolve a prepared adapter from the configured GGUF LoRA cache."""
    if name not in ADAPTERS:
        raise ValueError(f"Unknown GGUF LoRA adapter: {name}")
    spec = ADAPTERS[name]
    if not spec.gguf_file:
        raise ValueError(f"GGUF LoRA adapter {name!r} has no prepared GGUF filename")
    root = Path(os.environ.get(
        "GHOSTSWITCH_GGUF_LORA_DIR",
        str(user_dir(Path.home() / ".cache") / "gguf-lora" / (expert_library(_GGUF_BACKBONE_ID) if _GGUF_BACKBONE_ID else "unknown")),
    )).expanduser()
    return root / spec.gguf_file


@contextlib.contextmanager
def activate_gguf_adapter(model: Any, name: str | None) -> Iterator[dict[str, int | str]]:
    """Attach one GGUF LoRA to a shared llama.cpp context for a single call."""
    with _GGUF_ADAPTER_LOCK:
        if getattr(model, "_ghostswitch_adapter_state_invalid", False):
            raise RuntimeError("GGUF adapter state is invalid after a detach failure; restart the model before generation")
        if name is None:
            yield {}
            return
        path = gguf_adapter_path(name)
        if not path.is_file():
            raise FileNotFoundError(
                f"Prepared GGUF adapter not found: {path}. "
                "Run scripts/prepare_gguf_adapters.py before using this expert."
            )
        try:
            from llama_cpp import llama_cpp
            import ctypes
        except ImportError as error:
            raise RuntimeError("GGUF LoRA support needs llama-cpp-python") from error
        handle = llama_cpp.llama_adapter_lora_init(model._model.model, str(path).encode("utf-8"))
        if handle is None:
            raise RuntimeError(f"llama.cpp could not load GGUF LoRA adapter: {path}")
        adapters = (llama_cpp.llama_adapter_lora_p_ctypes * 1)(handle)
        scales = (ctypes.c_float * 1)(1.0)
        previous_cache = getattr(model, 'cache', None)
        try:
            # Never evaluate or restore base-weight KV/recurrent state under LoRA.
            if hasattr(model, 'set_cache'):
                model.set_cache(None)
            if hasattr(model, 'reset'):
                model.reset()
            if llama_cpp.llama_set_adapters_lora(model._ctx.ctx, adapters, 1, scales) != 0:
                raise RuntimeError(f"llama.cpp could not attach GGUF LoRA adapter: {path}")
            yield {"adapter_weight_bytes": path.stat().st_size}
        finally:
            clear_result = -1
            try:
                clear_result = llama_cpp.llama_set_adapters_lora(model._ctx.ctx, None, 0, None)
            finally:
                if clear_result != 0:
                    model._ghostswitch_adapter_state_invalid = True
                llama_cpp.llama_adapter_lora_free(handle)
                # State produced with expert weights must not leak into base calls.
                if hasattr(model, 'reset'):
                    model.reset()
                if hasattr(model, 'set_cache'):
                    model.set_cache(previous_cache)
            if clear_result != 0:
                raise RuntimeError(f"llama.cpp could not detach GGUF LoRA adapter: {name}")


def display_adapter_id(expert_id: str) -> str:
    """Return the stable user-facing LoRA id shown in CLI results."""
    return ADAPTER_DISPLAY_IDS.get(expert_id, expert_id.replace("_", "-"))

_PEFT_KEY = re.compile(
    r"^base_model\.model\.model\.(?:language_model\.)?layers\.(\d+)\.(.+)\.lora_([AB])\.weight$"
)
_SAFE_DTYPES = {"F32", "F16", "BF16"}


def declared_module(key: str, target_names: set[str]) -> bool:
    """Whether a PEFT tensor's module is in target_modules, as PEFT itself matches a list:
    the module's full name or a dotted suffix of it, such as "o_proj" or
    "model.language_model.layers.31.self_attn.o_proj"."""
    module = key.removeprefix("base_model.model.").rsplit(".lora_", 1)[0]
    return any(module == name or module.endswith(f".{name}") for name in target_names)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_CACHE_ROOT = _PROJECT_ROOT / "base" / active_backbone() / "lora-cache"
_DEFAULT_CACHE_ROOT = (
    _LOCAL_CACHE_ROOT
    if _LOCAL_CACHE_ROOT.is_dir()
    else user_dir(Path.home() / ".cache") / "lora"
)
_CACHE_ROOT = Path(os.environ.get(
    "GHOSTSWITCH_LORA_CACHE",
    str(_DEFAULT_CACHE_ROOT),
))


def native_adapter_path(name: str) -> Path:
    spec = ADAPTERS[name]
    return _CACHE_ROOT / name / spec.revision


def ensure_native_adapter(name: str, progress: ProgressCallback | None = None) -> tuple[Path, float, float]:
    """Convert once per revision, including across concurrent processes."""
    if mx is None or safe_open is None:
        raise RuntimeError("LoRA adapters require the MLX runtime") from _MLX_IMPORT_ERROR
    destination = native_adapter_path(name)
    destination.mkdir(parents=True, exist_ok=True)
    with exclusive(destination / ".conversion.lock"):
        return _ensure_native_adapter_locked(name, progress)


def _ensure_native_adapter_locked(name: str, progress: ProgressCallback | None = None) -> tuple[Path, float, float]:
    spec = ADAPTERS[name]
    destination = native_adapter_path(name)
    config_path = destination / "adapter_config.json"
    weights_path = destination / "adapters.safetensors"
    manifest_path = destination / "source_manifest.json"
    if config_path.is_file() and weights_path.is_file():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = {}
        if not isinstance(existing, dict):
            existing = {}
        if (
            existing.get("source_repo") == spec.repo_id
            and existing.get("source_revision") == spec.revision
            and existing.get("runtime_format_version") == 2
            and existing.get("source_config_filename", "adapter_config.json") == spec.source_config_filename
            and existing.get("source_filename", "adapter_model.safetensors") == spec.source_filename
            and manifest_path.is_file()
        ):
            return destination, 0.0, 0.0

    started = time.perf_counter_ns()
    # Fetch only these explicitly allowlisted data files from the pinned commit.
    source_paths = {
        logical: Path(download_file(spec.repo_id, filename, revision=spec.revision, progress=progress))
        for logical, filename in (("adapter_config.json", spec.source_config_filename),
                                  ("adapter_model.safetensors", spec.source_filename))
    }
    if progress:
        progress({"phase": "checking"})
    source_config = source_paths["adapter_config.json"]
    source_weights = source_paths["adapter_model.safetensors"]
    file_manifest = []
    for filename, path in source_paths.items():
        digest = hashlib.sha256()
        with path.open("rb") as source_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(chunk)
        file_manifest.append({"name": filename, "size": path.stat().st_size, "sha256": digest.hexdigest()})

    # Validate the safetensors envelope and tensor metadata with its safe reader
    # before MLX loads any values. No model-repository code is downloaded or run.
    with source_weights.open("rb") as weight_file:
        prefix = weight_file.read(8)
        if len(prefix) != 8:
            raise ValueError("Safetensors file is truncated before its header")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size <= 0 or header_size > 100 * 1024 * 1024:
            raise ValueError("Safetensors header size is invalid")
    with safe_open(str(source_weights), framework="numpy") as safe_file:
        source_keys = list(safe_file.keys())
        if not source_keys:
            raise ValueError("Adapter safetensors has no tensors")
        for key in source_keys:
            if _PEFT_KEY.fullmatch(key) is None:
                raise ValueError(f"Unsupported PEFT tensor: {key}")
            tensor_slice = safe_file.get_slice(key)
            shape = tensor_slice.get_shape()
            dtype = str(tensor_slice.get_dtype())
            if len(shape) != 2 or any(int(size) <= 0 for size in shape):
                raise ValueError(f"LoRA tensor has an invalid shape: {key} {shape}")
            if dtype not in _SAFE_DTYPES:
                raise ValueError(f"LoRA tensor dtype is not supported: {key} {dtype}")

    fetch_ms = (time.perf_counter_ns() - started) / 1_000_000
    config = json.loads(source_config.read_text(encoding="utf-8"))
    if config.get("peft_type") != "LORA" or config.get("use_dora") or config.get("rank_pattern"):
        raise ValueError("Only uniform PEFT LoRA adapters are supported")
    if config.get("base_model_name_or_path") not in spec.accepted_base_names:
        raise ValueError("Adapter was trained for a different backbone")
    rank = int(config["r"])
    if rank <= 0:
        raise ValueError("LoRA rank must be positive")
    target_modules = config["target_modules"]
    target_names = set(target_modules) if isinstance(target_modules, list) else None
    target_pattern = re.compile(target_modules) if isinstance(target_modules, str) else None
    if progress:
        progress({"phase": "converting"})
    source = mx.load(str(source_weights))
    native: dict[str, Any] = {}
    module_sides: dict[str, set[str]] = {}
    max_layer = -1
    for key, value in source.items():
        match = _PEFT_KEY.fullmatch(key)
        if match is None:
            raise ValueError(f"Unsupported PEFT tensor: {key}")
        layer_index, suffix, side = match.groups()
        normalized_module = f"model.layers.{layer_index}.{suffix}"
        if target_names is not None and not declared_module(key, target_names):
            raise ValueError(f"Tensor targets an undeclared module: {key}")
        if target_pattern is not None and not target_pattern.search(f"language_model.{normalized_module}"):
            raise ValueError(f"Tensor targets an undeclared module: {key}")
        if target_names is None and target_pattern is None:
            raise ValueError("Unsupported target_modules format")
        max_layer = max(max_layer, int(layer_index))
        module_name = f"language_model.model.layers.{layer_index}.{suffix}"
        module_sides.setdefault(module_name, set()).add(side)
        native[f"{module_name}.lora_{side.lower()}"] = value.T
    if not native or any(sides != {"A", "B"} for sides in module_sides.values()):
        raise ValueError("Adapter is empty or is missing paired LoRA tensors")
    for module_name in module_sides:
        a = native[f"{module_name}.lora_a"]
        b = native[f"{module_name}.lora_b"]
        if int(a.shape[1]) != rank or int(b.shape[0]) != rank:
            raise ValueError(f"Adapter tensor rank does not match adapter_config.json: {module_name}")
    mlx_config = {
        "fine_tune_type": "lora",
        "runtime_format_version": 2,
        "num_layers": max_layer + 1,
        "lora_parameters": {
            "keys": sorted(module_sides),
            "rank": rank,
            "scale": float(config["lora_alpha"]) / rank,
            "dropout": float(config.get("lora_dropout", 0.0)),
        },
        "source_repo": spec.repo_id,
        "source_revision": spec.revision,
        "source_config_filename": spec.source_config_filename,
        "source_filename": spec.source_filename,
        "base_model_id": spec.base_model_id,
        "accepted_base_name": config["base_model_name_or_path"],
        "rank": rank,
        "target_modules": target_modules,
    }
    destination.mkdir(parents=True, exist_ok=True)
    temporary_weights = destination / "adapters.partial.safetensors"
    temporary_config = destination / "adapter_config.json.partial"
    if progress:
        progress({"phase": "saving"})
    mx.save_safetensors(str(temporary_weights), native)
    os.replace(temporary_weights, weights_path)
    temporary_config.write_text(json.dumps(mlx_config, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_config, config_path)
    temporary_manifest = destination / "source_manifest.json.partial"
    temporary_manifest.write_text(json.dumps({
        "repo": spec.repo_id,
        "revision": spec.revision,
        "files": file_manifest,
        "tensor_count": len(source_keys),
        "tensor_names": source_keys,
    }, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    convert_ms = (time.perf_counter_ns() - started) / 1_000_000 - fetch_ms
    return destination, fetch_ms, convert_ms


@dataclass
class PreparedAdapter:
    base_modules: dict[str, Any]
    lora_modules: dict[str, LoRALinear]
    weight_bytes: int
    parameter_count: int


def _memory_budget_bytes() -> int:
    """GHOSTSWITCH_LORA_MEMORY_MB as bytes; 0 keeps only the adapter in use."""
    value = os.environ.get("GHOSTSWITCH_LORA_MEMORY_MB", "")
    return (int(value) if value.isdigit() else 512) * 1024 * 1024


class LoraManager:
    """Prepared adapters stay in memory so that switching back to one is instant.

    Without a limit every expert a session touches would stay resident: on a 16 GB Mac,
    where the 9B base model already takes about 8 GB, a few dozen 100-300 MB adapters
    would push it into swap. The least recently used adapters beyond the memory budget
    are dropped and prepared again from the disk cache when they are needed (about a
    second). The adapter in use is always kept, so the expert library can grow without
    the memory need growing with it.
    """

    def __init__(self, memory_budget_bytes: int | None = None) -> None:
        self._lock = threading.RLock()
        self._base_model: Any | None = None
        self._prepared: dict[str, PreparedAdapter] = {}  # least recently used first
        self._active: str | None = None
        self._memory_budget = _memory_budget_bytes() if memory_budget_bytes is None else memory_budget_bytes

    @property
    def active_adapter(self) -> str | None:
        return self._active

    @property
    def cached_adapters(self) -> tuple[str, ...]:
        return tuple(self._prepared)

    def _bind(self, model: Any) -> None:
        if self._base_model is None:
            self._base_model = model
        elif model is not self._base_model:
            raise ValueError("LoRA manager is bound to a different backbone instance")

    def _prepare(self, name: str, model: Any, base_model_id: str) -> tuple[PreparedAdapter, float, float, float]:
        if mx is None or load_adapters is None or remove_lora_layers is None or tree_unflatten is None:
            raise RuntimeError("LoRA adapters require the MLX runtime") from _MLX_IMPORT_ERROR
        spec = ADAPTERS[name]
        if base_model_id != spec.base_model_id:
            raise ValueError(f"Adapter {name} requires {spec.base_model_id}; received {base_model_id}")
        destination, fetch_ms, convert_ms = ensure_native_adapter(name)
        started = time.perf_counter_ns()
        config = json.loads((destination / "adapter_config.json").read_text(encoding="utf-8"))
        expected_modules = set(config["lora_parameters"]["keys"])
        base_lookup = dict(model.named_modules())
        base_modules: dict[str, Any] = {}
        for module_name in expected_modules:
            if module_name not in base_lookup:
                raise ValueError(f"Missing adapter tensor or backbone module: {module_name}")
            base_modules[module_name] = base_lookup[module_name]
        try:
            load_adapters(model, str(destination))
            loaded = dict(model.named_modules())
            lora_modules = {module_name: loaded[module_name] for module_name in expected_modules}
            if any(not isinstance(module, LoRALinear) for module in lora_modules.values()):
                raise ValueError("MLX did not install every declared LoRA module")
            mx.eval(*(value for wrapper in lora_modules.values() for value in (wrapper.lora_a, wrapper.lora_b)))
        finally:
            remove_lora_layers(model)
        if any(dict(model.named_modules())[key] is not value for key, value in base_modules.items()):
            raise RuntimeError("Adapter preparation did not restore backbone modules")
        prepare_ms = (time.perf_counter_ns() - started) / 1_000_000
        parameter_count = sum(int(wrapper.lora_a.size + wrapper.lora_b.size) for wrapper in lora_modules.values())
        return PreparedAdapter(base_modules, lora_modules, (destination / "adapters.safetensors").stat().st_size, parameter_count), fetch_ms, convert_ms, prepare_ms

    @contextlib.contextmanager
    def activate(self, name: str, model: Any, base_model_id: str) -> Iterator[dict[str, object]]:
        if name not in ADAPTERS:
            raise ValueError(f"Unknown LoRA expert: {name}")
        if base_model_id != ADAPTERS[name].base_model_id:
            raise ValueError(f"Adapter {name} requires {ADAPTERS[name].base_model_id}; received {base_model_id}")
        with self._lock:
            self._bind(model)
            if self._active is not None:
                raise RuntimeError("An adapter is already active")
            cache_hit = name in self._prepared
            fetch_ms = convert_ms = prepare_ms = 0.0
            evicted: list[str] = []
            if cache_hit:
                self._prepared[name] = self._prepared.pop(name)
            else:
                prepared, fetch_ms, convert_ms, prepare_ms = self._prepare(name, model, base_model_id)
                self._prepared[name] = prepared
                evicted = self._evict(keep=name)
            prepared = self._prepared[name]
            started = time.perf_counter_ns()
            try:
                model.update_modules(tree_unflatten(list(prepared.lora_modules.items())))
            except Exception:
                model.update_modules(tree_unflatten(list(prepared.base_modules.items())))
                raise
            self._active = name
            timings: dict[str, object] = {
                "adapter": name,
                "cache_hit": cache_hit,
                "fetch_ms": round(fetch_ms, 3),
                "convert_ms": round(convert_ms, 3),
                "prepare_ms": round(prepare_ms, 3),
                "attach_ms": round((time.perf_counter_ns() - started) / 1_000_000, 3),
                "detach_ms": 0.0,
                "adapter_weight_bytes": prepared.weight_bytes,
                "adapter_params": prepared.parameter_count,
                "adapted_modules": len(prepared.lora_modules),
                "evicted_adapters": evicted,
            }
            try:
                yield timings
            finally:
                started = time.perf_counter_ns()
                model.update_modules(tree_unflatten(list(prepared.base_modules.items())))
                self._active = None
                timings["detach_ms"] = round((time.perf_counter_ns() - started) / 1_000_000, 3)

    def _evict(self, keep: str) -> list[str]:
        """Drop the least recently used adapters until the rest fit the memory budget."""
        evicted: list[str] = []
        resident = sum(item.weight_bytes for item in self._prepared.values())
        for name in list(self._prepared):
            if resident <= self._memory_budget:
                break
            if name in (keep, self._active):
                continue
            resident -= self._prepared.pop(name).weight_bytes
            evicted.append(name)
        if evicted:
            self._release_memory()
        return evicted

    def _release_memory(self) -> None:
        gc.collect()
        if mx is not None:
            mx.clear_cache()

    def unload(self, name: str) -> bool:
        with self._lock:
            if self._active == name:
                raise RuntimeError("Cannot unload an active adapter")
            if self._prepared.pop(name, None) is None:
                return False
            self._release_memory()
            return True


LORA_MANAGER = LoraManager()
select_adapter_base(active_backbone(), get_backbone(active_backbone(), validate_checkpoint=False).model_id)
