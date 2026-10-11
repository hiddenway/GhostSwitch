"""Role-aware MLX model runtime with lazy loading and per-call telemetry.

Thinker and router are resident once called, specialist models use a bounded
LRU cache, and ``logic`` aliases Thinker so the shared backbone is loaded once.
"""

from __future__ import annotations

import contextlib
import copy
import ctypes
import gc
import hashlib
import os
import sys
import time
from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterator

try:
    import resource
except ImportError:  # Windows
    resource = None  # type: ignore[assignment]

try:
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.models.cache import LRUPromptCache, make_prompt_cache
    from mlx_lm.sample_utils import make_sampler
    _MLX_IMPORT_ERROR: ImportError | None = None
except ImportError as error:  # GGUF is also supported on Linux without MLX.
    mx = None  # type: ignore[assignment]
    load = stream_generate = make_prompt_cache = make_sampler = None  # type: ignore[assignment]
    LRUPromptCache = None  # type: ignore[assignment,misc]
    _MLX_IMPORT_ERROR = error
from core_models import active_backbone, find_mmproj, get_backbone, gguf_context, mmproj_source
from . import attachments, vision
from .lora_runtime import (
    LORA_MANAGER, activate_gguf_adapter, select_adapter_base,
    select_gguf_adapter_base,
)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    total_params: int
    resident: bool = False
    shared_key: str | None = None
    backend: str = "mlx"
    tokenizer_path: str | None = None


_DEFAULT_PROFILE = get_backbone(active_backbone(), validate_checkpoint=False)
ACTIVE_BACKBONE = _DEFAULT_PROFILE.id
THINKER_MODEL = _DEFAULT_PROFILE.model_id
ROUTER_MODEL = "mlx-community/functiongemma-270m-it-bf16"

MODEL_SPECS: dict[str, ModelSpec] = {
    "thinker": ModelSpec(THINKER_MODEL, _DEFAULT_PROFILE.total_params, resident=True),
    "logic": ModelSpec(THINKER_MODEL, _DEFAULT_PROFILE.total_params, resident=True, shared_key="thinker"),
    "router": ModelSpec(ROUTER_MODEL, 270_000_000, resident=True),
    "facts": ModelSpec(THINKER_MODEL, _DEFAULT_PROFILE.total_params, resident=True, shared_key="thinker"),
}


@dataclass
class LoadedModel:
    key: str
    spec: ModelSpec
    model: Any
    tokenizer: Any


def _memory_counter(name: str) -> int:
    if mx is None:
        return 0
    for owner in (mx, getattr(mx, "metal", None)):
        getter = getattr(owner, name, None)
        if callable(getter):
            try:
                return int(getter())
            except Exception:
                pass
    return 0


def _reset_peak_memory() -> None:
    if mx is None:
        return
    for owner in (mx, getattr(mx, "metal", None)):
        reset = getattr(owner, "reset_peak_memory", None)
        if callable(reset):
            try:
                reset()
                return
            except Exception:
                pass


def _rss_bytes() -> int:
    """Peak resident memory of this process."""
    if resource is not None:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    return _windows_peak_working_set()


def _windows_peak_working_set() -> int:
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    *((name, ctypes.c_size_t) for name in (
                        "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                        "PagefileUsage", "PeakPagefileUsage"))]

    counters = Counters(cb=ctypes.sizeof(Counters))
    try:
        query = ctypes.windll.kernel32.K32GetProcessMemoryInfo
        query.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
        query.restype = wintypes.BOOL
        # A HANDLE of -1 is this process; declared types keep it 64 bits wide.
        if query(wintypes.HANDLE(-1), ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize)
    except (AttributeError, OSError):
        pass
    return 0


def runs_on_gpu() -> bool:
    """Whether generation runs on a GPU: MLX always does; llama.cpp when its build finds one."""
    if MODEL_SPECS["thinker"].backend != "gguf":
        return True
    if os.environ.get("GHOSTSWITCH_GGUF_GPU_LAYERS", "-1") == "0":
        return False
    from . import cuda_libraries

    cuda_libraries.prepare()
    try:
        import llama_cpp
    except Exception:  # A missing or broken build cannot use a GPU either.
        return False
    return bool(llama_cpp.llama_supports_gpu_offload())


class ModelManager:
    """Load resident roles once and keep at most two expert models in an LRU."""

    def __init__(self, max_expert_models: int | None = None) -> None:
        configured = max_expert_models or int(os.environ.get("GHOSTSWITCH_EXPERT_CACHE_SIZE", "2"))
        self.max_expert_models = max(1, min(configured, 2))
        self._models: dict[str, LoadedModel] = {}
        self._expert_lru: OrderedDict[str, None] = OrderedDict()
        self.last_load_latency_ns = 0

    def canonical_role(self, role: str) -> str:
        if role not in MODEL_SPECS:
            raise ValueError(f"Unknown model role: {role}")
        return MODEL_SPECS[role].shared_key or role

    def get_model(self, role: str) -> LoadedModel:
        key = self.canonical_role(role)
        self.last_load_latency_ns = 0
        loaded = self._models.get(key)
        if loaded is not None:
            self._touch_expert(key)
            return loaded
        spec = MODEL_SPECS[key]
        started = time.perf_counter_ns()
        if spec.backend == "gguf":
            from . import cuda_libraries

            cuda_libraries.prepare()
            try:
                from llama_cpp import Llama
            except ImportError as error:
                raise RuntimeError(
                    "GGUF backend needs llama-cpp-python. Install a build for your "
                    "platform (CUDA for NVIDIA, Metal for Apple Silicon, or CPU); "
                    "see the GGUF setup in README.md."
                ) from error
            try:
                from transformers import AutoTokenizer
            except ImportError as error:
                raise RuntimeError(
                    "GGUF backend needs transformers to load the local tokenizer; "
                    "install requirements-gguf.txt."
                ) from error
            tokenizer = AutoTokenizer.from_pretrained(
                spec.tokenizer_path, local_files_only=True, trust_remote_code=False,
            )
            # llama.cpp reserves the whole KV cache for n_ctx up front, so the
            # context is bounded per backbone (GHOSTSWITCH_GGUF_CONTEXT overrides it).
            model = Llama(
                model_path=spec.model_id,
                n_ctx=gguf_context(ACTIVE_BACKBONE),
                n_gpu_layers=int(os.environ.get("GHOSTSWITCH_GGUF_GPU_LAYERS", "-1")),
                # Flash attention avoids a context-sized attention buffer next to the KV cache.
                flash_attn=os.environ.get("GHOSTSWITCH_GGUF_FLASH_ATTN", "1") != "0",
                verbose=False,
            )
        else:
            if load is None:
                raise RuntimeError("MLX backend needs mlx and mlx-lm installed") from _MLX_IMPORT_ERROR
            _limit_mlx_cache()
            model, tokenizer = load(spec.model_id)
        self.last_load_latency_ns = time.perf_counter_ns() - started
        loaded = LoadedModel(key=key, spec=spec, model=model, tokenizer=tokenizer)
        self._models[key] = loaded
        self._touch_expert(key)
        return loaded

    def _touch_expert(self, key: str) -> None:
        if MODEL_SPECS[key].resident:
            return
        self._expert_lru.pop(key, None)
        self._expert_lru[key] = None
        while len(self._expert_lru) > self.max_expert_models:
            evicted, _ = self._expert_lru.popitem(last=False)
            self._unload_key(evicted)

    def _unload_key(self, key: str) -> bool:
        if key not in self._models or MODEL_SPECS[key].resident:
            return False
        del self._models[key]
        self._expert_lru.pop(key, None)
        gc.collect()
        _clear_mlx_cache()
        return True

    def unload_model(self, role: str) -> bool:
        return self._unload_key(self.canonical_role(role))

    def unload_experts(self) -> None:
        for key in list(self._expert_lru):
            self._unload_key(key)

    def loaded_model_ids(self) -> dict[str, str]:
        return {key: loaded.spec.model_id for key, loaded in self._models.items()}


MODEL_MANAGER = ModelManager()


def configure_backbone(
    backbone_id: str, *, model_path: str | None = None, backend: str = "mlx",
) -> str:
    """Choose one backbone at startup, before any model or LoRA is loaded."""
    global ACTIVE_BACKBONE, THINKER_MODEL
    if MODEL_MANAGER.loaded_model_ids():
        raise RuntimeError("Backbone cannot change after a model has been loaded")
    profile = get_backbone(backbone_id, model_path=model_path, backend=backend)
    if profile.backend == "gguf":
        select_gguf_adapter_base(profile.id)
    else:
        select_adapter_base(profile.id, profile.model_id)
    for role in ("thinker", "logic", "facts"):
        MODEL_SPECS[role] = replace(
            MODEL_SPECS[role], model_id=profile.model_id, total_params=profile.total_params,
            backend=profile.backend, tokenizer_path=profile.tokenizer_path,
        )
    ACTIVE_BACKBONE = profile.id
    THINKER_MODEL = profile.model_id
    return profile.model_id
# Node prompts are small and independently keyed; keep recent branches resident
# without allowing the KV cache to grow without bound.
def _env_bytes(name: str, default_mib: int) -> int:
    """A positive MiB budget from the environment, else the default, in bytes."""
    value = os.environ.get(name, "")
    return (int(value) if value.isdigit() and int(value) > 0 else default_mib) * 1024 * 1024


# Budgets sized for a 16 GB Mac, where the model weights already take most of the
# memory. Without a limit MLX keeps freed GPU buffers up to its recommended working
# set (about 12 GB on a 16 GB M1 Pro). Raise these through the environment on larger machines.
MLX_CACHE_LIMIT_BYTES = _env_bytes("GHOSTSWITCH_MLX_CACHE_MB", 1024)
PROMPT_CACHE_BYTES = _env_bytes("GHOSTSWITCH_PROMPT_CACHE_MB", 256)
# An agent step resends its task's whole conversation with one more exchange, so the model
# state after the conversation is kept for the next step. On Qwen3.5 a state takes 32 KB
# per token plus the linear-attention state: about 200 MB after 5,000 tokens.
HISTORY_CACHE_BYTES = _env_bytes("GHOSTSWITCH_HISTORY_CACHE_MB", 1024)


class _HistoryStore:
    """Model states after agent conversations, least recently used first, within a budget.

    A step's state replaces the states of the earlier steps of its conversation: they cannot
    be extended to a longer prompt more cheaply than it, so a task holds one state.
    """

    def __init__(self, budget: int):
        self.budget = budget
        self.entries: list[tuple[str, list[int], Any, int]] = []  # owner, tokens, state, bytes

    @staticmethod
    def _extends(tokens: list[int], entry: tuple[str, list[int], Any, int]) -> bool:
        return len(entry[1]) <= len(tokens) and tokens[:len(entry[1])] == entry[1]

    def find(self, owner: str, tokens: list[int]) -> tuple[list[int], Any] | None:
        """The longest kept conversation that ``tokens`` start with, and its state."""
        found = max((entry for entry in self.entries if entry[0] == owner and self._extends(tokens, entry)),
                    key=lambda entry: len(entry[1]), default=None)
        if found is None:
            return None
        self.entries.remove(found)
        self.entries.append(found)
        return found[1], found[2]

    def keep(self, owner: str, tokens: list[int], size: int, snapshot: Callable[[], Any]) -> None:
        """Keep the state after ``tokens``, taken by ``snapshot`` only when it fits."""
        if size > self.budget:  # It would only evict the others.
            return
        self.entries = [entry for entry in self.entries
                        if entry[0] != owner or not self._extends(tokens, entry)]
        self.entries.append((owner, list(tokens), snapshot(), size))
        while sum(entry[3] for entry in self.entries) > self.budget:
            self.entries.pop(0)

    @property
    def nbytes(self) -> int:
        return sum(entry[3] for entry in self.entries)


_HISTORY = _HistoryStore(HISTORY_CACHE_BYTES)


def _limit_mlx_cache() -> None:
    set_cache_limit = getattr(mx, "set_cache_limit", None) if mx is not None else None
    if not callable(set_cache_limit):
        set_cache_limit = getattr(getattr(mx, "metal", None), "set_cache_limit", None)
    if callable(set_cache_limit):
        set_cache_limit(MLX_CACHE_LIMIT_BYTES)


def _clear_mlx_cache() -> None:
    clear_cache = getattr(mx, "clear_cache", None) if mx is not None else None
    if not callable(clear_cache):
        clear_cache = getattr(getattr(mx, "metal", None), "clear_cache", None)
    if callable(clear_cache):
        try:
            clear_cache()
        except Exception:
            pass


def release_memory() -> None:
    """Return freed MLX buffers to the system between requests; the weights stay loaded."""
    _clear_mlx_cache()


_PROMPT_CACHE = LRUPromptCache(max_size=256, max_bytes=PROMPT_CACHE_BYTES) if LRUPromptCache else None
# mlx_lm's chunk size for prompts.
PREFILL_STEP_TOKENS = 2048
# A cached prefix also stores the recurrent layers' state (about 70 MB for 4B), so only
# fixed instructions at least this long get an entry: the verifier's rules and long
# expert prompts, not the short search planner.
MIN_CACHED_PREFIX_TOKENS = 256


def _common_prefix(first: list[int], second: list[int]) -> int:
    length = 0
    while length < min(len(first), len(second)) and first[length] == second[length]:
        length += 1
    return length


def _prefill(model: Any, tokens: list[int], cache: list[Any]) -> None:
    """Advance ``cache`` through ``tokens``.

    Only the cache state is evaluated: logits over Qwen3.5's 248K-token vocabulary would
    take 0.5 MB per prompt token.
    """
    for start in range(0, len(tokens), PREFILL_STEP_TOKENS):
        check_cancelled()
        model(mx.array(tokens[start:start + PREFILL_STEP_TOKENS], dtype=mx.uint32)[None], cache=cache)
        mx.eval([entry.state for entry in cache])


# Router prefixes have a dedicated budget and are never populated under LoRA.
_ROUTER_PROMPT_CACHE = None
_ROUTER_CACHE_BYTES = _env_bytes("GHOSTSWITCH_ROUTER_CACHE_MB", 512)


def _router_mlx_cache():
    global _ROUTER_PROMPT_CACHE
    if _ROUTER_PROMPT_CACHE is None:
        _ROUTER_PROMPT_CACHE = LRUPromptCache(max_size=256, max_bytes=_ROUTER_CACHE_BYTES)
    return _ROUTER_PROMPT_CACHE


def get_model(role: str) -> LoadedModel:
    """Return the lazily loaded model for ``role``."""

    return MODEL_MANAGER.get_model(role)


def unload_model(role: str) -> bool:
    """Unload a non-resident model; resident roles remain available."""

    return MODEL_MANAGER.unload_model(role)


class TelemetryCollector:
    """Per-task MLX latency, token, parameter, and memory accounting."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def task_metrics(self) -> dict[str, int | float]:
        unique: dict[str, int] = {}
        for call in self.calls:
            unique[str(call["model_id"])] = int(call["total_params"])
            if call.get("adapter"):
                adapter_metrics = call.get("adapter_metrics") or {}
                unique[f"adapter:{call['adapter']}"] = int(adapter_metrics.get("adapter_params", 0))
        work = sum(
            int(call.get("P_work", int(call["active_params"]) * (int(call["input_tokens"]) + int(call["output_tokens"]))))
            for call in self.calls
        )
        thinker_latency = sum(float(call["latency_ms"]) for call in self.calls if call["role"] == "thinker")
        router_latency = sum(float(call["latency_ms"]) for call in self.calls if call["role"] == "router")
        expert_latency = sum(float(call["latency_ms"]) for call in self.calls if call.get("adapter"))
        return {
            "task_unique_params": sum(unique.values()),
            "peak_active_params": max((int(c["active_params"]) for c in self.calls), default=0),
            "parameter_token_work": work,
            "model_load_latency_ms": round(sum(float(c["load_latency_ms"]) for c in self.calls), 2),
            "thinker_latency_ms": round(thinker_latency, 2),
            "router_latency_ms": round(router_latency, 2),
            "experts_latency_ms": round(expert_latency, 2),
            "prompt_prepare_latency_ms": round(sum(float(c.get("prompt_prepare_ms", 0.0)) for c in self.calls), 2),
            "adapter_activation_latency_ms": round(sum(float(c.get("adapter_activation_ms", 0.0)) for c in self.calls), 2),
            "adapter_detach_latency_ms": round(sum(float(c.get("adapter_detach_ms", 0.0)) for c in self.calls), 2),
            "generation_wall_latency_ms": round(sum(float(c.get("generation_wall_ms", 0.0)) for c in self.calls), 2),
            "first_token_latency_ms": round(sum(float(c.get("time_to_first_token_ms", 0.0) or 0.0) for c in self.calls), 2),
            "cache_setup_latency_ms": round(sum(float(c.get("cache_setup_ms", 0.0)) for c in self.calls), 2),
            "estimated_prefill_latency_ms": round(sum(float(c.get("prefill_ms_estimated", 0.0) or 0.0) for c in self.calls), 2),
            "estimated_decode_latency_ms": round(sum(float(c.get("decode_ms_estimated", 0.0) or 0.0) for c in self.calls), 2),
            "call_wall_latency_ms": round(sum(float(c.get("call_wall_ms", 0.0)) for c in self.calls), 2),
            "peak_memory_bytes": max((int(c["mlx_peak_memory_bytes"]) for c in self.calls), default=0),
            "peak_process_rss_bytes": max((int(c["process_rss_bytes"]) for c in self.calls), default=0),
        }


class GenerationCancelled(Exception):
    """The caller disconnected; no further model work should run."""


_CANCEL_CHECK: ContextVar[Callable[[], bool] | None] = ContextVar("ghostswitch_cancel", default=None)


@contextlib.contextmanager
def cancellation_scope(check: Callable[[], bool] | None):
    token = _CANCEL_CHECK.set(check)
    try:
        yield
    finally:
        _CANCEL_CHECK.reset(token)


def check_cancelled() -> None:
    check = _CANCEL_CHECK.get()
    if check is not None and check():
        raise GenerationCancelled("Client disconnected")


_COLLECTOR: ContextVar[TelemetryCollector | None] = ContextVar("ghostswitch_telemetry", default=None)


@contextlib.contextmanager
def capture_telemetry() -> Iterator[TelemetryCollector]:
    collector = TelemetryCollector()
    parent = _COLLECTOR.get()
    token = _COLLECTOR.set(collector)
    try:
        yield collector
    finally:
        _COLLECTOR.reset(token)
        if parent is not None:
            parent.calls.extend(collector.calls)


def _apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    continue_final_message: bool = False,
    enable_thinking: bool = False,
    add_generation_prompt: bool = True,
) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt and not continue_final_message,
        "continue_final_message": continue_final_message,
    }
    if tools is not None:
        kwargs["tools"] = tools
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("continue_final_message", None)
            return tokenizer.apply_chat_template(messages, **kwargs)


def _template_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Messages for the chat template, with an image slot right after each image label.

    This follows Qwen's own "Picture 1: <image>" labelling.
    """
    rendered = []
    for message in messages:
        item = {key: value for key, value in message.items() if key != "images"}
        if message.get("images"):
            text = item.get("content") or ""
            parts: list[dict[str, str]] = []
            for image in message["images"]:
                # An image whose label is missing from the text goes before the remaining text.
                before, found, after = text.partition(image.label) if image.label else ("", "", "")
                if found:
                    parts.append({"type": "text", "text": before + found})
                    text = after
                parts.append({"type": "image"})
            if text:
                parts.append({"type": "text", "text": text})
            item["content"] = parts
        rendered.append(item)
    return rendered


_VISION: vision.VisionRuntime | None = None


def _mlx_checkpoint(model_id: str) -> Path | None:
    path = Path(model_id).expanduser()
    if path.is_dir():
        return path
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(model_id, local_files_only=True))
    except Exception:
        return None


def vision_available() -> bool:
    """Whether the active model accepts images (a GGUF projector may still need a download)."""
    spec = MODEL_SPECS["thinker"]
    if spec.backend == "gguf":
        return bool(find_mmproj(ACTIVE_BACKBONE, spec.model_id) or mmproj_source(ACTIVE_BACKBONE))
    checkpoint = _mlx_checkpoint(spec.model_id)
    return bool(checkpoint and vision.has_vision_weights(checkpoint))


def prepare_vision(progress: Callable[[dict], None] | None = None) -> None:
    """Load the MLX vision tower, or find or download the GGUF projector, before an image request."""
    global _VISION
    spec = MODEL_SPECS["thinker"]
    if spec.backend == "gguf":
        if find_mmproj(ACTIVE_BACKBONE, spec.model_id) is not None:
            return
        source = mmproj_source(ACTIVE_BACKBONE)
        if source is None:
            raise RuntimeError("Image input on GGUF needs a vision projector (mmproj*.gguf) next to the "
                               "model file, or set GHOSTSWITCH_GGUF_MMPROJ")
        from .download_progress import download_file
        repo_id, filename, revision = source
        download_file(repo_id, filename, revision=revision or "main", progress=progress)
        return
    if _VISION is None or _VISION.model_dir != _mlx_checkpoint(spec.model_id):
        checkpoint = _mlx_checkpoint(spec.model_id)
        if checkpoint is None or not vision.has_vision_weights(checkpoint):
            raise RuntimeError("The current model checkpoint has no vision weights; "
                               "choose a Qwen3.5 checkpoint that includes vision_tower weights")
        _VISION = vision.VisionRuntime(checkpoint, attachments.MAX_IMAGE_PIXELS)
    _limit_mlx_cache()
    _VISION.load()


def _gguf_projector(loaded: LoadedModel) -> Any:
    """The llama.cpp multimodal context for the loaded GGUF model, created once."""
    context = getattr(loaded.model, "_ghostswitch_mtmd", None)
    if context is not None:
        return context
    import llama_cpp
    from llama_cpp import mtmd_cpp
    from llama_cpp._utils import suppress_stdout_stderr

    path = find_mmproj(ACTIVE_BACKBONE, loaded.spec.model_id)
    if path is None or not path.is_file():
        raise RuntimeError("GGUF vision projector not found; send the request again to download it")
    with suppress_stdout_stderr(disable=False):
        params = mtmd_cpp.mtmd_context_params_default()
        params.use_gpu = os.environ.get("GHOSTSWITCH_GGUF_GPU_LAYERS", "-1") != "0"
        params.print_timings = False
        params.n_threads = loaded.model.n_threads
        params.flash_attn_type = (
            llama_cpp.LLAMA_FLASH_ATTN_TYPE_ENABLED
            if loaded.model.context_params.flash_attn_type == llama_cpp.LLAMA_FLASH_ATTN_TYPE_ENABLED
            else llama_cpp.LLAMA_FLASH_ATTN_TYPE_DISABLED
        )
        context = mtmd_cpp.mtmd_init_from_file(str(path).encode(), loaded.model.model, params)
    if not context:
        raise RuntimeError(f"Could not load the GGUF vision projector {path}")
    if not mtmd_cpp.mtmd_support_vision(context):
        mtmd_cpp.mtmd_free(context)
        raise RuntimeError(f"{path} is not a vision projector")
    loaded.model._stack.callback(mtmd_cpp.mtmd_free, context)
    loaded.model._ghostswitch_mtmd = context
    return context


def _gguf_multimodal_prefill(loaded: LoadedModel, prompt: str, images: list[Any]) -> tuple[list[int], int]:
    """Evaluate a prompt with images into the llama.cpp context.

    Returns the prompt as llama-cpp-python tracks it (one entry per position;
    an image spans fewer positions than tokens) and the number of prompt tokens.
    """
    from llama_cpp._utils import suppress_stdout_stderr

    # llama.cpp reports every image slice and batch on stderr.
    with suppress_stdout_stderr(disable=False):
        return _gguf_multimodal_eval(loaded, prompt, images)


def _gguf_multimodal_eval(loaded: LoadedModel, prompt: str, images: list[Any]) -> tuple[list[int], int]:
    import ctypes

    import llama_cpp
    from llama_cpp import mtmd_cpp

    context = _gguf_projector(loaded)
    llama = loaded.model
    text = prompt.replace(vision.IMAGE_PLACEHOLDER, mtmd_cpp.mtmd_default_marker().decode("utf-8"))
    image_pad = llama.tokenize(vision.IMAGE_PAD.encode("utf-8"), add_bos=False, special=True)[0]
    bitmaps = []
    used = 0
    try:
        for image in images:
            data = image.png_bytes()
            buffer = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
            bitmap = mtmd_cpp.mtmd_helper_bitmap_init_from_buf(context, buffer, len(data), False)
            if not bitmap:
                raise ValueError(f"Could not decode image {image.name}")
            bitmaps.append(bitmap)
        encoded = text.encode("utf-8")
        input_text = mtmd_cpp.mtmd_input_text()
        input_text.text = encoded
        input_text.text_len = len(encoded)
        input_text.add_special = False
        input_text.parse_special = True
        chunks = mtmd_cpp.mtmd_input_chunks_init()
        if not chunks:
            raise RuntimeError("Could not allocate multimodal input")
        try:
            array = (mtmd_cpp.mtmd_bitmap_p_ctypes * len(bitmaps))(*bitmaps)
            if mtmd_cpp.mtmd_tokenize(context, chunks, ctypes.byref(input_text), array, len(bitmaps)) != 0:
                raise ValueError("The prompt does not have one image slot per image")
            llama.reset()
            llama._ctx.kv_cache_clear()
            for index in range(mtmd_cpp.mtmd_input_chunks_size(chunks)):
                check_cancelled()
                chunk = mtmd_cpp.mtmd_input_chunks_get(chunks, index)
                if mtmd_cpp.mtmd_input_chunk_get_type(chunk) == mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_TEXT:
                    count = ctypes.c_size_t()
                    pointer = mtmd_cpp.mtmd_input_chunk_get_tokens_text(chunk, ctypes.byref(count))
                    tokens = [pointer[i] for i in range(count.value)]
                    used += len(tokens)
                    if used > llama.n_ctx():
                        raise ValueError("The prompt with images exceeds the context window")
                    llama.eval(tokens)
                    continue
                used += mtmd_cpp.mtmd_input_chunk_get_n_tokens(chunk)
                if used > llama.n_ctx():
                    raise ValueError("The prompt with images exceeds the context window")
                start = llama.n_tokens
                new_past = llama_cpp.llama_pos(0)
                if mtmd_cpp.mtmd_helper_eval_chunk_single(
                        context, llama._ctx.ctx, chunk, llama_cpp.llama_pos(start),
                        llama_cpp.llama_seq_id(0), llama.n_batch, False, ctypes.byref(new_past)) != 0:
                    raise RuntimeError("Could not evaluate an image")
                llama.n_tokens = new_past.value
                # Image positions have no text tokens; mark them so penalties and
                # prefix matching never treat them as earlier text.
                llama.input_ids[start:llama.n_tokens] = image_pad
            return llama.input_ids[: llama.n_tokens].tolist(), used
        finally:
            mtmd_cpp.mtmd_input_chunks_free(chunks)
    finally:
        for bitmap in bitmaps:
            mtmd_cpp.mtmd_bitmap_free(bitmap)


# llama.cpp context state right after a fixed prompt prefix, keyed by model, adapter and
# tokens. Only the context state is kept: llama-cpp-python's own LlamaState would also
# copy up to n_batch rows of logits (about 0.5 GB for this vocabulary).
_GGUF_PREFIX_STATES: OrderedDict[str, bytes] = OrderedDict()
# Router node prompts have their own budget, as on MLX.
_GGUF_ROUTER_STATES: OrderedDict[str, bytes] = OrderedDict()


def _gguf_state_size(llama: Any) -> int:
    import llama_cpp

    return llama_cpp.llama_state_get_size(llama._ctx.ctx)


def _gguf_save_state(llama: Any) -> bytes:
    import llama_cpp

    size = _gguf_state_size(llama)
    buffer = (ctypes.c_uint8 * size)()
    written = llama_cpp.llama_state_get_data(llama._ctx.ctx, buffer, size)
    return ctypes.string_at(buffer, written)


def _gguf_restore_state(llama: Any, state: bytes, tokens: list[int]) -> None:
    """Put the context back in a kept state whose evaluated tokens were ``tokens``."""
    import llama_cpp

    buffer = (ctypes.c_uint8 * len(state)).from_buffer_copy(state)
    if llama_cpp.llama_state_set_data(llama._ctx.ctx, buffer, len(state)) != len(state):
        raise RuntimeError("Could not restore a cached prompt state")
    llama.input_ids[: len(tokens)] = tokens
    llama.n_tokens = len(tokens)
    llama._requires_eval = True


def _gguf_prefix_state(loaded: LoadedModel, adapter: str | None, prefix: list[int],
                       store: OrderedDict[str, bytes], budget: int) -> bool:
    """Put the context in the state after ``prefix``: restore it, or evaluate and keep it.

    Returns whether the state came from the store. The following completion then
    evaluates only the tokens after the prefix.
    """
    llama = loaded.model
    key = hashlib.sha256(f"{loaded.spec.model_id}:{adapter or 'base'}:{prefix}".encode()).hexdigest()
    state = store.get(key)
    if state is not None:
        store.move_to_end(key)
        _gguf_restore_state(llama, state, prefix)
        return True
    llama.reset()
    llama.eval(prefix)
    store[key] = _gguf_save_state(llama)
    while len(store) > 1 and sum(map(len, store.values())) > budget:
        store.popitem(last=False)
    return False


def _chat_gguf(
    loaded: LoadedModel,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    *,
    role: str,
    tools: list[dict[str, Any]] | None,
    prefill: str | None,
    debug: dict[str, Any] | None,
    enable_thinking: bool,
    adapter: str | None = None,
    cache_prompt: bool = False,
    cache_namespace: str | None = None,
    cache_prefix: str | None = None,
    cache_history: bool = False,
    on_token: Callable[[str], None] | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> str:
    call_started = time.perf_counter_ns()
    rendered_messages = messages if prefill is None else [
        *messages, {"role": "assistant", "content": prefill},
    ]
    images = attachments.message_images(rendered_messages)
    prompt = _apply_chat_template(
        loaded.tokenizer, _template_messages(rendered_messages), tools, prefill is not None, enable_thinking,
    )
    input_token_ids = [] if images else list(
        loaded.model.tokenize(prompt.encode("utf-8"), add_bos=False, special=True))
    input_tokens = len(input_token_ids)
    # Router node prompts and other fixed instructions keep the context state after
    # them; image prompts are evaluated with the adapter active below and never cached.
    router_prefix = cache_prompt and adapter is None and (cache_namespace or "").startswith("capability-v2:")
    prefix_store, prefix_budget = (_GGUF_ROUTER_STATES, _ROUTER_CACHE_BYTES) if router_prefix else (
        _GGUF_PREFIX_STATES, PROMPT_CACHE_BYTES)
    prefix_tokens: list[int] = []
    if not images and (router_prefix or cache_prefix is not None):
        static_system = (rendered_messages[0] if cache_prefix is None
                         else {"role": "system", "content": cache_prefix})
        # Qwen3.5 puts the tool definitions ahead of the system text, so they belong to the prefix.
        static_prompt = _apply_chat_template(
            loaded.tokenizer, [static_system, {"role": "user", "content": ""}], tools, False, enable_thinking,
        )
        static_ids = list(loaded.model.tokenize(static_prompt.encode("utf-8"), add_bos=False, special=True))
        prefix_tokens = input_token_ids[:_common_prefix(static_ids, input_token_ids)]
        if not router_prefix and len(prefix_tokens) < MIN_CACHED_PREFIX_TOKENS:
            prefix_tokens = []
    # The conversation before the reply starts; the next agent step resends it.
    history_tokens = 0
    if cache_history and not images and prefill is None:
        history_prompt = _apply_chat_template(
            loaded.tokenizer, _template_messages(messages), tools,
            enable_thinking=enable_thinking, add_generation_prompt=False,
        )
        history_ids = list(loaded.model.tokenize(history_prompt.encode("utf-8"), add_bos=False, special=True))
        history_tokens = min(input_tokens - 1, _common_prefix(history_ids, input_token_ids))
    # Prompt tokens whose state an earlier call computed.
    reused_tokens = 0
    history_cache_build_ms = 0.0
    prompt_prepare_ms = round((time.perf_counter_ns() - call_started) / 1_000_000, 2)
    context_window = loaded.model.n_ctx() if callable(getattr(loaded.model, "n_ctx", None)) else None
    started = time.perf_counter_ns()
    adapter_activation_ms = 0.0
    adapter_detach_ms = 0.0
    cache_setup_ms = 0.0
    prefix_cache_build_ms = 0.0
    generation_wall_ms = 0.0
    time_to_first_token_ms: float | None = None
    output_parts: list[str] = []
    output_token_count = 0
    finish_reason = None
    generation_started: int | None = None
    with activate_gguf_adapter(loaded.model, adapter) as adapter_metrics:
        adapter_activation_ms = round((time.perf_counter_ns() - started) / 1_000_000, 2)
        try:
            cache_started = time.perf_counter_ns()
            history_owner = f"{loaded.spec.model_id}:{adapter or 'base'}"
            kept = _HISTORY.find(history_owner, input_token_ids[:history_tokens]) if history_tokens else None
            if kept is not None:
                # The previous step of the same task left the state after its conversation.
                _gguf_restore_state(loaded.model, kept[1], kept[0])
                reused_tokens = len(kept[0])
            if prefix_tokens and not reused_tokens:
                if _gguf_prefix_state(loaded, adapter, prefix_tokens, prefix_store, prefix_budget):
                    reused_tokens = len(prefix_tokens)
                else:
                    prefix_cache_build_ms = round((time.perf_counter_ns() - cache_started) / 1_000_000, 2)
            if history_tokens > reused_tokens:
                history_started = time.perf_counter_ns()
                if not reused_tokens and not prefix_tokens:
                    # The context still holds the previous request.
                    loaded.model.reset()
                loaded.model.eval(input_token_ids[loaded.model.n_tokens:history_tokens])
                _HISTORY.keep(history_owner, input_token_ids[:history_tokens],
                              _gguf_state_size(loaded.model), lambda: _gguf_save_state(loaded.model))
                history_cache_build_ms = round((time.perf_counter_ns() - history_started) / 1_000_000, 2)
            if prefix_tokens or history_tokens:
                cache_setup_ms = round((time.perf_counter_ns() - cache_started) / 1_000_000, 2)
            generation_started = time.perf_counter_ns()
            progress_last = time.monotonic() - 2
            if images:
                input_token_ids, input_tokens = _gguf_multimodal_prefill(loaded, prompt, images)
            response = loaded.model.create_completion(
                prompt=input_token_ids if images or prefix_tokens or history_tokens else prompt,
                max_tokens=max_tokens,
                temperature=temperature, stop=["<|im_end|>", "<|im_start|>"], echo=False, stream=True,
            )
            for chunk in response:
                check_cancelled()
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice_chunk = choices[0]
                piece = str(choice_chunk.get("text") or "")
                if piece:
                    output_token_count += 1
                if piece and time_to_first_token_ms is None:
                    time_to_first_token_ms = round(
                        (time.perf_counter_ns() - generation_started) / 1_000_000, 2
                    )
                output_parts.append(piece)
                if on_progress is not None and time.monotonic() - progress_last >= 2:
                    progress_last = time.monotonic()
                    # llama.cpp chunks may contain several tokens or a UTF-8
                    # fragment; count with the model tokenizer rather than chunks.
                    on_progress(len(loaded.model.tokenize(
                        "".join(output_parts).encode("utf-8"), add_bos=False, special=True)))
                if piece and on_token is not None:
                    on_token(piece)
                if choice_chunk.get("finish_reason") is not None:
                    finish_reason = choice_chunk.get("finish_reason")
            generation_wall_ms = round((time.perf_counter_ns() - generation_started) / 1_000_000, 2)
        finally:
            if images:
                # The next prompt must not reuse context that holds this request's images.
                loaded.model.reset()
        detach_started = time.perf_counter_ns()
    adapter_detach_ms = round((time.perf_counter_ns() - detach_started) / 1_000_000, 2)
    latency_ns = time.perf_counter_ns() - started
    result = "".join(output_parts)
    output_token_ids = list(loaded.model.tokenize(result.encode("utf-8"), add_bos=False, special=True))
    output_tokens = output_token_count
    if debug is not None:
        debug.update({
            "model_id": loaded.spec.model_id,
            "messages": rendered_messages,
            "tools": tools,
            "rendered_prompt": prompt,
            "input_token_ids": input_token_ids,
            "context_window_tokens": context_window,
            "generated_token_ids": output_token_ids,
            "generated_tokens": output_tokens,
            "decoded_raw_output": result,
            "finish_reason": finish_reason,
            "max_tokens": max_tokens,
        })
    collector = _COLLECTOR.get()
    if collector is not None:
        total_params = loaded.spec.total_params
        collector.calls.append({
            "role": role,
            "adapter": adapter,
            "adapter_metrics": adapter_metrics,
            "model_id": loaded.spec.model_id,
            "prompt": prompt,
            "raw_output": result,
            "trace_stage": str((debug or {}).get("stage", "generation")),
            "trace_metadata": {
                key: value for key, value in (debug or {}).items()
                if key in {"stage", "attempt", "node", "expert_id"}
            },
            "total_params": total_params,
            "active_params": total_params,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "prompt_tokens": input_tokens,
            "prefill_ms": None,
            "decode_ms": None,
            "prefill_ms_estimated": None,
            "decode_ms_estimated": None,
            "prompt_tokens_per_second": None,
            "generation_tokens_per_second": (
                round(output_tokens / (generation_wall_ms / 1000), 3)
                if generation_wall_ms else 0.0
            ),
            "prompt_prepare_ms": prompt_prepare_ms,
            "adapter_activation_ms": adapter_activation_ms,
            "adapter_detach_ms": adapter_detach_ms,
            "cache_setup_ms": cache_setup_ms,
            "time_to_first_token_ms": time_to_first_token_ms,
            "generation_wall_ms": generation_wall_ms,
            "generation_after_first_token_ms": (
                round(max(0.0, generation_wall_ms - time_to_first_token_ms), 2)
                if time_to_first_token_ms is not None else None
            ),
            "generated_tokens": output_tokens,
            "max_tokens": max_tokens,
            "finish_reason": finish_reason,
            "total_ms": round(latency_ns / 1_000_000, 2),
            "call_wall_ms": round((time.perf_counter_ns() - call_started) / 1_000_000, 2),
            "cached_prompt_tokens": reused_tokens,
            "cached_prefix_tokens": reused_tokens,
            "prompt_tokens_uncached": input_tokens - reused_tokens,
            "prompt_cache_hit": reused_tokens > 0,
            "cache_status": "hit" if reused_tokens else "miss" if prefix_tokens or history_tokens else "disabled",
            "cache_backend": "prefix_state" if prefix_tokens or history_tokens else None,
            "prefix_cache_build_ms": prefix_cache_build_ms,
            "history_cache_build_ms": history_cache_build_ms,
            "prompt_cache_bytes": (sum(map(len, prefix_store.values())) if prefix_tokens else 0) + (
                _HISTORY.nbytes if history_tokens else 0),
            "enable_thinking": enable_thinking,
            "P_work": total_params * (input_tokens + output_tokens),
            "latency_ms": round(latency_ns / 1_000_000, 2),
            "load_latency_ms": round(MODEL_MANAGER.last_load_latency_ns / 1_000_000, 2),
            "tokens_per_second": round(output_tokens / (latency_ns / 1_000_000_000), 3) if latency_ns else 0,
            "mlx_peak_memory_bytes": 0,
            "mlx_active_memory_bytes": 0,
            "mlx_cache_memory_bytes": 0,
            "process_rss_bytes": _rss_bytes(),
        })
    return f"{prefill}{result}" if prefill is not None else result


def chat(
    messages: list[dict[str, Any]],
    max_tokens: int = 256,
    temperature: float = 0.0,
    *,
    role: str = "thinker",
    tools: list[dict[str, Any]] | None = None,
    prefill: str | None = None,
    adapter: str | None = None,
    debug: dict[str, Any] | None = None,
    enable_thinking: bool = False,
    cache_prompt: bool = False,
    cache_namespace: str | None = None,
    cache_prefix: str | None = None,
    cache_history: bool = False,
    on_token: Callable[[str], None] | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> str:
    """Generate using a role's model and append one telemetry event if active.

    ``cache_prefix`` is fixed text that the system message starts with in every
    request of its kind; the model state after it is computed once and reused.
    ``cache_history`` also keeps the state after the conversation, for a request that
    comes back with more messages: an agent step.
    """

    check_cancelled()
    loaded = get_model("thinker" if adapter else role)
    check_cancelled()
    if loaded.spec.backend == "gguf":
        return _chat_gguf(
            loaded, messages, max_tokens, temperature, role=role, tools=tools,
            prefill=prefill, debug=debug, enable_thinking=enable_thinking,
            adapter=adapter, cache_prompt=cache_prompt, cache_namespace=cache_namespace,
            cache_prefix=cache_prefix, cache_history=cache_history, on_token=on_token, on_progress=on_progress,
        )
    call_started = time.perf_counter_ns()
    rendered_messages = messages
    if prefill is not None:
        rendered_messages = [*messages, {"role": "assistant", "content": prefill}]
    images = attachments.message_images(rendered_messages)
    prompt = _apply_chat_template(
        loaded.tokenizer, _template_messages(rendered_messages), tools, prefill is not None, enable_thinking
    )
    input_token_ids = [int(token) for token in loaded.tokenizer.encode(prompt)]
    prepared = None
    if images:
        # Images enter as embeddings with multimodal positions; such prompts are not cached.
        prepare_vision()
        prepared = _VISION.prepare(loaded.model, input_token_ids, images)
        input_token_ids = prepared.token_ids
        cache_prompt = False
    elif cache_prefix is not None:
        cache_prompt = True
    prefix_tokens: list[int] = []
    # The conversation before the reply starts; the next agent step resends it.
    history_tokens = 0
    if cache_prompt:
        static_system = (rendered_messages[0] if cache_prefix is None
                         else {"role": "system", "content": cache_prefix})
        # Qwen3.5 puts the tool definitions ahead of the system text, so they belong to the prefix.
        static_prompt = _apply_chat_template(
            loaded.tokenizer, [static_system, {"role": "user", "content": ""}], tools, False, enable_thinking,
        )
        prefix_tokens = input_token_ids[:_common_prefix(
            [int(token) for token in loaded.tokenizer.encode(static_prompt)], input_token_ids)]
        if cache_history and prefill is None:
            history_prompt = _apply_chat_template(
                loaded.tokenizer, _template_messages(messages), tools,
                enable_thinking=enable_thinking, add_generation_prompt=False,
            )
            history_tokens = min(len(input_token_ids) - 1, _common_prefix(
                [int(token) for token in loaded.tokenizer.encode(history_prompt)], input_token_ids))
        if cache_prefix is not None and len(prefix_tokens) < MIN_CACHED_PREFIX_TOKENS:
            prefix_tokens = []
            cache_prompt = history_tokens > 0
    input_tokens = len(input_token_ids)
    prompt_prepare_ms = round((time.perf_counter_ns() - call_started) / 1_000_000, 2)
    _reset_peak_memory()
    started = time.perf_counter_ns()
    adapter_activation_ms = 0.0
    adapter_detach_ms = 0.0
    time_to_first_token_ms: float | None = None
    generation_wall_ms = 0.0
    with (LORA_MANAGER.activate(adapter, loaded.model, loaded.spec.model_id) if adapter else contextlib.nullcontext({})) as adapter_metrics, \
            (_VISION.positions(loaded.model, prepared) if prepared else contextlib.nullcontext()):
        adapter_activation_ms = round((time.perf_counter_ns() - started) / 1_000_000, 2)
        sampler = make_sampler(temp=temperature)
        finish_reason: str | None = None
        output_token_ids: list[int] = []
        output_parts: list[str] = []
        response_metrics: dict[str, Any] = {}
        cached_tokens = 0
        # Prompt tokens whose state an earlier call computed.
        reused_tokens = 0
        cache_hit = False
        prefix_cache_build_ms = 0.0
        history_cache_build_ms = 0.0
        cache_bytes = 0
        prompt_cache = None
        cache_setup_ms = 0.0
        generation_prompt: str | list[int] = prompt
        cache_model_key = loaded.spec.model_id
        if cache_prompt:
            cache_started = time.perf_counter_ns()
            cache_store = (_router_mlx_cache() if adapter is None and (cache_namespace or "").startswith("capability-v2:") else _PROMPT_CACHE)
            prefix_digest = hashlib.sha256(",".join(map(str, prefix_tokens)).encode()).hexdigest()
            namespace = f":{cache_namespace}" if cache_namespace else ""
            cache_model_key = f"{loaded.spec.model_id}:adapter={adapter or 'base'}{namespace}:{prefix_digest}"
            remaining_tokens = input_token_ids
            kept = _HISTORY.find(cache_model_key, input_token_ids[:history_tokens]) if history_tokens else None
            if kept is not None:
                # The previous step of the same task left the state after its conversation.
                prompt_cache, remaining_tokens = copy.deepcopy(kept[1]), input_token_ids[len(kept[0]):]
            if prompt_cache is None and prefix_tokens:
                prompt_cache, remaining_tokens = cache_store.fetch_nearest_cache(cache_model_key, input_token_ids)
                if prompt_cache is None:
                    prefix_started = time.perf_counter_ns()
                    prefix_cache = make_prompt_cache(loaded.model)
                    _prefill(loaded.model, prefix_tokens, prefix_cache)
                    prefix_cache_build_ms = round((time.perf_counter_ns() - prefix_started) / 1_000_000, 2)
                    cache_store.insert_cache(
                        cache_model_key, prefix_tokens, prefix_cache, cache_type="system"
                    )
                    prompt_cache, remaining_tokens = cache_store.fetch_nearest_cache(cache_model_key, input_token_ids)
                else:
                    cache_hit = True
            else:
                cache_hit = prompt_cache is not None
            cached_tokens = len(input_token_ids) - len(remaining_tokens)
            reused_tokens = cached_tokens if cache_hit else 0
            if prompt_cache is None:
                prompt_cache = make_prompt_cache(loaded.model)
            if history_tokens > cached_tokens:
                history_started = time.perf_counter_ns()
                _prefill(loaded.model, input_token_ids[cached_tokens:history_tokens], prompt_cache)
                _HISTORY.keep(cache_model_key, input_token_ids[:history_tokens],
                              sum(entry.nbytes for entry in prompt_cache),
                              lambda: copy.deepcopy(prompt_cache))
                history_cache_build_ms = round((time.perf_counter_ns() - history_started) / 1_000_000, 2)
                cached_tokens = history_tokens
                remaining_tokens = input_token_ids[history_tokens:]
            generation_prompt = remaining_tokens
            cache_bytes = cache_store.nbytes + (_HISTORY.nbytes if history_tokens else 0)
            cache_setup_ms = round((time.perf_counter_ns() - cache_started) / 1_000_000, 2)
        generation_kwargs: dict[str, Any] = {}
        if prompt_cache is not None:
            generation_kwargs["prompt_cache"] = prompt_cache
        if prepared is not None:
            generation_prompt = prepared.token_ids
            generation_kwargs["input_embeddings"] = prepared.embeddings
        generation_started = time.perf_counter_ns()
        for response in stream_generate(
            loaded.model,
            loaded.tokenizer,
            prompt=generation_prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            **generation_kwargs,
        ):
            check_cancelled()
            if on_progress is not None:
                on_progress(int(getattr(response, "generation_tokens", len(output_token_ids) + 1)))
            if time_to_first_token_ms is None:
                time_to_first_token_ms = round(
                    (time.perf_counter_ns() - generation_started) / 1_000_000, 2
                )
            output_parts.append(response.text)
            if response.text and on_token is not None:
                on_token(response.text)
            output_token_ids.append(int(response.token))
            finish_reason = response.finish_reason
            response_metrics = {
                "prompt_tokens": int(getattr(response, "prompt_tokens", input_tokens)),
                "prompt_tps": float(getattr(response, "prompt_tps", 0.0) or 0.0),
                "generation_tokens": int(getattr(response, "generation_tokens", len(output_token_ids))),
                "generation_tps": float(getattr(response, "generation_tps", 0.0) or 0.0),
            }
        generation_wall_ms = round((time.perf_counter_ns() - generation_started) / 1_000_000, 2)
        result = "".join(output_parts)
        if debug is not None:
            debug.update({
                "model_id": loaded.spec.model_id,
                "messages": rendered_messages,
                "tools": tools,
                "rendered_prompt": prompt,
                "input_token_ids": [int(token) for token in input_token_ids or []],
                "generated_token_ids": output_token_ids,
                "generated_tokens": int(response_metrics.get("generation_tokens", len(output_token_ids))),
                "decoded_raw_output": result,
                "finish_reason": finish_reason,
                "max_tokens": max_tokens,
            })
        detach_started = time.perf_counter_ns()
    adapter_detach_ms = round((time.perf_counter_ns() - detach_started) / 1_000_000, 2)
    latency_ns = time.perf_counter_ns() - started
    output_tokens = int(response_metrics.get("generation_tokens", len(output_token_ids)))
    prompt_tps = float(response_metrics.get("prompt_tps", 0.0))
    generation_tps = float(response_metrics.get("generation_tps", 0.0))
    prefill_ms = round(
        prefix_cache_build_ms + history_cache_build_ms
        + (max(0, input_tokens - cached_tokens) / prompt_tps * 1000 if prompt_tps else 0.0),
        2,
    )
    decode_ms = round(output_tokens / generation_tps * 1000, 2) if generation_tps else 0.0
    collector = _COLLECTOR.get()
    if collector is not None:
        total_params = loaded.spec.total_params
        active_params = total_params + int(adapter_metrics.get("adapter_params", 0))
        collector.calls.append({
            "role": role if not adapter else adapter,
            "adapter": adapter,
            "adapter_metrics": dict(adapter_metrics),
            "model_id": loaded.spec.model_id,
            "prompt": prompt,
            "raw_output": result,
            "trace_stage": str((debug or {}).get("stage", "generation")),
            "trace_metadata": {
                key: value for key, value in (debug or {}).items()
                if key in {"stage", "attempt", "node", "expert_id"}
            },
            "total_params": total_params,
            "active_params": active_params,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "prompt_tokens": input_tokens,
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "prefill_ms_estimated": prefill_ms,
            "decode_ms_estimated": decode_ms,
            "prompt_tokens_per_second": prompt_tps or None,
            "generation_tokens_per_second": generation_tps or (
                round(output_tokens / (generation_wall_ms / 1000), 3) if generation_wall_ms else 0.0
            ),
            "prompt_prepare_ms": prompt_prepare_ms,
            "adapter_activation_ms": adapter_activation_ms,
            "adapter_detach_ms": adapter_detach_ms,
            "time_to_first_token_ms": time_to_first_token_ms,
            "generation_wall_ms": generation_wall_ms,
            "generation_after_first_token_ms": (
                round(max(0.0, generation_wall_ms - time_to_first_token_ms), 2)
                if time_to_first_token_ms is not None else None
            ),
            "generated_tokens": output_tokens,
            "max_tokens": max_tokens,
            "finish_reason": finish_reason,
            "total_ms": round(latency_ns / 1_000_000, 2),
            "call_wall_ms": round((time.perf_counter_ns() - call_started) / 1_000_000, 2),
            # A prefix built during this call was computed here, so only a hit counts as cached.
            "cached_prompt_tokens": reused_tokens,
            "cached_prefix_tokens": reused_tokens,
            "prompt_tokens_uncached": max(0, len(input_token_ids) - reused_tokens),
            "prompt_cache_hit": cache_hit,
            "cache_status": "hit" if cache_hit else "miss" if cache_prompt else "disabled",
            "prefix_cache_build_ms": prefix_cache_build_ms,
            "history_cache_build_ms": history_cache_build_ms,
            "cache_setup_ms": cache_setup_ms,
            "prompt_cache_bytes": cache_bytes,
            "enable_thinking": enable_thinking,
            "P_work": active_params * (input_tokens + output_tokens),
            "latency_ms": round(latency_ns / 1_000_000, 2),
            "load_latency_ms": round(MODEL_MANAGER.last_load_latency_ns / 1_000_000, 2),
            "tokens_per_second": round(output_tokens / (latency_ns / 1_000_000_000), 3) if latency_ns else 0,
            "mlx_peak_memory_bytes": _memory_counter("get_peak_memory"),
            "mlx_active_memory_bytes": _memory_counter("get_active_memory"),
            "mlx_cache_memory_bytes": _memory_counter("get_cache_memory"),
            "process_rss_bytes": _rss_bytes(),
        })
    return f"{prefill}{result}" if prefill is not None else result
