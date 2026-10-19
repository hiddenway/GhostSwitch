"""Explicit backbone profiles backed by local, git-ignored model weights."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_BACKBONE = "qwen35-9b"
# A server process serves one backbone; its catalogs are built for this one at import.
ACTIVE_BACKBONE_ENV = "GHOSTSWITCH_BACKBONE"
_PROFILES = Path(__file__).with_name("backbones.json")
_PROJECT_ROOT = _PROFILES.resolve().parents[1]

def map_legacy_environment(environ=os.environ) -> None:
    """The project was called OmniBrain and, before that, MiniBrain: their OMNIBRAIN_* and
    MINIBRAIN_* variables stand in for unset GHOSTSWITCH_* ones, the newer name first."""
    for legacy in ("OMNIBRAIN_", "MINIBRAIN_"):
        for name, value in list(environ.items()):
            if name.startswith(legacy):
                environ.setdefault("GHOSTSWITCH_" + name.removeprefix(legacy), value)


map_legacy_environment()


def data_root() -> Path:
    """Where downloaded models and runtime files live: GHOSTSWITCH_HOME, or this project folder.

    The installer sets GHOSTSWITCH_HOME, so replacing the app folder on an update keeps the models.
    """
    configured = os.environ.get("GHOSTSWITCH_HOME")
    return Path(configured).expanduser().resolve() if configured else _PROJECT_ROOT


def user_dir(root: Path) -> Path:
    """GhostSwitch's folder under root, or the OmniBrain or MiniBrain folder while only that one exists.

    With GHOSTSWITCH_HOME set, settings and caches live there instead (root ~/.config → config),
    so an installed copy keeps everything it writes in one folder.
    """
    if os.environ.get("GHOSTSWITCH_HOME"):
        return data_root() / root.name.lstrip(".")
    current = root / "ghostswitch"
    if not current.exists():
        for legacy in (root / "omnibrain", root / "minibrain"):
            if legacy.is_dir():
                return legacy
    return current


def _rows() -> dict[str, dict]:
    return json.loads(_PROFILES.read_text(encoding="utf-8"))


def _row(backbone_id: str) -> dict:
    rows = _rows()
    if backbone_id not in rows:
        raise ValueError(f"Unknown backbone {backbone_id!r}; choose one of {sorted(rows)}")
    return rows[backbone_id]


def active_backbone() -> str:
    """The backbone this process serves: GHOSTSWITCH_BACKBONE, or the default."""
    backbone_id = os.environ.get(ACTIVE_BACKBONE_ENV) or DEFAULT_BACKBONE
    _row(backbone_id)
    return backbone_id


def display_name(backbone_id: str) -> str:
    return str(_row(backbone_id).get("display_name", backbone_id))


def uncensored(backbone_id: str) -> bool:
    """Whether the backbone is a fine-tune with its refusal training removed."""
    return _row(backbone_id).get("uncensored") is True


def profile_name(backbone_id: str, backend: str) -> str:
    """Execution profile file stem for a backbone and backend, such as mlx9b_v6."""
    name = _row(backbone_id).get("profiles", {}).get(backend)
    if not isinstance(name, str) or not name:
        raise ValueError(f"No {backend} execution profile for backbone {backbone_id!r}")
    return name


def hf_base_repo(backbone_id: str) -> str:
    return str(_row(backbone_id)["hf_base_repo"])


def gguf_context(backbone_id: str) -> int:
    configured = os.environ.get("GHOSTSWITCH_GGUF_CONTEXT", "")
    if configured.isdigit() and int(configured) > 0:
        return int(configured)
    return int(_row(backbone_id).get("gguf", {}).get("context", 4096))


def kv_bytes_per_token(backbone_id: str) -> int:
    """KV cache bytes per context token across the full-attention layers (f16)."""
    return int(_row(backbone_id).get("kv_bytes_per_token", 0))


def local_model_dir(backbone_id: str) -> Path:
    return (data_root() / _row(backbone_id)["local_model_path"]).resolve()


def download_source(backbone_id: str, backend: str) -> dict:
    """The pinned Hugging Face repository, revision, and weight size the app downloads."""
    source = _row(backbone_id).get(backend)
    if not isinstance(source, dict) or not all(isinstance(source.get(key), str) for key in ("repo_id", "revision")):
        raise ValueError(f"No {backend} download source for backbone {backbone_id!r}")
    return dict(source)


def find_gguf(backbone_id: str) -> Path | None:
    """Find the backbone's GGUF file in the project, LM Studio, or the Hugging Face cache."""
    gguf = _row(backbone_id).get("gguf", {})
    filename = gguf.get("filename")
    if not isinstance(filename, str) or not filename:
        return None
    candidates = [data_root() / "base" / backbone_id / "gguf" / filename]
    if os.environ.get("GHOSTSWITCH_GGUF_DIR"):
        candidates.append(Path(os.environ["GHOSTSWITCH_GGUF_DIR"]).expanduser() / filename)
    lmstudio = Path.home() / ".lmstudio" / "models"
    if lmstudio.is_dir():
        candidates += sorted(lmstudio.glob(f"*/*/{filename}"))
    repo_id = gguf.get("repo_id")
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path.home() / ".cache" / "huggingface" / "hub")
    if isinstance(repo_id, str) and hub.is_dir():
        candidates += sorted((hub / f"models--{repo_id.replace('/', '--')}").glob(f"snapshots/*/{filename}"))
    return next((path for path in candidates if path.is_file()), None)


def mmproj_source(backbone_id: str) -> tuple[str, str, str | None] | None:
    """Repository, file name and revision of the backbone's GGUF vision projector."""
    gguf = _row(backbone_id).get("gguf", {})
    filename, repo_id = gguf.get("mmproj"), gguf.get("repo_id")
    if not isinstance(filename, str) or not isinstance(repo_id, str):
        return None
    revision = gguf.get("revision")
    return repo_id, filename, revision if isinstance(revision, str) else None


def find_mmproj(backbone_id: str, gguf_path: str | None = None) -> Path | None:
    """Find a GGUF vision projector: GHOSTSWITCH_GGUF_MMPROJ, next to the model, or a known location."""
    configured = os.environ.get("GHOSTSWITCH_GGUF_MMPROJ")
    if configured:
        return Path(configured).expanduser()
    candidates: list[Path] = []
    if gguf_path:
        # LM Studio and llama.cpp keep a model's projector in its folder.
        candidates += sorted(Path(gguf_path).expanduser().parent.glob("mmproj*.gguf"))
    source = mmproj_source(backbone_id)
    if source:
        repo_id, filename, _ = source
        candidates.append(data_root() / "base" / backbone_id / "gguf" / filename)
        lmstudio = Path.home() / ".lmstudio" / "models"
        if lmstudio.is_dir():
            candidates += sorted(lmstudio.glob(f"*/*/{filename}"))
        hub = Path(os.environ.get("HF_HUB_CACHE") or Path.home() / ".cache" / "huggingface" / "hub")
        if hub.is_dir():
            candidates += sorted((hub / f"models--{repo_id.replace('/', '--')}").glob(f"snapshots/*/{filename}"))
    return next((path for path in candidates if path.is_file()), None)


@dataclass(frozen=True)
class BackboneProfile:
    id: str
    model_id: str
    total_params: int
    model_type: str
    quantization_bits: int
    text_hidden_size: int | None = None
    text_num_layers: int | None = None
    backend: str = "mlx"
    tokenizer_path: str | None = None


def backbone_ids() -> tuple[str, ...]:
    return tuple(_rows())


def adapter_base_names(backbone_id: str) -> frozenset[str]:
    names = _row(backbone_id).get("adapter_base_names", [])
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        raise ValueError(f"Invalid adapter lineage for {backbone_id!r}")
    return frozenset(names)


def get_backbone(
    backbone_id: str, *, model_path: str | None = None, backend: str = "mlx",
    validate_checkpoint: bool = True,
) -> BackboneProfile:
    profiles = json.loads(_PROFILES.read_text(encoding="utf-8"))
    if backbone_id not in profiles:
        raise ValueError(f"Unknown backbone {backbone_id!r}; choose one of {sorted(profiles)}")
    row = profiles[backbone_id]
    if backend not in {"mlx", "gguf"}:
        raise ValueError(f"Unknown model backend: {backend!r}")
    if backend == "gguf":
        if not model_path:
            raise ValueError("GGUF backend requires --model-path to a local .gguf file")
        tokenizer_path = (data_root() / row["local_model_path"]).resolve()
        profile = BackboneProfile(
            # Keep the snapshot symlink basename: the blob target in the Hub
            # cache is content-addressed and has no .gguf file extension.
            backbone_id, str(Path(model_path).expanduser().absolute()), int(row["total_params"]),
            str(row["model_type"]), 0,
            int(row["text_hidden_size"]), int(row["text_num_layers"]),
            backend="gguf", tokenizer_path=str(tokenizer_path),
        )
        validate_local_gguf(profile)
        return profile
    configured_local = row.get("local_model_path")
    local_path = (data_root() / configured_local).resolve() if configured_local else None
    use_local_checkpoint = model_path is None and local_path is not None and local_path.is_dir()
    selected = model_path or (str(local_path) if use_local_checkpoint else row["model_id"])
    if model_path is not None:
        model_id = str(Path(selected).expanduser().resolve())
    elif use_local_checkpoint:
        model_id = str(local_path)
    elif selected.startswith(("~", "/")):
        model_id = str(Path(selected).expanduser().resolve())
    else:
        model_id = selected
    profile = BackboneProfile(
        backbone_id, model_id, int(row["total_params"]),
        str(row["model_type"]), int(row["quantization_bits"]),
        int(row["text_hidden_size"]) if "text_hidden_size" in row else None,
        int(row["text_num_layers"]) if "text_num_layers" in row else None,
    )
    if validate_checkpoint and (
        model_path is not None or use_local_checkpoint or selected.startswith(("~", "/", "."))
    ):
        validate_local_model(profile)
    return profile


def validate_local_model(profile: BackboneProfile) -> None:
    """Reject an incomplete or wrong local checkpoint before runtime selection."""
    directory = Path(profile.model_id)
    config_file = directory / "config.json"
    index_file = directory / "model.safetensors.index.json"
    tokenizer_file = directory / "tokenizer.json"
    if not config_file.is_file() or not index_file.is_file() or not tokenizer_file.is_file():
        raise FileNotFoundError(f"Incomplete MLX checkpoint: {directory}")
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if config.get("model_type") != profile.model_type:
        raise ValueError(f"Wrong model type for {profile.id}: {config.get('model_type')!r}")
    if config.get("quantization_config", {}).get("bits") != profile.quantization_bits:
        raise ValueError(f"Wrong quantization for {profile.id}")
    text_config = config.get("text_config", {})
    if profile.text_hidden_size is not None and text_config.get("hidden_size") != profile.text_hidden_size:
        raise ValueError(f"Wrong text hidden size for {profile.id}")
    if profile.text_num_layers is not None and text_config.get("num_hidden_layers") != profile.text_num_layers:
        raise ValueError(f"Wrong text layer count for {profile.id}")
    index = json.loads(index_file.read_text(encoding="utf-8"))
    shards = set(index.get("weight_map", {}).values())
    if not shards or any(not (directory / shard).is_file() for shard in shards):
        raise FileNotFoundError(f"Missing model weight shard in {directory}")


def validate_local_gguf(profile: BackboneProfile) -> None:
    """Check the local GGUF architecture before loading it with llama.cpp."""
    path = Path(profile.model_id)
    if path.suffix.lower() != ".gguf" or not path.is_file():
        raise FileNotFoundError(f"GGUF model file not found: {path}")
    with path.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise ValueError(f"Invalid GGUF file header: {path}")
    from gguf import GGUFReader

    try:
        reader = GGUFReader(path)
        def field(name: str) -> object:
            value = reader.get_field(name)
            return value.contents() if value is not None else None
        architecture = field("general.architecture")
        if architecture != "qwen35":
            raise ValueError(f"Wrong GGUF architecture for {profile.id}: {architecture!r}")
        hidden_size = field("qwen35.embedding_length")
        layers = field("qwen35.block_count")
        if hidden_size != profile.text_hidden_size:
            raise ValueError(f"Wrong GGUF hidden size for {profile.id}: {hidden_size!r}")
        # Qwen3.5 GGUF exports may count the single MTP/next-token prediction
        # block in block_count, although the backbone profile records only the
        # regular transformer layers from the HF config.
        nextn_layers = field("qwen35.nextn_predict_layers")
        has_single_nextn_layer = nextn_layers is not None and int(nextn_layers) == 1
        expected_layers = profile.text_num_layers
        if layers != expected_layers and not (
            has_single_nextn_layer and layers == expected_layers + 1
        ):
            raise ValueError(f"Wrong GGUF layer count for {profile.id}: {layers!r}")
        if field("split.count") not in (None, 1):
            raise ValueError("Split GGUF checkpoints are not supported; provide one complete file")
    except (OSError, IndexError, OverflowError) as error:
        raise ValueError(f"Invalid GGUF metadata: {path}") from error
    tokenizer_path = Path(profile.tokenizer_path or "")
    if not all((tokenizer_path / name).is_file() for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")):
        raise FileNotFoundError(f"Local Qwen3.5 tokenizer and chat template required: {tokenizer_path}")
