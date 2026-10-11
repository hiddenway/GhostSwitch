"""Data-only, community-contributed expert packages. Never execute repository code."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1] / "experts"


def user_root() -> Path:
    """Adapters the user added from Hugging Face; they are chosen by hand and never routed.

    Adapters added while the project was called OmniBrain stay in ~/.config/omnibrain/experts.
    """
    import os
    configured = os.environ.get("GHOSTSWITCH_USER_EXPERTS")
    if configured:
        return Path(configured).expanduser()
    config = Path.home() / ".config"
    current, legacy = config / "ghostswitch" / "experts", config / "omnibrain" / "experts"
    return legacy if legacy.is_dir() and not current.exists() else current


def manual_only(package: dict) -> bool:
    return bool(package["profile"].get("manual_only"))
VERDICTS = {"helps", "no_gain", "hurts"}


def _evaluation(path: Path) -> dict | None:
    """The optional measured comparison with the base model; a blocked expert is never routed."""
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("evaluation.json needs schema_version 1")
    if value.get("verdict") not in VERDICTS or type(value.get("blocked")) is not bool:
        raise ValueError("evaluation.json needs a verdict (helps, no_gain, hurts) and blocked")
    if value["blocked"] and (not isinstance(value.get("reason"), str) or not value["reason"].strip()):
        raise ValueError("A blocked expert needs a reason")
    return value


def blocked(package: dict) -> bool:
    return bool((package.get("evaluation") or {}).get("blocked"))


def _relative(value: object, suffix: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Source filenames must be relative POSIX paths")
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in {"..", ".", ""} for p in value.split("/")) or not value.endswith(suffix):
        raise ValueError(f"Invalid source filename: {value!r}")
    return value


def _source(raw: dict) -> tuple[str, str]:
    if not isinstance(raw, dict) or not isinstance(raw.get("url"), str):
        raise ValueError("source must contain a Hugging Face URL")
    url = urlparse(raw["url"])
    repo = url.path.strip("/")
    if (url.scheme != "https" or url.netloc != "huggingface.co" or url.query or url.fragment
            or re.fullmatch(r"[\w.-]+/[\w.-]+", repo) is None):
        raise ValueError("Use a https://huggingface.co/owner/repository URL")
    revision = raw.get("revision")
    if not isinstance(revision, str) or re.fullmatch(r"[a-f0-9]{40}", revision) is None:
        raise ValueError("Hugging Face revision must be a pinned 40-character commit SHA")
    return repo, revision


def load_packages(backbone: str | None = None) -> dict[str, dict]:
    """Expert packages for a backbone; defaults to the backbone this process serves."""
    from core_models import active_backbone
    return _load_packages(backbone or active_backbone())


def reload_packages() -> None:
    """Forget cached packages, for example after the user added an adapter."""
    _load_packages.cache_clear()


@lru_cache(maxsize=None)
def _load_packages(backbone: str) -> dict[str, dict]:
    packages = {}
    folders = sorted((ROOT / backbone).glob("*")) + sorted((user_root() / backbone).glob("hub_*"))
    for folder in folders:
        if not folder.is_dir():
            continue
        try:
            if folder.is_symlink() or any(p.is_symlink() for p in folder.rglob("*")):
                raise ValueError("Package symlinks are not allowed")
            for required in ("expert.json", "lora.json", "prompt.txt", "README.md"):
                if not (folder / required).is_file() or not (folder / required).read_text().strip():
                    raise ValueError(f"Missing or empty {required}")
            if any(p.suffix.lower() in {".gguf", ".safetensors", ".bin", ".pt", ".pth"} for p in folder.rglob("*")):
                raise ValueError("Store weights in Hugging Face, not in the expert package")
            profile = json.loads((folder / "expert.json").read_text())
            link = json.loads((folder / "lora.json").read_text())
            if not isinstance(profile, dict) or not isinstance(link, dict):
                raise ValueError("expert.json and lora.json must be objects")
            from core_models import adapter_base_names
            accepted = link.get("accepted_base_names")
            if (not isinstance(accepted, list) or not accepted
                    or any(not isinstance(name, str) for name in accepted)
                    or not set(accepted).issubset(adapter_base_names(backbone))):
                raise ValueError("accepted_base_names must match this backbone")
            if not isinstance(link.get("display_id"), str) or not link["display_id"].strip():
                raise ValueError("display_id is required")
            for field in ("name", "description"):
                if not isinstance(profile.get(field), str) or not profile[field].strip():
                    raise ValueError(f"expert.{field} is required")
            for field in ("capabilities", "aliases", "keywords"):
                if not isinstance(profile.get(field), list) or any(not isinstance(v, str) or not v.strip() for v in profile[field]):
                    raise ValueError(f"expert.{field} must be a list of nonempty strings")
            if not profile["capabilities"]:
                raise ValueError("At least one capability is required")
            if profile.get("schema_version") != 1 or link.get("schema_version") != 1:
                raise ValueError("Unsupported package schema_version")
            if profile.get("id") != folder.name or re.fullmatch(r"[a-z][a-z0-9_]*", folder.name) is None:
                raise ValueError("Package folder must match the lowercase expert id")
            if profile.get("backbone") != backbone or not isinstance(profile.get("default_enabled", False), bool):
                raise ValueError("Invalid backbone or default_enabled")
            if type(profile.get("manual_only", False)) is not bool:
                raise ValueError("manual_only must be a boolean")
            routing = profile.get("routing") if not profile.get("manual_only") else profile.get("routing", {})
            if not isinstance(routing, dict):
                raise ValueError("routing must be an object")
            if not profile.get("manual_only"):
                if not isinstance(routing.get("parent"), str) or not routing["parent"]:
                    raise ValueError("routing.parent is required")
                if not isinstance(routing.get("description"), str) or not routing["description"].strip():
                    raise ValueError("routing.description is required")
                if not isinstance(routing.get("node_id", folder.name), str) or not routing.get("node_id", folder.name).strip():
                    raise ValueError("routing.node_id must be a string")
            source = link["source"]
            repo, revision = _source(source)
            config = _relative(source["config"], "adapter_config.json")
            weights = _relative(source["weights"], ".safetensors")
            row = {"repo_id": repo, "revision": revision, "display_id": link["display_id"],
                   "accepted_base_names": link["accepted_base_names"],
                   "source_config_filename": config, "source_filename": weights,
                   "source_subdir": str(PurePosixPath(config).parent) if "/" in config else ""}
            if "gguf" in link:
                gguf = link["gguf"]
                if not isinstance(gguf, dict):
                    raise ValueError("gguf must be an object")
                filename = _relative(gguf["filename"], ".gguf")
                if "/" in filename:
                    raise ValueError("GGUF cache filename must be a basename")
                row["gguf_file"] = filename
                if "source" in gguf:
                    gr, gv = _source(gguf["source"])
                    row.update(gguf_repo_id=gr, gguf_revision=gv,
                               gguf_source_filename=_relative(gguf["source"]["weights"], ".gguf"))
            policy_path = folder / "response_policy.json"
            policy = json.loads(policy_path.read_text()) if policy_path.exists() else {}
            if not isinstance(policy, dict) or any(not isinstance(policy.get(k, ""), str)
                                                  for k in ("final_guidance", "review_guidance")):
                raise ValueError("Policy guidance must be strings")
            if type(policy.get("thinking", False)) is not bool:
                raise ValueError("Policy thinking must be a boolean")
            packages[folder.name] = {"profile": profile, "adapter": row, "routing": routing,
                                     "prompt": (folder / "prompt.txt").read_text().strip(),
                                     "policy": policy, "path": folder,
                                     "evaluation": _evaluation(folder / "evaluation.json")}
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise ValueError(f"Invalid expert package {folder}: {exc}") from exc
    gguf_files = [p["adapter"].get("gguf_file") for p in packages.values() if p["adapter"].get("gguf_file")]
    if len(gguf_files) != len(set(gguf_files)):
        raise ValueError("GGUF cache filenames must be unique")
    return packages


def package_router_config(raw: dict, packages: dict) -> dict:
    """Compose and validate routes without importing an inference backend."""
    by_id = {node["id"]: node for node in raw["nodes"]}
    if len(by_id) != len(raw["nodes"]):
        raise ValueError("Duplicate router node id")
    for expert_id, package in packages.items():
        if blocked(package) or manual_only(package):
            # Blocked: measured as no better than the base model. Manual: added from Hugging
            # Face without a curated description, so only the user chooses it.
            continue
        routing = package["routing"]
        parent = by_id.get(routing["parent"])
        if parent is None or parent.get("expert_id"):
            raise ValueError(f"Invalid routing parent for {expert_id}: {routing['parent']}")
        node_id = routing.get("node_id", expert_id)
        if node_id in by_id:
            raise ValueError(f"Duplicate routing node: {node_id}")
        node = {"id": node_id, "description": routing["description"], "expert_id": expert_id}
        parent.setdefault("children", []).append(node_id)
        raw["nodes"].append(node)
        by_id[node_id] = node
    # Each backbone has its own packages: drop scaffold branches left without experts.
    empty = [node for node in raw["nodes"] if node["id"] != raw["root_id"]
             and not node.get("expert_id") and not node.get("children")]
    while empty:
        for node in empty:
            raw["nodes"].remove(node)
            del by_id[node["id"]]
            for parent in raw["nodes"]:
                if node["id"] in parent.get("children", []):
                    parent["children"].remove(node["id"])
        empty = [node for node in raw["nodes"] if node["id"] != raw["root_id"]
                 and not node.get("expert_id") and not node.get("children")]
    def visit(node_id: str, ancestors: set[str], depth: int) -> set[str]:
        if node_id in ancestors or depth > min(raw.get("max_depth", 6), 6):
            raise ValueError("Routing cycle or depth limit exceeded")
        node = by_id[node_id]
        children = node.get("children", [])
        if len(children) != len(set(children)) or len(children) > 15:
            raise ValueError(f"Invalid children or branching limit exceeded: {node_id}")
        if bool(children) == bool(node.get("expert_id")) and node_id != raw["root_id"]:
            raise ValueError(f"Route must be a branch or expert leaf: {node_id}")
        reached = {node_id}
        for child in children:
            reached.update(visit(child, ancestors | {node_id}, depth + 1))
        return reached
    if visit(raw["root_id"], set(), 1) != set(by_id):
        raise ValueError("Unreachable routing nodes")
    return raw
