"""Community LoRA adapters found on Hugging Face for the active base model.

These are unverified: they have no curated description, prompt, or measured benefit, so
the router never picks them and the user chooses one by hand. Adding one writes a small
expert package under the user's configuration folder, pinned to the repository's current
commit. A curated package in experts/<backbone>/ with a passing evaluation.json is how an
adapter becomes a verified GhostSwitch expert.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from core_models import adapter_base_names

from .expert_packages import ROOT, blocked, load_packages, reload_packages, user_root

HUB = "https://huggingface.co"
_CACHE_SECONDS = 1800
_cache: dict[str, tuple[float, list[dict]]] = {}
_lock = threading.Lock()


def _get(url: str, raw: bool = False):
    request = urllib.request.Request(url, headers={"User-Agent": "GhostSwitch"})
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read()
    return body.decode("utf-8", "replace") if raw else json.loads(body)


def _hub_bases(backbone: str) -> list[str]:
    """Base model repositories adapters may declare for this backbone, such as Qwen/Qwen3.5-9B."""
    return sorted(name for name in adapter_base_names(backbone) if re.fullmatch(r"[\w.-]+/[\w.-]+", name))


def blocked_repos(backbone: str) -> dict[str, str]:
    """Repositories never offered: listed in blocklist.json, or measured and blocked as experts."""
    result = {}
    path = ROOT / backbone / "blocklist.json"
    if path.exists():
        value = json.loads(path.read_text())
        for row in value.get("repos", []):
            result[row["repo_id"]] = row.get("reason", "")
    for package in load_packages(backbone).values():
        if blocked(package):
            result[package["adapter"]["repo_id"]] = package["evaluation"]["reason"]
    return result


def _adapter_dir(files: list[str]) -> str | None:
    """The folder holding a PEFT LoRA (config and weights side by side), the shallowest first."""
    configs = {name.rsplit("/", 1)[0] if "/" in name else "" for name in files if name.endswith("adapter_config.json")}
    found = [folder for folder in configs
             if (f"{folder}/adapter_model.safetensors" if folder else "adapter_model.safetensors") in files
             and not re.search(r"(^|/)checkpoint-\d+", folder)]
    return min(found, key=lambda folder: (folder.count("/"), len(folder))) if found else None


def _slug(repo_id: str) -> str:
    return "hub_" + re.sub(r"[^a-z0-9]+", "_", repo_id.lower()).strip("_")[:60]


def discover(backbone: str, refresh: bool = False) -> list[dict]:
    """LoRA adapters on Hugging Face for this backbone, most downloaded first."""
    with _lock:
        cached = _cache.get(backbone)
        if cached and not refresh and time.monotonic() - cached[0] < _CACHE_SECONDS:
            return cached[1]
    curated = {package["adapter"]["repo_id"] for package in load_packages(backbone).values()
               if not package["profile"].get("manual_only")}
    skip = blocked_repos(backbone)
    installed = {package["adapter"]["repo_id"]: expert_id for expert_id, package in load_packages(backbone).items()
                 if package["profile"].get("manual_only")}
    rows: dict[str, dict] = {}
    for base in _hub_bases(backbone):
        query = urllib.parse.urlencode({"filter": f"base_model:adapter:{base}", "sort": "downloads",
                                        "direction": -1, "limit": 1000, "full": "true"})
        for model in _get(f"{HUB}/api/models?{query}"):
            repo = model["id"]
            folder = _adapter_dir([sibling["rfilename"] for sibling in model.get("siblings", [])])
            if (folder is None or model.get("private") or model.get("gated") or repo in rows
                    or repo in curated or repo in skip):
                continue
            rows[repo] = {"repo_id": repo, "name": repo.split("/", 1)[1], "author": model.get("author"),
                          "downloads": model.get("downloads", 0), "likes": model.get("likes", 0),
                          "updated": model.get("lastModified"), "adapter_dir": folder,
                          "url": f"{HUB}/{repo}", "installed": installed.get(repo)}
    result = sorted(rows.values(), key=lambda row: (-row["downloads"], -row["likes"], row["repo_id"]))
    with _lock:
        _cache[backbone] = (time.monotonic(), result)
    return result


def _compatible(config: dict, backbone: str) -> bool:
    """The adapter was trained on this instruct model, not a base or distilled variant."""
    base = str(config.get("base_model_name_or_path") or "")
    names = adapter_base_names(backbone)
    return base in names or base.rstrip("/").rsplit("/", 1)[-1] in {name.rsplit("/", 1)[-1] for name in names}


def _summary(readme: str, repo: str) -> str:
    """The first prose paragraph of a model card, without front matter, headings, or templates."""
    text = re.sub(r"\A---.*?---", "", readme, flags=re.S)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    for block in re.split(r"\n\s*\n", text):
        # Cards often open with HTML banners and badges; keep only the prose.
        block = " ".join(re.sub(r"<[^>]*>|!\[[^\]]*\]\([^)]*\)", " ", block).split())
        if (len(block) > 40 and not block.startswith(("#", "|", "-", "*", "```", ">", "[", "!["))
                and "More Information Needed" not in block):
            return block[:400]
    return f"Community LoRA adapter from {repo} on Hugging Face."


def install(backbone: str, repo_id: str) -> str:
    """Add an adapter as a manual-only expert package pinned to the current commit."""
    if repo_id in blocked_repos(backbone):
        raise ValueError(f"{repo_id} is blocked: {blocked_repos(backbone)[repo_id]}")
    try:
        model = _get(f"{HUB}/api/models/{urllib.parse.quote(repo_id, safe='/')}")
    except urllib.error.HTTPError as error:
        raise ValueError(f"{repo_id} was not found on Hugging Face ({error.code})") from error
    if model.get("private") or model.get("gated"):
        raise ValueError(f"{repo_id} is private or gated")
    folder = _adapter_dir([sibling["rfilename"] for sibling in model.get("siblings", [])])
    if folder is None:
        raise ValueError(f"{repo_id} has no PEFT LoRA adapter (adapter_config.json with adapter_model.safetensors)")
    revision = model["sha"]
    prefix = f"{folder}/" if folder else ""
    config = _get(f"{HUB}/{repo_id}/raw/{revision}/{prefix}adapter_config.json")
    if str(config.get("peft_type", "LORA")).upper() != "LORA":
        raise ValueError(f"{repo_id} is a {config.get('peft_type')} adapter, not LoRA")
    if not _compatible(config, backbone):
        raise ValueError(f"{repo_id} was trained on {config.get('base_model_name_or_path')!r}, "
                         f"not on this base model")
    try:
        readme = _get(f"{HUB}/{repo_id}/raw/{revision}/README.md", raw=True)
    except urllib.error.HTTPError:
        readme = ""
    expert_id = _slug(repo_id)
    target = user_root() / backbone / expert_id
    temporary = target.with_name(target.name + ".partial")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    name = repo_id.split("/", 1)[1].replace("-", " ").replace("_", " ")
    description = _summary(readme, repo_id)
    profile = {"id": expert_id, "name": name, "description": description, "capabilities": [name],
               "aliases": [repo_id], "keywords": [], "schema_version": 1, "backbone": backbone,
               "default_enabled": False, "manual_only": True, "source": "huggingface"}
    base = sorted(_hub_bases(backbone))[0]
    link = {"schema_version": 1, "display_id": expert_id.removeprefix("hub_")[:48],
            "accepted_base_names": [base],
            "source": {"url": f"{HUB}/{repo_id}", "revision": revision,
                       "config": f"{prefix}adapter_config.json", "weights": f"{prefix}adapter_model.safetensors"},
            "gguf": {"filename": f"{expert_id}.gguf"}}
    (temporary / "expert.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n")
    (temporary / "lora.json").write_text(json.dumps(link, indent=2) + "\n")
    # Unverified adapters get no task prompt: the user's request and the base contract apply.
    (temporary / "prompt.txt").write_text("Answer the user's request.\n")
    (temporary / "README.md").write_text(
        f"# {name}\n\nAdded from {HUB}/{repo_id} at commit `{revision}`. Not verified for GhostSwitch: "
        "no curated description, prompt, or measured benefit. Chosen by hand only.\n")
    shutil.rmtree(target, ignore_errors=True)
    temporary.rename(target)
    reload_packages()
    try:
        load_packages(backbone)[expert_id]  # Validates the package like a curated one.
    except (KeyError, ValueError) as error:
        shutil.rmtree(target, ignore_errors=True)
        reload_packages()
        raise ValueError(f"{repo_id} could not be added: {error}") from error
    with _lock:
        _cache.pop(backbone, None)
    return expert_id


def remove(backbone: str, expert_id: str) -> None:
    """Delete an adapter the user added; curated experts cannot be removed this way."""
    target = user_root() / backbone / expert_id
    if not expert_id.startswith("hub_") or not target.is_dir():
        raise KeyError(expert_id)
    shutil.rmtree(target)
    reload_packages()
    with _lock:
        _cache.pop(backbone, None)
