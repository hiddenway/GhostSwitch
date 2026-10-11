"""Unified v6 view of adapter, prompt and semantic expert metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import llm
from .expert_catalog import EXPERT_CATALOG, ExpertProfile
from .expert_packages import blocked, load_packages, manual_only
from .lora_runtime import ADAPTERS, AdapterSpec, display_adapter_id

_PROMPT_ROOT = Path(__file__).resolve().parents[1] / "base"


@dataclass(frozen=True)
class ExpertDefinition:
    expert_id: str
    adapter_id: str
    profile: ExpertProfile
    adapter: AdapterSpec
    prompt: str | None
    # response_policy.json "thinking": True or False fixes reasoning for this specialization;
    # None leaves it to the automatic check (core/thinking_decision.py).
    thinking: bool | None = None


class ExpertRegistry:
    def get(self, expert_id: str) -> ExpertDefinition:
        profile = EXPERT_CATALOG.by_id.get(expert_id)
        adapter = ADAPTERS.get(expert_id)
        if profile is None or adapter is None:
            raise KeyError(expert_id)
        path = _PROMPT_ROOT / llm.ACTIVE_BACKBONE / "prompts" / f"{expert_id}.txt"
        package = load_packages(llm.ACTIVE_BACKBONE).get(expert_id)
        prompt = package["prompt"] if package else (path.read_text(encoding="utf-8").strip() if path.is_file() else None)
        if package and package["policy"].get("final_guidance"):
            prompt += "\n\n" + package["policy"]["final_guidance"]
        thinking = package["policy"].get("thinking") if package else None
        return ExpertDefinition(expert_id, display_adapter_id(expert_id), profile, adapter, prompt, thinking)

    def list_available(self) -> list[ExpertDefinition]:
        backend = llm.MODEL_SPECS["thinker"].backend
        return [self.get(expert_id) for expert_id in sorted(ADAPTERS)
                if expert_id in EXPERT_CATALOG.by_id and self.is_available(expert_id, backend)]

    def is_available(self, expert_id: str, backend: str) -> bool:
        if expert_id not in ADAPTERS or expert_id not in EXPERT_CATALOG.by_id:
            return False
        from .expert_library import LIBRARY
        if not LIBRARY.enabled(expert_id):
            return False
        if backend == "gguf":
            return ADAPTERS[expert_id].gguf_file is not None and LIBRARY.downloaded(expert_id)
        return backend == "mlx" and LIBRARY.downloaded(expert_id)

    def review_guidance(self, expert_id: str) -> str:
        package = load_packages(llm.ACTIVE_BACKBONE).get(expert_id)
        return package["policy"].get("review_guidance", "") if package else ""

    def resolve_adapter(self, expert_id: str) -> AdapterSpec:
        return self.get(expert_id).adapter

    def get_prompt(self, expert_id: str) -> str | None:
        return self.get(expert_id).prompt

    def check_backbone_compatibility(self, expert_id: str) -> bool:
        definition = self.get(expert_id)
        backend = llm.MODEL_SPECS["thinker"].backend
        return (backend == "gguf" and definition.adapter.gguf_file is not None
                or backend == "mlx" and definition.adapter.base_model_id == llm.MODEL_SPECS["thinker"].model_id)

    def descriptions(self) -> dict[str, str]:
        """Experts routing may choose: available, and not blocked by an evaluation."""
        packages = load_packages(llm.ACTIVE_BACKBONE)
        return {item.expert_id: item.profile.description for item in self.list_available()
                if item.expert_id in packages and not blocked(packages[item.expert_id])
                and not manual_only(packages[item.expert_id])}


EXPERT_REGISTRY = ExpertRegistry()
