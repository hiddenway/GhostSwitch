"""Small-footprint lexical candidate retrieval for large expert registries.

The catalog narrows thousands of expert descriptions to a small candidate set;
the router model only reranks that set. Metadata lives outside routing code so
adding an expert does not require adding a Python branch.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


CATALOG_PATH = Path(__file__).with_name("config") / "expert_catalog.json"
MAX_ROUTER_CANDIDATES = 8
_TOKEN = re.compile(r"[\w]+", re.UNICODE)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "can", "do", "for", "from", "how", "i",
    "in", "is", "it", "of", "on", "or", "please", "should", "that", "the", "this", "to",
    "use", "what", "when", "where", "which", "who", "with", "would", "эксперт", "задача",
    "можно", "нужно", "пожалуйста", "сделай", "сделать", "это", "для", "как", "что", "ли",
}


def _normalize(value: str) -> str:
    return " ".join(_TOKEN.findall(value.casefold().replace("_", " ").replace("-", " ")))


def _tokens(value: str) -> list[str]:
    return [token for token in _TOKEN.findall(value.casefold().replace("_", " ").replace("-", " ")) if token not in _STOPWORDS and len(token) > 1]


@dataclass(frozen=True)
class ExpertProfile:
    id: str
    name: str
    capabilities: tuple[str, ...]
    aliases: tuple[str, ...]
    description: str
    keywords: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> "ExpertProfile":
        expert_id = str(value.get("id", "")).strip()
        if not expert_id or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", expert_id):
            raise ValueError(f"Invalid expert id: {expert_id!r}")

        def strings(key: str) -> tuple[str, ...]:
            raw = value.get(key, [])
            if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
                raise ValueError(f"Expert {expert_id!r} field {key!r} must be a list of strings")
            return tuple(item.strip() for item in raw if item.strip())

        name = str(value.get("name", "")).strip()
        description = str(value.get("description", "")).strip()
        if not name or not description:
            raise ValueError(f"Expert {expert_id!r} requires name and description")
        return cls(expert_id, name, strings("capabilities"), strings("aliases"), description, strings("keywords"))

    def to_prompt_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "capabilities": list(self.capabilities),
            "description": self.description,
        }


@dataclass(frozen=True)
class ExpertCandidate:
    profile: ExpertProfile
    score: float
    query_coverage: float


class ExpertCatalog:
    """In-memory inverted index; query cost scales with matching postings."""

    _FIELD_WEIGHTS = {
        "id": 4.0,
        "name": 3.5,
        "capabilities": 4.0,
        "aliases": 4.5,
        "keywords": 2.2,
        "description": 1.0,
    }

    def __init__(self, profiles: Iterable[ExpertProfile | dict[str, object]]) -> None:
        normalized = [item if isinstance(item, ExpertProfile) else ExpertProfile.from_mapping(item) for item in profiles]
        self.profiles = tuple(normalized)
        self.by_id = {profile.id: profile for profile in self.profiles}
        if len(self.by_id) != len(self.profiles):
            raise ValueError("Expert ids must be unique")
        self._term_weights: dict[str, dict[str, float]] = {}
        self._exact_labels: dict[str, set[str]] = {}
        for profile in self.profiles:
            fields = {
                "id": profile.id.replace("_", " ").replace("-", " "),
                "name": profile.name,
                "capabilities": " ".join(profile.capabilities),
                "aliases": " ".join(profile.aliases),
                "keywords": " ".join(profile.keywords),
                "description": profile.description,
            }
            weights: dict[str, float] = {}
            for field, content in fields.items():
                for token in set(_tokens(content)):
                    weights[token] = max(weights.get(token, 0.0), self._FIELD_WEIGHTS[field])
            for token, weight in weights.items():
                self._term_weights.setdefault(token, {})[profile.id] = weight
            for label in (profile.id, profile.name, *profile.capabilities, *profile.aliases):
                normalized_label = _normalize(label)
                if normalized_label:
                    self._exact_labels.setdefault(normalized_label, set()).add(profile.id)

    def reload(self) -> None:
        """Rebuild in place, so modules holding EXPERT_CATALOG see added experts."""
        self.__dict__.update(ExpertCatalog.load().__dict__)

    @classmethod
    def load(cls, path: Path = CATALOG_PATH) -> "ExpertCatalog":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
            raise ValueError("Expert catalog must be a JSON array of profile objects")
        if path == CATALOG_PATH:
            from .expert_packages import load_packages
            value += [item["profile"] for item in load_packages().values()]
        return cls(value)

    def resolve_exact(self, query: str, allowed_ids: Sequence[str] | None = None) -> str | None:
        target = _normalize(query)
        if not target:
            return None
        allowed = set(allowed_ids) if allowed_ids is not None else set(self.by_id)
        matches = self._exact_labels.get(target, set()) & allowed
        return next(iter(matches)) if len(matches) == 1 else None

    def search(
        self,
        query: str,
        *,
        allowed_ids: Sequence[str] | None = None,
        limit: int = MAX_ROUTER_CANDIDATES,
    ) -> list[ExpertCandidate]:
        if limit <= 0:
            return []
        allowed = set(allowed_ids) if allowed_ids is not None else set(self.by_id)
        terms = set(_tokens(query))
        if not terms or not allowed:
            return []
        total_docs = max(1, len(allowed))
        idf: dict[str, float] = {}
        scores: dict[str, float] = {}
        matched_query_weight = 0.0
        query_weight = 0.0
        for term in terms:
            posting = self._term_weights.get(term, {})
            df = sum(expert_id in allowed for expert_id in posting)
            term_idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5)) if df else math.log(1 + total_docs)
            idf[term] = term_idf
            query_weight += term_idf
            if df:
                matched_query_weight += term_idf
                for expert_id, field_weight in posting.items():
                    if expert_id in allowed:
                        scores[expert_id] = scores.get(expert_id, 0.0) + term_idf * field_weight

        if not scores:
            return []

        normalized_query = _normalize(query)
        padded_query = f" {normalized_query} "
        for expert_id in scores:
            profile = self.by_id[expert_id]
            phrases = (profile.id, profile.name, *profile.capabilities, *profile.aliases)
            if any(f" {_normalize(phrase)} " in padded_query for phrase in phrases if _normalize(phrase)):
                scores[expert_id] += 5.0

        coverage = matched_query_weight / query_weight if query_weight else 0.0
        return [
            ExpertCandidate(self.by_id[expert_id], score, coverage)
            for expert_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
        ]


EXPERT_CATALOG = ExpertCatalog.load()
