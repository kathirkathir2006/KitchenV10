"""V3 typed outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Hypothesis:
    identity: str | None
    family: str | None
    kind: str | None  # ingredient | packaged | prepared_meal | non_food
    food_state: str | None
    score: float
    sources: list[str] = field(default_factory=list)
    evidence: dict[str, float] = field(default_factory=dict)


@dataclass
class RegionDecision:
    status: str  # KNOWN|PROBABLE_KNOWN|UNKNOWN_FOOD|UNKNOWN_PREPARED_MEAL|NON_FOOD|CONFLICTING_EVIDENCE|INSUFFICIENT_EVIDENCE
    identity: str | None
    family: str | None
    kind: str | None
    food_state: str | None
    confidence: float
    hypotheses: list[Hypothesis]
    abstain_reason: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class V3Observation:
    status: str
    identity: str | None
    raw_label: str | None
    visual_class: str | None
    food_state: str | None
    confidence: float
    bbox: list[float] | None
    is_food: bool
    is_prepared_meal: bool
    abstain_reason: str | None
    provenance: dict[str, Any]
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
