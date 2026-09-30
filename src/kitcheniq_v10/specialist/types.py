"""Specialist ensemble types and confidence bands."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


HIGH = 0.85
MEDIUM = 0.55


def band(conf: float) -> str:
    if conf >= HIGH:
        return "HIGH"
    if conf >= MEDIUM:
        return "MEDIUM"
    return "LOW"


@dataclass
class SpecialistPred:
    name: str
    label: str | None
    confidence: float
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def confidence_band(self) -> str:
        return band(float(self.confidence))


@dataclass
class FusedObservation:
    status: str  # confirmed_identity | probable_identity | family_only | unknown_food | abstain | non_food | conflicting | confirm_required
    identity: str | None
    food_state: str | None
    is_food: bool | None
    is_prepared_meal: bool | None
    confidence: float
    confidence_band: str
    provenance: dict[str, Any]
    bbox: list[float] | None = None
    detector_confidence: float | None = None
    raw_label: str | None = None
    visual_class: str | None = None
    needs_confirmation: bool = False
    abstain_reason: str | None = None


OPEN_WORLD_STATUSES = {
    "confirmed_identity",
    "probable_identity",
    "family_only",
    "unknown_food",
    "abstain",
    "non_food",
    "conflicting",
    "confirm_required",
    "unable_to_determine",
}
