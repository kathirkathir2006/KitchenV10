"""Open-set decision engine — fail closed; never force unknown into known."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kitcheniq_v10.v3.types import Hypothesis, RegionDecision

# Calibrated on validation heuristics ONLY — never tuned on 41-scene GT.
# These constants were chosen from retrieval score distributions on prototype
# library holdout-style splits (random identity folds), not acceptance scenes.
SIM_KNOWN = 0.62
SIM_PROBABLE = 0.48
MARGIN_KNOWN = 0.08
MARGIN_PROBABLE = 0.04
MIN_SOURCES_KNOWN = 2


@dataclass
class OpenSetDecision:
    sim_known: float = SIM_KNOWN
    sim_probable: float = SIM_PROBABLE
    margin_known: float = MARGIN_KNOWN
    margin_probable: float = MARGIN_PROBABLE
    min_sources_known: int = MIN_SOURCES_KNOWN


def decide_region(
    *,
    hypotheses: list[Hypothesis],
    food_gate: dict[str, Any],
    image_quality_ok: bool = True,
    cfg: OpenSetDecision | None = None,
) -> RegionDecision:
    cfg = cfg or OpenSetDecision()
    if not image_quality_ok:
        return RegionDecision(
            status="INSUFFICIENT_EVIDENCE",
            identity=None,
            family=None,
            kind=None,
            food_state=None,
            confidence=0.0,
            hypotheses=hypotheses,
            abstain_reason="image_quality",
        )

    if food_gate.get("is_food") is False or food_gate.get("label") == "non_food":
        return RegionDecision(
            status="NON_FOOD",
            identity="non_food",
            family="non_food",
            kind="non_food",
            food_state=None,
            confidence=float(food_gate.get("confidence") or 0.7),
            hypotheses=hypotheses,
            provenance={"food_gate": food_gate},
        )

    if not hypotheses:
        kind = food_gate.get("kind") or "ingredient"
        if kind == "prepared_meal":
            status = "UNKNOWN_PREPARED_MEAL"
        else:
            status = "UNKNOWN_FOOD"
        return RegionDecision(
            status=status,
            identity=None,
            family=food_gate.get("family"),
            kind=kind,
            food_state=food_gate.get("food_state"),
            confidence=float(food_gate.get("confidence") or 0.4),
            hypotheses=[],
            abstain_reason="no_hypotheses",
            provenance={"food_gate": food_gate},
        )

    top = hypotheses[0]
    second = hypotheses[1] if len(hypotheses) > 1 else None
    margin = top.score - (second.score if second else 0.0)
    n_src = len(set(top.sources))

    # Conflicting: two strong near-ties with different identities
    if second and margin < cfg.margin_probable and second.score >= cfg.sim_probable:
        if top.identity != second.identity:
            return RegionDecision(
                status="CONFLICTING_EVIDENCE",
                identity=None,
                family=top.family,
                kind=top.kind,
                food_state=top.food_state,
                confidence=float(margin),
                hypotheses=hypotheses[:5],
                abstain_reason="top_margin_conflict",
                provenance={"top": top.identity, "second": second.identity},
            )

    if (
        top.score >= cfg.sim_known
        and margin >= cfg.margin_known
        and n_src >= cfg.min_sources_known
        and top.identity
    ):
        return RegionDecision(
            status="KNOWN",
            identity=top.identity,
            family=top.family,
            kind=top.kind,
            food_state=top.food_state,
            confidence=float(min(0.99, top.score)),
            hypotheses=hypotheses[:5],
            provenance={"sources": top.sources, "evidence": top.evidence},
        )

    # Single strong specialist signal allowed for KNOWN when multi-source unavailable
    # (retrieval library may be scaffold-only on Windows). Still requires margin.
    spec = float((top.evidence or {}).get("specialist_support") or 0.0)
    if top.identity and spec >= 0.55 and margin >= cfg.margin_known and "specialist_support" in top.sources:
        return RegionDecision(
            status="KNOWN",
            identity=top.identity,
            family=top.family,
            kind=top.kind,
            food_state=top.food_state,
            confidence=float(min(0.95, spec)),
            hypotheses=hypotheses[:5],
            provenance={"sources": top.sources, "path": "strong_specialist"},
        )

    if top.score >= cfg.sim_probable and margin >= cfg.margin_probable and top.identity:
        return RegionDecision(
            status="PROBABLE_KNOWN",
            identity=top.identity,
            family=top.family,
            kind=top.kind,
            food_state=top.food_state,
            confidence=float(top.score * 0.85),
            hypotheses=hypotheses[:5],
            abstain_reason=None,
            provenance={"sources": top.sources, "band": "probable"},
        )

    if top.identity and spec >= 0.35 and margin >= 0.02:
        return RegionDecision(
            status="PROBABLE_KNOWN",
            identity=top.identity,
            family=top.family,
            kind=top.kind,
            food_state=top.food_state,
            confidence=float(spec * 0.8),
            hypotheses=hypotheses[:5],
            provenance={"sources": top.sources, "path": "probable_specialist"},
        )

    # Open-world: food known as category but identity unsupported
    if (food_gate.get("kind") or top.kind) == "prepared_meal":
        status = "UNKNOWN_PREPARED_MEAL"
    else:
        status = "UNKNOWN_FOOD"
    return RegionDecision(
        status=status,
        identity=None,
        family=top.family or food_gate.get("family"),
        kind=top.kind or food_gate.get("kind"),
        food_state=top.food_state or food_gate.get("food_state"),
        confidence=float(max(top.score, food_gate.get("confidence") or 0.3)),
        hypotheses=hypotheses[:5],
        abstain_reason="below_calibrated_threshold",
        provenance={"top_score": top.score, "margin": margin, "n_sources": n_src},
    )
