"""Deterministic evidence fusion + conflict resolution for specialist ensemble."""

from __future__ import annotations

from typing import Any

from kitcheniq_v10.specialist.types import FusedObservation, SpecialistPred, band


PREPARED_MEAL_LABELS = {
    "biryani",
    "fried_rice",
    "pizza",
    "omelette",
    "pasta",
    "salad",
    "sandwich",
    "burrito",
    "taco",
    "sushi",
    "lasagna",
    "hamburger",
    "hot_dog",
    "bibimbap",
    "club_sandwich",
    "prepared_meal",
    "fast_food",
}

NONFOOD_HINTS = {
    "non_food",
    "empty_plate",
    "hand",
    "human_hand",
    "kitchen_utensil",
    "utensil",
    "cookware",
    "sink",
    "smartphone_screen",
    "cutting_board_empty",
    "empty_packaging",
}


def fuse_specialists(
    *,
    food_nonfood: SpecialistPred,
    identity: SpecialistPred,
    prepared: SpecialistPred,
    food_state: SpecialistPred,
    bbox: list[float] | None = None,
    detector_confidence: float | None = None,
    ocr: dict[str, Any] | None = None,
    barcode: dict[str, Any] | None = None,
) -> FusedObservation:
    """Fuse specialist outputs with deterministic conflict rules."""
    provenance: dict[str, Any] = {
        "food_nonfood": {"label": food_nonfood.label, "confidence": food_nonfood.confidence, "source": food_nonfood.name},
        "identity": {"label": identity.label, "confidence": identity.confidence, "source": identity.name},
        "prepared_meal": {"label": prepared.label, "confidence": prepared.confidence, "source": prepared.name},
        "food_state": {"label": food_state.label, "confidence": food_state.confidence, "source": food_state.name},
        "ocr": ocr,
        "barcode": barcode,
    }

    fn_lab = (food_nonfood.label or "").lower()
    fn_conf = float(food_nonfood.confidence)
    id_lab = identity.label
    id_conf = float(identity.confidence)
    prep_lab = (prepared.label or "").lower() if prepared.label else None
    prep_conf = float(prepared.confidence)
    state_lab = food_state.label
    state_conf = float(food_state.confidence)

    # Hard conflict: food vs nonfood both high
    is_food_pred = fn_lab in {
        "food",
        "raw_ingredient",
        "prepared_meal",
        "packaged",
        "packaged_food",
        "mixed",
    }
    is_nonfood_pred = fn_lab in {"non_food", "nonfood", "empty", "unknown"} or fn_lab in NONFOOD_HINTS
    if is_food_pred and is_nonfood_pred and fn_conf >= 0.85:
        # shouldn't happen with single label; treat ambiguous visual_class
        pass

    # Generic closed-set dump labels are not identities
    GENERIC_ID = {"food", "unknown", "unable_to_determine", "none", "null", "mixed", "mixed_scene"}

    # Explicit non-food branch
    if fn_lab in {"non_food", "nonfood"} and fn_conf >= 0.55:
        # If identity is high-confidence food label, conflict
        if id_lab and id_lab.lower() not in NONFOOD_HINTS and id_conf >= 0.85 and fn_conf >= 0.85:
            return FusedObservation(
                status="conflicting",
                identity=None,
                food_state=None,
                is_food=None,
                is_prepared_meal=False,
                confidence=min(fn_conf, id_conf),
                confidence_band=band(min(fn_conf, id_conf)),
                provenance=provenance,
                bbox=bbox,
                detector_confidence=detector_confidence,
                needs_confirmation=True,
                abstain_reason="food_nonfood_identity_conflict",
            )
        return FusedObservation(
            status="non_food",
            identity="non_food",
            food_state=None,
            is_food=False,
            is_prepared_meal=False,
            confidence=fn_conf,
            confidence_band=band(fn_conf),
            provenance=provenance,
            bbox=bbox,
            detector_confidence=detector_confidence,
            raw_label="non_food",
            visual_class="non_food",
        )

    # Prepared-meal dominance over weak raw identity
    if prep_lab in PREPARED_MEAL_LABELS or prep_lab in {"prepared", "prepared_meal", "plated", "leftover"}:
        if prep_conf >= 0.55 and (id_conf < 0.85 or (id_lab or "").lower() in {"rice", "food", "vegetable", "pasta"}):
            meal_id = prep_lab if prep_lab in PREPARED_MEAL_LABELS else (id_lab if id_conf >= 0.55 else "prepared_meal")
            status = "confirmed_identity" if prep_conf >= 0.85 else "probable_identity"
            if prep_conf < 0.55:
                status = "abstain"
            return FusedObservation(
                status=status if status != "abstain" else "probable_identity",
                identity=meal_id,
                food_state=(
                    "plated"
                    if (state_lab in {None, "cooked", "prepared", "plated"} or state_conf < 0.55)
                    else state_lab
                ),
                is_food=True,
                is_prepared_meal=True,
                confidence=max(prep_conf, id_conf * 0.5),
                confidence_band=band(max(prep_conf, id_conf * 0.5)),
                provenance=provenance,
                bbox=bbox,
                detector_confidence=detector_confidence,
                raw_label=meal_id,
                visual_class="prepared_meal",
            )

    id_norm = (id_lab or "").strip().lower().replace(" ", "_")
    # Treat generic / dump labels as no identity
    if id_norm in GENERIC_ID:
        id_lab = None
        id_conf = 0.0
        id_norm = ""

    # Identity path
    if not id_lab or id_conf < 0.55:
        if is_food_pred and fn_conf >= 0.55:
            return FusedObservation(
                status="unknown_food",
                identity=None,
                food_state=state_lab if state_conf >= 0.55 else None,
                is_food=True,
                is_prepared_meal=False,
                confidence=fn_conf,
                confidence_band=band(fn_conf),
                provenance=provenance,
                bbox=bbox,
                detector_confidence=detector_confidence,
                visual_class=fn_lab,
                needs_confirmation=True,
                abstain_reason="identity_below_medium",
            )
        # Whole-image low-evidence: prefer abstain over false food identity
        return FusedObservation(
            status="abstain",
            identity=None,
            food_state=None,
            is_food=None,
            is_prepared_meal=None,
            confidence=max(id_conf, fn_conf),
            confidence_band=band(max(id_conf, fn_conf)),
            provenance=provenance,
            bbox=bbox,
            detector_confidence=detector_confidence,
            needs_confirmation=True,
            abstain_reason="insufficient_evidence",
        )

    status = "confirmed_identity" if id_conf >= 0.85 else "probable_identity"
    is_prep = id_norm in PREPARED_MEAL_LABELS or (
        prep_conf >= 0.7 and prep_lab in {"prepared", "prepared_meal", "plated"}
    )
    # Prefer plated for prepared meals when state is cooked/prepared
    out_state = state_lab if state_conf >= 0.55 else None
    if is_prep and (out_state in {None, "cooked", "prepared"}):
        out_state = "plated" if (prep_lab in {"prepared_meal", "plated"} or is_prep) else out_state
    return FusedObservation(
        status=status,
        identity=id_lab,
        food_state=out_state,
        is_food=True,
        is_prepared_meal=is_prep,
        confidence=id_conf,
        confidence_band=band(id_conf),
        provenance=provenance,
        bbox=bbox,
        detector_confidence=detector_confidence,
        raw_label=id_lab,
        visual_class="prepared_meal" if is_prep else (fn_lab if is_food_pred else "raw_ingredient"),
    )
