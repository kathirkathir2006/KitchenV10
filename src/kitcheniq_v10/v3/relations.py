"""Cross-object relationship reasoning — modifies confidence only."""

from __future__ import annotations

from typing import Any


def relate_regions(region_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return context evidence edges (confidence modifiers), never forced identities."""
    edges: list[dict[str, Any]] = []
    kinds = [r.get("kind") for r in region_summaries]
    states = [r.get("food_state") for r in region_summaries]
    ids = [r.get("identity") for r in region_summaries]

    has_plate = any(i in {"empty_plate", "plate"} or k == "non_food" and "plate" in str(i) for i, k in zip(ids, kinds))
    has_prepared = any(k == "prepared_meal" or (s in {"prepared", "plated"}) for k, s in zip(kinds, states))
    has_packaged = any(k == "packaged" for k in kinds)
    has_ingredient = any(k == "ingredient" for k in kinds)

    for i, r in enumerate(region_summaries):
        rid = r.get("region_id") or f"r{i}"
        if has_prepared and r.get("kind") == "ingredient" and r.get("identity") in {"cooked_rice", "rice", "pasta"}:
            edges.append(
                {
                    "source": "scene_context",
                    "target": r.get("identity") or "cooked_rice",
                    "kind": "co_occurrence",
                    "weight": 0.05,
                    "detail": {"context": "prepared_meal_scene", "region": rid},
                }
            )
        if has_plate and r.get("kind") in {"prepared_meal", "ingredient"} and r.get("food_state") in {None, "prepared", "unknown"}:
            edges.append(
                {
                    "source": "scene_context",
                    "target": r.get("identity") or "unknown",
                    "kind": "spatial_compatibility",
                    "weight": 0.04,
                    "detail": {"context": "plated", "region": rid},
                }
            )
        if has_packaged and r.get("identity") == "cheese":
            edges.append(
                {
                    "source": "scene_context",
                    "target": "packaged_cheese",
                    "kind": "co_occurrence",
                    "weight": 0.06,
                    "detail": {"context": "packaging_present", "region": rid},
                }
            )
        if has_ingredient and has_prepared and r.get("identity") in {"biryani", "fried_rice", "pizza", "salad"}:
            edges.append(
                {
                    "source": "scene_context",
                    "target": r["identity"],
                    "kind": "spatial_compatibility",
                    "weight": 0.05,
                    "detail": {"context": "multi_object_prepared", "region": rid},
                }
            )
    return edges
