"""Specialist recognizers — independent evidence sources (not final identity)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from kitcheniq_v10.v3.retrieval import FAMILY_OF, KIND_OF, STATE_HINT


class FoodNonFoodSpecialist:
    """Conservative food gate — prefer food when uncertain (fail soft on FN).

    V3 failure matrix @0.2927: earliest food/non-food = 25, subtype C = 24.
    Therefore: never veto food when any top-k food identity has non-trivial mass.
    """

    def __call__(
        self,
        *,
        visual_class: str | None,
        label: str | None,
        conf: float,
        retrieval_food_score: float,
        topk_food_mass: float = 0.0,
        topk_identities: list[str] | None = None,
    ) -> dict[str, Any]:
        vc = (visual_class or "").lower()
        lab = (label or "").lower()
        tops = [t.lower() for t in (topk_identities or [])]
        non_food_labels = {"non_food", "human_hand", "empty_plate", "hand"}
        foodish = [t for t in tops if t and t not in non_food_labels]
        # Hard non-food only when label evidence is exclusively non-food and no food mass
        exclusive_nonfood = bool(tops) and all(t in non_food_labels for t in tops)
        if exclusive_nonfood and topk_food_mass < 0.08 and retrieval_food_score < 0.25:
            return {"is_food": False, "label": "non_food", "confidence": conf, "kind": "non_food"}
        if (
            lab in non_food_labels
            and conf >= 0.92
            and topk_food_mass < 0.05
            and retrieval_food_score < 0.2
            and not foodish
        ):
            return {"is_food": False, "label": "non_food", "confidence": conf, "kind": "non_food"}
        # Default: food (open-world allows unknown dish later)
        use_lab = foodish[0] if foodish else (lab if lab not in non_food_labels else None)
        fam = FAMILY_OF.get(use_lab or "", "unknown")
        kind = KIND_OF.get(fam, "ingredient")
        if vc == "prepared_meal" or kind == "prepared_meal":
            kind = "prepared_meal"
        elif vc == "packaged_food":
            kind = "packaged"
        return {
            "is_food": True,
            "label": use_lab,
            "confidence": max(conf, retrieval_food_score, topk_food_mass),
            "kind": kind,
            "family": fam if fam != "unknown" else None,
            "food_state": STATE_HINT.get(use_lab or "", None),
            "visual_class": vc if vc != "non_food" else ("prepared_meal" if kind == "prepared_meal" else "raw_ingredient"),
        }


class SoftmaxSpecialist:
    """Wrap existing multi-head classifier as ONE evidence source (top-k hypotheses)."""

    def __init__(self, model, inv: dict[int, str], device: str, visual_labels: list[str], state_labels: list[str]):
        self.model = model
        self.inv = inv
        self.device = device
        self.visual_labels = visual_labels
        self.state_labels = state_labels

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> dict[str, Any]:
        out = self.model(x)
        logits = out["label"][0]
        probs = F.softmax(logits, dim=-1)
        k = min(5, probs.numel())
        top = probs.topk(k)
        identities = []
        for p, i in zip(top.values.tolist(), top.indices.tolist()):
            lab = self.inv.get(int(i), f"idx_{i}")
            fam = FAMILY_OF.get(lab, "unknown")
            identities.append(
                {
                    "identity": lab,
                    "score": float(p),
                    "family": fam,
                    "kind": KIND_OF.get(fam, "ingredient"),
                    "food_state": STATE_HINT.get(lab, "unknown"),
                }
            )
        vi = int(out["visual_class"][0].argmax().item())
        si = int(out["food_state"][0].argmax().item())
        vprobs = F.softmax(out["visual_class"][0], dim=-1)
        sprobs = F.softmax(out["food_state"][0], dim=-1)
        return {
            "identities": identities,
            "visual_class": self.visual_labels[vi] if vi < len(self.visual_labels) else "unknown",
            "visual_conf": float(vprobs[vi].item()),
            "food_state": self.state_labels[si] if si < len(self.state_labels) else "unknown",
            "state_conf": float(sprobs[si].item()),
            "top1": identities[0]["identity"] if identities else None,
            "top1_conf": identities[0]["score"] if identities else 0.0,
        }


class StateSpecialist:
    def __call__(self, clf_state: str | None, retrieval_state: str | None, clf_conf: float, ret_conf: float) -> dict[str, Any]:
        if clf_state and retrieval_state and clf_state != retrieval_state:
            # conflict — lower confidence, keep clf as primary evidence only if much stronger
            if clf_conf > ret_conf + 0.15:
                return {"food_state": clf_state, "confidence": clf_conf * 0.7, "conflict": True}
            if ret_conf > clf_conf + 0.15:
                return {"food_state": retrieval_state, "confidence": ret_conf * 0.7, "conflict": True}
            return {"food_state": None, "confidence": 0.2, "conflict": True}
        st = clf_state or retrieval_state
        return {"food_state": st, "confidence": max(clf_conf, ret_conf), "conflict": False}
