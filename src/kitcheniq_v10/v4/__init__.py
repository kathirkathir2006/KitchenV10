"""V4 open-world scene engine — top-k candidates + evidence fusion + open-set.

Addresses measured bottlenecks: OPEN_SET (primary), FOOD_NONFOOD (secondary).
Detector upgrade deferred (tertiary). No flat argmax-as-final-identity.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from kitcheniq_v10.specialist.types import FusedObservation, band

NEW_ROOT = Path(r"C:\Projects\KitchenIQ-V10-AI")
OLD_ROOT = Path(r"C:\Projects\KitchenIQ-OS")

VISUAL_CLASS_LABELS = (
    "raw_ingredient",
    "packaged_food",
    "prepared_meal",
    "recipe_document",
    "non_food",
    "mixed",
)
FOOD_STATE_LABELS = ("raw", "prepared", "packaged", "processed", "unknown")
KIND_LABELS = ("ingredient", "meal", "non_food", "document", "packaged")

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
    "doughnut",
    "french_fries",
    "ice_cream",
    "cooked_rice",
    "soup",
    "curry",
}

NONFOOD = {"non_food", "human_hand", "hand", "empty_plate"}

FAMILY = {
    "tomato": "produce",
    "onion": "produce",
    "potato": "produce",
    "carrot": "produce",
    "garlic": "produce",
    "ginger": "produce",
    "cheese": "dairy",
    "packaged_cheese": "packaged_dairy",
    "egg": "protein",
    "fried_egg": "prepared_egg",
    "cooked_rice": "rice_grain",
    "fried_rice": "rice_grain",
    "biryani": "rice_grain",
    "pasta": "grain_starch",
    "pizza": "prepared_meal",
    "salad": "prepared_meal",
    "sandwich": "prepared_meal",
    "doughnut": "prepared_sweet",
    "ice_cream": "prepared_sweet",
    "french_fries": "prepared_side",
}


class FoodVisionV2(nn.Module):
    def __init__(self, n_labels: int, backbone):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        dim = 768
        self.fc_visual = nn.Linear(dim, len(VISUAL_CLASS_LABELS))
        self.fc_state = nn.Linear(dim, len(FOOD_STATE_LABELS))
        self.fc_kind = nn.Linear(dim, len(KIND_LABELS))
        self.fc_label = nn.Linear(dim, max(1, n_labels))

    def forward(self, x):
        with torch.no_grad():
            h = self.backbone(x)
        if isinstance(h, (tuple, list)):
            h = h[0]
        if h.ndim > 2:
            h = h.mean(dim=1)
        h = h.float()
        return {
            "visual_class": self.fc_visual(h),
            "food_state": self.fc_state(h),
            "kind": self.fc_kind(h),
            "label": self.fc_label(h),
        }


def preprocess(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB").resize((224, 224), Image.BILINEAR)).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")
    arr = (arr - mean) / std
    return torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0)


def map_state(raw: str | None, *, visual: str | None, prepared: bool) -> str | None:
    """Preserve 0.2927 specialist state mapping (V3 regression must not repeat)."""
    r = (raw or "").lower()
    v = (visual or "").lower()
    if prepared:
        if r in {"leftover"}:
            return "leftover"
        if r in {"cooked", "prepared", "plated"} or v == "prepared_meal":
            return "plated"
        return "plated"
    mapping = {
        "raw": "raw",
        "prepared": "partially_prepared",
        "packaged": "packaged",
        "processed": "cooked",
        "unknown": None,
    }
    return mapping.get(r, r if r else None)


class ReferenceLibrary:
    def __init__(self, items: list[dict[str, Any]], model_version: str = ""):
        self.items = items
        self.model_version = model_version
        if items:
            mat = np.stack([np.asarray(it["embedding"], dtype=np.float32) for it in items], 0)
            mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
            self._mat = mat
        else:
            self._mat = np.zeros((0, 768), dtype=np.float32)

    @classmethod
    def load(cls, path: Path) -> "ReferenceLibrary":
        data = json.loads(path.read_text(encoding="utf-8"))
        items = list(data.get("items") or [])
        n = int(data.get("n") or len(items))
        if n <= 0 or not items:
            raise RuntimeError(f"reference library n_items must be > 0 (got n={n} path={path})")
        for it in items:
            if not it.get("embedding"):
                raise RuntimeError("reference item missing embedding")
            if not it.get("canonical_identity"):
                raise RuntimeError("reference item missing label")
        return cls(items=items, model_version=str(data.get("model_version") or ""))

    def query(self, emb: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        if self._mat.shape[0] == 0:
            return []
        q = emb.reshape(-1).astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)
        sims = self._mat @ q
        idx = np.argsort(-sims)[:top_k]
        out = []
        for i in idx:
            it = self.items[int(i)]
            out.append(
                {
                    "identity": it["canonical_identity"],
                    "score": float(sims[int(i)]),
                    "family": it.get("family"),
                    "kind": it.get("kind"),
                    "food_state": it.get("food_state"),
                }
            )
        return out


class VisionV4Engine:
    """One serious candidate: open-world scene engine on existing detector + DINOv2 specialist."""

    def __init__(self, *, weights_path: Path | None = None, library_path: Path | None = None):
        self.weights_path = weights_path or (
            NEW_ROOT
            / "backend/instance/dev_experiments/v10-vision-recovery/models/identity_coverage_best.pt"
        )
        self.library_path = library_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._clf = None
        self._inv: dict[int, str] = {}
        self._library: ReferenceLibrary | None = None
        self._app = None
        self._kiq_detector = None
        self.retrieval_enabled = False
        self.backbone_name = "DINOv2:dinov2_vitb14"

    def load(self) -> None:
        blob = torch.load(self.weights_path, map_location="cpu", weights_only=False)
        vocab = blob.get("label_vocab") or {}
        n_labels = int(blob.get("n_labels") or len(vocab))
        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True)
        model = FoodVisionV2(n_labels, backbone)
        model.load_state_dict(blob.get("state_dict") or blob, strict=False)
        model.eval()
        model.to(self.device)
        self._clf = model
        self._inv = {int(v): k for k, v in vocab.items()}
        self._temperature = float(((blob.get("calibration") or {}) or {}).get("temperature") or 1.0)

        if self.library_path and Path(self.library_path).is_file():
            lib = ReferenceLibrary.load(Path(self.library_path))
            self._library = lib
            self.retrieval_enabled = True
        else:
            self._library = None
            self.retrieval_enabled = False

        self._load_detector()

    def _load_detector(self) -> None:
        import shutil

        if str(OLD_ROOT / "backend") not in sys.path:
            sys.path.insert(0, str(OLD_ROOT / "backend"))
        from app import create_app
        from app.config import Config
        from app.services.food_vision.detection import select_object_detector

        class C(Config):
            TESTING = True
            AUTH_DEV_MODE = True
            SECURITY_ENFORCE_AUTH = False
            SQLALCHEMY_ENGINE_OPTIONS = {}

        reg = OLD_ROOT / "backend/instance/dev_experiments/fv-production-final/registry_final.db"
        tmp_db = NEW_ROOT / "backend/instance/dev_experiments/v10-vision-v4/models/det_tmp.db"
        tmp_db.parent.mkdir(parents=True, exist_ok=True)
        if reg.is_file():
            shutil.copy2(reg, tmp_db)
        C.SQLALCHEMY_DATABASE_URI = f"sqlite:///{tmp_db.as_posix()}"
        self._app = create_app(C)
        with self._app.app_context():
            self._kiq_detector = select_object_detector()

    def detect(self, image: Image.Image) -> list[dict[str, Any]]:
        import io

        w, h = image.size
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        with self._app.app_context():
            det_out = self._kiq_detector.detect(
                image_bytes=buf.getvalue(),
                mime="image/jpeg",
                evidence_ref="v10-vision-v4",
                session_context={},
            )
        dets = []
        for d in sorted(list(det_out.detections or []), key=lambda x: float(x.confidence), reverse=True)[:12]:
            bbox = list(d.bbox)
            if d.normalized or max(abs(x) for x in bbox) <= 1.5:
                bbox = [bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h]
            dets.append({"bbox": [float(x) for x in bbox], "confidence": float(d.confidence)})
        if not dets:
            dets = [{"bbox": [0.0, 0.0, float(w), float(h)], "confidence": 0.0, "whole_image": True}]
        return dets

    @torch.no_grad()
    def _specialist(self, crop: Image.Image) -> dict[str, Any]:
        assert self._clf is not None
        x = preprocess(crop).to(self.device)
        out = self._clf(x)
        t = max(1e-3, float(self._temperature))
        probs = F.softmax(out["label"][0] / t, dim=-1)
        k = min(5, probs.numel())
        top = probs.topk(k)
        identities = []
        for p, i in zip(top.values.tolist(), top.indices.tolist()):
            lab = self._inv.get(int(i), f"idx_{i}")
            identities.append(
                {
                    "identity": lab,
                    "score": float(p),
                    "family": FAMILY.get(lab),
                    "prepared": lab in PREPARED_MEAL_LABELS,
                }
            )
        vi = int(out["visual_class"][0].argmax().item())
        si = int(out["food_state"][0].argmax().item())
        vprobs = F.softmax(out["visual_class"][0], dim=-1)
        sprobs = F.softmax(out["food_state"][0], dim=-1)
        # embedding for retrieval = backbone pooled
        h = self._clf.backbone(x)
        if isinstance(h, (tuple, list)):
            h = h[0]
        if h.ndim > 2:
            h = h.mean(dim=1)
        emb = F.normalize(h.float(), dim=-1)[0].detach().cpu().numpy()
        return {
            "identities": identities,
            "visual_class": VISUAL_CLASS_LABELS[vi],
            "visual_conf": float(vprobs[vi].item()),
            "food_state_raw": FOOD_STATE_LABELS[si],
            "state_conf": float(sprobs[si].item()),
            "embedding": emb,
        }

    def _food_gate(self, clf: dict[str, Any], retrieval: list[dict[str, Any]]) -> dict[str, Any]:
        """Repair FOOD_NONFOOD false negatives — measured secondary bottleneck."""
        idents = clf.get("identities") or []
        tops = [str(h.get("identity") or "") for h in idents]
        food_mass = float(
            sum(float(h["score"]) for h in idents if str(h.get("identity")) not in NONFOOD)
        )
        ret_food = max((float(r["score"]) for r in retrieval if r.get("identity") not in NONFOOD), default=0.0)
        vc = clf.get("visual_class") or ""
        top1 = tops[0] if tops else ""
        exclusive_nf = bool(tops) and all(t in NONFOOD for t in tops)
        if exclusive_nf and food_mass < 0.08 and ret_food < 0.25 and float(idents[0]["score"]) >= 0.85:
            return {"is_food": False, "kind": "non_food", "confidence": float(idents[0]["score"])}
        # Prefer food when any food mass exists (tomato→non_food failure pattern)
        foodish = [t for t in tops if t not in NONFOOD]
        use = foodish[0] if foodish else (top1 if top1 not in NONFOOD else None)
        prepared = (use in PREPARED_MEAL_LABELS) or (
            vc == "prepared_meal" and (use in PREPARED_MEAL_LABELS or use is None)
        )
        # raw produce must not inherit prepared_meal visual alone
        if use in {"tomato", "onion", "potato", "carrot", "garlic", "ginger", "cheese", "egg", "apple", "banana"}:
            prepared = False
        kind = "prepared_meal" if prepared else ("packaged" if vc == "packaged_food" or (use or "").startswith("packaged") else "ingredient")
        return {
            "is_food": True,
            "kind": kind,
            "confidence": max(food_mass, ret_food, float(clf.get("visual_conf") or 0)),
            "hint_identity": use,
            "visual_class": (
                "prepared_meal"
                if prepared
                else ("packaged_food" if kind == "packaged" else ("raw_ingredient" if vc == "non_food" else vc))
            ),
        }

    def _fuse_candidates(
        self,
        clf: dict[str, Any],
        retrieval: list[dict[str, Any]],
        gate: dict[str, Any],
        ctx_boosts: dict[str, float],
    ) -> list[dict[str, Any]]:
        scores: dict[str, float] = defaultdict(float)
        sources: dict[str, set[str]] = defaultdict(set)
        meta: dict[str, dict[str, Any]] = {}
        for h in clf.get("identities") or []:
            lab = str(h["identity"])
            if lab in NONFOOD and gate.get("is_food"):
                continue
            scores[lab] += float(h["score"])
            sources[lab].add("specialist")
            meta[lab] = {"family": h.get("family"), "prepared": h.get("prepared")}
        for r in retrieval:
            lab = str(r["identity"])
            scores[lab] += 0.85 * float(r["score"])
            sources[lab].add("retrieval")
            meta[lab] = {**meta.get(lab, {}), "family": r.get("family"), "kind": r.get("kind")}
        for lab, b in ctx_boosts.items():
            if lab in scores or b > 0:
                scores[lab] += b
                sources[lab].add("scene_context")
        hyps = []
        for lab, sc in scores.items():
            # multi-source diversity bonus
            div = min(1.0, len(sources[lab]) / 2.0)
            hyps.append(
                {
                    "identity": lab,
                    "score": float(sc * (0.6 + 0.4 * div)),
                    "sources": sorted(sources[lab]),
                    **meta.get(lab, {}),
                }
            )
        hyps.sort(key=lambda x: -x["score"])
        return hyps

    def _decide(self, hyps: list[dict[str, Any]], gate: dict[str, Any], clf: dict[str, Any]) -> dict[str, Any]:
        if not gate.get("is_food"):
            return {
                "status": "non_food",
                "identity": "non_food",
                "raw_label": "non_food",
                "confidence": float(gate.get("confidence") or 0.7),
                "is_food": False,
                "is_prepared_meal": False,
                "decision": "NON_FOOD",
            }
        if not hyps:
            kind = gate.get("kind")
            return {
                "status": "unknown_food",
                "identity": None,
                "raw_label": None,
                "confidence": float(gate.get("confidence") or 0.4),
                "is_food": True,
                "is_prepared_meal": kind == "prepared_meal",
                "decision": "UNKNOWN_PREPARED_MEAL" if kind == "prepared_meal" else "UNKNOWN_FOOD",
                "abstain_reason": "no_hypotheses",
            }
        top = hyps[0]
        second = hyps[1] if len(hyps) > 1 else None
        margin = top["score"] - (second["score"] if second else 0.0)
        spec = next((float(h["score"]) for h in (clf.get("identities") or []) if h["identity"] == top["identity"]), 0.0)

        # Strong specialist or multi-source → KNOWN (addresses OPEN_SET over-abstention when evidence exists)
        hint = gate.get("hint_identity")
        if top["identity"] not in NONFOOD and (
            (spec >= 0.28 and margin >= 0.015)
            or (top["score"] >= 0.40 and len(top["sources"]) >= 2 and margin >= 0.02)
            or (spec >= 0.45)
            or (hint and hint == top["identity"] and spec >= 0.18 and margin >= 0.01)
        ):
            prepared = bool(top.get("prepared")) or top["identity"] in PREPARED_MEAL_LABELS
            strong = spec >= 0.45 or top["score"] >= 0.55
            return {
                "status": "confirmed_identity" if strong else "probable_identity",
                "identity": top["identity"],
                "raw_label": top["identity"],
                "confidence": float(min(0.99, max(spec, top["score"]))),
                "is_food": True,
                "is_prepared_meal": prepared,
                "decision": "KNOWN" if strong else "PROBABLE_KNOWN",
                "hypotheses": hyps[:5],
            }

        if second and margin < 0.03 and second["score"] >= 0.25 and second["identity"] != top["identity"]:
            return {
                "status": "conflicting",
                "identity": None,
                "raw_label": None,
                "confidence": float(margin),
                "is_food": True,
                "is_prepared_meal": gate.get("kind") == "prepared_meal",
                "decision": "CONFLICTING_EVIDENCE",
                "abstain_reason": "top_margin_conflict",
                "hypotheses": hyps[:5],
            }

        prepared = gate.get("kind") == "prepared_meal"
        return {
            "status": "unknown_food",
            "identity": None,
            "raw_label": None,
            "confidence": float(max(top["score"], gate.get("confidence") or 0.3)),
            "is_food": True,
            "is_prepared_meal": prepared,
            "decision": "UNKNOWN_PREPARED_MEAL" if prepared else "UNKNOWN_FOOD",
            "abstain_reason": "below_calibrated_threshold",
            "hypotheses": hyps[:5],
        }

    def _scene_context_boosts(self, region_summaries: list[dict[str, Any]]) -> list[dict[str, float]]:
        """Per-region identity boosts from co-occurrence — confidence only, never invent."""
        kinds = [r.get("kind") for r in region_summaries]
        has_prepared = any(k == "prepared_meal" for k in kinds)
        has_ingredient = any(k == "ingredient" for k in kinds)
        out = []
        for r in region_summaries:
            boosts: dict[str, float] = {}
            lab = r.get("hint_identity")
            if has_prepared and lab in {"cooked_rice", "rice", "pasta", "biryani", "fried_rice"}:
                boosts[lab] = boosts.get(lab, 0) + 0.05
            if has_ingredient and lab in PREPARED_MEAL_LABELS:
                boosts[lab] = boosts.get(lab, 0) + 0.04
            out.append(boosts)
        return out

    def infer_image(self, image: Image.Image) -> list[FusedObservation]:
        dets = self.detect(image)
        region_data = []
        for det in dets:
            x1, y1, x2, y2 = det["bbox"]
            crop = image.crop((int(x1), int(y1), int(max(x2, x1 + 1)), int(max(y2, y1 + 1))))
            clf = self._specialist(crop)
            retrieval = []
            if self.retrieval_enabled and self._library is not None:
                retrieval = self._library.query(clf["embedding"], top_k=5)
            gate = self._food_gate(clf, retrieval)
            region_data.append({"det": det, "clf": clf, "retrieval": retrieval, "gate": gate})

        summaries = [
            {"kind": r["gate"].get("kind"), "hint_identity": r["gate"].get("hint_identity")}
            for r in region_data
        ]
        boosts_list = self._scene_context_boosts(summaries)

        observations: list[FusedObservation] = []
        for r, boosts in zip(region_data, boosts_list):
            hyps = self._fuse_candidates(r["clf"], r["retrieval"], r["gate"], boosts)
            dec = self._decide(hyps, r["gate"], r["clf"])
            prepared = bool(dec.get("is_prepared_meal"))
            state = map_state(
                r["clf"].get("food_state_raw"),
                visual=r["gate"].get("visual_class") or r["clf"].get("visual_class"),
                prepared=prepared,
            )
            conf = float(dec.get("confidence") or 0)
            observations.append(
                FusedObservation(
                    status=str(dec.get("status") or "abstain"),
                    identity=dec.get("identity"),
                    raw_label=dec.get("raw_label"),
                    visual_class=r["gate"].get("visual_class") or r["clf"].get("visual_class"),
                    food_state=state,
                    confidence=conf,
                    confidence_band=band(conf),
                    bbox=r["det"]["bbox"],
                    detector_confidence=float(r["det"].get("confidence") or 0),
                    is_food=bool(dec.get("is_food")),
                    is_prepared_meal=prepared,
                    abstain_reason=dec.get("abstain_reason"),
                    needs_confirmation=dec.get("decision") == "PROBABLE_KNOWN",
                    provenance={
                        "v4": True,
                        "decision": dec.get("decision"),
                        "retrieval_enabled": self.retrieval_enabled,
                        "hypotheses": (dec.get("hypotheses") or [])[:5],
                        "retrieval_top": r["retrieval"][:3],
                        "backbone": self.backbone_name,
                    },
                )
            )
        return observations
