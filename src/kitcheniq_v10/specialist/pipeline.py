"""Specialist ensemble inference pipeline (V10).

Loads Faster R-CNN + production FoodVisionV2 multi-head as specialist sources,
applies dedicated fusion/conflict/abstention. Does not mutate the old project.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from kitcheniq_v10.specialist.fusion import PREPARED_MEAL_LABELS, fuse_specialists
from kitcheniq_v10.specialist.types import FusedObservation, SpecialistPred

OLD_ROOT = Path(r"C:\Projects\KitchenIQ-OS")
PROD_WEIGHTS = (
    OLD_ROOT
    / "backend/instance/dev_experiments/fv-production-final/registry_artifacts"
    / "food-vision-classifier-v1/classifier-prod-v1.0.0/model.pt"
)


class SpecialistEnsemble:
    def __init__(
        self,
        *,
        mode: str = "A",
        hardneg_strict: bool = False,
        device: str | None = None,
    ) -> None:
        self.mode = mode
        self.hardneg_strict = hardneg_strict
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._clf = None
        self._detector = None
        self._inv: dict[int, str] = {}
        self._temperature = 1.0

    def load(self) -> None:
        if str(OLD_ROOT / "backend") not in sys.path:
            sys.path.insert(0, str(OLD_ROOT / "backend"))
        from app.services.food_vision.model_arch import (
            FOOD_STATE_LABELS,
            VISUAL_CLASS_LABELS,
            build_model_from_arch,
        )
        from app.services.food_vision.inference import _resolve_arch_from_blob

        blob = torch.load(PROD_WEIGHTS, map_location="cpu", weights_only=False)
        arch = _resolve_arch_from_blob(blob)
        vocab = arch.get("label_vocab") or blob.get("label_vocab") or {}
        arch["label_vocab"] = vocab
        arch.setdefault("architecture", "FoodVisionV2")
        arch.setdefault("train_mode", "A")
        arch.setdefault("n_labels", len(vocab) or blob.get("n_labels") or 1)
        model = build_model_from_arch(arch, load_backbone=True)
        model.load_state_dict(blob["state_dict"], strict=False)
        model.eval()
        model.to(self.device)
        self._clf = model
        self._inv = {int(v): k for k, v in vocab.items()}
        cal = blob.get("calibration") or {}
        self._temperature = float((cal.get("temperature") if isinstance(cal, dict) else None) or 1.0)
        self._visual_labels = list(VISUAL_CLASS_LABELS)
        self._state_labels = list(FOOD_STATE_LABELS)

        # Prefer KitchenIQ production detector (read-only from old project).
        # COCO torchvision FRCNN is fallback only — wrong domain for 41-scene GT boxes.
        self._kiq_detector = None
        self._detector = None
        try:
            from app import create_app
            from app.config import Config
            from app.services.food_vision.detection import select_object_detector
            import shutil

            class C(Config):
                TESTING = True
                AUTH_DEV_MODE = True
                SECURITY_ENFORCE_AUTH = False
                SQLALCHEMY_ENGINE_OPTIONS = {}

            reg = (
                OLD_ROOT
                / "backend/instance/dev_experiments/fv-production-final/registry_final.db"
            )
            tmp_db = Path(__file__).resolve().parents[3] / "backend/instance/dev_experiments/v10-specialist-ensemble/models/det_tmp.db"
            tmp_db.parent.mkdir(parents=True, exist_ok=True)
            if reg.is_file():
                shutil.copy2(reg, tmp_db)
                C.SQLALCHEMY_DATABASE_URI = f"sqlite:///{tmp_db.as_posix()}"
            else:
                C.SQLALCHEMY_DATABASE_URI = f"sqlite:///{tmp_db.as_posix()}"
            self._app = create_app(C)
            with self._app.app_context():
                self._kiq_detector = select_object_detector()
        except Exception as e:  # noqa: BLE001
            self._kiq_detector_error = str(e)
            from torchvision.models.detection import (
                fasterrcnn_resnet50_fpn,
                FasterRCNN_ResNet50_FPN_Weights,
            )

            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            det = fasterrcnn_resnet50_fpn(weights=weights)
            det.eval()
            det.to(self.device)
            self._detector = det

    def _preprocess(self, img: Image.Image) -> torch.Tensor:
        import numpy as np

        img = img.convert("RGB").resize((224, 224), Image.BILINEAR)
        arr = np.asarray(img).astype("float32") / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype="float32")
        std = np.array([0.229, 0.224, 0.225], dtype="float32")
        arr = (arr - mean) / std
        return torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0).to(self.device)

    @torch.no_grad()
    def _classify_crop(self, crop: Image.Image) -> dict[str, Any]:
        assert self._clf is not None
        x = self._preprocess(crop)
        out = self._clf(x)
        logits = out["label"][0]
        t = max(1e-3, float(self._temperature))
        probs = F.softmax(logits / t, dim=-1)
        conf, idx = probs.max(dim=-1)
        top1 = self._inv.get(int(idx.item()), f"idx_{int(idx.item())}")
        vi = int(out["visual_class"][0].argmax().item())
        si = int(out["food_state"][0].argmax().item())
        vprobs = F.softmax(out["visual_class"][0], dim=-1)
        sprobs = F.softmax(out["food_state"][0], dim=-1)
        visual = self._visual_labels[vi] if vi < len(self._visual_labels) else "unknown"
        state_raw = self._state_labels[si] if si < len(self._state_labels) else "unknown"
        return {
            "top1": top1,
            "identity_conf": float(conf.item()),
            "visual_class": visual,
            "visual_conf": float(vprobs[vi].item()),
            "food_state_raw": state_raw,
            "state_conf": float(sprobs[si].item()),
            "top5": [
                {"label": self._inv.get(int(i), str(i)), "prob": float(p)}
                for p, i in zip(*probs.topk(min(5, probs.numel())))
            ],
        }

    def _map_state(self, raw: str | None, *, visual: str | None, prepared: bool) -> str | None:
        r = (raw or "").lower()
        v = (visual or "").lower()
        if prepared:
            if r in {"leftover"}:
                return "leftover"
            # Acceptance GT for prepared meals typically uses "plated"
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

    def _food_nonfood_specialist(self, clf: dict[str, Any]) -> SpecialistPred:
        visual = (clf["visual_class"] or "").lower()
        vconf = clf["visual_conf"]
        top1 = (clf["top1"] or "").lower().replace(" ", "_")
        id_conf = float(clf["identity_conf"])
        # Candidate B/C: stricter hard-negative — treat medium non_food / empty-like as nonfood
        if visual == "non_food":
            conf = vconf
            if self.hardneg_strict:
                conf = max(conf, 0.6)
            return SpecialistPred("food_nonfood_specialist", "non_food", conf)
        if top1 in {"non_food", "empty_plate", "hand", "human_hand"}:
            return SpecialistPred(
                "food_nonfood_specialist", "non_food", max(id_conf, 0.7)
            )
        # food-like visual classes (include packaged_food synonym)
        if visual in {"raw_ingredient", "prepared_meal", "packaged", "packaged_food", "mixed"}:
            mapped = "packaged" if visual == "packaged_food" else visual
            return SpecialistPred("food_nonfood_specialist", mapped, vconf)
        # Candidate C: whole-image / low-specificity identity → do not claim food
        if self.mode == "C" or self.hardneg_strict:
            if top1 in {"food", "unknown", "mixed"} or id_conf < 0.55:
                if vconf < 0.70:
                    return SpecialistPred(
                        "food_nonfood_specialist",
                        "non_food",
                        max(0.56, 1.0 - id_conf),
                        {"reason": "hardneg_low_specificity"},
                    )
        if self.hardneg_strict and id_conf < 0.55 and vconf < 0.55:
            return SpecialistPred(
                "food_nonfood_specialist", "non_food", 0.56, {"reason": "low_conf_hardneg"}
            )
        return SpecialistPred("food_nonfood_specialist", visual or "unknown", vconf)

    def _identity_specialist(self, clf: dict[str, Any]) -> SpecialistPred:
        # Hierarchical stopping: low conf → family_only handled in fusion via unknown_food
        return SpecialistPred(
            "identity_specialist",
            clf["top1"],
            clf["identity_conf"],
            {"top5": clf["top5"]},
        )

    def _prepared_specialist(self, clf: dict[str, Any]) -> SpecialistPred:
        top1 = (clf["top1"] or "").lower().replace(" ", "_")
        visual = clf["visual_class"]
        if top1 in PREPARED_MEAL_LABELS or visual == "prepared_meal":
            conf = clf["identity_conf"] if top1 in PREPARED_MEAL_LABELS else clf["visual_conf"]
            lab = top1 if top1 in PREPARED_MEAL_LABELS else "prepared_meal"
            return SpecialistPred("prepared_meal_specialist", lab, conf)
        return SpecialistPred("prepared_meal_specialist", "not_prepared", 1.0 - clf["visual_conf"] * 0.5)

    def _state_specialist(self, clf: dict[str, Any], prepared: bool) -> SpecialistPred:
        mapped = self._map_state(clf["food_state_raw"], visual=clf["visual_class"], prepared=prepared)
        return SpecialistPred("food_state_specialist", mapped, clf["state_conf"])

    @torch.no_grad()
    def detect(self, image: Image.Image) -> list[dict[str, Any]]:
        w, h = image.size
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        image_bytes = buf.getvalue()

        if self._kiq_detector is not None:
            with self._app.app_context():
                det_out = self._kiq_detector.detect(
                    image_bytes=image_bytes,
                    mime="image/jpeg",
                    evidence_ref="v10-specialist",
                    session_context={},
                )
            dets = []
            for d in sorted(
                list(det_out.detections or []),
                key=lambda x: float(x.confidence),
                reverse=True,
            )[:12]:
                bbox = list(d.bbox)
                if d.normalized or max(abs(x) for x in bbox) <= 1.5:
                    bbox = [bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h]
                dets.append(
                    {
                        "bbox": [float(x) for x in bbox],
                        "confidence": float(d.confidence),
                        "normalized": False,
                    }
                )
            if not dets:
                dets = [
                    {
                        "bbox": [0.0, 0.0, float(w), float(h)],
                        "confidence": 0.0,
                        "whole_image": True,
                    }
                ]
            return dets

        assert self._detector is not None
        import torchvision.transforms.functional as TF

        t = TF.to_tensor(image.convert("RGB")).to(self.device)
        out = self._detector([t])[0]
        boxes = out["boxes"].detach().cpu()
        scores = out["scores"].detach().cpu()
        keep = scores >= 0.4
        boxes, scores = boxes[keep], scores[keep]
        if scores.numel() > 12:
            idx = scores.topk(12).indices
            boxes, scores = boxes[idx], scores[idx]
        dets = []
        for b, s in zip(boxes.tolist(), scores.tolist()):
            x1, y1, x2, y2 = b
            dets.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(s),
                    "normalized": False,
                }
            )
        if not dets:
            dets = [
                {
                    "bbox": [0.0, 0.0, float(w), float(h)],
                    "confidence": 0.0,
                    "whole_image": True,
                }
            ]
        return dets

    def infer_image(self, image: Image.Image) -> list[FusedObservation]:
        if self._clf is None:
            self.load()
        dets = self.detect(image)
        obs: list[FusedObservation] = []
        for d in dets:
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.size[0], x2), min(image.size[1], y2)
            if x2 <= x1 + 2 or y2 <= y1 + 2:
                crop = image
            else:
                crop = image.crop((x1, y1, x2, y2))
            clf = self._classify_crop(crop)
            fn = self._food_nonfood_specialist(clf)
            ident = self._identity_specialist(clf)
            prep = self._prepared_specialist(clf)
            prepared_flag = (prep.label or "") in PREPARED_MEAL_LABELS or prep.label == "prepared_meal"
            state = self._state_specialist(clf, prepared_flag)
            fused = fuse_specialists(
                food_nonfood=fn,
                identity=ident,
                prepared=prep,
                food_state=state,
                bbox=d["bbox"],
                detector_confidence=d.get("confidence"),
            )
            obs.append(fused)
        return obs

    def infer_path(self, path: Path) -> list[FusedObservation]:
        img = Image.open(path).convert("RGB")
        return self.infer_image(img)
