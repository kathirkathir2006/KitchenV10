"""Vision V3 end-to-end pipeline (Candidate E primary target)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from kitcheniq_v10.specialist.types import FusedObservation, band
from kitcheniq_v10.v3.backbone import DenseEncoder, load_primary_backbone
from kitcheniq_v10.v3.decision import OpenSetDecision, decide_region
from kitcheniq_v10.v3.evidence import EvidenceGraph
from kitcheniq_v10.v3.relations import relate_regions
from kitcheniq_v10.v3.retrieval import HierarchicalRetriever, ReferenceLibrary
from kitcheniq_v10.v3.specialists import FoodNonFoodSpecialist, SoftmaxSpecialist, StateSpecialist

OLD_ROOT = Path(r"C:\Projects\KitchenIQ-OS")
NEW_ROOT = Path(r"C:\Projects\KitchenIQ-V10-AI")

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


def _preprocess(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB").resize((224, 224), Image.BILINEAR)).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")
    arr = (arr - mean) / std
    return torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0)


class VisionV3Pipeline:
    """Candidate modes: A(baseline wrapper), B(det+retrieval), E(full graph)."""

    def __init__(
        self,
        *,
        mode: str = "E",
        weights_path: Path | None = None,
        library_path: Path | None = None,
        device: str | None = None,
    ):
        self.mode = mode
        self.weights_path = weights_path
        self.library_path = library_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder: DenseEncoder | None = None
        self.retriever: HierarchicalRetriever | None = None
        self.specialist: SoftmaxSpecialist | None = None
        self.food_gate = FoodNonFoodSpecialist()
        self.state_spec = StateSpecialist()
        self.decision_cfg = OpenSetDecision()
        self._kiq_detector = None
        self._app = None
        self.backbone_info: dict[str, Any] = {}

    def load(self) -> None:
        self.encoder = load_primary_backbone(self.device)
        self.backbone_info = {
            "name": self.encoder.info.name,
            "version": self.encoder.info.version,
            "source": self.encoder.info.source,
            "embed_dim": self.encoder.info.embed_dim,
            "weights_hash": self.encoder.info.weights_hash,
            "fallback_reason": self.encoder.info.fallback_reason,
        }
        if self.library_path and Path(self.library_path).is_file():
            lib = ReferenceLibrary.load(Path(self.library_path))
        else:
            from kitcheniq_v10.v3.retrieval import build_library_from_prototypes, identities_from_manifests

            lib = build_library_from_prototypes(self.encoder, identities_from_manifests(NEW_ROOT))
        self.retriever = HierarchicalRetriever(lib, top_k=5)

        # specialist identity head (coverage model) — evidence only
        wp = self.weights_path or (
            NEW_ROOT
            / "backend/instance/dev_experiments/v10-vision-recovery/models/identity_coverage_best.pt"
        )
        blob = torch.load(wp, map_location="cpu", weights_only=False)
        vocab = blob.get("label_vocab") or {}
        n_labels = int(blob.get("n_labels") or len(vocab))
        # share a fresh dinov2 backbone for the specialist head (frozen)
        bb = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True)
        model = FoodVisionV2(n_labels, bb)
        model.load_state_dict(blob.get("state_dict") or blob, strict=False)
        model.eval()
        model.to(self.device)
        inv = {int(v): k for k, v in vocab.items()}
        self.specialist = SoftmaxSpecialist(
            model, inv, self.device, list(VISUAL_CLASS_LABELS), list(FOOD_STATE_LABELS)
        )
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
        tmp_db = NEW_ROOT / "backend/instance/dev_experiments/v10-vision-v3/models/det_tmp.db"
        tmp_db.parent.mkdir(parents=True, exist_ok=True)
        if reg.is_file():
            shutil.copy2(reg, tmp_db)
        C.SQLALCHEMY_DATABASE_URI = f"sqlite:///{tmp_db.as_posix()}"
        self._app = create_app(C)
        with self._app.app_context():
            self._kiq_detector = select_object_detector()

    def _detect(self, image: Image.Image) -> list[dict[str, Any]]:
        assert self._app and self._kiq_detector
        import io

        w, h = image.size
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        image_bytes = buf.getvalue()
        with self._app.app_context():
            det_out = self._kiq_detector.detect(
                image_bytes=image_bytes,
                mime="image/jpeg",
                evidence_ref="v10-vision-v3",
                session_context={},
            )
        dets: list[dict[str, Any]] = []
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

    def infer_image(self, image: Image.Image) -> list[FusedObservation]:
        assert self.encoder and self.retriever and self.specialist
        dets = self._detect(image)
        region_summaries: list[dict[str, Any]] = []
        region_graphs: list[EvidenceGraph] = []
        crops_meta: list[dict[str, Any]] = []

        for i, det in enumerate(dets):
            x1, y1, x2, y2 = det["bbox"]
            crop = image.crop((int(x1), int(y1), int(max(x2, x1 + 1)), int(max(y2, y1 + 1))))
            emb = self.encoder.embed_image(crop)
            g = emb["global"]
            retrieval = self.retriever.query(g)
            x = _preprocess(crop).to(self.device)
            clf = self.specialist(x)

            # retrieval food-ness: max score among non-non_food identities
            food_scores = [
                m["score"]
                for m in retrieval.get("identity") or []
                if m.get("identity") not in {"non_food", "human_hand", "empty_plate"}
            ]
            retrieval_food = max(food_scores) if food_scores else 0.0
            idents = clf.get("identities") or []
            topk_ids = [h.get("identity") or "" for h in idents]
            topk_food_mass = float(
                sum(
                    float(h.get("score") or 0)
                    for h in idents
                    if (h.get("identity") or "") not in {"non_food", "human_hand", "empty_plate"}
                )
            )
            gate = self.food_gate(
                visual_class=clf.get("visual_class"),
                label=clf.get("top1"),
                conf=float(clf.get("top1_conf") or 0),
                retrieval_food_score=float(retrieval_food),
                topk_food_mass=topk_food_mass,
                topk_identities=topk_ids,
            )
            st = self.state_spec(
                clf.get("food_state"),
                (retrieval.get("state") or [{}])[0].get("food_state"),
                float(clf.get("state_conf") or 0),
                float((retrieval.get("state") or [{}])[0].get("score") or 0),
            )
            gate["food_state"] = st.get("food_state") or gate.get("food_state")

            graph = EvidenceGraph()
            graph.add_node(f"region_{i}", bbox=det["bbox"], det_conf=det["confidence"])
            # specialist hypotheses
            for hyp in clf.get("identities") or []:
                graph.add_edge(
                    "specialist",
                    hyp["identity"],
                    "specialist_support",
                    float(hyp["score"]),
                    family=hyp.get("family"),
                    entity_kind=hyp.get("kind"),
                    food_state=hyp.get("food_state"),
                )
            # retrieval hypotheses
            for hyp in retrieval.get("identity") or []:
                graph.add_edge(
                    "retrieval",
                    hyp["identity"],
                    "visual_similarity",
                    float(hyp["score"]),
                    family=hyp.get("family"),
                    entity_kind=None,
                    food_state=None,
                )
            for fam in retrieval.get("family") or []:
                # boost identities of that family weakly via family_support on family name node
                graph.add_edge(
                    "retrieval_family",
                    fam["family"],
                    "family_support",
                    float(fam["score"]) * 0.25,
                    family=fam["family"],
                )
            if st.get("conflict"):
                for hyp in clf.get("identities") or []:
                    graph.add_edge("state", hyp["identity"], "negative_evidence", 0.1)

            region_graphs.append(graph)
            crops_meta.append({"det": det, "gate": gate, "clf": clf, "retrieval": retrieval, "emb": emb})
            region_summaries.append(
                {
                    "region_id": f"region_{i}",
                    "identity": (clf.get("identities") or [{}])[0].get("identity"),
                    "kind": gate.get("kind"),
                    "food_state": gate.get("food_state"),
                }
            )

        # cross-object context edges
        if self.mode in {"D", "E"}:
            ctx_edges = relate_regions(region_summaries)
            for e in ctx_edges:
                # attach to all graphs that mention target identity
                for g in region_graphs:
                    g.add_edge(e["source"], e["target"], e["kind"], e["weight"], **(e.get("detail") or {}))

        observations: list[FusedObservation] = []
        for i, (meta, graph) in enumerate(zip(crops_meta, region_graphs)):
            hyps = graph.score_identities()
            # drop pure family nodes that aren't identities
            hyps = [h for h in hyps if h.identity and h.identity not in {"produce", "dairy", "unknown"}]
            if self.mode == "A":
                # baseline-like: take specialist top1 only
                top = (meta["clf"].get("identities") or [{}])[0]
                from kitcheniq_v10.v3.types import Hypothesis

                hyps = [
                    Hypothesis(
                        identity=top.get("identity"),
                        family=top.get("family"),
                        kind=top.get("kind"),
                        food_state=top.get("food_state"),
                        score=float(top.get("score") or 0),
                        sources=["specialist_support"],
                        evidence={"specialist_support": float(top.get("score") or 0)},
                    )
                ]
            decision = decide_region(hypotheses=hyps, food_gate=meta["gate"], cfg=self.decision_cfg)
            status_map = {
                "KNOWN": "confirmed_identity",
                "PROBABLE_KNOWN": "probable_identity",
                "UNKNOWN_FOOD": "unknown_food",
                "UNKNOWN_PREPARED_MEAL": "unknown_food",
                "NON_FOOD": "non_food",
                "CONFLICTING_EVIDENCE": "conflicting",
                "INSUFFICIENT_EVIDENCE": "unable_to_determine",
            }
            fused_status = status_map.get(decision.status, "abstain")
            identity = decision.identity
            raw = identity
            visual = meta["gate"].get("visual_class") or meta["clf"].get("visual_class")
            if decision.status == "NON_FOOD":
                raw = "non_food"
                visual = "non_food"
            elif decision.status in {"UNKNOWN_FOOD", "UNKNOWN_PREPARED_MEAL"}:
                # Open-world: keep family-level signal; do not force a leaf identity
                raw = None
                visual = (
                    "prepared_meal"
                    if decision.status == "UNKNOWN_PREPARED_MEAL"
                    else (visual or "raw_ingredient")
                )
            elif decision.status in {"CONFLICTING_EVIDENCE", "INSUFFICIENT_EVIDENCE"}:
                raw = None

            conf = float(decision.confidence)
            observations.append(
                FusedObservation(
                    status=fused_status,
                    identity=identity if decision.status in {"KNOWN", "PROBABLE_KNOWN"} else None,
                    raw_label=raw if decision.status in {"KNOWN", "PROBABLE_KNOWN", "NON_FOOD"} else None,
                    visual_class=visual,
                    food_state=decision.food_state,
                    confidence=conf,
                    confidence_band=band(conf),
                    bbox=meta["det"]["bbox"],
                    is_food=decision.status != "NON_FOOD",
                    is_prepared_meal=(decision.kind == "prepared_meal")
                    or (decision.status == "UNKNOWN_PREPARED_MEAL"),
                    abstain_reason=decision.abstain_reason,
                    needs_confirmation=decision.status == "PROBABLE_KNOWN",
                    provenance={
                        "v3_mode": self.mode,
                        "decision": decision.status,
                        "backbone": self.backbone_info.get("name"),
                        "hypotheses": [
                            {
                                "identity": h.identity,
                                "score": h.score,
                                "sources": h.sources,
                            }
                            for h in decision.hypotheses[:5]
                        ],
                        "retrieval_top": (meta["retrieval"].get("identity") or [])[:3],
                        "evidence_graph_n_edges": len(graph.edges),
                    },
                )
            )
        return observations
