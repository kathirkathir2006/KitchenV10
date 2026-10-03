"""KitchenIQ V10-VR region proposals, matched-budget control crops, multi-view crops and arm layout.

Shared, unchanged, by the Kaggle VAL embedding kernel and the local 41-scene diagnostic. Nothing here scores or
decides; it only says WHICH image boxes become embedding queries. Every arm issues exactly PROTOCOL["N"] queries.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROTOCOL = {
    "protocol_id": "kiq_v10_m1_m3_protocol_v1",
    "N": 8,
    "topk_per_view": 20,
    "union_limit": 20,
    "seed": 1234,
    "control": {"side_fraction": [0.4, 0.9], "aspect": [0.75, 1.3333], "rng_key": "seed:image_sha256"},
    "proposals": {
        "max_per_source": 100, "nms_iou": 0.5, "min_side_px": 16, "class_agnostic": True,
        "frcnn": {"builder": "torchvision.models.detection.fasterrcnn_resnet50_fpn_v2",
                  "weights": "FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1", "box_score_thresh": 0.05,
                  "box_detections_per_img": 300, "preprocessing": "weights.transforms() (ToTensor, no resize; model resizes min 800 / max 1333)"},
        "owlv2": {"model": "google/owlv2-base-patch16-ensemble", "revision": "cfd3195ba4ea9592eec887ded089f4c08eff231d",
                  "licence": "apache-2.0", "score_thresh": 0.01, "pre_nms_topk": 300,
                  "preprocessing": "Owlv2Processor (pad to square bottom/right, resize 960x960); boxes decoded vs padded square side",
                  "queries": ["food", "a dish of food", "a plate of food", "a bowl of food", "fruit", "a vegetable",
                              "a packaged food product", "a jar", "a bottle", "a carton", "a can", "bread", "meat",
                              "a snack", "a dessert"]},
        "sam3": {"model": "facebook/sam3", "revision": "3c879f39826c281e95690f02c7821c4de09afae7",
                 "licence": "Meta SAM License (custom, 'other')", "status": "UNAVAILABLE",
                 "reason": "Hugging Face repo is gated (gated='manual': Meta approval form); no approved token in this environment. Not substituted."},
    },
    "multiview": {"pad": 0.15, "context": 0.5,
                  "layout": "full + top1 x {orig, padded, square, context} + top2 x {orig, padded, context}"},
    "padding_rule": "if a source yields fewer boxes than its arm needs, the remaining slots take the CONTROL crops in order (counted and reported)",
    "arms": {
        "E_FULL": "1 query: full image (frozen Adapter E single-view reference; NOT budget-matched)",
        "CONTROL": "full + 7 seeded random crops",
        "FRCNN": "full + top-7 Faster R-CNN boxes",
        "OWLV2": "full + top-7 OWLv2 boxes",
        "FRCNN_MV": "full + Faster R-CNN top-2 multi-view",
        "OWLV2_MV": "full + OWLv2 top-2 multi-view",
    },
    "matched_arms": ["CONTROL", "FRCNN", "OWLV2", "FRCNN_MV", "OWLV2_MV"],
    "v10_arms": ["FRCNN", "OWLV2", "FRCNN_MV", "OWLV2_MV"],
    "v10_primary_rule": "highest VAL Recall@5 among v10_arms; ties -> higher Recall@10, then arm order",
    "union_rule": "per view top-20 labels; candidate score = max over views listing it; order by score desc, then view_support desc, then label; keep 20",
    "decision_rule": "frozen V5 decide() with frozen Adapter E thresholds on the union score vector (labels in no view top-20 = -9)",
    "scene_region_match": {"iou": 0.5, "box": "view region_box (the proposal / crop the view derives from)",
                           "region_candidates": "union over the arm's views whose region_box IoU >= 0.5 with the GT box"},
    "region_recall_ious": [0.25, 0.5, 0.75],
    "spurious_iou": 0.25,
    "bootstrap": {"n": 10000, "seed": 1234, "ci": 0.95, "unit": "query id (VAL); scene id (41-scene)"},
    "gates": {
        "A_recall5_min_gain_pp": 5.0, "A_baseline": "stronger of E_FULL and CONTROL (VAL Recall@5)",
        "B_min_recovered_top20": 8, "B_population": "12 historical Adapter E VAL errors with correct label outside top-5",
        "C_max_top1_drop_pp": 2.0, "C_reference": "E_FULL VAL raw top-1",
        "D_unknown_rejection_min": "E_FULL VAL unknown rejection", "D_max_false_confirmation_rise_pp": 0.5,
        "E_photo_scene_complete_min": "6/22 (verified Adapter E GT-region reference)",
    },
    "coverage": {"R_min": 15, "S_min": 10, "D_min": 8, "diversity_cluster_cos": 0.70, "near_dup_cos": 0.85,
                 "change_note": "diversity_cluster_cos set 0.80 -> 0.70 before freeze: at 0.80 cluster count equalled reference count for all 27 labels (library already dedups at 0.90), so it exposed no diversity limit"},
    "test_split": "NOT OPENED",
}


def protocol_sha() -> str:
    return hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()


def file_sha(p: Path | str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------- box geometry
def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def clip(b, W, H) -> list[float]:
    return [max(0.0, min(float(W), b[0])), max(0.0, min(float(H), b[1])),
            max(0.0, min(float(W), b[2])), max(0.0, min(float(H), b[3]))]


def nms(boxes: list, scores: list, thr: float) -> list[int]:
    order = sorted(range(len(boxes)), key=lambda i: -scores[i])
    keep = []
    for i in order:
        if all(iou(boxes[i], boxes[k]) < thr for k in keep):
            keep.append(i)
    return keep


def expand(b, r: float, W, H) -> list[float]:
    w, h = b[2] - b[0], b[3] - b[1]
    return clip([b[0] - r * w, b[1] - r * h, b[2] + r * w, b[3] + r * h], W, H)


def square(b, W, H) -> list[float]:
    s = min(max(b[2] - b[0], b[3] - b[1]), W, H)
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    x0 = min(max(0.0, cx - s / 2), W - s)
    y0 = min(max(0.0, cy - s / 2), H - s)
    return [x0, y0, x0 + s, y0 + s]


def control_crops(W: int, H: int, key: str, n: int) -> list[list[float]]:
    c = PROTOCOL["control"]
    rng = random.Random(f"{PROTOCOL['seed']}:{key}")
    out = []
    for _ in range(n):
        s = rng.uniform(*c["side_fraction"])
        a = rng.uniform(*c["aspect"])
        w = min(float(W), s * W * a ** 0.5)
        h = min(float(H), s * H / a ** 0.5)
        x0 = rng.uniform(0, W - w)
        y0 = rng.uniform(0, H - h)
        out.append([x0, y0, x0 + w, y0 + h])
    return out


def _postprocess(boxes, scores, labels, W, H) -> list[dict]:
    p = PROTOCOL["proposals"]
    boxes = [clip(b, W, H) for b in boxes]
    ok = [i for i, b in enumerate(boxes) if b[2] - b[0] >= p["min_side_px"] and b[3] - b[1] >= p["min_side_px"]]
    boxes, scores, labels = [boxes[i] for i in ok], [scores[i] for i in ok], [labels[i] for i in ok]
    keep = nms(boxes, scores, p["nms_iou"])[: p["max_per_source"]]
    return [{"box": [round(v, 2) for v in boxes[i]], "confidence": round(float(scores[i]), 5),
             "class_hint": labels[i], "rank": r} for r, i in enumerate(keep)]


def duplicate_rate(boxes: list, scores: list, thr: float = 0.5) -> float:
    if not boxes:
        return 0.0
    order = sorted(range(len(boxes)), key=lambda i: -scores[i])
    dup = sum(1 for n, i in enumerate(order) if any(iou(boxes[i], boxes[j]) >= thr for j in order[:n]))
    return dup / len(boxes)


# ------------------------------------------------------------- proposal sources
class FasterRCNNSource:
    name = "frcnn"

    def __init__(self, dev: torch.device):
        import torchvision
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
        cfg = PROTOCOL["proposals"]["frcnn"]
        self.dev = dev
        self.w = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        self.model = fasterrcnn_resnet50_fpn_v2(weights=self.w, box_score_thresh=cfg["box_score_thresh"],
                                                box_detections_per_img=cfg["box_detections_per_img"]).eval().to(dev)
        self.tf = self.w.transforms()
        self.cats = self.w.meta["categories"]
        ck = Path(torch.hub.get_dir()) / "checkpoints" / os.path.basename(self.w.url)
        self.info = {"model": "Faster R-CNN ResNet-50 FPN v2", "source_repository": "pytorch/vision",
                     "revision": f"torchvision {torchvision.__version__}", "checkpoint": self.w.url,
                     "checkpoint_sha256": file_sha(ck) if ck.is_file() else None,
                     "licence": "BSD-3-Clause (torchvision code); weights trained on COCO 2017",
                     "licence_url": "https://github.com/pytorch/vision/blob/main/LICENSE",
                     "weights_available": True, "environment_compatible": True, **cfg}
        self.raw_dup = []

    @torch.no_grad()
    def __call__(self, img: Image.Image) -> list[dict]:
        W, H = img.size
        out = self.model([self.tf(img).to(self.dev)])[0]
        b = out["boxes"].cpu().tolist()
        s = out["scores"].cpu().tolist()
        lab = [self.cats[i] for i in out["labels"].cpu().tolist()]
        self.raw_dup.append(duplicate_rate(b[:300], s[:300]))
        return _postprocess(b, s, lab, W, H)


class OWLv2Source:
    name = "owlv2"

    def __init__(self, dev: torch.device):
        import transformers
        from huggingface_hub import hf_hub_download
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        cfg = PROTOCOL["proposals"]["owlv2"]
        self.dev = dev
        self.proc = Owlv2Processor.from_pretrained(cfg["model"], revision=cfg["revision"])
        self.model = Owlv2ForObjectDetection.from_pretrained(cfg["model"], revision=cfg["revision"]).eval().to(dev)
        wpath = hf_hub_download(cfg["model"], "model.safetensors", revision=cfg["revision"])
        self.text = self.proc(text=[cfg["queries"]], return_tensors="pt")
        self.info = {"model": "OWLv2 base patch16 ensemble", "source_repository": f"huggingface.co/{cfg['model']}",
                     "revision": cfg["revision"], "checkpoint": "model.safetensors",
                     "checkpoint_sha256": file_sha(wpath), "licence": cfg["licence"],
                     "licence_url": "https://www.apache.org/licenses/LICENSE-2.0",
                     "transformers_version": transformers.__version__,
                     "weights_available": True, "environment_compatible": True, **cfg}
        self.raw_dup = []

    @torch.no_grad()
    def __call__(self, img: Image.Image) -> list[dict]:
        cfg = PROTOCOL["proposals"]["owlv2"]
        W, H = img.size
        side = float(max(W, H))
        pix = self.proc(images=img, return_tensors="pt")["pixel_values"].to(self.dev)
        out = self.model(input_ids=self.text["input_ids"].to(self.dev),
                         attention_mask=self.text["attention_mask"].to(self.dev), pixel_values=pix)
        prob = torch.sigmoid(out.logits[0]).max(-1)
        sc, qi = prob.values.cpu(), prob.indices.cpu()
        cxcywh = out.pred_boxes[0].cpu()
        top = torch.argsort(sc, descending=True)[: cfg["pre_nms_topk"]]
        top = [int(i) for i in top if float(sc[i]) >= cfg["score_thresh"]]
        boxes, scores, labs = [], [], []
        for i in top:
            cx, cy, w, h = (float(v) * side for v in cxcywh[i])
            boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            scores.append(float(sc[i]))
            labs.append(cfg["queries"][int(qi[i])])
        self.raw_dup.append(duplicate_rate([clip(b, W, H) for b in boxes], scores))
        return _postprocess(boxes, scores, labs, W, H)


def sam3_registry() -> dict:
    s = PROTOCOL["proposals"]["sam3"]
    return {"model": "SAM 3", "source_repository": f"huggingface.co/{s['model']}", "revision": s["revision"],
            "checkpoint": "sam3.pt / model.safetensors", "checkpoint_sha256": None, "licence": s["licence"],
            "licence_url": f"https://huggingface.co/{s['model']}/blob/main/LICENSE", "weights_available": False,
            "environment_compatible": None, "evaluated": False, "status": s["status"], "reason": s["reason"]}


# ------------------------------------------------------------- views + matched-budget arms
def build_views(W: int, H: int, key: str, props: dict[str, list[dict]]) -> tuple[list[dict], dict[str, list[int]], dict]:
    """Unique views (crop boxes) and, per arm, the list of N view indices. Returns (views, arms, padding counts)."""
    N, mv = PROTOCOL["N"], PROTOCOL["multiview"]
    views: list[dict] = []
    index: dict[tuple, int] = {}

    def add(crop, region, kind, source, rank, conf) -> int:
        k = (tuple(round(v, 1) for v in crop), kind, source)
        if k not in index:
            index[k] = len(views)
            views.append({"view_id": len(views), "crop_box": [round(v, 2) for v in crop],
                          "region_box": [round(v, 2) for v in region], "kind": kind, "source": source,
                          "proposal_rank": rank, "confidence": conf})
        return index[k]

    full = [0.0, 0.0, float(W), float(H)]
    f = add(full, full, "full", "full_image", 0, 1.0)
    ctrl = [add(b, b, "control_crop", "control", r, None) for r, b in enumerate(control_crops(W, H, key, N - 1))]
    arms = {"E_FULL": [f], "CONTROL": [f] + ctrl}
    padding = {}
    for src in ("frcnn", "owlv2"):
        P = props.get(src) or []
        reg = [add(p["box"], p["box"], "orig", src, p["rank"], p["confidence"]) for p in P[: N - 1]]
        padding[src.upper()] = N - 1 - len(reg)
        arms[src.upper()] = [f] + reg + ctrl[: N - 1 - len(reg)]
        m = []
        for r, kinds in ((0, ("orig", "padded", "square", "context")), (1, ("orig", "padded", "context"))):
            if r >= len(P):
                continue
            b = P[r]["box"]
            for kd in kinds:
                crop = {"orig": b, "padded": expand(b, mv["pad"], W, H), "square": square(b, W, H),
                        "context": expand(b, mv["context"], W, H)}[kd]
                m.append(add(crop, b, kd, src, P[r]["rank"], P[r]["confidence"]))
        padding[src.upper() + "_MV"] = N - 1 - len(m)
        arms[src.upper() + "_MV"] = [f] + m + ctrl[: N - 1 - len(m)]
    for a in PROTOCOL["matched_arms"]:
        assert len(arms[a]) == N, (a, len(arms[a]))
    return views, arms, padding


def crop(img: Image.Image, box) -> Image.Image:
    return img.crop(tuple(int(round(v)) for v in box))


def timed(fn, *a):
    t = time.perf_counter()
    r = fn(*a)
    return r, time.perf_counter() - t
