"""Export leakage fingerprints of the frozen 41-scene benchmark (no images leave the machine).

Per scene image and per required GT crop: sha256 (scene only), 64-bit dHash, DINOv2 CLS embedding.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(r"C:\Projects\KitchenIQ-V10-AI")
CATALOG = ROOT / "backend/instance/dev_experiments/v10-specialist-ensemble/acceptance/fixtures/catalog/fixture_catalog.json"
GEN = Path(r"C:\Projects\KitchenIQ-OS\backend\instance\dev_experiments\fv-real-world-acceptance-v2\fixtures\generated")
CATALOG_SHA = "38fc3eaee75c76fa5fd1dc3d4177af4abde2cfeb3e410d290fb524f9e30a1523"
OUT = ROOT / "kaggle/datasets/kiq-v5-benchmark-fingerprints"


def dhash(img: Image.Image) -> str:
    g = np.asarray(img.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def preprocess(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB").resize((224, 224), Image.BILINEAR)).astype("float32") / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], "float32")) / np.array([0.229, 0.224, 0.225], "float32")
    return torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0)


@torch.no_grad()
def embed(model, img: Image.Image) -> list[float]:
    v = F.normalize(model.forward_features(preprocess(img))["x_norm_clstoken"].float(), dim=-1)[0]
    return [round(float(x), 6) for x in v]


def main() -> None:
    assert hashlib.sha256(CATALOG.read_bytes()).hexdigest() == CATALOG_SHA, "benchmark catalog changed — STOP"
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).eval()
    cat = json.loads(CATALOG.read_text(encoding="utf-8"))
    prints = []
    for sc in cat["scenes"]:
        p = Path((sc.get("image") or {}).get("path") or "")
        if not p.is_file():
            p = sorted(GEN.glob(f"{sc['scene_id']}*"))[0]
        img = Image.open(p).convert("RGB")
        W, H = img.size
        prints.append({"scene_id": sc["scene_id"], "kind": "scene", "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                       "dhash": dhash(img), "embedding": embed(model, img)})
        for inst in sc.get("instances") or []:
            b = inst.get("bbox")
            if not b:
                continue
            crop = img.crop((max(0, int(b[0])), max(0, int(b[1])), min(W, int(b[2])), min(H, int(b[3]))))
            prints.append({"scene_id": sc["scene_id"], "kind": "gt_crop", "instance_id": inst.get("instance_id"),
                           "dhash": dhash(crop), "embedding": embed(model, crop)})
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"catalog_sha256": CATALOG_SHA, "n": len(prints), "backbone": "DINOv2:dinov2_vitb14",
               "contains_images": False, "fingerprints": prints}
    (OUT / "benchmark_fingerprints.json").write_text(json.dumps(payload), encoding="utf-8")
    (OUT / "dataset-metadata.json").write_text(json.dumps({
        "title": "kiq-v5-benchmark-fingerprints", "id": "kathiresannatarajan/kiq-v5-benchmark-fingerprints",
        "licenses": [{"name": "other"}]}, indent=2), encoding="utf-8")
    print("fingerprints", len(prints))


if __name__ == "__main__":
    main()
