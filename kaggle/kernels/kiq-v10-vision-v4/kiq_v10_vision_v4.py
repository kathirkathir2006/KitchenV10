"""Kaggle V4: build production-eligible reference library with ASSERT n_items > 0."""

from __future__ import annotations

import hashlib
import json
import random
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SEED = 42
WORK = Path("/kaggle/working/v10-vision-v4")
REPORTS = WORK / "reports"
IMG = WORK / "images"
MAX_PER = 10
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

FAMILY = {
    "tomato": "produce", "onion": "produce", "potato": "produce", "carrot": "produce",
    "garlic": "produce", "ginger": "produce", "cheese": "dairy",
    "packaged_cheese": "packaged_dairy", "cooked_rice": "rice_grain", "biryani": "rice_grain",
    "fried_rice": "rice_grain", "fried_egg": "prepared_egg", "pasta": "grain_starch",
    "pizza": "prepared_meal", "salad": "prepared_meal", "sandwich": "prepared_meal",
    "doughnut": "prepared_sweet", "ice_cream": "prepared_sweet", "french_fries": "prepared_side",
    "yogurt": "dairy", "cream": "dairy", "non_food": "non_food",
}


def status(msg: str) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (WORK / "LIVE_STATUS.txt").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def find_pack() -> Path:
    for p in [Path("/kaggle/input/kiq-v10-recovery-pack"), Path("/kaggle/input/datasets/kathiresannatarajan/kiq-v10-recovery-pack")]:
        if p.exists():
            return p
    raise FileNotFoundError("recovery pack missing")


def download(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "KitchenIQ-V4/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        if len(data) < 1500:
            return False
        dest.write_bytes(data)
        Image.open(dest).verify()
        return True
    except Exception:
        dest.unlink(missing_ok=True)
        return False


def preprocess(img, device):
    img = img.convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.asarray(img).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")
    arr = (arr - mean) / std
    return torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0).to(device)


@torch.no_grad()
def embed(model, x):
    if hasattr(model, "forward_features"):
        feats = model.forward_features(x)
        if isinstance(feats, dict) and "x_norm_clstoken" in feats:
            return F.normalize(feats["x_norm_clstoken"].float(), dim=-1)
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if out.ndim > 2:
        out = out.mean(dim=1)
    return F.normalize(out.float(), dim=-1)


def main():
    for d in (WORK, REPORTS, IMG):
        d.mkdir(parents=True, exist_ok=True)
    status("START v4 reference library")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DINOv2 — DINOv3 gated 403; do not use untrained weights
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True)
    model.eval().to(device)
    status("backbone=DINOv2:dinov2_vitb14")
    pack = find_pack()
    rows = []
    man = pack / "production_eligible_manifest.jsonl"
    for line in man.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    by = defaultdict(list)
    for r in rows:
        lab = (r.get("target_class") or r.get("label") or r.get("fine_label") or "").lower().replace(" ", "_")
        if lab:
            by[lab].append(r)
    items = []
    for lab, lst in sorted(by.items()):
        random.shuffle(lst)
        n_ok = 0
        for r in lst:
            if n_ok >= MAX_PER:
                break
            url = r.get("image_url") or r.get("download_url") or r.get("url")
            if not url:
                continue
            dest = IMG / f"{lab}_{n_ok}.jpg"
            if not dest.is_file() and not download(str(url), dest):
                continue
            try:
                im = Image.open(dest).convert("RGB")
            except Exception:
                continue
            g = embed(model, preprocess(im, device)).detach().cpu().numpy().reshape(-1)
            fam = FAMILY.get(lab, "unknown")
            kind = (
                "prepared_meal"
                if fam in {"prepared_meal", "prepared_egg", "prepared_sweet", "prepared_side", "rice_grain"}
                else ("packaged" if "packaged" in fam else ("non_food" if fam == "non_food" else "ingredient"))
            )
            state = "prepared" if kind == "prepared_meal" else ("packaged" if kind == "packaged" else "raw")
            h = hashlib.sha256(dest.read_bytes()).hexdigest()
            items.append(
                {
                    "id": f"{lab}_{n_ok}_{h[:8]}",
                    "canonical_identity": lab,
                    "family": fam,
                    "kind": kind,
                    "food_state": state,
                    "embedding": g.astype(float).tolist(),
                    "provenance": r.get("source") or "production_eligible",
                    "licence": r.get("licence") or r.get("license") or "unknown",
                    "source_hash": h,
                    "model_version": "DINOv2:dinov2_vitb14",
                    "data_zone": r.get("data_zone") or "PRODUCTION_ELIGIBLE",
                }
            )
            n_ok += 1
        status(f"label={lab} embedded={n_ok}")

    n = len(items)
    status(f"n_items={n}")
    if n <= 0:
        status("FATAL library n_items==0 — abort candidate")
        raise SystemExit("ASSERT_FAIL n_items>0")

    out = {
        "model_version": "DINOv2:dinov2_vitb14",
        "n": n,
        "items": items,
        "experiment_only_mixed": False,
        "assert_n_items_gt_0": True,
    }
    raw = json.dumps(out)
    (WORK / "reference_library.json").write_text(raw, encoding="utf-8")
    summary = {
        "n_items": n,
        "labels": sorted({i["canonical_identity"] for i in items}),
        "sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "backbone": "DINOv2:dinov2_vitb14",
    }
    (REPORTS / "reference_library_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    status(f"DONE sha={summary['sha256']}")


if __name__ == "__main__":
    main()
