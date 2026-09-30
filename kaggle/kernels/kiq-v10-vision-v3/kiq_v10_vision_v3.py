"""Kaggle: build DINOv3 (or DINOv2 fallback) reference library from production-eligible crops.

Exports embeddings JSON only — no bulk image mirror to Windows.
"""

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
WORK = Path("/kaggle/working/v10-vision-v3")
REPORTS = WORK / "reports"
IMG = WORK / "images"
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def status(msg: str) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (WORK / "LIVE_STATUS.txt").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def find_pack() -> Path:
    for p in [
        Path("/kaggle/input/kiq-v10-recovery-pack"),
        Path("/kaggle/input/datasets/kathiresannatarajan/kiq-v10-recovery-pack"),
    ]:
        if p.exists():
            return p
    raise FileNotFoundError("recovery pack not found")


def load_backbone(device):
    errors = []
    for repo, name in [
        ("facebookresearch/dinov3", "dinov3_vitb16"),
        ("facebookresearch/dinov3", "dinov3_vits16"),
    ]:
        try:
            m = torch.hub.load(repo, name, pretrained=True, trust_repo=True)
            return m, f"DINOv3:{name}", None
        except Exception as e:
            errors.append(f"{repo}/{name}:{e}")
    m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True)
    return m, "DINOv2:dinov2_vitb14", ";".join(errors)[:500]


def preprocess(img, device):
    img = img.convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.asarray(img).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")
    arr = (arr - mean) / std
    t = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0).to(device)
    return t


@torch.no_grad()
def embed(model, x):
    if hasattr(model, "forward_features"):
        feats = model.forward_features(x)
        if isinstance(feats, dict) and "x_norm_clstoken" in feats:
            g = feats["x_norm_clstoken"]
            return F.normalize(g.float(), dim=-1)
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if out.ndim > 2:
        out = out.mean(dim=1)
    return F.normalize(out.float(), dim=-1)


def download(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "KitchenIQ-V3/1.0"})
        with urllib.request.urlopen(req, timeout=25) as r:
            data = r.read()
        if len(data) < 1000:
            return False
        dest.write_bytes(data)
        try:
            Image.open(dest).verify()
        except Exception:
            dest.unlink(missing_ok=True)
            return False
        return True
    except Exception:
        return False


FAMILY = {
    "tomato": "produce",
    "onion": "produce",
    "potato": "produce",
    "carrot": "produce",
    "garlic": "produce",
    "ginger": "produce",
    "cheese": "dairy",
    "packaged_cheese": "packaged_dairy",
    "cooked_rice": "rice_grain",
    "biryani": "rice_grain",
    "fried_egg": "prepared_egg",
    "pasta": "grain_starch",
    "pizza": "prepared_meal",
    "salad": "prepared_meal",
    "sandwich": "prepared_meal",
    "doughnut": "prepared_sweet",
    "ice_cream": "prepared_sweet",
    "french_fries": "prepared_side",
    "non_food": "non_food",
}


def main():
    for d in (WORK, REPORTS, IMG):
        d.mkdir(parents=True, exist_ok=True)
    status("START v3 reference library")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pack = find_pack()
    model, name, fb = load_backbone(device)
    model.eval().to(device)
    status(f"backbone={name} fallback={fb}")

    # rows from production eligible + acquired if present
    rows = []
    man = pack / "production_eligible_manifest.jsonl"
    if man.is_file():
        for line in man.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))
    # also training manifest labels list for targets
    tmax = 12  # per label cap to stay within session
    by = defaultdict(list)
    for r in rows:
        lab = (
            r.get("fine_label")
            or r.get("target_class")
            or r.get("label")
            or ""
        ).lower().replace(" ", "_")
        if not lab:
            continue
        by[lab].append(r)

    items = []
    for lab, lst in sorted(by.items()):
        random.shuffle(lst)
        n_ok = 0
        for r in lst:
            if n_ok >= tmax:
                break
            url = (
                r.get("image_url")
                or r.get("url")
                or r.get("download_url")
                or r.get("http_url")
            )
            # Open Images often stores only image_id — construct AWS URL when possible
            if not url and (r.get("source") or "").lower() in {"open_images", "openimages", "oi"}:
                iid = r.get("image_id") or ""
                if iid and "/" not in str(iid):
                    # best-effort thumbnail path used by many OI mirrors; skip if unknown
                    url = None
            if not url:
                continue
            dest = IMG / f"{lab}_{n_ok}.jpg"
            if not dest.is_file():
                if not download(str(url), dest):
                    continue
            try:
                im = Image.open(dest).convert("RGB")
            except Exception:
                continue
            x = preprocess(im, device)
            g = embed(model, x).detach().cpu().numpy().reshape(-1)
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
                    "provenance": r.get("source") or r.get("provenance") or "production_eligible",
                    "licence": r.get("license") or r.get("licence") or "unknown",
                    "source_hash": h,
                    "model_version": name,
                }
            )
            n_ok += 1
        status(f"label={lab} embedded={n_ok}")

    out = {"model_version": name, "fallback_reason": fb, "n": len(items), "items": items}
    (WORK / "reference_library.json").write_text(json.dumps(out), encoding="utf-8")
    summary = {
        "backbone": name,
        "fallback_reason": fb,
        "n_items": len(items),
        "labels": sorted({i["canonical_identity"] for i in items}),
        "sha256": hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest(),
    }
    (REPORTS / "reference_library_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    status(f"DONE n={len(items)} sha={summary['sha256']}")


if __name__ == "__main__":
    main()
