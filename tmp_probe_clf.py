"""Probe tomato classification logits and fixture catalog bbox fields."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(r"C:/Projects/KitchenIQ-V10-AI")
sys.path.insert(0, str(ROOT / "src"))

from kitcheniq_v10.specialist.pipeline import SpecialistEnsemble  # noqa: E402

cat = json.loads(
    (ROOT / "backend/instance/dev_experiments/v10-specialist-ensemble/acceptance/fixtures/catalog/fixture_catalog.json").read_text(
        encoding="utf-8"
    )
)
for sc in cat["scenes"][:3]:
    inst = sc.get("instances") or []
    print(
        sc["scene_id"],
        "n_inst",
        len(inst),
        "bboxes",
        [i.get("bbox") for i in inst[:4]],
        "path",
        (sc.get("image") or {}).get("path"),
    )

gen = ROOT / "backend/instance/dev_experiments/v10-specialist-ensemble/acceptance/fixtures/generated"
print("gen_exists", gen.is_dir(), "n", len(list(gen.glob("*"))) if gen.is_dir() else 0)
img_path = gen / "01_tomato_pilot.jpg"
print("tomato_exists", img_path.is_file(), img_path.stat().st_size if img_path.is_file() else None)

e = SpecialistEnsemble(mode="C", hardneg_strict=False)
e.load()
img = Image.open(img_path).convert("RGB")
print("size", img.size)
clf = e._classify_crop(img)
print("clf", {k: clf[k] for k in ("top1", "identity_conf", "visual_class", "visual_conf", "food_state_raw", "state_conf")})
print("top5", clf["top5"][:5])
fn = e._food_nonfood_specialist(clf)
ident = e._identity_specialist(clf)
prep = e._prepared_specialist(clf)
print("fn", fn)
print("ident", ident)
print("prep", prep)
obs = e.infer_image(img)
print("obs0", obs[0])
