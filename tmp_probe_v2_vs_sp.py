"""Probe KIQ detector on multi-object fixture from official catalog path."""
from __future__ import annotations

import io
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(r"C:/Projects/KitchenIQ-V10-AI")
OLD = Path(r"C:/Projects/KitchenIQ-OS")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(OLD / "backend"))

cat = json.loads(
    (
        ROOT
        / "backend/instance/dev_experiments/v10-specialist-ensemble/acceptance/fixtures/catalog/fixture_catalog.json"
    ).read_text(encoding="utf-8")
)
# find multi + a known good scene from production
for sid in ("01_tomato_pilot", "02_multi_raw_composite", "13_kitchen_mood", "12_recipe_doc"):
    sc = next((s for s in cat["scenes"] if s["scene_id"] == sid), None)
    if not sc:
        print(sid, "MISSING_IN_CATALOG")
        continue
    p = Path((sc.get("image") or {}).get("path") or "")
    print(sid, "path_ok", p.is_file(), p, "size", p.stat().st_size if p.is_file() else None)

from app import create_app
from app.config import Config
from app.services.food_vision.detection import select_object_detector
from app.services.food_vision.v2.pipeline import run_vision_engine_v2
from app.services.food_vision.real_world_acceptance.metrics import (
    match_instances,
    scene_completeness,
)


class C(Config):
    TESTING = True
    AUTH_DEV_MODE = True
    SECURITY_ENFORCE_AUTH = False
    SQLALCHEMY_ENGINE_OPTIONS = {}


reg = OLD / "instance/dev_experiments/fv-production-final/registry_final.db"
tmp_db = ROOT / "backend/instance/dev_experiments/v10-specialist-ensemble/models/det_tmp.db"
tmp_db.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(reg, tmp_db)
C.SQLALCHEMY_DATABASE_URI = f"sqlite:///{tmp_db.as_posix()}"
app = create_app(C)

# Compare V2 vs specialist on a few scenes
from kitcheniq_v10.specialist.pipeline import SpecialistEnsemble

ens = SpecialistEnsemble(mode="C")
ens.load()

with app.app_context():
    det = select_object_detector()
    n_strict_v2 = 0
    n_strict_sp = 0
    n = 0
    for sc in cat["scenes"]:
        p = Path((sc.get("image") or {}).get("path") or "")
        if not p.is_file():
            # try V10 local copy
            local = (
                ROOT
                / "backend/instance/dev_experiments/v10-specialist-ensemble/acceptance/fixtures/generated"
                / p.name
            )
            if local.is_file():
                p = local
            else:
                print("SKIP missing", sc["scene_id"])
                continue
        image_bytes = p.read_bytes()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # detector raw
        dout = det.detect(
            image_bytes=image_bytes,
            mime="image/jpeg",
            evidence_ref="probe",
            session_context={},
        )
        nd = len(list(dout.detections or []))
        # V2
        v2 = run_vision_engine_v2(
            image_bytes=image_bytes,
            mime="image/jpeg",
            evidence_ref=f"probe/{sc['scene_id']}",
            skip_secure_boundary=True,
            boundary_allowed=True,
        )
        v2_obs = list(v2.to_dict().get("observations") or [])
        m_v2 = match_instances(sc.get("instances") or [], v2_obs)
        sc_v2 = scene_completeness(m_v2)
        # Specialist → map to official obs schema
        sp_obs_raw = ens.infer_image(img)
        sp_obs = []
        for o in sp_obs_raw:
            sp_obs.append(
                {
                    "raw_label": o.identity or o.raw_label,
                    "fine_label": o.identity,
                    "bbox": o.bbox,
                    "confidence": o.confidence,
                    "visual_class": o.visual_class,
                    "food_state": o.food_state,
                    "status": o.status,
                }
            )
        m_sp = match_instances(sc.get("instances") or [], sp_obs)
        sc_sp = scene_completeness(m_sp)
        # empty / nonfood scenes: official runner has special handling — approximate
        if sc_v2["strict_scene_complete"]:
            n_strict_v2 += 1
        if sc_sp["strict_scene_complete"]:
            n_strict_sp += 1
        n += 1
        if sc["scene_id"] in {
            "01_tomato_pilot",
            "02_multi_raw_composite",
            "13_kitchen_mood",
            "08_empty_plate",
            "05_ingredient_prepared",
        } or (sc_v2["strict_scene_complete"] != sc_sp["strict_scene_complete"]):
            print(
                sc["scene_id"],
                "ndet",
                nd,
                "v2_nobs",
                len(v2_obs),
                "v2_strict",
                sc_v2["strict_scene_complete"],
                "v2_comp",
                sc_v2["scene_completeness"],
                "sp_nobs",
                len(sp_obs),
                "sp_strict",
                sc_sp["strict_scene_complete"],
                "sp_comp",
                sc_sp["scene_completeness"],
                "sp0",
                sp_obs[0] if sp_obs else None,
                "v2_0",
                {k: v2_obs[0].get(k) for k in ("raw_label", "confidence", "visual_class", "bbox", "food_state")}
                if v2_obs
                else None,
            )

print("TOTAL", n, "V2_strict_rate", round(n_strict_v2 / max(n, 1), 4), "SP_strict_rate", round(n_strict_sp / max(n, 1), 4))
