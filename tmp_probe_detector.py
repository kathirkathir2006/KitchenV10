"""Probe KIQ detector + FoodVisionV2 on tomato fixture."""
from __future__ import annotations

import io
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(r"C:/Projects/KitchenIQ-V10-AI")
OLD = Path(r"C:/Projects/KitchenIQ-OS")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(OLD / "backend"))

from app import create_app  # noqa: E402
from app.config import Config  # noqa: E402
from app.services.food_vision.detection import select_object_detector  # noqa: E402
from app.services.food_vision.v2.engine import FoodVisionV2Engine  # noqa: E402

img_path = (
    ROOT
    / "backend/instance/dev_experiments/v10-specialist-ensemble/acceptance/fixtures/generated/01_tomato_pilot.jpg"
)
img = Image.open(img_path).convert("RGB")
rgb = np.asarray(img)
print("img", rgb.shape, img.size)

buf = io.BytesIO()
img.save(buf, format="JPEG")
image_bytes = buf.getvalue()


class C(Config):
    TESTING = True
    AUTH_DEV_MODE = True
    SECURITY_ENFORCE_AUTH = False
    SQLALCHEMY_ENGINE_OPTIONS = {}


reg = OLD / "instance/dev_experiments/fv-production-final/registry_final.db"
tmp_db = ROOT / "backend/instance/dev_experiments/v10-specialist-ensemble/models/det_tmp.db"
tmp_db.parent.mkdir(parents=True, exist_ok=True)
if reg.is_file():
    shutil.copy2(reg, tmp_db)
C.SQLALCHEMY_DATABASE_URI = f"sqlite:///{tmp_db.as_posix()}"

app = create_app(C)
with app.app_context():
    det = select_object_detector()
    print("det_type", type(det), det)
    out = det.detect(
        image_bytes=image_bytes,
        mime="image/jpeg",
        evidence_ref="probe",
        session_context={},
    )
    print("det_out", type(out), getattr(out, "detections", None))
    dets = list(out.detections or [])
    print("ndets", len(dets))
    for d in dets[:5]:
        print(" ", d)

    eng = FoodVisionV2Engine()
    eng.load()
    res = eng.infer_image(img)
    obs = getattr(res, "observations", res)
    print("v2_type", type(res), "nobs", len(obs) if hasattr(obs, "__len__") else None)
    if hasattr(obs, "__iter__") and not isinstance(obs, (str, bytes)):
        for o in list(obs)[:5]:
            print(
                " ",
                getattr(o, "status", o),
                getattr(o, "identity", None),
                getattr(o, "confidence", None),
                getattr(o, "bbox", None),
            )
