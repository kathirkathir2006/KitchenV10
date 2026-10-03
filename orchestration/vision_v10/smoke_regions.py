"""Local smoke test of kiq_v10_regions on one 41-scene image (no scoring)."""
import importlib.util
import json
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(r"C:\Projects\KitchenIQ-V10-AI")
spec = importlib.util.spec_from_file_location("kiq_v10_regions", ROOT / "kaggle/v10_src/kiq_v10_regions.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
cat = json.loads((ROOT / "backend/instance/dev_experiments/v10-specialist-ensemble/acceptance/fixtures/catalog/fixture_catalog.json").read_text(encoding="utf-8"))
sc = max(cat["scenes"], key=lambda s: len(s.get("instances") or []))
img = Image.open(sc["image"]["path"]).convert("RGB")
dev = torch.device("cpu")
props = {}
for cls in (R.FasterRCNNSource, R.OWLv2Source):
    s = cls(dev)
    props[s.name], dt = R.timed(s, img)
    print(s.name, f"{dt:.1f}s", len(props[s.name]), props[s.name][:2], json.dumps(s.info)[:400])
views, arms, pad = R.build_views(img.size[0], img.size[1], sc["image"]["sha256"], props)
print(sc["scene_id"], img.size, "views", len(views), {a: len(v) for a, v in arms.items()}, pad)
print("gt", [(i["label"], i["bbox"]) for i in sc["instances"]])
print("protocol_sha", R.protocol_sha())
