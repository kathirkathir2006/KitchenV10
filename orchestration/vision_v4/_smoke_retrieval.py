"""Smoke: load V4 with library and observe one scene."""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

NEW_ROOT = Path(r"C:\Projects\KitchenIQ-V10-AI")
sys.path.insert(0, str(NEW_ROOT / "src"))
sys.path.insert(0, str(NEW_ROOT))

from PIL import Image

from kitcheniq_v10.v4 import VisionV4Engine
from orchestration.vision_v4.run_benchmark_and_finalize import prepare_catalog, find_library


def main() -> None:
    lib = find_library()
    print("library", lib, flush=True)
    eng = VisionV4Engine(library_path=lib)
    eng.load()
    print("retrieval_enabled", eng.retrieval_enabled, "n", len(eng._library.items) if eng._library else 0, flush=True)
    catalog, fixtures = prepare_catalog()
    cat = json.loads(catalog.read_text(encoding="utf-8"))
    sc = cat["scenes"][0]
    img_path = Path(sc["image"]["path"])
    print("scene", sc.get("scene_id"), "img", img_path, img_path.is_file(), flush=True)
    im = Image.open(img_path)
    try:
        fused = eng.infer_image(im)
    except Exception:
        traceback.print_exc()
        raise
    print("n_obs", len(fused), flush=True)
    for o in fused[:5]:
        print(
            {
                "status": o.status,
                "identity": o.identity,
                "conf": o.confidence,
                "is_food": o.is_food,
                "state": o.food_state,
                "prov": o.provenance,
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
