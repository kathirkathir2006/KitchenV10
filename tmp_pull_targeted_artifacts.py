"""Download Kaggle kernel output and keep only weights/reports (no images)."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

OUT_ROOT = Path(r"C:/Projects/KitchenIQ-V10-AI/kaggle/outputs/kathiresannatarajan__kiq-v10-targeted-clf")
KAGGLE = Path(r"C:\Users\kathi\AppData\Roaming\Python\Python313\Scripts\kaggle.exe")
KEEP_SUFFIX = {".pt", ".json", ".txt"}


def want(name: str) -> bool:
    low = name.replace("\\", "/").lower()
    if "/images/" in low or low.endswith((".jpg", ".jpeg", ".png", ".webp")):
        return False
    p = Path(name)
    if p.suffix.lower() in KEEP_SUFFIX:
        return True
    return "live_status" in p.name.lower()


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="kiq_out_"))
    print("downloading to", tmp, flush=True)
    subprocess.check_call(
        [str(KAGGLE), "kernels", "output", "kathiresannatarajan/kiq-v10-targeted-clf", "-p", str(tmp)]
    )
    zips = list(tmp.glob("*.zip"))
    print("zips", zips, flush=True)
    kept: list[tuple[str, int]] = []
    if zips:
        with zipfile.ZipFile(zips[0], "r") as zf:
            for name in zf.namelist():
                if not want(name):
                    continue
                target = OUT_ROOT / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                kept.append((name, target.stat().st_size))
    else:
        for p in tmp.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(tmp).as_posix()
            if not want(rel):
                continue
            target = OUT_ROOT / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
            kept.append((rel, target.stat().st_size))
    print("KEPT", kept, flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    # also purge any images that leaked
    img = OUT_ROOT / "v10-targeted-clf" / "images"
    if img.is_dir():
        shutil.rmtree(img, ignore_errors=True)
        print("purged images", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
