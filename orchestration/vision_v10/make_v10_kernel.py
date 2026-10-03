"""Build kaggle/kernels/kiq-v10-val-embed = unchanged V5 shared code + kiq_v10_regions.py + kiq_v10_val_embed.py."""
import json
from pathlib import Path

ROOT = Path(r"C:\Projects\KitchenIQ-V10-AI")
FUT = "from __future__ import annotations\n"
base = (ROOT / "kaggle/v5_src/kiq_v5_stage.py").read_text(encoding="utf-8")
shared = base[:base.index('if __name__ == "__main__":')]
code = shared.replace("__MODE__", "v10_val").replace("__LIBRARY_SHA__", "UNUSED").replace("__ADAPTER_CONFIG_SHA__", "UNUSED")
code += "\n# ===================== kiq_v10_regions.py (verbatim)\n"
code += (ROOT / "kaggle/v10_src/kiq_v10_regions.py").read_text(encoding="utf-8").replace(FUT, "")
code += (ROOT / "kaggle/v10_src/kiq_v10_val_embed.py").read_text(encoding="utf-8")
assert code.count(FUT) == 1 and code.startswith('"""')
out = ROOT / "kaggle/kernels/kiq-v10-val-embed"
out.mkdir(parents=True, exist_ok=True)
(out / "kiq_v10_val_embed.py").write_text(code, encoding="utf-8")
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": "kathiresannatarajan/kiq-v10-val-embed", "title": "kiq-v10-val-embed", "code_file": "kiq_v10_val_embed.py",
    "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
    "dataset_sources": [], "kernel_sources": ["kathiresannatarajan/kiq-v5-real-photo-v1"],
    "competition_sources": [], "model_sources": []}, indent=2), encoding="utf-8")
print(out)
