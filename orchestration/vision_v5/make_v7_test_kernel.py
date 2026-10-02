"""Build kaggle/kernels/kiq-v7-test-embed from the unchanged V5 shared code + kaggle/v7_src/kiq_v7_test_embed.py."""
import json
from pathlib import Path

ROOT = Path(r"C:\Projects\KitchenIQ-V10-AI")
base = (ROOT / "kaggle/v5_src/kiq_v5_stage.py").read_text(encoding="utf-8")
shared = base[:base.index('if __name__ == "__main__":')]
code = shared.replace("__MODE__", "v7_test_embed").replace("__LIBRARY_SHA__", "UNUSED").replace("__ADAPTER_CONFIG_SHA__", "UNUSED")
code += (ROOT / "kaggle/v7_src/kiq_v7_test_embed.py").read_text(encoding="utf-8")
out = ROOT / "kaggle/kernels/kiq-v7-test-embed"
out.mkdir(parents=True, exist_ok=True)
(out / "kiq_v5_stage.py").write_text(code, encoding="utf-8")
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": "kathiresannatarajan/kiq-v7-test-embed", "title": "kiq-v7-test-embed", "code_file": "kiq_v5_stage.py",
    "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
    "dataset_sources": [], "kernel_sources": ["kathiresannatarajan/kiq-v5-real-photo-v1"],
    "competition_sources": [], "model_sources": []}, indent=2), encoding="utf-8")
print(out)
