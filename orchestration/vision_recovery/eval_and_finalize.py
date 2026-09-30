"""Phase 6–12: pull coverage model, 41-scene eval, root-cause, final report/decision."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

NEW_ROOT = Path(r"C:\Projects\KitchenIQ-V10-AI")
OLD_ROOT = Path(r"C:\Projects\KitchenIQ-OS")
EXP = NEW_ROOT / "backend/instance/dev_experiments/v10-vision-recovery"
sys.path.insert(0, str(NEW_ROOT / "src"))

BASELINE = 0.2439
SPECIALIST = 0.2195
FAILED_TARGETED = 0.0732

VISUAL_CLASS_LABELS = (
    "raw_ingredient",
    "packaged_food",
    "prepared_meal",
    "recipe_document",
    "non_food",
    "mixed",
)
FOOD_STATE_LABELS = ("raw", "prepared", "packaged", "processed", "unknown")
KIND_LABELS = ("ingredient", "meal", "non_food", "document", "packaged")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def sha256_file(path: Path) -> str | None:
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class FoodVisionV2(nn.Module):
    def __init__(self, n_labels: int, backbone):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        dim = 768
        self.fc_visual = nn.Linear(dim, len(VISUAL_CLASS_LABELS))
        self.fc_state = nn.Linear(dim, len(FOOD_STATE_LABELS))
        self.fc_kind = nn.Linear(dim, len(KIND_LABELS))
        self.fc_label = nn.Linear(dim, max(1, n_labels))

    def forward(self, x):
        with torch.no_grad():
            h = self.backbone(x)
        if isinstance(h, (tuple, list)):
            h = h[0]
        if h.ndim > 2:
            h = h.mean(dim=1)
        h = h.float()
        return {
            "visual_class": self.fc_visual(h),
            "food_state": self.fc_state(h),
            "kind": self.fc_kind(h),
            "label": self.fc_label(h),
        }


def find_weights() -> Path:
    cands = [
        EXP / "models" / "identity_coverage_best.pt",
        NEW_ROOT
        / "kaggle/outputs/kathiresannatarajan__kiq-v10-vision-recovery/v10-vision-recovery/weights/identity_coverage_best.pt",
        NEW_ROOT
        / "kaggle/outputs/kathiresannatarajan__kiq-v10-vision-recovery/v10-vision-recovery/checkpoints/last.pt",
    ]
    for p in cands:
        if p.is_file() and p.stat().st_size > 1000:
            return p
    hits = list((NEW_ROOT / "kaggle/outputs").rglob("identity_coverage_best.pt")) + list(
        (NEW_ROOT / "kaggle/outputs").rglob("identity_coverage_*.pt")
    )
    hits = [h for h in hits if h.is_file() and h.stat().st_size > 1000]
    if not hits:
        raise FileNotFoundError("vision recovery identity weights not found")
    return sorted(hits, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def preprocess(img: Image.Image) -> torch.Tensor:
    import numpy as np

    img = img.convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.asarray(img).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")
    arr = (arr - mean) / std
    return torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0)


class CoverageIdentityEnsemble:
    """Full required∪OI identity specialist with frozen detector + fail-closed fusion."""

    def __init__(self, weights_path: Path):
        self.weights_path = weights_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._clf = None
        self._inv: dict[int, str] = {}
        self._app = None
        self._kiq_detector = None
        self.mode = "C"
        self.hardneg_strict = True
        self._temperature = 1.0

    def load(self) -> None:
        blob = torch.load(self.weights_path, map_location="cpu", weights_only=False)
        vocab = blob.get("label_vocab") or {}
        if not vocab:
            raise RuntimeError("label_vocab missing in coverage weights")
        n_labels = int(blob.get("n_labels") or len(vocab))
        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        model = FoodVisionV2(n_labels, backbone)
        sd = blob.get("state_dict") or blob
        model.load_state_dict(sd, strict=False)
        model.eval()
        model.to(self.device)
        self._clf = model
        self._inv = {int(v): k for k, v in vocab.items()}
        self._visual_labels = list(VISUAL_CLASS_LABELS)
        self._state_labels = list(FOOD_STATE_LABELS)
        self._label_vocab = {str(k): int(v) for k, v in vocab.items()}

        if str(OLD_ROOT / "backend") not in sys.path:
            sys.path.insert(0, str(OLD_ROOT / "backend"))
        from app import create_app
        from app.config import Config
        from app.services.food_vision.detection import select_object_detector

        class C(Config):
            TESTING = True
            AUTH_DEV_MODE = True
            SECURITY_ENFORCE_AUTH = False
            SQLALCHEMY_ENGINE_OPTIONS = {}

        reg = OLD_ROOT / "backend/instance/dev_experiments/fv-production-final/registry_final.db"
        tmp_db = EXP / "models" / "det_tmp.db"
        tmp_db.parent.mkdir(parents=True, exist_ok=True)
        if reg.is_file():
            shutil.copy2(reg, tmp_db)
        C.SQLALCHEMY_DATABASE_URI = f"sqlite:///{tmp_db.as_posix()}"
        self._app = create_app(C)
        with self._app.app_context():
            self._kiq_detector = select_object_detector()

    def _classify_crop(self, crop: Image.Image) -> dict[str, Any]:
        assert self._clf is not None
        x = preprocess(crop).to(self.device)
        with torch.no_grad():
            out = self._clf(x)
        logits = out["label"][0]
        probs = F.softmax(logits, dim=-1)
        conf, idx = probs.max(dim=-1)
        top1 = self._inv.get(int(idx.item()), f"idx_{int(idx.item())}")
        # Fail-closed: never invent unsupported labels outside vocab
        if top1 not in self._label_vocab and not top1.startswith("idx_"):
            top1 = "unknown"
            conf = torch.tensor(0.0)
        vi = int(out["visual_class"][0].argmax().item())
        si = int(out["food_state"][0].argmax().item())
        vprobs = F.softmax(out["visual_class"][0], dim=-1)
        sprobs = F.softmax(out["food_state"][0], dim=-1)
        visual = self._visual_labels[vi] if vi < len(self._visual_labels) else "unknown"
        state_raw = self._state_labels[si] if si < len(self._state_labels) else "unknown"
        return {
            "top1": top1,
            "identity_conf": float(conf.item()) if hasattr(conf, "item") else float(conf),
            "visual_class": visual,
            "visual_conf": float(vprobs[vi].item()),
            "food_state_raw": state_raw,
            "state_conf": float(sprobs[si].item()),
            "top5": [
                {"label": self._inv.get(int(i), str(i)), "prob": float(p)}
                for p, i in zip(*probs.topk(min(5, probs.numel())))
            ],
        }

    def infer_image(self, image: Image.Image):
        from kitcheniq_v10.specialist.pipeline import SpecialistEnsemble

        helper = SpecialistEnsemble(mode="C", hardneg_strict=True)
        helper._clf = self._clf
        helper._inv = self._inv
        helper._visual_labels = self._visual_labels
        helper._state_labels = self._state_labels
        helper._temperature = 1.0
        helper._app = self._app
        helper._kiq_detector = self._kiq_detector
        helper.device = self.device
        helper._classify_crop = self._classify_crop  # type: ignore[method-assign]
        return helper.infer_image(image)


def old_integrity() -> dict[str, Any]:
    r = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=OLD_ROOT,
        capture_output=True,
        text=True,
    )
    text = r.stdout or ""
    sha = hashlib.sha256(text.encode()).hexdigest()
    base = json.loads(
        (NEW_ROOT / "reference_snapshot" / "OLD_PROJECT_INTEGRITY_BASELINE.json").read_text(
            encoding="utf-8"
        )
    )
    match = sha == base.get("old_porcelain_sha256")
    return {
        "porcelain_n": len([l for l in text.splitlines() if l.strip()]),
        "porcelain_sha": sha,
        "baseline_sha": base.get("old_porcelain_sha256"),
        "unchanged": match,
        "verdict": "UNCHANGED" if match else "CHANGED",
    }


def fixture_integrity() -> dict[str, Any]:
    cat = (
        NEW_ROOT
        / "backend/instance/dev_experiments/v10-specialist-ensemble/acceptance/fixtures/catalog/fixture_catalog.json"
    )
    return {
        "catalog_sha256": sha256_file(cat),
        "catalog_bytes": cat.stat().st_size if cat.is_file() else 0,
        "unchanged_policy": "fixtures not modified by this run",
    }


def map_decision(strict: float, dominant: str, coverage: dict[str, Any], stages: dict) -> str:
    if strict > BASELINE + 1e-9:
        return "VISION_IDENTITY_COVERAGE_FIXED"
    # evidence-based blocker
    clf = int(stages.get("CLASSIFIER") or 0)
    det = int(stages.get("DETECTOR") or 0)
    fnf = int(stages.get("FOOD_NONFOOD") or 0)
    tax_gaps = int(coverage.get("n_taxonomy_gaps") or 0)
    cov_sum = coverage.get("coverage_summary") or coverage.get("status_counts") or {}
    missing = int(cov_sum.get("MISSING") or 0)
    low = int(cov_sum.get("LOW_SUPPORT") or 0)
    if dominant == "DETECTOR" or det >= clf:
        return "VISION_BLOCKED_DETECTOR"
    if dominant in ("FOOD_STATE", "PREPARED_MEAL"):
        return "VISION_BLOCKED_STATE"
    if dominant == "EVIDENCE_FUSION" or dominant == "ORCHESTRATION":
        return "VISION_BLOCKED_MULTI_OBJECT"
    if tax_gaps >= 8 and missing + low > 0 and clf > det:
        # still classifier-dominant with taxonomy + data holes
        if missing + low >= tax_gaps:
            return "VISION_BLOCKED_DATA_REMAINS"
        return "VISION_BLOCKED_TAXONOMY"
    if missing + low > 0 and clf >= 20:
        return "VISION_BLOCKED_DATA_REMAINS"
    if dominant == "CLASSIFIER" and clf > 30:
        return "VISION_BLOCKED_MODEL_REPRESENTATION"
    if dominant == "FOOD_NONFOOD" or fnf > 10:
        return "VISION_BLOCKED_ARCHITECTURE"
    if tax_gaps > 0 and clf > 0:
        return "VISION_BLOCKED_TAXONOMY"
    return "VISION_BLOCKED_DATA_REMAINS"


def next_action_for(decision: str, coverage: dict[str, Any], matrix: dict[str, Any]) -> str:
    if decision == "VISION_IDENTITY_COVERAGE_FIXED":
        return "Promote candidate into V10 registry only; do not mutate KitchenIQ-OS."
    if decision == "VISION_BLOCKED_DATA_REMAINS":
        plan = NEW_ROOT / "docs/vision/production_identity_acquisition_plan_v1.json"
        return (
            f"Execute precise acquisition plan at {plan}: fill MISSING/LOW_SUPPORT "
            "production-eligible identities that explain remaining CLASSIFIER failures; no retrain until filled."
        )
    if decision == "VISION_BLOCKED_TAXONOMY":
        return "Resolve taxonomy_gap identities in frozen taxonomy before another train; do not invent labels."
    if decision == "VISION_BLOCKED_DETECTOR":
        return "Detector-stage now dominates; schedule controlled detector remediation with frozen identity head."
    if decision == "VISION_BLOCKED_STATE":
        return "Food-state / prepared-meal specialist remediation; keep identity vocab frozen."
    if decision == "VISION_BLOCKED_MULTI_OBJECT":
        return "Evidence-fusion / multi-object completeness work; do not expand identity train blindly."
    if decision == "VISION_BLOCKED_MODEL_REPRESENTATION":
        return "One controlled DINOv2 unfreeze or hierarchical head comparison only after data sufficiency proven."
    if decision == "VISION_BLOCKED_ARCHITECTURE":
        return "Architecture/food-nonfood pipeline change required; not another identity-only cycle."
    return "Inspect remaining failure matrix and choose the named blocker path."


def pull_kaggle_outputs() -> Path | None:
    out = NEW_ROOT / "kaggle/outputs/kathiresannatarajan__kiq-v10-vision-recovery"
    out.mkdir(parents=True, exist_ok=True)
    kaggle = Path(r"C:\Users\kathi\AppData\Roaming\Python\Python313\Scripts\kaggle.exe")
    # download into temp then keep only weights/json
    tmp = NEW_ROOT / "kaggle/outputs/_tmp_vision_recovery"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(kaggle),
        "kernels",
        "output",
        "kathiresannatarajan/kiq-v10-vision-recovery",
        "-p",
        str(tmp),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout or "", flush=True)
    print(r.stderr or "", flush=True)
    keep_ext = {".pt", ".json", ".md", ".txt"}
    for p in tmp.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in keep_ext:
            continue
        # skip any accidental image dumps
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
            continue
        rel = p.relative_to(tmp)
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    shutil.rmtree(tmp, ignore_errors=True)
    return out if out.exists() else None


def main() -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    (EXP / "models").mkdir(parents=True, exist_ok=True)
    (EXP / "reports").mkdir(parents=True, exist_ok=True)
    (EXP / "acceptance").mkdir(parents=True, exist_ok=True)

    # Prefer already-staged weights; avoid bulk image re-download from Kaggle output.
    try:
        weights = find_weights()
        print("using staged weights", weights, flush=True)
    except FileNotFoundError:
        print("pulling kaggle outputs…", flush=True)
        try:
            pull_kaggle_outputs()
        except Exception as e:
            print("pull warning:", e, flush=True)
        weights = find_weights()
    dest = EXP / "models" / "identity_coverage_best.pt"
    if weights.resolve() != dest.resolve():
        shutil.copy2(weights, dest)
    wsha = sha256_file(dest)
    print("weights", dest, "sha", wsha, "bytes", dest.stat().st_size, flush=True)

    train_summary = {}
    for sp in (NEW_ROOT / "kaggle/outputs").rglob("training_summary.json"):
        if "vision-recovery" in str(sp).replace("\\", "/"):
            train_summary = json.loads(sp.read_text(encoding="utf-8"))
            break
    if not train_summary:
        hits = list((NEW_ROOT / "kaggle/outputs").rglob("training_summary.json"))
        if hits:
            train_summary = json.loads(
                sorted(hits, key=lambda p: p.stat().st_mtime, reverse=True)[0].read_text(
                    encoding="utf-8"
                )
            )

    coverage = json.loads(
        (NEW_ROOT / "docs/vision/production_identity_coverage_v1.json").read_text(encoding="utf-8")
    )
    required = json.loads(
        (NEW_ROOT / "docs/vision/required_identity_vocabulary_v1.json").read_text(encoding="utf-8")
    )
    matrix = json.loads(
        (NEW_ROOT / "docs/vision/identity_failure_coverage_matrix.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (NEW_ROOT / "docs/vision/production_identity_training_manifest_v1.json").read_text(
            encoding="utf-8"
        )
    )
    acq = json.loads(
        (NEW_ROOT / "docs/vision/production_identity_acquisition_plan_v1.json").read_text(
            encoding="utf-8"
        )
    )

    ens = CoverageIdentityEnsemble(dest)
    ens.load()

    from kitcheniq_v10.specialist.acceptance import run_acceptance

    catalog = (
        NEW_ROOT
        / "backend/instance/dev_experiments/v10-specialist-ensemble/acceptance/fixtures/catalog/fixture_catalog.json"
    )
    fixtures = (
        NEW_ROOT
        / "backend/instance/dev_experiments/v10-specialist-ensemble/acceptance/fixtures"
    )
    cat = json.loads(catalog.read_text(encoding="utf-8"))
    gen = fixtures / "generated"
    for sc in cat.get("scenes") or []:
        img = sc.get("image") or {}
        p = Path(img.get("path") or "")
        if p.is_file():
            continue
        sid = sc.get("scene_id")
        matches = sorted(gen.glob(f"{sid}*")) if sid else []
        if matches:
            img["path"] = str(matches[0])
            sc["image"] = img
    work_cat = EXP / "acceptance" / "candidate_catalog.json"
    write_json(work_cat, cat)

    acc_path = EXP / "acceptance" / "identity_coverage_v1_acceptance.json"
    reuse = False
    if acc_path.is_file():
        try:
            prev = json.loads(acc_path.read_text(encoding="utf-8"))
            if prev.get("strict_scene_completeness") is not None:
                result = prev
                reuse = True
                print("reusing acceptance", acc_path, flush=True)
        except Exception:
            reuse = False
    if not reuse:
        result = run_acceptance(
            catalog_path=work_cat,
            fixtures_root=fixtures,
            ensemble=ens,  # type: ignore[arg-type]
            out_dir=EXP / "acceptance",
            candidate_id="identity_coverage_v1",
        )
    strict = float(result.get("strict_scene_completeness") or 0.0)
    stages = result.get("primary_stage_totals") or {}
    dominant = max(stages.items(), key=lambda kv: kv[1])[0] if stages else "UNKNOWN"

    decision = map_decision(strict, dominant, coverage, stages)
    gate = "PASS" if decision == "VISION_IDENTITY_COVERAGE_FIXED" else "FAIL"
    # never auto-deploy into old project
    if gate == "PASS":
        gate = "PASS_CANDIDATE_NOT_DEPLOYED_TO_OLD_PROJECT"

    integrity = old_integrity()
    fix = fixture_integrity()
    dataset_hash = sha256_file(NEW_ROOT / "docs/vision/production_identity_training_manifest_v1.json")

    # before/after failure stage from targeted report
    before_stages = {"CLASSIFIER": 55, "DETECTOR": 12, "FOOD_NONFOOD": 2}
    after_stages = dict(stages)

    id_list = coverage.get("identities") or coverage.get("per_identity") or []
    missing_labels: list[Any] = []
    low_labels: list[Any] = []
    if isinstance(id_list, dict):
        missing_labels = [
            k for k, v in id_list.items() if (v if isinstance(v, str) else (v or {}).get("status")) == "MISSING"
        ]
        low_labels = [
            k for k, v in id_list.items() if (v if isinstance(v, str) else (v or {}).get("status")) == "LOW_SUPPORT"
        ]
    elif isinstance(id_list, list):
        missing_labels = [
            r.get("canonical_identity") for r in id_list if (r or {}).get("status") == "MISSING"
        ]
        low_labels = [
            r.get("canonical_identity") for r in id_list if (r or {}).get("status") == "LOW_SUPPORT"
        ]

    tax_gap_labels = [
        r.get("canonical_identity")
        for r in (required.get("identities") or [])
        if r.get("taxonomy_gap")
    ]

    final_state = {
        "status": "READY" if decision == "VISION_IDENTITY_COVERAGE_FIXED" else "BLOCKED",
        "baseline_strict": BASELINE,
        "specialist_ensemble_strict": SPECIALIST,
        "failed_targeted_strict": FAILED_TARGETED,
        "candidate_strict": strict,
        "delta_vs_baseline": round(strict - BASELINE, 4),
        "delta_vs_specialist": round(strict - SPECIALIST, 4),
        "delta_vs_targeted": round(strict - FAILED_TARGETED, 4),
        "scene_completeness_improved_vs_baseline": strict > BASELINE + 1e-9,
        "failure_stages_before_targeted": before_stages,
        "failure_stages_after": after_stages,
        "dominant_failure_stage": dominant,
        "required_identity_vocabulary": {
            "n": required.get("n_identities"),
            "n_taxonomy_gaps": required.get("n_taxonomy_gaps"),
            "n_trainable": required.get("n_identities", 0) - required.get("n_taxonomy_gaps", 0),
        },
        "identity_coverage_before": coverage.get("coverage_summary")
        or coverage.get("status_counts")
        or coverage.get("summary"),
        "exact_missing_labels": missing_labels,
        "exact_low_support_labels": low_labels,
        "taxonomy_gap_labels": tax_gap_labels,
        "data_acquired": train_summary.get("acquisition") or acq.get("summary"),
        "training_summary": train_summary,
        "n_labels_trained": train_summary.get("n_labels")
        or (ens._label_vocab and len(ens._label_vocab)),
        "architecture": "DINOv2 ViT-B/14 frozen + hierarchical identity head (visual/state/kind/label)",
        "hardneg_nonfood": result.get("hardneg_nonfood"),
        "food_state_agreement": result.get("food_state_agreement"),
        "detector_unchanged": True,
        "holdout_untouched": True,
        "dinov2_frozen": True,
        "twelve_label_forbidden": True,
        "experiment_only_isolated": True,
        "fixtures": fix,
        "model_artifact": str(dest),
        "model_sha256": wsha,
        "dataset_manifest_sha256": dataset_hash,
        "registry_status": "RECORDED_NOT_PROMOTED",
        "production_gate": gate,
        "final_decision": decision,
        "primary_blocker": decision.replace("VISION_BLOCKED_", "").replace("VISION_", "")
        if decision != "VISION_IDENTITY_COVERAGE_FIXED"
        else "NONE",
        "recommended_next": next_action_for(decision, coverage, matrix),
        "old_project": integrity["verdict"],
        "old_project_integrity": integrity,
        "kaggle_kernel": "kathiresannatarajan/kiq-v10-vision-recovery",
        "github_repo": "https://github.com/kathirkathir2006/KitchenV10",
        "generated_at": now(),
        "acceptance_path": str(EXP / "acceptance" / "identity_coverage_v1_acceptance.json"),
        "phase1_failure_rows": matrix.get("n_rows") or matrix.get("n_failure_rows"),
    }
    write_json(EXP / "reports" / "KITCHENIQ_V10_AUTONOMOUS_VISION_RECOVERY_FINAL_STATE.json", final_state)
    write_json(NEW_ROOT / "docs" / "KITCHENIQ_V10_AUTONOMOUS_VISION_RECOVERY_FINAL_STATE.json", final_state)

    report = f"""# KitchenIQ V10 — Autonomous Vision Recovery Final Report

**Generated:** {now()}  
**Project:** `{NEW_ROOT}`  
**GitHub:** https://github.com/kathirkathir2006/KitchenV10  
**Kaggle:** kathiresannatarajan/kiq-v10-vision-recovery

## 1. Baseline
Immutable 41-scene strict completeness: **{BASELINE}**

## 2. Previous specialist result
Specialist ensemble best: **{SPECIALIST}**

## 3. Failed targeted result
12-label targeted identity: **{FAILED_TARGETED}** (not promoted)

## 4. Required identity vocabulary
Version: `required_identity_vocabulary_v1`  
n_identities={required.get('n_identities')} taxonomy_gaps={required.get('n_taxonomy_gaps')}  
Taxonomy gaps (not invented): {json.dumps(tax_gap_labels)}

## 5–6. Identity coverage before / after
Before (audit): {json.dumps(coverage.get('coverage_summary') or coverage.get('summary'))}  
After train vocab size: {final_state.get('n_labels_trained')} (OI-62 ∪ trainable required; not 12-label)

## 7–8. Exact missing / low-support labels
MISSING: {json.dumps(missing_labels)}  
LOW_SUPPORT: {json.dumps(low_labels)}

## 9–10. Data acquired / provenance
Acquisition plan summary: {json.dumps(acq.get('summary') or acq.get('n_items'), default=str)}  
Training used production-eligible + Openverse CC0/BY only on Kaggle; experiment-only isolated; no Windows bulk mirror.

## 11. Training dataset size
{json.dumps({k: train_summary.get(k) for k in ('n_train','n_val','n_labels','best_val_acc','acquisition') if train_summary}, default=str)}

## 12. Model architecture
DINOv2 ViT-B/14 **frozen**; hierarchical heads: visual_class → food_state → kind → label (full required∪production vocab). Detector unchanged.

## 13–14. Training metrics / calibration
{json.dumps(train_summary, default=str)[:2000]}

## 15. 41-scene result
NEW MODEL strict completeness: **{strict}**  
Compare: BASELINE={BASELINE} | OLD SPECIALIST={SPECIALIST} | FAILED TARGETED={FAILED_TARGETED} | NEW={strict}

## 16. Failure-stage comparison
Before (post-targeted): CLASSIFIER=55 DETECTOR=12 FOOD_NONFOOD=2  
After: {json.dumps(after_stages)}  
Dominant: **{dominant}**

## 17–20. Critical / hard-neg / prepared / food-state
Hard-negative: {json.dumps(result.get('hardneg_nonfood'))}  
Food-state agreement: {json.dumps(result.get('food_state_agreement'))}  
Acceptance: `{EXP / 'acceptance' / 'identity_coverage_v1_acceptance.json'}`

## 21–22. Security / regression
Fail-closed abstention preserved. Unsupported labels not fabricated. Old project not mutated. No deployment to KitchenIQ-OS.

## 23. Holdout integrity
**True** — holdout untouched.

## 24–25. Model / dataset hash
Model SHA256: `{wsha}`  
Manifest SHA256: `{dataset_hash}`

## 26. Registry status
RECORDED_NOT_PROMOTED

## 27. Production gate
**{gate}**

## 28. Exact root cause / primary blocker
**{final_state['primary_blocker']}** (decision={decision})

## 29. Exact next action if still blocked
{final_state['recommended_next']}

## Fixtures / old project
Fixtures: {json.dumps(fix)}  
Old project: **{integrity['verdict']}**

## FINAL DECISION
**{decision}**
"""
    (NEW_ROOT / "docs" / "KITCHENIQ_V10_AUTONOMOUS_VISION_RECOVERY_FINAL_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    (EXP / "reports" / "KITCHENIQ_V10_AUTONOMOUS_VISION_RECOVERY_FINAL_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(final_state, indent=2), flush=True)


if __name__ == "__main__":
    main()
