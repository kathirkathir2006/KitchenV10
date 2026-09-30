"""V10 Vision Recovery — Phases 1–5: failure audit, required vocab, coverage, acquisition plan, training manifest.

Does NOT train. Does NOT download bulk images to Windows.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

NEW_ROOT = Path(r"C:\Projects\KitchenIQ-V10-AI")
OLD_ROOT = Path(r"C:\Projects\KitchenIQ-OS")
DOCS = NEW_ROOT / "docs" / "vision"
EXP = NEW_ROOT / "backend" / "instance" / "dev_experiments" / "v10-vision-recovery"

TARGETED_ACCEPT = (
    NEW_ROOT
    / "backend/instance/dev_experiments/v10-targeted-classifier/acceptance/targeted_identity_v1_acceptance.json"
)
SPECIALIST_ACCEPT = (
    NEW_ROOT
    / "backend/instance/dev_experiments/v10-specialist-ensemble/acceptance/candidate_a_acceptance.json"
)
CATALOG = (
    NEW_ROOT
    / "backend/instance/dev_experiments/v10-specialist-ensemble/acceptance/fixtures/catalog/fixture_catalog.json"
)
TAX_CSV = OLD_ROOT / "backend/app/services/food_vision/dataset/v1_label_taxonomy.csv"
PROD_WEIGHTS = (
    OLD_ROOT
    / "backend/instance/dev_experiments/fv-production-final/registry_artifacts"
    / "food-vision-classifier-v1/classifier-prod-v1.0.0/model.pt"
)
PROD_MAN = (
    NEW_ROOT
    / "backend/instance/dev_experiments/v10-targeted-classifier/data/production_eligible_manifest.jsonl"
)
HN_MAN = (
    NEW_ROOT
    / "backend/instance/dev_experiments/v10-targeted-classifier/data/hard_negative_manifest.jsonl"
)
EXP_MAN = (
    NEW_ROOT
    / "backend/instance/dev_experiments/v10-targeted-classifier/data/experiment_only_manifest.jsonl"
)

# Explicit scene→taxonomy canonical maps (no invention)
CANONICAL_MAP = {
    "hand": "human_hand",
    "human_hand": "human_hand",
    "non_food": "non_food_object",
    "empty_plate": "empty_plate",
    "fried_rice": "cooked_rice",  # closest tax-supported prepared rice identity
    "omelette": "fried_egg",
    "donuts": "doughnut",  # if absent → gap checked later
    "packaged_cheese": "packaged_cheese",
    "recipe_document": "recipe_document",  # may gap
    "club_sandwich": "sandwich",  # production OI has sandwich; tax may gap
    "salad": "salad",
    "lasagna": "lasagna",
    "hummus": "hummus",
    "guacamole": "guacamole",
    "pad_thai": "pad_thai",
    "bibimbap": "bibimbap",
    "spring_rolls": "spring_rolls",
    "cherry_tomatoes": "tomato",
    "tomatoes": "tomato",
    "greek_yogurt": "yogurt",
    "plain_yogurt": "yogurt",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def norm(s: str | None) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(s or "").strip().lower().replace("-", "_").replace(" ", "_")).strip("_")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_taxonomy() -> dict[str, dict[str, Any]]:
    rows = {}
    with TAX_CSV.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            lab = norm(r["label"])
            rows[lab] = {
                "label": lab,
                "visual_class": r.get("visual_class"),
                "food_state": r.get("food_state"),
                "kind": r.get("kind"),
                "canonical_id": r.get("canonical_id"),
                "parent_canonical_id": r.get("parent_canonical_id"),
                "is_hard_negative": str(r.get("is_hard_negative")).lower() == "true",
            }
    return rows


def load_prod_vocab() -> dict[str, int]:
    blob = torch.load(PROD_WEIGHTS, map_location="cpu", weights_only=False)
    return {norm(k): int(v) for k, v in (blob.get("label_vocab") or {}).items()}


def resolve_canonical(label: str, tax: dict[str, dict[str, Any]], prod: dict[str, int]) -> dict[str, Any]:
    raw = norm(label)
    mapped = norm(CANONICAL_MAP.get(raw, raw))
    in_tax = mapped in tax or raw in tax
    canon = mapped if mapped in tax else (raw if raw in tax else mapped)
    # doughnut alias
    if not in_tax and mapped == "doughnut" and "doughnut" not in tax:
        # production OI uses doughnut
        in_tax = False
    taxonomy_gap = not (canon in tax or raw in tax)
    # sandwich may only be in prod vocab
    model_vocab = canon in prod or raw in prod or mapped in prod
    if taxonomy_gap and model_vocab:
        # still a KitchenIQ taxonomy gap even if OI head knows it
        pass
    return {
        "raw": raw,
        "canonical": canon,
        "taxonomy_support": not taxonomy_gap,
        "taxonomy_gap": taxonomy_gap,
        "model_vocab_support": model_vocab,
        "tax_meta": tax.get(canon) or tax.get(raw),
    }


def phase1(tax: dict, prod: dict, prod_counts: Counter, hn_counts: Counter, exp_counts: Counter) -> dict[str, Any]:
    targeted = json.loads(TARGETED_ACCEPT.read_text(encoding="utf-8"))
    specialist = json.loads(SPECIALIST_ACCEPT.read_text(encoding="utf-8"))
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    cat_by = {s["scene_id"]: s for s in catalog.get("scenes") or []}
    spec_by = {s["scene_id"]: s for s in specialist.get("scenes") or []}

    rows: list[dict[str, Any]] = []
    group = Counter()

    for scene in targeted.get("scenes") or []:
        sid = scene["scene_id"]
        cat = cat_by.get(sid) or {}
        gt_inst = {norm(i.get("label")): i for i in (cat.get("instances") or [])}
        obs = scene.get("observations") or []
        spec_scene = spec_by.get(sid) or {}

        for fail in scene.get("failures") or []:
            gt = norm(fail.get("gt"))
            stage = str(fail.get("stage") or "UNKNOWN")
            pred = fail.get("pred")
            res = resolve_canonical(gt, tax, prod)
            gt_meta = gt_inst.get(gt) or gt_inst.get(res["canonical"]) or {}
            # detector status
            if stage == "DETECTOR":
                det_status = "MISS"
            elif fail.get("iou") is not None and float(fail.get("iou") or 0) <= 0 and stage == "CLASSIFIER":
                # matched somehow with low iou / class-only
                det_status = "HIT_OR_WHOLE_IMAGE"
            else:
                det_status = "HIT_OR_WHOLE_IMAGE" if stage != "DETECTOR" else "MISS"

            # root cause bucket
            if stage == "DETECTOR":
                root = "D. detector failure"
                code = "detector_failure"
            elif stage == "FOOD_STATE":
                root = "E. food-state failure"
                code = "food_state_failure"
            elif stage == "FOOD_NONFOOD":
                root = "H. genuine ambiguous/unknown case" if gt in {"non_food", "empty_plate", "hand", "human_hand"} else "G. multi-object/evidence-fusion failure"
                code = "food_nonfood_or_fusion"
            elif stage == "PREPARED_MEAL":
                root = "F. prepared-meal failure"
                code = "prepared_meal_failure"
            elif res["taxonomy_gap"]:
                root = "A. taxonomy missing"
                code = "taxonomy_missing"
            elif not res["model_vocab_support"] and prod_counts.get(res["canonical"], 0) == 0:
                root = "B. model vocabulary missing"
                code = "model_vocab_missing"
            elif prod_counts.get(res["canonical"], 0) + prod_counts.get(gt, 0) < 15:
                root = "C. training data insufficient"
                code = "training_data_insufficient"
            elif stage == "CLASSIFIER":
                # classifier stage but may be multi-object fusion
                if (scene.get("n_gt") or 0) >= 2 and (scene.get("n_obs") or 0) < (scene.get("n_gt") or 0):
                    root = "G. multi-object/evidence-fusion failure"
                    code = "multi_object_fusion"
                else:
                    root = "C. training data insufficient" if prod_counts.get(res["canonical"], 0) < 40 else "B. model vocabulary missing"
                    code = "classifier_identity"
            else:
                root = "H. genuine ambiguous/unknown case"
                code = "ambiguous"

            group[root] += 1
            obs0 = obs[0] if obs else {}
            rows.append(
                {
                    "scene_id": sid,
                    "category": scene.get("category"),
                    "expected_identity": gt,
                    "canonical_identity": res["canonical"],
                    "expected_kind": (res.get("tax_meta") or {}).get("kind") or gt_meta.get("visual_class_hint"),
                    "expected_food_state": gt_meta.get("food_state"),
                    "expected_component": gt_meta.get("instance_id"),
                    "detector_status": det_status,
                    "detector_box": gt_meta.get("bbox"),
                    "classifier_output": pred or obs0.get("raw_label") or obs0.get("identity"),
                    "classifier_status": obs0.get("status"),
                    "confidence": obs0.get("confidence"),
                    "failure_stage": stage,
                    "current_taxonomy_support": res["taxonomy_support"],
                    "taxonomy_gap": res["taxonomy_gap"],
                    "current_model_vocab_support": res["model_vocab_support"],
                    "production_training_count": int(prod_counts.get(res["canonical"], 0) + prod_counts.get(gt, 0)),
                    "validation_count": 0,
                    "experiment_only_count": int(exp_counts.get(res["canonical"], 0) + exp_counts.get(gt, 0)),
                    "hardneg_count": int(hn_counts.get(res["canonical"], 0) + hn_counts.get(gt, 0)),
                    "production_eligible": (prod_counts.get(res["canonical"], 0) + prod_counts.get(gt, 0)) > 0,
                    "likely_root_cause": root,
                    "likely_root_cause_code": code,
                    "specialist_stage_ref": [
                        f.get("stage") for f in (spec_scene.get("failures") or []) if norm(f.get("gt")) == gt
                    ],
                }
            )

    matrix = {
        "generated_at": now(),
        "source_acceptance": str(TARGETED_ACCEPT),
        "specialist_ref": str(SPECIALIST_ACCEPT),
        "n_failure_rows": len(rows),
        "group_counts": dict(group),
        "failures": rows,
    }
    write_json(DOCS / "identity_failure_coverage_matrix.json", matrix)

    # markdown
    lines = [
        "# Identity Failure Coverage Matrix",
        "",
        f"Generated: {now()}",
        f"Failure rows: {len(rows)}",
        "",
        "## Groups",
        "",
    ]
    for k, v in sorted(group.items(), key=lambda kv: -kv[1]):
        lines.append(f"- **{k}**: {v}")
    lines += ["", "## Per-failure (compact)", "", "| scene | expected | canonical | stage | taxonomy | model_vocab | prod_n | root |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['scene_id']} | {r['expected_identity']} | {r['canonical_identity']} | {r['failure_stage']} | "
            f"{r['current_taxonomy_support']} | {r['current_model_vocab_support']} | {r['production_training_count']} | {r['likely_root_cause_code']} |"
        )
    (DOCS / "identity_failure_coverage_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return matrix


def phase2(catalog: dict, tax: dict, prod: dict, matrix: dict) -> dict[str, Any]:
    needed: dict[str, dict[str, Any]] = {}
    for sc in catalog.get("scenes") or []:
        for inst in sc.get("instances") or []:
            if not inst.get("required", True):
                continue
            raw = norm(inst.get("label"))
            res = resolve_canonical(raw, tax, prod)
            key = res["canonical"]
            entry = needed.setdefault(
                key,
                {
                    "canonical_identity": key,
                    "raw_aliases": set(),
                    "taxonomy_gap": res["taxonomy_gap"],
                    "taxonomy_support": res["taxonomy_support"],
                    "model_vocab_support": res["model_vocab_support"],
                    "kind": (res.get("tax_meta") or {}).get("kind") or inst.get("visual_class_hint"),
                    "food_state_examples": set(),
                    "scene_ids": set(),
                    "visual_class": (res.get("tax_meta") or {}).get("visual_class"),
                },
            )
            entry["raw_aliases"].add(raw)
            if inst.get("food_state"):
                entry["food_state_examples"].add(inst["food_state"])
            entry["scene_ids"].add(sc["scene_id"])
            entry["taxonomy_gap"] = entry["taxonomy_gap"] and res["taxonomy_gap"]
            entry["model_vocab_support"] = entry["model_vocab_support"] or res["model_vocab_support"]

    # also include identities that failed even if somehow not in GT parse
    for f in matrix.get("failures") or []:
        key = f["canonical_identity"]
        entry = needed.setdefault(
            key,
            {
                "canonical_identity": key,
                "raw_aliases": set(),
                "taxonomy_gap": f.get("taxonomy_gap", True),
                "taxonomy_support": f.get("current_taxonomy_support", False),
                "model_vocab_support": f.get("current_model_vocab_support", False),
                "kind": f.get("expected_kind"),
                "food_state_examples": set(),
                "scene_ids": set(),
                "visual_class": None,
            },
        )
        entry["raw_aliases"].add(f["expected_identity"])
        entry["scene_ids"].add(f["scene_id"])

    # serialize sets
    identities = []
    for key, e in sorted(needed.items()):
        identities.append(
            {
                "canonical_identity": key,
                "raw_aliases": sorted(e["raw_aliases"]),
                "taxonomy_gap": bool(e["taxonomy_gap"]),
                "taxonomy_support": bool(e["taxonomy_support"]),
                "model_vocab_support": bool(e["model_vocab_support"]),
                "kind": e.get("kind"),
                "visual_class": e.get("visual_class"),
                "food_state_examples": sorted(e["food_state_examples"]),
                "scene_ids": sorted(e["scene_ids"]),
                "trainable_production_candidate": (not e["taxonomy_gap"]) or bool(e["model_vocab_support"]),
            }
        )

    vocab = {
        "version": "required_identity_vocabulary_v1",
        "generated_at": now(),
        "n_identities": len(identities),
        "n_taxonomy_gaps": sum(1 for i in identities if i["taxonomy_gap"]),
        "n_model_vocab_supported": sum(1 for i in identities if i["model_vocab_support"]),
        "production_classifier_vocab_size": len(prod),
        "taxonomy_size": len(tax),
        "identities": identities,
        "notes": "No invented labels. taxonomy_gap=true means no suitable 207-tax mapping; model_vocab_support reflects OI-62 production head.",
    }
    write_json(DOCS / "required_identity_vocabulary_v1.json", vocab)
    return vocab


def phase3(vocab: dict, prod_rows: list, hn_rows: list, exp_rows: list) -> dict[str, Any]:
    def counts(rows: list[dict], hardneg: bool | None = None) -> Counter:
        c: Counter = Counter()
        for r in rows:
            if hardneg is True and not r.get("is_hard_negative"):
                continue
            if hardneg is False and r.get("is_hard_negative"):
                continue
            lab = norm(r.get("label") or r.get("target_class"))
            c[lab] += 1
        return c

    pc = counts(prod_rows, False)
    hc = counts(hn_rows, True) if hn_rows else counts([r for r in prod_rows if r.get("is_hard_negative")], True)
    ec = counts(exp_rows)

    # provenance aggregation
    prov: dict[str, list] = defaultdict(list)
    for r in prod_rows:
        lab = norm(r.get("label") or r.get("target_class"))
        prov[lab].append(
            {
                "licence": r.get("licence"),
                "source": r.get("source"),
                "zone": r.get("data_zone"),
                "provenance_hash": r.get("provenance_hash"),
            }
        )

    items = []
    for ident in vocab["identities"]:
        key = ident["canonical_identity"]
        aliases = set(ident["raw_aliases"]) | {key}
        n_prod = sum(pc.get(a, 0) for a in aliases)
        n_hn = sum(hc.get(a, 0) for a in aliases)
        n_exp = sum(ec.get(a, 0) for a in aliases)
        # sufficiency: based on failure criticality not blanket 250
        # critical if appears in >=2 scenes or is prepared meal in failures
        n_scenes = len(ident.get("scene_ids") or [])
        if ident["taxonomy_gap"] and not ident["model_vocab_support"]:
            status = "TAXONOMY_GAP"
        elif n_prod == 0 and n_hn == 0:
            status = "MISSING"
        elif any((prov.get(a) or [{}])[0].get("licence") in {None, "unknown", "unclear"} for a in aliases if a in prov):
            status = "INVALID_PROVENANCE" if n_prod == 0 else "LOW_SUPPORT"
        else:
            # need more support if many scenes or prepared
            need = 25 if n_scenes >= 2 else 12
            if ident.get("kind") == "meal" or key in {"cooked_rice", "pizza", "fried_egg", "biryani"}:
                need = 30
            if key in {"non_food_object", "empty_plate", "human_hand"}:
                need = 20
                n_prod_eff = n_hn  # hardneg is the training signal
                status = "SUFFICIENT" if n_prod_eff >= need or n_hn >= need else ("LOW_SUPPORT" if n_hn > 0 else "MISSING")
            else:
                status = "SUFFICIENT" if n_prod >= need else ("LOW_SUPPORT" if n_prod > 0 else "MISSING")

        items.append(
            {
                "canonical_identity": key,
                "production_image_count": n_prod,
                "validation_image_count": 0,  # split later on Kaggle
                "test_image_count": 0,
                "hard_negative_count": n_hn,
                "experiment_only_count": n_exp,
                "status": status,
                "taxonomy_gap": ident["taxonomy_gap"],
                "model_vocab_support": ident["model_vocab_support"],
                "n_scenes": n_scenes,
                "sources": sorted({p.get("source") for a in aliases for p in prov.get(a, []) if p.get("source")}),
                "licences": sorted({p.get("licence") for a in aliases for p in prov.get(a, []) if p.get("licence")}),
            }
        )

    coverage = {
        "version": "production_identity_coverage_v1",
        "generated_at": now(),
        "status_counts": dict(Counter(i["status"] for i in items)),
        "identities": items,
    }
    write_json(DOCS / "production_identity_coverage_v1.json", coverage)
    return coverage


def phase4_5(vocab: dict, coverage: dict, matrix: dict) -> dict[str, Any]:
    """Build acquisition plan + training manifest (no Windows bulk download)."""
    cov_by = {i["canonical_identity"]: i for i in coverage["identities"]}
    fail_labels = Counter(f["canonical_identity"] for f in matrix["failures"] if f["failure_stage"] == "CLASSIFIER")

    acquisition = []
    for ident in vocab["identities"]:
        key = ident["canonical_identity"]
        cov = cov_by.get(key) or {}
        st = cov.get("status")
        if st in {"SUFFICIENT"}:
            continue
        if st == "TAXONOMY_GAP" and not ident.get("model_vocab_support"):
            acquisition.append(
                {
                    "canonical_identity": key,
                    "action": "TAXONOMY_RESOLUTION_REQUIRED",
                    "additional_images_needed": 0,
                    "priority": "P0" if fail_labels.get(key, 0) >= 2 else "P2",
                    "reason": "No 207-taxonomy mapping; cannot invent production class",
                    "acceptance_failures_addressed": fail_labels.get(key, 0),
                    "acceptable_sources": [],
                }
            )
            continue
        have = int(cov.get("production_image_count") or 0)
        if key in {"non_food_object", "empty_plate", "human_hand"}:
            have = int(cov.get("hard_negative_count") or 0)
            target = 40
        elif ident.get("kind") == "meal" or key in {"cooked_rice", "pizza", "fried_egg", "biryani"}:
            target = 60
        else:
            target = 40 if fail_labels.get(key, 0) >= 2 else 25
        need = max(0, target - have)
        if need <= 0 and st != "MISSING":
            continue
        acquisition.append(
            {
                "canonical_identity": key,
                "action": "ACQUIRE_PRODUCTION_ELIGIBLE",
                "additional_images_needed": need if need > 0 else (25 if st == "MISSING" else 0),
                "current_count": have,
                "target_count": target,
                "priority": "P0" if fail_labels.get(key, 0) >= 2 else ("P1" if fail_labels.get(key, 0) == 1 else "P2"),
                "reason": f"status={st}; classifier_failures={fail_labels.get(key, 0)}; scenes={len(ident.get('scene_ids') or [])}",
                "acceptance_failures_addressed": fail_labels.get(key, 0),
                "acceptable_sources": ["openverse_cc_by_or_cc0", "open_images_cc_by", "wikimedia_cc_by_or_cc0", "government_public_domain"],
                "required_diversity": ["lighting", "angle", "background", "scale"],
                "hard_negatives_required": key in {"non_food_object", "empty_plate", "human_hand"},
                "provenance_requirements": ["verified_licence", "attribution_where_required", "no_experiment_only"],
            }
        )

    write_json(
        DOCS / "production_identity_acquisition_plan_v1.json",
        {"generated_at": now(), "n_items": len(acquisition), "items": acquisition},
    )

    # Training manifest: all identities that are trainable and have OR will acquire data
    train_labels = []
    for ident in vocab["identities"]:
        key = ident["canonical_identity"]
        cov = cov_by.get(key) or {}
        if ident["taxonomy_gap"] and not ident["model_vocab_support"]:
            train_labels.append(
                {
                    "canonical_identity": key,
                    "include_in_training": False,
                    "reason": "TAXONOMY_GAP",
                    "existing_production_count": cov.get("production_image_count", 0),
                }
            )
            continue
        train_labels.append(
            {
                "canonical_identity": key,
                "include_in_training": True,
                "reason": "REQUIRED_IDENTITY",
                "existing_production_count": cov.get("production_image_count", 0)
                if key not in {"non_food_object", "empty_plate", "human_hand"}
                else cov.get("hard_negative_count", 0),
                "aliases": ident.get("raw_aliases"),
                "kind": ident.get("kind"),
                "visual_class": ident.get("visual_class"),
            }
        )

    manifest = {
        "version": "production_identity_training_manifest_v1",
        "generated_at": now(),
        "rules": {
            "experiment_only_mixed": False,
            "synthetic_allowed": False,
            "acceptance_scene_leakage_allowed": False,
            "holdout_leakage_allowed": False,
            "twelve_label_targeted_vocab_forbidden": True,
        },
        "n_trainable": sum(1 for t in train_labels if t["include_in_training"]),
        "n_excluded_taxonomy_gap": sum(1 for t in train_labels if not t["include_in_training"]),
        "labels": train_labels,
        "source_manifests": {
            "production_eligible": str(PROD_MAN),
            "hard_negative": str(HN_MAN),
            "experiment_only_isolated": str(EXP_MAN),
        },
        "acquisition_plan": str(DOCS / "production_identity_acquisition_plan_v1.json"),
    }
    write_json(DOCS / "production_identity_training_manifest_v1.json", manifest)
    return {"acquisition": acquisition, "manifest": manifest}


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    EXP.mkdir(parents=True, exist_ok=True)
    tax = load_taxonomy()
    prod = load_prod_vocab()
    prod_rows = load_jsonl(PROD_MAN)
    hn_rows = load_jsonl(HN_MAN)
    exp_rows = load_jsonl(EXP_MAN)
    prod_counts = Counter(norm(r.get("label") or r.get("target_class")) for r in prod_rows if not r.get("is_hard_negative"))
    hn_counts = Counter(norm(r.get("label") or r.get("target_class")) for r in hn_rows)
    # also hardneg labels inside prod file if any
    for r in prod_rows:
        if r.get("is_hard_negative"):
            hn_counts[norm(r.get("label") or r.get("target_class"))] += 1
    exp_counts = Counter(norm(r.get("label") or r.get("target_class")) for r in exp_rows)

    print("PHASE1", flush=True)
    matrix = phase1(tax, prod, prod_counts, hn_counts, exp_counts)
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    print("PHASE2", flush=True)
    vocab = phase2(catalog, tax, prod, matrix)
    print("PHASE3", flush=True)
    coverage = phase3(vocab, prod_rows, hn_rows, exp_rows)
    print("PHASE4_5", flush=True)
    plans = phase4_5(vocab, coverage, matrix)

    summary = {
        "phases_complete": ["1", "2", "3", "4_plan", "5_manifest"],
        "n_failure_rows": matrix["n_failure_rows"],
        "group_counts": matrix["group_counts"],
        "n_required_identities": vocab["n_identities"],
        "n_taxonomy_gaps": vocab["n_taxonomy_gaps"],
        "coverage_status_counts": coverage["status_counts"],
        "n_acquisition_items": len(plans["acquisition"]),
        "n_trainable_labels": plans["manifest"]["n_trainable"],
        "generated_at": now(),
    }
    write_json(EXP / "reports" / "phases_1_5_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
