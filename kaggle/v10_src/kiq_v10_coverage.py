"""KitchenIQ V10-VR Milestone 1: 207-label reference coverage audit of the SEARCHABLE corpus.

Searchable corpus today = reference_library_v1 (524 refs, 27 labels, frozen). A taxonomy label counts only through
references that are actually indexed for retrieval. Mapping library label -> taxonomy label is explicit and
state-qualified; nothing is matched by fuzzy text. Fields the corpus does not annotate are null, not 0.

usage: python kaggle/v10_src/kiq_v10_coverage.py
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Projects\KitchenIQ-V10-AI")
TAXONOMY = Path(r"C:\Projects\KitchenIQ-OS\backend\app\services\food_vision\dataset\v1_label_taxonomy.csv")
TAXONOMY_SHA = "cfc98a0ea122a6588698232e7da9cb63060636ef1cb41908c0a27a9308e59756"
LIBDIR = ROOT / "docs/v5/reference_library_v1"
LIB_SHA = "7e95ee5636af0a14b2839b1a242699bc02635b5d0956c6d6e89a5b9019fc0a02"
ART = ROOT / "artifacts/v10"
spec = importlib.util.spec_from_file_location("kiq_v10_regions", ROOT / "kaggle/v10_src/kiq_v10_regions.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
COV = R.PROTOCOL["coverage"]

# library label (state in frozen config) -> taxonomy label. Only where label AND state agree.
LIB_TO_TAXONOMY = {
    "cheese": ("cheese", "EXACT"), "cream": ("cream", "EXACT"), "garlic": ("garlic", "EXACT"),
    "ginger": ("ginger", "EXACT"), "tomato": ("tomato", "EXACT"), "yogurt": ("yogurt", "EXACT"),
    "biryani": ("biryani", "EXACT"), "pizza": ("pizza", "EXACT"),
    "packaged_cheese": ("packaged cheese", "EXACT"), "tomato_paste": ("tomato paste", "EXACT"),
    "rice": ("cooked rice", "STATE_QUALIFIED (library rice is state=cooked; taxonomy 'rice' is raw)"),
    "pasta": ("prepared pasta", "STATE_QUALIFIED (library pasta is state=plated; taxonomy 'pasta' is raw)"),
    "non_food": ("non food object", "GENERIC (library non_food is a pooled non-food class)"),
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def clusters(E: np.ndarray, thr: float) -> int:
    """Greedy leader clustering: a reference joins the first cluster whose leader has cos >= thr."""
    leaders: list[int] = []
    for i in range(len(E)):
        if not any(float(E[i] @ E[j]) >= thr for j in leaders):
            leaders.append(i)
    return len(leaders)


def main() -> None:
    status = "FOUND" if TAXONOMY.is_file() and sha(TAXONOMY) == TAXONOMY_SHA else ("MISMATCH" if TAXONOMY.is_file() else "MISSING")
    if status != "FOUND":
        raise SystemExit(f"TARGET_LABEL_MANIFEST_STATUS = {status}: {TAXONOMY}")
    rows = list(csv.DictReader(TAXONOMY.open(encoding="utf-8")))
    assert len(rows) == 207 and len({r["label"] for r in rows}) == 207
    assert sha(LIBDIR / "reference_library_v1_manifest.json") == LIB_SHA
    lib = json.loads((LIBDIR / "reference_library_v1_manifest.json").read_bytes())
    emb = json.loads((LIBDIR / "reference_library_v1_dinov2_embeddings.json").read_bytes())["embeddings"]
    refs = defaultdict(list)
    for r in lib["items"]:
        refs[r["label"]].append(r)
    tax_names = {r["label"] for r in rows}
    for lab, (t, _) in LIB_TO_TAXONOMY.items():
        assert t in tax_names and lab in refs, (lab, t)
    outside = sorted(set(refs) - set(LIB_TO_TAXONOMY))
    by_tax = {t: lab for lab, (t, _) in LIB_TO_TAXONOMY.items()}

    hn = defaultdict(set)
    for r in rows:
        if r["hard_negative_pair_with"]:
            hn[r["label"]].add(r["hard_negative_pair_with"])
            hn[r["hard_negative_pair_with"]].add(r["label"])

    out = []
    for r in rows:
        t = r["label"]
        lab = by_tax.get(t)
        rs = refs.get(lab, []) if lab else []
        E = np.stack([np.asarray(emb[x["ref_id"]], np.float32) for x in rs]) if rs else np.zeros((0, 768), np.float32)
        S = E @ E.T if len(E) else np.zeros((0, 0))
        near = int(((S >= COV["near_dup_cos"]).sum() - len(E)) // 2) if len(E) else 0
        pair_mean = round(float(S[np.triu_indices(len(E), 1)].mean()), 4) if len(E) > 1 else None
        if len(E) > 1:
            ev = np.linalg.svd(E - E.mean(0), compute_uv=False) ** 2
            p = ev / ev.sum()
            eff_rank = round(float(np.exp(-(p * np.log(p + 1e-12)).sum())), 2)
        else:
            eff_rank = None
        states = Counter(x["state"] for x in rs)
        n, src = len(rs), len({x.get("creator") or x["group"] for x in rs})
        div = clusters(E, COV["diversity_cluster_cos"]) if n else 0
        partners = sorted(hn.get(t, set()))
        if n == 0:
            st = "MISSING"
        elif n >= COV["R_min"] and src >= COV["S_min"] and div >= COV["D_min"]:
            st = "COVERED"
        else:
            st = "PARTIAL"
        out.append({
            "label": t, "family": r["parent_canonical_id"] or r["canonical_id"] or r["visual_class"],
            "visual_class": r["visual_class"], "kind": r["kind"], "food_state": r["food_state"],
            "library_label": lab, "mapping": LIB_TO_TAXONOMY[lab][1] if lab else None,
            "reference_count": n, "unique_source_count": src, "visual_diversity_count": div,
            "near_duplicate_pairs_cos085": near, "mean_pairwise_cos": pair_mean, "effective_rank": eff_rank,
            "household_context_count": None, "packaged_product_count": states.get("packaged", 0) if n else 0,
            "raw_count": states.get("raw", 0), "cooked_count": states.get("cooked", 0) + states.get("plated", 0),
            "opened_count": None, "leftover_count": None,
            "barcode_available": False, "ocr_available": False,
            "hard_negative_count": len(partners), "hard_negative_partners": partners,
            "hard_negative_partners_searchable": [p for p in partners if p in by_tax],
            "coverage_status": st,
            "limits": ([] if n == 0 else
                       [m for m, bad in (("reference_count<R_min", n < COV["R_min"]), ("unique_sources<S_min", src < COV["S_min"]),
                                         ("visual_diversity<D_min", div < COV["D_min"])) if bad]),
        })
    cnt = Counter(o["coverage_status"] for o in out)
    by_kind = defaultdict(Counter)
    for o in out:
        by_kind[o["visual_class"]][o["coverage_status"]] += 1
    canon = defaultdict(set)
    for o in out:
        if o["coverage_status"] != "MISSING":
            canon[o["family"]].add(o["label"])
    summary = {
        "TARGET_LABEL_MANIFEST_STATUS": status, "taxonomy_file": str(TAXONOMY), "taxonomy_sha256": TAXONOMY_SHA,
        "searchable_corpus": "reference_library_v1", "library_manifest_sha256": LIB_SHA,
        "library_labels": len(refs), "library_refs": len(lib["items"]),
        "library_labels_mapped_to_taxonomy": len(LIB_TO_TAXONOMY), "library_labels_outside_taxonomy": outside,
        "thresholds": COV, "protocol_sha256": R.protocol_sha(),
        "labels_covered": cnt["COVERED"], "labels_partial": cnt["PARTIAL"], "labels_missing": cnt["MISSING"],
        "coverage_percentage_covered": round(100 * cnt["COVERED"] / 207, 2),
        "coverage_percentage_any_reference": round(100 * (cnt["COVERED"] + cnt["PARTIAL"]) / 207, 2),
        "by_visual_class": {k: dict(v) for k, v in by_kind.items()},
        "canonical_entities_with_any_reference": len(canon),
        "canonical_entities_total": len({o["family"] for o in out}),
        "not_annotated_in_corpus": ["household_context_count", "opened_count", "leftover_count"],
        "notes": ["Counts are of indexed references only; candidate source pools (Kaggle staging, OS datasets) are not searchable and are not counted.",
                  "visual_diversity_count = greedy leader clusters at DINOv2 cos >= 0.70 (library already dedups at 0.90); mean_pairwise_cos and effective_rank are threshold-free diversity measures.",
                  "hard_negative_count = taxonomy hard-negative partners (both directions); searchable partners listed separately."],
    }
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "v10_207_label_coverage.json").write_text(json.dumps({"summary": summary, "labels": out}, indent=1), encoding="utf-8")
    cols = ["label", "family", "visual_class", "kind", "reference_count", "unique_source_count", "visual_diversity_count",
            "near_duplicate_pairs_cos085", "mean_pairwise_cos", "effective_rank", "household_context_count", "packaged_product_count", "raw_count", "cooked_count",
            "opened_count", "leftover_count", "barcode_available", "ocr_available", "hard_negative_count", "coverage_status",
            "library_label", "mapping"]
    with (ART / "v10_207_label_coverage.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for o in out:
            w.writerow({k: ("NOT_ANNOTATED" if o[k] is None and k in summary["not_annotated_in_corpus"] else o[k]) for k in cols})
    print(json.dumps(summary, indent=1))
    for o in out:
        if o["coverage_status"] != "MISSING":
            print(o["label"], o["coverage_status"], o["reference_count"], o["unique_source_count"], o["visual_diversity_count"], o["limits"])


if __name__ == "__main__":
    main()
