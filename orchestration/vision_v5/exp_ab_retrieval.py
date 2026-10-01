"""V5 Experiments A + B (no training).

A  = frozen DINOv2 + current V4 retrieval (flat top-k cosine, single nearest label).
     A_v4query additionally reproduces V4's query path (backbone from the specialist
     checkpoint) to test embedding-space consistency with the library.
B  = frozen DINOv2 + hierarchical retrieval (prototype + kNN vote, identity -> family
     -> unknown back-off, calibrated thresholds).

All thresholds are fitted on the reference library only (leave-one-out for knowns,
leave-one-label-out for unknowns). The frozen 41-scene benchmark is read-only and
used only for scoring on ground-truth regions.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(r"C:\Projects\KitchenIQ-V10-AI")
sys.path.insert(0, str(ROOT / "src"))

EXP_ID = "v5_exp_AB_retrieval_001"
OUT = ROOT / "backend/instance/dev_experiments/v10-vision-v5" / EXP_ID
LIB = ROOT / "backend/instance/dev_experiments/v10-vision-v4/reference_library/reference_library.json"
CATALOG = ROOT / "backend/instance/dev_experiments/v10-specialist-ensemble/acceptance/fixtures/catalog/fixture_catalog.json"
FIXTURES = CATALOG.parent.parent
CKPT = ROOT / "backend/instance/dev_experiments/v10-vision-recovery/models/identity_coverage_best.pt"
CATALOG_SHA = "38fc3eaee75c76fa5fd1dc3d4177af4abde2cfeb3e410d290fb524f9e30a1523"


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def preprocess(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB").resize((224, 224), Image.BILINEAR)).astype("float32") / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], "float32")) / np.array([0.229, 0.224, 0.225], "float32")
    return torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0)


@torch.no_grad()
def embed(model, img: Image.Image) -> np.ndarray:
    feats = model.forward_features(preprocess(img))
    return F.normalize(feats["x_norm_clstoken"].float(), dim=-1)[0].numpy()


def resolve_image(scene: dict) -> Path:
    p = Path((scene.get("image") or {}).get("path") or "")
    if p.is_file():
        return p
    found = sorted((FIXTURES / "generated").glob(f"{scene.get('scene_id')}*"))
    return found[0] if found else p


# ---------------------------------------------------------------- retrieval
class Index:
    def __init__(self, emb: np.ndarray, labels: list[str], families: dict[str, str]):
        self.emb, self.labels, self.families = emb, np.array(labels), families
        self.label_set = sorted(set(labels))
        self.protos = {l: _norm(emb[self.labels == l].mean(0)) for l in self.label_set}

    def flat(self, q: np.ndarray, exclude: np.ndarray | None = None) -> tuple[str, float]:
        sims = self.emb @ q
        if exclude is not None:
            sims = np.where(exclude, -9, sims)
        i = int(sims.argmax())
        return str(self.labels[i]), float(sims[i])

    def hier_scores(self, q: np.ndarray, exclude: np.ndarray | None = None, k: int = 3) -> dict[str, float]:
        sims = self.emb @ q
        if exclude is not None:
            sims = np.where(exclude, -9, sims)
        out = {}
        for l in self.label_set:
            m = (self.labels == l) & (sims > -9)
            if not m.any():
                continue
            s = np.sort(sims[m])[::-1][:k]
            mask = m if exclude is None else (self.labels == l) & ~exclude
            proto = _norm(self.emb[mask].mean(0))
            out[l] = 0.5 * float(proto @ q) + 0.5 * float(s.mean())
        return out


def _norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)


def hier_decide(scores: dict[str, float], families: dict[str, str], t_id: float, m_id: float, t_fam: float) -> dict:
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    top, s1 = ranked[0]
    s2 = ranked[1][1] if len(ranked) > 1 else -1
    if s1 >= t_id and (s1 - s2) >= m_id:
        return {"level": "identity", "label": top, "family": families.get(top), "score": s1, "margin": s1 - s2}
    fam_scores: dict[str, float] = defaultdict(lambda: -9.0)
    for l, s in ranked:
        f = families.get(l) or "unknown"
        fam_scores[f] = max(fam_scores[f], s)
    fam, fs = max(fam_scores.items(), key=lambda x: x[1])
    if fam != "unknown" and fs >= t_fam:
        return {"level": "family", "label": None, "family": fam, "score": fs, "margin": s1 - s2}
    return {"level": "unknown", "label": None, "family": None, "score": s1, "margin": s1 - s2}


# ---------------------------------------------------------------- calibration (library only)
def calibrate(index: Index) -> dict:
    n = len(index.labels)
    known_rows, unknown_rows = [], []
    for i in range(n):
        ex = np.zeros(n, bool)
        ex[i] = True
        known_rows.append((index.labels[i], index.hier_scores(index.emb[i], ex), index.flat(index.emb[i], ex)))
    for l in index.label_set:
        ex = index.labels == l
        for i in np.where(ex)[0]:
            unknown_rows.append((l, index.hier_scores(index.emb[i], ex), index.flat(index.emb[i], ex)))

    best = None
    for t_id in np.arange(0.30, 0.96, 0.02):
        for m_id in (0.0, 0.01, 0.02, 0.03, 0.05, 0.08):
            for t_fam in (t_id - 0.02, t_id - 0.05, t_id - 0.1, 9.0):
                k_ok = sum(
                    1 for g, s, _ in known_rows
                    if (d := hier_decide(s, index.families, t_id, m_id, t_fam))["level"] == "identity" and d["label"] == g
                    or d["level"] == "family" and d["family"] == index.families.get(g) and d["family"] != "unknown"
                )
                k_false = sum(
                    1 for g, s, _ in known_rows
                    if (d := hier_decide(s, index.families, t_id, m_id, t_fam))["level"] == "identity" and d["label"] != g
                )
                u_rej = sum(
                    1 for g, s, _ in unknown_rows
                    if hier_decide(s, index.families, t_id, m_id, t_fam)["level"] != "identity"
                )
                obj = k_ok / len(known_rows) + u_rej / len(unknown_rows) - 2.0 * k_false / len(known_rows)
                if best is None or obj > best[0]:
                    best = (obj, float(t_id), float(m_id), float(t_fam), k_ok, k_false, u_rej)
    obj, t_id, m_id, t_fam, k_ok, k_false, u_rej = best

    # flat (A) open-set threshold on the same library protocol
    flat_best = None
    for t in np.arange(0.30, 0.96, 0.01):
        ok = sum(1 for g, _, (p, s) in known_rows if s >= t and p == g)
        false = sum(1 for g, _, (p, s) in known_rows if s >= t and p != g)
        rej = sum(1 for _, _, (_, s) in unknown_rows if s < t)
        o = ok / len(known_rows) + rej / len(unknown_rows) - 2.0 * false / len(known_rows)
        if flat_best is None or o > flat_best[0]:
            flat_best = (o, float(t))

    loo_top1 = sum(1 for g, _, (p, _) in known_rows if p == g) / len(known_rows)
    loo_fam = sum(
        1 for g, _, (p, _) in known_rows
        if index.families.get(p) == index.families.get(g) and index.families.get(g) != "unknown"
    ) / max(1, sum(1 for g, *_ in known_rows if index.families.get(g) != "unknown"))
    pos = [s for g, _, (p, s) in known_rows if p == g]
    unk = [s for _, _, (_, s) in unknown_rows]
    return {
        "B": {"t_id": t_id, "m_id": m_id, "t_fam": t_fam, "objective": round(obj, 4),
              "known_correct_rate": round(k_ok / len(known_rows), 4),
              "known_false_identity_rate": round(k_false / len(known_rows), 4),
              "unknown_rejection_rate": round(u_rej / len(unknown_rows), 4)},
        "A_flat_threshold": flat_best[1],
        "library_loo_identity_top1": round(loo_top1, 4),
        "library_loo_family_top1": round(loo_fam, 4),
        "sim_known_correct_mean": round(float(np.mean(pos)), 4) if pos else None,
        "sim_unknown_top1_mean": round(float(np.mean(unk)), 4),
        "n_known_queries": len(known_rows),
        "n_unknown_queries": len(unknown_rows),
    }


# ---------------------------------------------------------------- main
def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    assert sha_file(CATALOG) == CATALOG_SHA, "frozen benchmark catalog hash changed — STOP"

    lib = json.loads(LIB.read_text(encoding="utf-8"))
    items = lib["items"]
    assert len(items) > 0
    emb = np.stack([_norm(np.asarray(i["embedding"], np.float32)) for i in items])
    labels = [i["canonical_identity"] for i in items]
    families = {i["canonical_identity"]: (i.get("family") or "unknown") for i in items}
    index = Index(emb, labels, families)

    cal = calibrate(index)
    print("calibration", json.dumps(cal), flush=True)

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).eval()

    # embedding-space consistency check: specialist checkpoint backbone vs pretrained
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = blob.get("state_dict") or blob
    bb = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    ref = model.state_dict()
    diffs = [float((bb[k].float() - ref[k].float()).abs().max()) for k in bb if k in ref and bb[k].shape == ref[k].shape]
    consistency = {
        "ckpt_backbone_tensors": len(bb),
        "matched_tensors": len(diffs),
        "max_abs_weight_diff": max(diffs) if diffs else None,
        "backbone_differs_from_pretrained": bool(diffs and max(diffs) > 1e-6),
    }
    v4_model = None
    if consistency["backbone_differs_from_pretrained"]:
        v4_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).eval()
        v4_model.load_state_dict(bb, strict=False)
    print("consistency", json.dumps(consistency), flush=True)

    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    lib_labels = set(index.label_set)
    rows = []
    for sc in catalog["scenes"]:
        img = Image.open(resolve_image(sc)).convert("RGB")
        W, H = img.size
        for inst in sc.get("instances") or []:
            if not inst.get("required", True):
                continue
            b = inst.get("bbox")
            crop = img if not b else img.crop((max(0, int(b[0])), max(0, int(b[1])), min(W, int(b[2])), min(H, int(b[3]))))
            q = embed(model, crop)
            gt = str(inst.get("label"))
            covered = gt in lib_labels
            a_lab, a_sim = index.flat(q)
            a_accept = a_sim >= cal["A_flat_threshold"]
            hs = index.hier_scores(q)
            ranked = sorted(hs.items(), key=lambda x: -x[1])
            bd = hier_decide(hs, families, cal["B"]["t_id"], cal["B"]["m_id"], cal["B"]["t_fam"])
            row = {
                "scene_id": sc["scene_id"], "category": sc.get("category"), "gt": gt,
                "gt_state": inst.get("food_state"), "covered_by_library": covered,
                "gt_family": families.get(gt),
                "A_top1": a_lab, "A_sim": round(a_sim, 4), "A_accept": bool(a_accept),
                "B_decision": bd["level"], "B_label": bd["label"], "B_family": bd["family"],
                "B_score": round(bd["score"], 4), "B_margin": round(bd["margin"], 4),
                "B_top5": [l for l, _ in ranked[:5]],
            }
            if v4_model is not None:
                qv = embed(v4_model, crop)
                v_lab, v_sim = index.flat(qv)
                row.update({"A_v4query_top1": v_lab, "A_v4query_sim": round(v_sim, 4)})
            rows.append(row)
    print(f"scored {len(rows)} required GT regions", flush=True)

    def metrics(prefix: str) -> dict:
        cov = [r for r in rows if r["covered_by_library"]]
        unc = [r for r in rows if not r["covered_by_library"]]
        if prefix == "A":
            acc = lambda r: r["A_accept"] and r["A_top1"] == r["gt"]
            confirmed = lambda r: r["A_accept"]
            wrong = lambda r: r["A_accept"] and r["A_top1"] != r["gt"]
            top1 = lambda r: r["A_top1"] == r["gt"]
            fam_ok = lambda r: families.get(r["A_top1"]) == r["gt_family"] and r["gt_family"] != "unknown"
        elif prefix == "A_v4query":
            if v4_model is None:
                return {"skipped": "backbone identical to pretrained"}
            acc = lambda r: r["A_v4query_top1"] == r["gt"]
            confirmed = lambda r: True
            wrong = lambda r: r["A_v4query_top1"] != r["gt"]
            top1 = acc
            fam_ok = lambda r: families.get(r["A_v4query_top1"]) == r["gt_family"] and r["gt_family"] != "unknown"
        else:
            acc = lambda r: r["B_decision"] == "identity" and r["B_label"] == r["gt"]
            confirmed = lambda r: r["B_decision"] == "identity"
            wrong = lambda r: r["B_decision"] == "identity" and r["B_label"] != r["gt"]
            top1 = lambda r: r["B_top5"][0] == r["gt"]
            fam_ok = lambda r: (r["B_family"] or families.get(r["B_top5"][0])) == r["gt_family"] and r["gt_family"] != "unknown"
        fam_cov = [r for r in cov if r["gt_family"] != "unknown"]
        out = {
            "covered_identity_top1_raw": round(sum(map(top1, cov)) / max(1, len(cov)), 4),
            "covered_identity_accepted_correct": round(sum(map(acc, cov)) / max(1, len(cov)), 4),
            "covered_family_top1": round(sum(map(fam_ok, fam_cov)) / max(1, len(fam_cov)), 4),
            "uncovered_unknown_rejection": round(sum(1 for r in unc if not confirmed(r)) / max(1, len(unc)), 4),
            "false_confirmation_rate_all": round(sum(map(wrong, rows)) / max(1, len(rows)), 4),
        }
        if prefix == "B":
            out["covered_top5"] = round(sum(1 for r in cov if r["gt"] in r["B_top5"]) / max(1, len(cov)), 4)
            out["covered_family_level_backoff"] = sum(1 for r in cov if r["B_decision"] == "family")
            out["covered_identity_ok_or_correct_family_backoff"] = round(
                sum(1 for r in cov if acc(r) or (r["B_decision"] == "family" and r["B_family"] == r["gt_family"]))
                / max(1, len(cov)), 4)
        # oracle-region strict completeness (diagnostic only; not the official detector metric)
        by_scene = defaultdict(list)
        for r in rows:
            by_scene[r["scene_id"]].append(acc(r) or (r["gt"] == "non_food" and not confirmed(r)))
        out["oracle_region_scene_complete"] = round(sum(all(v) for v in by_scene.values()) / 41, 4)
        return out

    scene_labels = defaultdict(set)
    for r in rows:
        scene_labels[r["scene_id"]].add(r["gt"])
    ceiling = sum(1 for labs in scene_labels.values() if labs <= (lib_labels | {"non_food"})) / 41

    results = {
        "experiment_id": EXP_ID,
        "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        "benchmark_catalog_sha256": CATALOG_SHA,
        "reference_library_sha256": sha_file(LIB),
        "reference_library_n": len(items),
        "reference_library_labels": index.label_set,
        "hardware": "CPU (6 logical cores, 7.9 GB RAM, no CUDA)",
        "training": "none",
        "benchmark_usage": "scoring only on GT regions; no fitting, selection or threshold tuning",
        "calibration_library_only": cal,
        "embedding_space_consistency": consistency,
        "n_required_regions": len(rows),
        "n_covered_regions": sum(r["covered_by_library"] for r in rows),
        "library_coverage_scene_ceiling": round(ceiling, 4),
        "A": metrics("A"),
        "A_v4query": metrics("A_v4query"),
        "B": metrics("B"),
        "uncovered_gt_labels": dict(Counter(r["gt"] for r in rows if not r["covered_by_library"]).most_common()),
        "covered_confusions_B": dict(Counter(f"{r['gt']}->{r['B_top5'][0]}" for r in rows
                                             if r["covered_by_library"] and r["B_top5"][0] != r["gt"]).most_common()),
        "duration_s": round(time.time() - t0, 1),
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (OUT / "per_region_failures.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({k: results[k] for k in ["embedding_space_consistency", "library_coverage_scene_ceiling",
                                               "n_covered_regions", "A", "A_v4query", "B", "covered_confusions_B"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
