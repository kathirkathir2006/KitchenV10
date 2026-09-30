"""Hierarchical visual reference library + retrieval (not flat label softmax)."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from kitcheniq_v10.v3.backbone import DenseEncoder

FAMILY_OF = {
    "tomato": "produce",
    "onion": "produce",
    "potato": "produce",
    "carrot": "produce",
    "garlic": "produce",
    "ginger": "produce",
    "bell_pepper": "produce",
    "cucumber": "produce",
    "lettuce": "produce",
    "apple": "produce",
    "banana": "produce",
    "cheese": "dairy",
    "packaged_cheese": "packaged_dairy",
    "egg": "protein",
    "fried_egg": "prepared_egg",
    "cooked_rice": "rice_grain",
    "fried_rice": "rice_grain",
    "biryani": "rice_grain",
    "pasta": "grain_starch",
    "bread": "grain_starch",
    "pizza": "prepared_meal",
    "salad": "prepared_meal",
    "sandwich": "prepared_meal",
    "soup": "prepared_meal",
    "doughnut": "prepared_sweet",
    "ice_cream": "prepared_sweet",
    "french_fries": "prepared_side",
    "non_food": "non_food",
    "human_hand": "non_food",
    "empty_plate": "non_food",
}

KIND_OF = {
    "produce": "ingredient",
    "dairy": "ingredient",
    "protein": "ingredient",
    "grain_starch": "ingredient",
    "packaged_dairy": "packaged",
    "prepared_meal": "prepared_meal",
    "prepared_egg": "prepared_meal",
    "prepared_sweet": "prepared_meal",
    "prepared_side": "prepared_meal",
    "rice_grain": "prepared_meal",
    "non_food": "non_food",
}

STATE_HINT = {
    "tomato": "raw",
    "onion": "raw",
    "potato": "raw",
    "carrot": "raw",
    "garlic": "raw",
    "ginger": "raw",
    "cheese": "raw",
    "packaged_cheese": "packaged",
    "egg": "raw",
    "fried_egg": "prepared",
    "cooked_rice": "prepared",
    "fried_rice": "prepared",
    "biryani": "prepared",
    "pasta": "prepared",
    "pizza": "prepared",
    "salad": "prepared",
    "sandwich": "prepared",
    "doughnut": "prepared",
    "ice_cream": "prepared",
    "french_fries": "prepared",
}


@dataclass
class RefItem:
    id: str
    canonical_identity: str
    family: str
    kind: str
    food_state: str
    embedding: np.ndarray
    provenance: str
    licence: str
    source_hash: str
    model_version: str


@dataclass
class ReferenceLibrary:
    items: list[RefItem] = field(default_factory=list)
    model_version: str = ""

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_version": self.model_version,
            "n": len(self.items),
            "items": [
                {
                    "id": it.id,
                    "canonical_identity": it.canonical_identity,
                    "family": it.family,
                    "kind": it.kind,
                    "food_state": it.food_state,
                    "embedding": it.embedding.tolist(),
                    "provenance": it.provenance,
                    "licence": it.licence,
                    "source_hash": it.source_hash,
                    "model_version": it.model_version,
                }
                for it in self.items
            ],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ReferenceLibrary":
        data = json.loads(path.read_text(encoding="utf-8"))
        items = [
            RefItem(
                id=r["id"],
                canonical_identity=r["canonical_identity"],
                family=r["family"],
                kind=r["kind"],
                food_state=r["food_state"],
                embedding=np.asarray(r["embedding"], dtype=np.float32),
                provenance=r.get("provenance") or "",
                licence=r.get("licence") or "",
                source_hash=r.get("source_hash") or "",
                model_version=r.get("model_version") or "",
            )
            for r in data.get("items") or []
        ]
        return cls(items=items, model_version=data.get("model_version") or "")


class HierarchicalRetriever:
    def __init__(self, library: ReferenceLibrary, top_k: int = 5):
        self.library = library
        self.top_k = top_k
        if library.items:
            self._mat = np.stack([it.embedding for it in library.items], axis=0)
            # L2 normalize
            norms = np.linalg.norm(self._mat, axis=1, keepdims=True) + 1e-8
            self._mat = self._mat / norms
        else:
            self._mat = np.zeros((0, 768), dtype=np.float32)

    def query(self, emb: torch.Tensor | np.ndarray) -> dict[str, Any]:
        if self._mat.shape[0] == 0:
            return {
                "visual": [],
                "family": [],
                "identity": [],
                "state": [],
            }
        if torch.is_tensor(emb):
            q = emb.detach().cpu().float().numpy().reshape(-1)
        else:
            q = np.asarray(emb, dtype=np.float32).reshape(-1)
        q = q / (np.linalg.norm(q) + 1e-8)
        sims = self._mat @ q
        idx = np.argsort(-sims)[: max(self.top_k * 3, self.top_k)]
        visual = []
        for i in idx[: self.top_k]:
            it = self.library.items[int(i)]
            visual.append(
                {
                    "identity": it.canonical_identity,
                    "family": it.family,
                    "kind": it.kind,
                    "food_state": it.food_state,
                    "score": float(sims[int(i)]),
                    "ref_id": it.id,
                }
            )
        # aggregate family / identity / state
        fam_scores: dict[str, float] = defaultdict(float)
        id_scores: dict[str, float] = defaultdict(float)
        state_scores: dict[str, float] = defaultdict(float)
        for i in idx:
            it = self.library.items[int(i)]
            s = float(sims[int(i)])
            fam_scores[it.family] = max(fam_scores[it.family], s)
            id_scores[it.canonical_identity] = max(id_scores[it.canonical_identity], s)
            state_scores[it.food_state] = max(state_scores[it.food_state], s)
        family = sorted(
            [{"family": k, "score": v} for k, v in fam_scores.items()],
            key=lambda x: -x["score"],
        )[: self.top_k]
        identity = sorted(
            [{"identity": k, "score": v, "family": FAMILY_OF.get(k, "unknown")} for k, v in id_scores.items()],
            key=lambda x: -x["score"],
        )[: self.top_k]
        state = sorted(
            [{"food_state": k, "score": v} for k, v in state_scores.items()],
            key=lambda x: -x["score"],
        )[: self.top_k]
        return {"visual": visual, "family": family, "identity": identity, "state": state}


def build_library_from_prototypes(encoder: DenseEncoder, identities: list[str]) -> ReferenceLibrary:
    """Build a minimal prototype library using solid-color / text-free synthetic crops.

    NOTE: Synthetic prototypes are EXPERIMENT scaffolding only for embedding space
    topology when production crops are unavailable locally. Production promotion
    still requires production-eligible embedded references (manifest-backed).
    For V3 local candidate B/E we also ingest production label names as
    identity centroids via random natural crops is avoided on Windows (no bulk).
    Instead we store label-level unit vectors derived from encoder of simple
    patterned images + explicit metadata marking provenance=prototype_scaffold.
    """
    items: list[RefItem] = []
    # Use distinct procedural textures so embeddings separate somewhat
    rng = np.random.RandomState(42)
    for i, lab in enumerate(sorted(set(identities))):
        arr = rng.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
        # tint by identity hash for stability
        h = abs(hash(lab)) % 200 + 30
        arr[:, :, i % 3] = np.clip(arr[:, :, i % 3].astype(int) + h, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        emb = encoder.embed_image(img)["global"].detach().cpu().numpy().reshape(-1)
        fam = FAMILY_OF.get(lab, "unknown")
        items.append(
            RefItem(
                id=f"proto_{lab}",
                canonical_identity=lab,
                family=fam,
                kind=KIND_OF.get(fam, "ingredient"),
                food_state=STATE_HINT.get(lab, "unknown"),
                embedding=emb.astype(np.float32),
                provenance="prototype_scaffold_non_production",
                licence="n/a",
                source_hash=hashlib_sha(lab + str(emb[:8].tolist())),
                model_version=encoder.info.name,
            )
        )
    return ReferenceLibrary(items=items, model_version=encoder.info.name)


def hashlib_sha(s: str) -> str:
    import hashlib

    return hashlib.sha256(s.encode()).hexdigest()[:32]


def identities_from_manifests(new_root: Path) -> list[str]:
    ids: set[str] = set()
    req = new_root / "docs/vision/required_identity_vocabulary_v1.json"
    if req.is_file():
        data = json.loads(req.read_text(encoding="utf-8"))
        for r in data.get("identities") or []:
            if r.get("trainable_production_candidate") or r.get("model_vocab_support"):
                ids.add(r["canonical_identity"])
    vocab = (
        new_root
        / "kaggle/datasets/kiq-v10-recovery-pack/production_label_vocab.json"
    )
    if vocab.is_file():
        v = json.loads(vocab.read_text(encoding="utf-8"))
        ids.update(str(k).lower().replace(" ", "_") for k in v.keys())
    ids.update(FAMILY_OF.keys())
    ids.discard("non_food_object")
    ids.add("non_food")
    return sorted(ids)
