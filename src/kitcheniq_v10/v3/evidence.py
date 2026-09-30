"""Evidence graph: competing hypotheses from independent evidence sources."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from kitcheniq_v10.v3.types import Hypothesis


@dataclass
class EvidenceEdge:
    source: str
    target: str
    kind: str
    weight: float
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceGraph:
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: list[EvidenceEdge] = field(default_factory=list)

    def add_node(self, node_id: str, **attrs: Any) -> None:
        self.nodes[node_id] = dict(attrs)

    def add_edge(self, source: str, target: str, kind: str, weight: float, **detail: Any) -> None:
        self.edges.append(EvidenceEdge(source, target, kind, weight, detail))

    def score_identities(self) -> list[Hypothesis]:
        """Aggregate competing identity hypotheses — NOT a single softmax."""
        scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        sources: dict[str, list[str]] = defaultdict(list)
        meta: dict[str, dict[str, Any]] = {}

        for e in self.edges:
            if e.kind in {
                "visual_similarity",
                "family_support",
                "state_compatibility",
                "specialist_support",
                "ocr_agreement",
                "barcode_agreement",
                "spatial_compatibility",
                "co_occurrence",
                "negative_evidence",
            }:
                ident = e.target
                if e.kind == "negative_evidence":
                    scores[ident]["neg"] += abs(e.weight)
                else:
                    scores[ident][e.kind] += e.weight
                sources[ident].append(e.kind)
                meta[ident] = {**meta.get(ident, {}), **(e.detail or {})}

        hyps: list[Hypothesis] = []
        for ident, parts in scores.items():
            pos = sum(v for k, v in parts.items() if k != "neg")
            neg = parts.get("neg", 0.0)
            # require multi-source support for high confidence
            uniq_src = sorted(set(sources[ident]))
            diversity = min(1.0, len([s for s in uniq_src if s != "negative_evidence"]) / 3.0)
            raw = max(0.0, pos - neg)
            score = raw * (0.55 + 0.45 * diversity)
            fam = meta[ident].get("family")
            kind = meta[ident].get("entity_kind") or meta[ident].get("kind")
            state = meta[ident].get("food_state")
            hyps.append(
                Hypothesis(
                    identity=ident,
                    family=fam,
                    kind=kind,
                    food_state=state,
                    score=float(score),
                    sources=uniq_src,
                    evidence={k: float(v) for k, v in parts.items()},
                )
            )
        hyps.sort(key=lambda h: -h.score)
        return hyps

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "kind": e.kind,
                    "weight": e.weight,
                    "detail": e.detail,
                }
                for e in self.edges
            ],
        }
