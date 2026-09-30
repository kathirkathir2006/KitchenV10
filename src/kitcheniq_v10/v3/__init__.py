"""KitchenIQ Vision V3 — open-set hierarchical retrieval + evidence graph pipeline.

Does NOT use a single softmax as the final identity decision.
"""

from __future__ import annotations

from .backbone import DenseEncoder, load_primary_backbone
from .decision import OpenSetDecision, decide_region
from .evidence import EvidenceGraph, Hypothesis
from .pipeline import VisionV3Pipeline
from .retrieval import HierarchicalRetriever, ReferenceLibrary
from .types import RegionDecision, V3Observation

__all__ = [
    "DenseEncoder",
    "load_primary_backbone",
    "OpenSetDecision",
    "decide_region",
    "EvidenceGraph",
    "Hypothesis",
    "VisionV3Pipeline",
    "HierarchicalRetriever",
    "ReferenceLibrary",
    "RegionDecision",
    "V3Observation",
]
