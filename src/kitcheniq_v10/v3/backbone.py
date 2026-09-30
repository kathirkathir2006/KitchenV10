"""Primary visual backbone for V3: DINOv3 preferred, DINOv2 controlled fallback."""

from __future__ import annotations

import hashlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class BackboneInfo:
    name: str
    version: str
    source: str
    embed_dim: int
    weights_hash: str | None
    fallback_reason: str | None = None


class DenseEncoder:
    """Frozen encoder producing global + region + patch embeddings."""

    def __init__(self, model: Any, info: BackboneInfo, device: str = "cpu"):
        self.model = model
        self.info = info
        self.device = device
        self.model.eval()
        self.model.to(device)

    @torch.no_grad()
    def preprocess(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB").resize((224, 224), Image.BILINEAR)
        arr = np.asarray(img).astype("float32") / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype="float32")
        std = np.array([0.229, 0.224, 0.225], dtype="float32")
        arr = (arr - mean) / std
        return torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0).to(self.device)

    @torch.no_grad()
    def embed_image(self, img: Image.Image) -> dict[str, Any]:
        x = self.preprocess(img)
        t0 = time.perf_counter()
        global_vec, patch_vec = self._forward(x)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "global": global_vec,
            "patch": patch_vec,
            "latency_ms": latency_ms,
            "dim": int(global_vec.shape[-1]),
            "backbone": self.info.name,
        }

    def _forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if hasattr(self.model, "forward_features"):
            feats = self.model.forward_features(x)
            if isinstance(feats, dict):
                if "x_norm_clstoken" in feats:
                    g = feats["x_norm_clstoken"]
                    p = feats.get("x_norm_patchtokens")
                    return F.normalize(g.float(), dim=-1), (
                        F.normalize(p.float(), dim=-1) if p is not None else None
                    )
            if torch.is_tensor(feats):
                if feats.ndim == 3:
                    g = feats[:, 0]
                    p = feats[:, 1:]
                    return F.normalize(g.float(), dim=-1), F.normalize(p.float(), dim=-1)
                return F.normalize(feats.float(), dim=-1), None
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim > 2:
            g = out.mean(dim=1)
            return F.normalize(g.float(), dim=-1), F.normalize(out.float(), dim=-1)
        return F.normalize(out.float(), dim=-1), None


def _hash_state_dict(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in list(model.state_dict().items())[:32]:
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes()[:4096])
    return h.hexdigest()


def _try_dinov3(device: str) -> DenseEncoder | None:
    """Load DINOv3 ViT-B/16 with pretrained weights if Meta CDN allows."""
    errors: list[str] = []
    hub = Path.home() / ".cache/torch/hub/facebookresearch_dinov3_main"
    if not hub.is_dir():
        try:
            torch.hub.load("facebookresearch/dinov3", "dinov3_vitb16", trust_repo=True, pretrained=False)
        except Exception as e:  # noqa: BLE001
            errors.append(f"hub_clone:{type(e).__name__}:{e}")
    if hub.is_dir() and str(hub) not in sys.path:
        sys.path.insert(0, str(hub))
    try:
        from dinov3.hub.backbones import dinov3_vitb16  # type: ignore
    except Exception as e:  # noqa: BLE001
        errors.append(f"import:{type(e).__name__}:{e}")
        return None
    try:
        model = dinov3_vitb16(pretrained=True)
        info = BackboneInfo(
            name="DINOv3:dinov3_vitb16",
            version="dinov3_vitb16_pretrain_lvd1689m",
            source="dinov3.hub.backbones.dinov3_vitb16(pretrained=True)",
            embed_dim=int(getattr(model, "embed_dim", 768) or 768),
            weights_hash=_hash_state_dict(model)[:64],
        )
        return DenseEncoder(model, info, device=device)
    except Exception as e:  # noqa: BLE001
        errors.append(f"pretrained:{type(e).__name__}:{e}")
        # Do NOT use randomly initialized DINOv3 — that would silently destroy retrieval.
        reason = (
            "DINOv3 pretrained weights gated/unavailable (HTTP 403 or auth required on "
            "dl.fbaipublicfiles.com). Refusing untrained DINOv3. "
            f"Detail: {errors[-1]}"
        )
        # stash reason on a sentinel by raising to caller via None + global note
        _try_dinov3.last_error = reason  # type: ignore[attr-defined]
        return None


_try_dinov3.last_error = None  # type: ignore[attr-defined]


def load_primary_backbone(device: str | None = None) -> DenseEncoder:
    """Load DINOv3 if pretrained weights available; else DINOv2 ViT-B/14 (documented)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    enc = _try_dinov3(device)
    if enc is not None:
        return enc
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True)
    info = BackboneInfo(
        name="DINOv2:dinov2_vitb14",
        version="dinov2_fallback",
        source="torch.hub:facebookresearch/dinov2/dinov2_vitb14",
        embed_dim=768,
        weights_hash=_hash_state_dict(model)[:64],
        fallback_reason=_try_dinov3.last_error  # type: ignore[attr-defined]
        or "DINOv3 unavailable; using frozen DINOv2 with full V3 retrieval/evidence-graph algorithm.",
    )
    return DenseEncoder(model, info, device=device)
