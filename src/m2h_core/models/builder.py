"""Convenience builders for M2H models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch.nn as nn

from ..backbone.dinov2 import load_backbone
from . import full, lightweight

BackboneName = Literal["vit_small", "vit_base", "vit_large", "vit_giant2"]
ModelFlavor = Literal["default", "lightweight"]


@dataclass
class ModelConfig:
    arch: BackboneName = "vit_small"
    flavor: ModelFlavor = "default"
    min_depth: float = 0.001
    max_depth: float = 10.0
    num_classes: int = 41


def build_model(cfg: Optional[ModelConfig] = None, **overrides):
    """Instantiate an M2H model according to the provided configuration."""
    cfg = cfg or ModelConfig()
    for key, value in overrides.items():
        setattr(cfg, key, value)

    backbone = load_backbone(cfg.arch)
    embed_dim = backbone.embed_dim
    post_process_channels = [embed_dim // 2 ** (3 - i) for i in range(4)]

    if cfg.flavor == "lightweight":
        head = lightweight.MLTHead(
            embed_dims=embed_dim,
            post_process_channels=post_process_channels,
            readout_type="project",
            min_depth=cfg.min_depth,
            max_depth=cfg.max_depth,
            num_classes=cfg.num_classes,
            act_layer=nn.GELU,
        )
        model = lightweight.DepthEncoderDecoder(backbone=backbone, mlt_head=head)
    else:
        head = full.MLTHead(
            channels=256,
            embed_dims=embed_dim,
            post_process_channels=post_process_channels,
            readout_type="project",
            min_depth=cfg.min_depth,
            max_depth=cfg.max_depth,
            num_classes=cfg.num_classes,
            act_layer=nn.GELU,
        )
        model = full.DepthEncoderDecoder(backbone=backbone, mlt_head=head)

    return model


__all__ = ["ModelConfig", "build_model"]
