"""Helpers for configuring DINOv2 backbones for M2H."""
from __future__ import annotations

from functools import partial
from typing import Dict

import torch

from ..models.full import CenterPadding

_ARCH_TO_HUB: Dict[str, str] = {
    "vit_small": "dinov2_vits14",
    "vit_base": "dinov2_vitb14",
    "vit_large": "dinov2_vitl14",
    "vit_giant2": "dinov2_vitg14",
}

_OUT_INDEX = {
    "vit_small": [2, 5, 8, 11],
    "vit_base": [2, 5, 8, 11],
    "vit_large": [4, 11, 17, 23],
    "vit_giant2": [9, 19, 29, 39],
}


def load_backbone(arch: str):
    if arch not in _ARCH_TO_HUB:
        raise ValueError(f"Unsupported arch '{arch}'.")

    repo_id = _ARCH_TO_HUB[arch]
    backbone = torch.hub.load("facebookresearch/dinov2", repo_id)
    out_index = _OUT_INDEX[arch]

    backbone.forward = partial(
        backbone.get_intermediate_layers,
        n=out_index,
        reshape=True,
        return_class_token=True,
        norm=False,
    )

    backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone.patch_size)(x[0]))
    return backbone

