"""Utilities shared by the optional boundary-aware decoder branch."""

import torch
import torch.nn.functional as F


def boundary_target(mask):
    """Create a class-agnostic one-pixel neighbourhood boundary target.

    ``mask`` may be a binary mask shaped ``[B,1,H,W]`` or a multiclass
    label map shaped ``[B,H,W]``.  A pixel is marked when its 3x3
    neighbourhood contains more than one value.
    """
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4:
        raise ValueError("Expected mask with shape [B,H,W] or [B,1,H,W], got {}".format(tuple(mask.shape)))
    if mask.size(1) != 1:
        mask = torch.argmax(mask, dim=1, keepdim=True)
    values = mask.float()
    local_max = F.max_pool2d(values, kernel_size=3, stride=1, padding=1)
    local_min = -F.max_pool2d(-values, kernel_size=3, stride=1, padding=1)
    return (local_max - local_min > 0).to(dtype=values.dtype)


def boundary_loss(logits, mask):
    """Binary cross-entropy for the optional boundary prediction head."""
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    mask = F.interpolate(mask.float(), size=logits.shape[-2:], mode="nearest")
    target = boundary_target(mask)
    return F.binary_cross_entropy_with_logits(logits, target)
