"""Signed-distance feature calibration and mask-derived supervision."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.ndimage import distance_transform_edt


class SignedDistanceCalibration(nn.Module):
    """Predict foreground signed distances and use them to gate the last decoder feature."""

    def __init__(self, channels, num_classes, residual_scale=0.1):
        super().__init__()
        if residual_scale <= 0:
            raise ValueError("sdf_residual_scale must be positive")
        distance_channels = 1 if num_classes == 1 else num_classes - 1
        self.distance_head = nn.Sequential(
            nn.Conv2d(channels, max(16, channels // 4), 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(16, channels // 4), distance_channels, 1),
        )
        self.gate = nn.Conv2d(distance_channels, channels, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.residual_scale = residual_scale

    def forward(self, feature):
        distance = torch.tanh(self.distance_head(feature))
        calibrated = feature * (1 + self.residual_scale * torch.tanh(self.gate(distance)))
        return calibrated, distance


def signed_distance_loss(prediction, mask, radius=16):
    """Smooth L1 against a clipped per-class signed EDT at prediction resolution.

    Positive values are inside a target, negative values outside. Empty and full
    class masks have no observed boundary and are excluded from the loss.
    """
    if radius <= 0:
        raise ValueError("sdf_radius must be positive")
    if mask.ndim == 4:
        if mask.shape[1] != 1:
            raise ValueError("mask must have one channel")
        mask = mask[:, 0]
    if mask.ndim != 3:
        raise ValueError("mask must be [B,H,W] or [B,1,H,W]")
    labels = F.interpolate(mask[:, None].float(), size=prediction.shape[-2:], mode="nearest")[:, 0]
    labels = labels.detach().cpu().numpy()
    binary = prediction.shape[1] == 1
    targets = np.zeros(prediction.shape, dtype=np.float32)
    valid = np.zeros(prediction.shape[:2], dtype=np.bool_)
    for batch_index, label in enumerate(labels):
        for channel in range(prediction.shape[1]):
            region = label >= 0.5 if binary else label == channel + 1
            if not region.any() or region.all():
                continue
            signed = distance_transform_edt(region) - distance_transform_edt(~region)
            targets[batch_index, channel] = np.clip(signed / radius, -1, 1)
            valid[batch_index, channel] = True
    if not valid.any():
        return prediction.sum() * 0
    target = torch.from_numpy(targets).to(device=prediction.device, dtype=prediction.dtype)
    valid = torch.from_numpy(valid).to(device=prediction.device)
    error = F.smooth_l1_loss(prediction.float(), target.float(), reduction="none")
    boundary_weight = 1 + 2 * (target.abs() < 0.25).float()
    weights = valid[:, :, None, None] * boundary_weight
    return (error * weights).sum() / weights.sum().clamp_min(1)
