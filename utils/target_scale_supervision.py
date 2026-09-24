"""Target-scale curriculum supervision for the existing EMCAD heads.

The module deliberately leaves the EMCAD network and inference path unchanged.
It only supplies sample/head weights to the existing CE + Dice supervision.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion


BUCKET_NAMES = ("empty", "small", "medium", "large")

# Factors are indexed in EMCAD output order [P4, P3, P2, P1].
AREA_HEAD_FACTORS = {
    "small": (0.70, 0.90, 1.20, 1.50),
    "medium": (1.00, 1.00, 1.00, 1.00),
    "large": (1.50, 1.20, 0.90, 0.70),
}
BOUNDARY_HEAD_MODULATION = (0.90, 0.95, 1.05, 1.10)


def _target_2d(target):
    """Return integer labels as [B,H,W] without changing the caller tensor."""
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim != 3:
        raise ValueError("target_scale supervision expects [B,H,W] labels, got {}".format(tuple(target.shape)))
    return target.long()


# 创新点：目标尺度课程监督的线性课程强度，训练后期恢复统一监督 时间：20260924
def curriculum_strength(mode, epoch, max_epochs):
    """Return the configured schedule strength for one zero-based epoch."""
    if mode == "off":
        return 0.0
    if mode == "static":
        return 1.0
    if mode in {"early_only", "late_only"}:
        # Use the same number of weighted epochs in each arm; an odd middle
        # epoch remains unweighted in both arms.
        half_count = max_epochs // 2
        if mode == "early_only":
            return float(epoch < half_count)
        return float(epoch >= max_epochs - half_count)
    if mode != "curriculum":
        raise ValueError("Unknown target_scale_mode: {}".format(mode))
    if max_epochs <= 1:
        return 1.0
    return float(max(0.0, 1.0 - float(epoch) / float(max_epochs - 1)))


# 创新点：依据 GT 前景面积和 3x3 腐蚀边界复杂度生成四头监督权重 时间：20260924
def build_target_scale_weights(
    target,
    mode="off",
    factor_mode="area_boundary",
    small_threshold=0.05,
    large_threshold=0.20,
    boundary_reference=0.25,
    epoch=0,
    max_epochs=1,
    num_heads=4,
):
    """Build normalized sample/head weights and auditable bucket statistics.

    ``target`` is a batch of integer label maps.  A boundary is defined on the
    foreground mask as ``foreground - erosion_3x3(foreground)``.  Empty samples
    receive all-one weights because they have no foreground scale to emphasize.
    """
    target = _target_2d(target)
    if not (0.0 <= small_threshold < large_threshold <= 1.0):
        raise ValueError("Require 0 <= small_threshold < large_threshold <= 1")
    if boundary_reference <= 0:
        raise ValueError("boundary_reference must be positive")
    if factor_mode not in {"area", "area_boundary"}:
        raise ValueError("Unknown target_scale_factors: {}".format(factor_mode))
    if mode not in {"off", "static", "curriculum", "early_only", "late_only"}:
        raise ValueError("Unknown target_scale_mode: {}".format(mode))
    if mode != "off" and num_heads != 4:
        raise ValueError("target-scale supervision requires EMCAD's four output heads")

    foreground = target > 0
    foreground_float = foreground.float().unsqueeze(1)
    foreground_pixels = foreground_float.sum(dim=(1, 2, 3))
    image_area = float(target.shape[-2] * target.shape[-1])
    area_ratio = foreground_pixels / image_area

    # Pad the inverse mask with background so image-border foreground is boundary.
    background = 1.0 - foreground_float
    padded_background = F.pad(background, (1, 1, 1, 1), value=1.0)
    eroded = 1.0 - F.max_pool2d(padded_background, kernel_size=3, stride=1)
    boundary_pixels = (foreground_float - eroded).clamp_min(0.0).sum(dim=(1, 2, 3))
    boundary_complexity = boundary_pixels / foreground_pixels.clamp_min(1.0)

    bucket_ids = torch.zeros(target.shape[0], dtype=torch.long, device=target.device)
    nonempty = foreground_pixels > 0
    bucket_ids[nonempty & (area_ratio < small_threshold)] = 1
    bucket_ids[nonempty & (area_ratio >= small_threshold) & (area_ratio < large_threshold)] = 2
    bucket_ids[nonempty & (area_ratio >= large_threshold)] = 3

    weights = torch.ones(
        (target.shape[0], num_heads), dtype=torch.float32, device=target.device
    )
    if mode != "off":
        factors = torch.ones_like(weights)
        for bucket_id, bucket_name in ((1, "small"), (2, "medium"), (3, "large")):
            selected = bucket_ids == bucket_id
            if selected.any():
                factors[selected] = torch.as_tensor(
                    AREA_HEAD_FACTORS[bucket_name], dtype=weights.dtype, device=weights.device
                )
        if factor_mode == "area_boundary":
            boundary_fraction = (boundary_complexity / boundary_reference).clamp(0.0, 1.0)
            modulation = torch.as_tensor(
                BOUNDARY_HEAD_MODULATION, dtype=weights.dtype, device=weights.device
            ).view(1, num_heads)
            factors = factors * (1.0 + boundary_fraction.view(-1, 1) * (modulation - 1.0))
        strength = curriculum_strength(mode, epoch, max_epochs)
        weights = 1.0 + float(strength) * (factors - 1.0)
        # Each sample has mean head weight 1, so the candidate changes relative
        # supervision emphasis without introducing a free global loss multiplier.
        weights = weights / weights.mean(dim=1, keepdim=True).clamp_min(1e-6)
    else:
        strength = 0.0

    stats = {
        "area_ratio": area_ratio.detach(),
        "boundary_complexity": boundary_complexity.detach(),
        "bucket_ids": bucket_ids.detach(),
        "weights": weights.detach(),
        "curriculum_strength": float(strength),
    }
    return weights, stats


# 创新点：把目标尺度权重施加到原 EMCAD 交叉熵的样本归约上 时间：20260924
def weighted_cross_entropy(logits, target, sample_weights):
    """Apply sample weights while retaining CrossEntropyLoss's spatial mean."""
    target = _target_2d(target)
    per_pixel = F.cross_entropy(logits, target, reduction="none")
    per_sample = per_pixel.flatten(1).mean(dim=1)
    return (per_sample * sample_weights.to(dtype=per_sample.dtype)).mean()


# 创新点：把目标尺度权重施加到原 EMCAD batch-level soft Dice 归约上 时间：20260924
def weighted_dice_loss(logits, target, sample_weights, num_classes):
    """Apply sample weights to the same squared-denominator soft Dice formula."""
    target = _target_2d(target)
    probabilities = torch.softmax(logits, dim=1)
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    batch_weights = sample_weights.to(dtype=probabilities.dtype).view(-1, 1, 1, 1)
    dims = (0, 2, 3)
    intersection = torch.sum(probabilities * target_one_hot * batch_weights, dim=dims)
    denominator = torch.sum(
        (probabilities * probabilities + target_one_hot * target_one_hot) * batch_weights,
        dim=dims,
    )
    dice = (2.0 * intersection + 1e-5) / (denominator + 1e-5)
    return 1.0 - dice.mean()


# 创新点：对目标尺度监督批次输出可复核的分桶、课程和四头权重统计 时间：20260924
def summarize_target_scale_batch(stats, num_heads=4):
    """Convert detached batch statistics to JSON/log-friendly Python values."""
    bucket_counts = torch.bincount(stats["bucket_ids"], minlength=len(BUCKET_NAMES))
    weights = stats["weights"]
    return {
        "bucket_counts": [int(value) for value in bucket_counts.cpu().tolist()],
        "area_mean": float(stats["area_ratio"].mean().cpu().item()),
        "boundary_complexity_mean": float(stats["boundary_complexity"].mean().cpu().item()),
        "curriculum_strength": float(stats["curriculum_strength"]),
        "head_weight_mean": [float(value) for value in weights.mean(dim=0).cpu().tolist()[:num_heads]],
        "head_weight_min": float(weights.min().cpu().item()),
        "head_weight_max": float(weights.max().cpu().item()),
    }


# 创新点：对完整患者标签计算与训练一致的前景尺度分桶和边界复杂度 时间：20260924
def target_scale_bucket_numpy(label, small_threshold=0.05, large_threshold=0.20):
    """Return patient-level area/boundary bucket statistics for result tables."""
    label = np.asarray(label)
    if label.ndim not in {2, 3}:
        raise ValueError("Expected [H,W] or [D,H,W] label array, got {}".format(label.shape))
    foreground = label > 0
    area_ratio = float(foreground.mean())
    if label.ndim == 2:
        structure = np.ones((3, 3), dtype=bool)
    else:
        # Keep the z axis independent because EMCAD predicts 2D slices.
        structure = np.ones((1, 3, 3), dtype=bool)
    eroded = binary_erosion(foreground, structure=structure, border_value=0)
    boundary = foreground & ~eroded
    foreground_count = int(foreground.sum())
    boundary_complexity = float(boundary.sum() / max(foreground_count, 1))
    if foreground_count == 0:
        bucket = "empty"
    elif area_ratio < small_threshold:
        bucket = "small"
    elif area_ratio < large_threshold:
        bucket = "medium"
    else:
        bucket = "large"
    return {
        "bucket": bucket,
        "area_ratio": area_ratio,
        "boundary_complexity": boundary_complexity,
    }
