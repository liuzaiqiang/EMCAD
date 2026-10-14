"""Area adaptive foreground Tversky supervision for binary and multiclass tasks."""

import torch
import torch.nn.functional as F


def area_adaptive_tversky_loss(logits, target, alpha=0.4, beta=0.6,
                               max_class_weight=3.0, eps=1e-6):
    """One foreground-area-weighted objective for binary and multiclass masks.

    A unit is one (image, foreground class) pair. All present units in the
    batch share the same weighting and reduction, regardless of class count.
    The one-logit and multi-logit cases only differ when decoding probabilities
    and labels into the common [B, K, H, W] foreground representation.
    """
    if logits.ndim != 4 or target.ndim != 3 or target.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError("Expected logits [B,C,H,W] and target [B,H,W]")
    if not 0 < alpha < 1 or not 0 < beta < 1 or max_class_weight < 1:
        raise ValueError("Invalid Tversky parameters")

    if logits.shape[1] == 1:
        probabilities = torch.sigmoid(logits.float())
        labels = target.unsqueeze(1).to(probabilities.dtype)
    else:
        probabilities = torch.softmax(logits.float(), dim=1)[:, 1:]
        labels = F.one_hot(target.long(), num_classes=logits.shape[1])
        labels = labels.permute(0, 3, 1, 2)[:, 1:].to(probabilities.dtype)

    dims = (2, 3)
    area = labels.sum(dim=dims)
    present = area > 0
    if not present.any():
        return logits.sum() * 0.0

    tp = (probabilities * labels).sum(dim=dims)
    fp = (probabilities * (1 - labels)).sum(dim=dims)
    fn = ((1 - probabilities) * labels).sum(dim=dims)
    score = (tp + eps) / (tp + alpha * fp + beta * fn + eps)

    inverse_area = area.clamp_min(1).rsqrt()
    relative_weight = inverse_area / inverse_area[present].mean().clamp_min(eps)
    weights = relative_weight.clamp(max=max_class_weight) * present
    return ((1 - score) * weights).sum() / weights.sum().clamp_min(eps)
