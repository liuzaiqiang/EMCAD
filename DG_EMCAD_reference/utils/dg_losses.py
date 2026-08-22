"""Training-only losses and EMA teacher for DG-EMCAD."""

import copy
import inspect
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


class ModelEMA:
    """Exponential-moving-average copy used only as a training teacher."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0,1)")
        self.decay = float(decay)
        self.module = copy.deepcopy(unwrap_model(model)).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        student_state = unwrap_model(student).state_dict()
        for name, ema_value in self.module.state_dict().items():
            student_value = student_state[name].detach().to(device=ema_value.device)
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(student_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(student_value)


def unpack_output(model_output) -> Tuple[List[torch.Tensor], Optional[Dict]]:
    if isinstance(model_output, dict):
        return list(model_output["logits"]), model_output.get("adaptive_aux")
    if isinstance(model_output, (list, tuple)):
        return list(model_output), None
    return [model_output], None


def _probability(logits: torch.Tensor, temperature: float, eps: float = 1e-6):
    scaled = logits / temperature
    if logits.shape[1] == 1:
        foreground = torch.sigmoid(scaled).clamp(eps, 1.0 - eps)
        return torch.cat((1.0 - foreground, foreground), dim=1)
    return torch.softmax(scaled, dim=1).clamp_min(eps)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def ground_truth_boundary(
    target: torch.Tensor,
    prediction_channels: int,
    radius: int = 2,
) -> torch.Tensor:
    """Return a [B,1,H,W] boundary band from integer or binary labels."""
    if target.ndim == 4:
        if target.shape[1] != 1:
            raise ValueError("4D target must have one channel")
        labels = target[:, 0]
    elif target.ndim == 3:
        labels = target
    else:
        raise ValueError("target must be [B,H,W] or [B,1,H,W]")

    class_count = 2 if prediction_channels == 1 else prediction_channels
    labels = labels.long().clamp(0, class_count - 1)
    one_hot = F.one_hot(labels, num_classes=class_count).permute(0, 3, 1, 2).float()
    dilated = F.max_pool2d(one_hot, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-one_hot, kernel_size=3, stride=1, padding=1)
    boundary = (dilated - eroded).amax(dim=1, keepdim=True).gt(0).float()
    if radius > 0:
        kernel_size = 2 * int(radius) + 1
        boundary = F.max_pool2d(
            boundary,
            kernel_size=kernel_size,
            stride=1,
            padding=int(radius),
        )
    return boundary.clamp(0.0, 1.0)


def _gradient_consistency(
    student_probability: torch.Tensor,
    teacher_probability: torch.Tensor,
    boundary: torch.Tensor,
) -> torch.Tensor:
    student_dx = student_probability[:, :, :, 1:] - student_probability[:, :, :, :-1]
    teacher_dx = teacher_probability[:, :, :, 1:] - teacher_probability[:, :, :, :-1]
    mask_x = torch.maximum(boundary[:, :, :, 1:], boundary[:, :, :, :-1])
    loss_x = _masked_mean((student_dx - teacher_dx).abs().mean(dim=1, keepdim=True), mask_x)

    student_dy = student_probability[:, :, 1:, :] - student_probability[:, :, :-1, :]
    teacher_dy = teacher_probability[:, :, 1:, :] - teacher_probability[:, :, :-1, :]
    mask_y = torch.maximum(boundary[:, :, 1:, :], boundary[:, :, :-1, :])
    loss_y = _masked_mean((student_dy - teacher_dy).abs().mean(dim=1, keepdim=True), mask_y)
    return 0.5 * (loss_x + loss_y)


class BoundaryPartitionDistillationLoss(nn.Module):
    """Stable-region KL + boundary JS + boundary gradient matching."""

    def __init__(
        self,
        temperature: float = 2.0,
        confidence_threshold: float = 0.70,
        boundary_radius: int = 2,
        stable_weight: float = 1.0,
        boundary_weight: float = 1.0,
        gradient_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self.confidence_threshold = float(confidence_threshold)
        self.boundary_radius = int(boundary_radius)
        self.stable_weight = float(stable_weight)
        self.boundary_weight = float(boundary_weight)
        self.gradient_weight = float(gradient_weight)

    def forward(
        self,
        student_outputs: Sequence[torch.Tensor],
        teacher_final: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # The final student head is already directly supervised; distil coarse heads.
        coarse_outputs = list(student_outputs[:-1])
        if not coarse_outputs:
            zero = teacher_final.new_tensor(0.0)
            return zero, {"stable_kl": zero, "boundary_js": zero, "gradient": zero}

        teacher_probability = _probability(teacher_final.detach(), self.temperature)
        boundary = ground_truth_boundary(
            target,
            prediction_channels=teacher_final.shape[1],
            radius=self.boundary_radius,
        )
        stable_mask = 1.0 - boundary
        confidence = teacher_probability.max(dim=1, keepdim=True).values
        stable_mask = stable_mask * (confidence >= self.confidence_threshold).float()

        stable_losses, boundary_losses, gradient_losses = [], [], []
        for logits in coarse_outputs:
            if logits.shape[-2:] != teacher_final.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=teacher_final.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            student_probability = _probability(logits, self.temperature)

            kl_map = (
                teacher_probability
                * (teacher_probability.log() - student_probability.log())
            ).sum(dim=1, keepdim=True)
            stable_losses.append(_masked_mean(kl_map, stable_mask))

            mixture = 0.5 * (student_probability + teacher_probability)
            js_map = 0.5 * (
                student_probability * (student_probability.log() - mixture.log())
            ).sum(dim=1, keepdim=True)
            js_map = js_map + 0.5 * (
                teacher_probability * (teacher_probability.log() - mixture.log())
            ).sum(dim=1, keepdim=True)
            boundary_losses.append(_masked_mean(js_map, boundary))
            gradient_losses.append(
                _gradient_consistency(student_probability, teacher_probability, boundary)
            )

        stable_kl = torch.stack(stable_losses).mean() * (self.temperature ** 2)
        boundary_js = torch.stack(boundary_losses).mean() * (self.temperature ** 2)
        gradient = torch.stack(gradient_losses).mean()
        total = (
            self.stable_weight * stable_kl
            + self.boundary_weight * boundary_js
            + self.gradient_weight * gradient
        )
        return total, {
            "stable_kl": stable_kl.detach(),
            "boundary_js": boundary_js.detach(),
            "gradient": gradient.detach(),
        }


def _call_multiclass_dice(dice_loss, logits: torch.Tensor, target: torch.Tensor):
    parameters = inspect.signature(dice_loss.forward).parameters
    if "softmax" in parameters:
        return dice_loss(logits, target, softmax=True)
    return dice_loss(logits, target)


def routing_prediction_loss(
    auxiliary: Optional[Dict],
    target: torch.Tensor,
    ce_loss=None,
    dice_loss=None,
) -> torch.Tensor:
    """Directly supervise q_i so entropy and cross-scale JS are meaningful."""
    if auxiliary is None:
        return target.new_tensor(0.0, dtype=torch.float32)
    losses = []
    target_size = target.shape[-2:]
    for logits in auxiliary["routing_logits"]:
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
        if logits.shape[1] == 1:
            mask = target.float()
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            bce = F.binary_cross_entropy_with_logits(logits, mask)
            probability = torch.sigmoid(logits)
            intersection = (probability * mask).sum(dim=(1, 2, 3))
            denominator = (probability + mask).sum(dim=(1, 2, 3))
            dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
            losses.append(0.5 * bce + 0.5 * dice)
        else:
            if ce_loss is None or dice_loss is None:
                raise ValueError("multiclass routing loss requires ce_loss and dice_loss")
            labels = target[:, 0] if target.ndim == 4 else target
            losses.append(
                0.3 * ce_loss(logits, labels.long())
                + 0.7 * _call_multiclass_dice(dice_loss, logits, labels)
            )
    return torch.stack(losses).mean()


def routing_regularization(
    auxiliary: Optional[Dict],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Prevent branch collapse and align larger kernels with higher uncertainty."""
    if auxiliary is None:
        zero = torch.tensor(0.0)
        return zero, {"balance": zero, "order": zero}
    balance_losses, order_losses = [], []
    for weights, uncertainty in zip(
        auxiliary["routing_weights"], auxiliary["uncertainty"]
    ):
        branch_count = weights.shape[1]
        usage = weights.mean(dim=(0, 2, 3))
        uniform = torch.full_like(usage, 1.0 / branch_count)
        balance_losses.append((usage - uniform).pow(2).mean())

        scale = torch.linspace(0.0, 1.0, branch_count, device=weights.device)
        expected_scale = (weights * scale.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
        order_losses.append(F.mse_loss(expected_scale, uncertainty.detach()))
    balance = torch.stack(balance_losses).mean()
    order = torch.stack(order_losses).mean()
    return balance + order, {"balance": balance.detach(), "order": order.detach()}


def linear_ramp(epoch: int, warmup_epochs: int, ramp_epochs: int) -> float:
    if epoch < warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch - warmup_epochs + 1) / float(ramp_epochs))
