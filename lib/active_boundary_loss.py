"""Active Boundary Loss (ABL) for 2-D semantic segmentation.

Implements the two-phase loss described in Wang et al., "Active Boundary Loss
for Semantic Segmentation", AAAI 2022. It adds no model parameters and is only
called by training code when the corresponding command-line switch is enabled.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F


# Eight-neighborhood order is shared by target-direction construction and the
# predicted local KL distribution. The order itself is arbitrary as long as it
# is identical in both places.
_DIRECTIONS_8 = (
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1),
)


def _shift_map(value, dy, dx, fill_value):
    """Return a spatially shifted 2-D tensor and fill positions outside it."""
    height, width = value.shape
    shifted = torch.full_like(value, fill_value)

    src_y0 = max(dy, 0)
    src_y1 = height + min(dy, 0)
    dst_y0 = max(-dy, 0)
    dst_y1 = height - max(dy, 0)
    src_x0 = max(dx, 0)
    src_x1 = width + min(dx, 0)
    dst_x0 = max(-dx, 0)
    dst_x1 = width - max(dx, 0)

    shifted[dst_y0:dst_y1, dst_x0:dst_x1] = value[
        src_y0:src_y1, src_x0:src_x1
    ]
    return shifted


def _shift_probabilities(value, dy, dx):
    """Return ``value[:, y + dy, x + dx]`` for a [C,H,W] probability map."""
    channels, height, width = value.shape
    shifted = value.new_zeros((channels, height, width))

    src_y0 = max(dy, 0)
    src_y1 = height + min(dy, 0)
    dst_y0 = max(-dy, 0)
    dst_y1 = height - max(dy, 0)
    src_x0 = max(dx, 0)
    src_x1 = width + min(dx, 0)
    dst_x0 = max(-dx, 0)
    dst_x1 = width - max(dx, 0)

    shifted[:, dst_y0:dst_y1, dst_x0:dst_x1] = value[
        :, src_y0:src_y1, src_x0:src_x1
    ]
    return shifted


def _ground_truth_boundary_and_distance(label):
    """Build the GT boundary and its Euclidean distance transform on CPU."""
    # ABL's reference implementation uses scipy's exact Euclidean distance
    # transform. Import lazily so baseline training does not load this utility's
    # optional dependency unless ABL is enabled.
    from scipy.ndimage import distance_transform_edt

    label = np.asarray(label)
    height, width = label.shape
    boundary = np.zeros((height, width), dtype=np.bool_)

    # The paper defines a two-neighborhood using the downward and rightward
    # offsets. A label transition at either offset marks a GT boundary pixel.
    boundary[:-1, :] |= label[:-1, :] != label[1:, :]
    boundary[:, :-1] |= label[:, :-1] != label[:, 1:]

    if not boundary.any():
        return boundary, np.zeros((height, width), dtype=np.float32)

    # distance_transform_edt measures each non-boundary pixel's distance to
    # the nearest zero; negating the boundary mask makes GT boundary pixels 0.
    distance = distance_transform_edt(~boundary).astype(np.float32)
    return boundary, distance


def active_boundary_loss(
        logits,
        target,
        max_boundary_ratio=0.01,
        distance_clip=20.0,
        label_smoothing=0.2,
        eps=1e-7):
    """Compute ABL for binary or multi-class segmentation logits.

    Args:
        logits: Raw logits with shape [B,C,H,W]. For binary segmentation C=1;
            for multi-class segmentation C is the number of semantic classes.
        target: Integer labels [B,H,W], or binary labels [B,1,H,W].
        max_boundary_ratio: Maximum fraction of pixels selected as predicted
            boundary points in each image (the paper uses approximately 1%).
        distance_clip: Distance normalization cap theta from the paper (20).
        label_smoothing: Smoothing mass distributed over the eight directions
            (0.2 gives target probabilities 0.8 and 0.2/7).
        eps: Numerical floor used when taking probability logarithms.

    Returns:
        A scalar tensor. A zero tensor connected to ``logits`` is returned when
        a sample has no usable predicted or ground-truth boundary.
    """
    if logits.ndim != 4:
        raise ValueError("ABL expects logits with shape [B,C,H,W]")
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim != 3:
        raise ValueError("ABL expects labels with shape [B,H,W] or [B,1,H,W]")
    if logits.shape[0] != target.shape[0] or logits.shape[-2:] != target.shape[-2:]:
        raise ValueError(
            "ABL logits and target must match in batch and spatial dimensions; "
            "got {} and {}".format(tuple(logits.shape), tuple(target.shape))
        )
    if not 0.0 < max_boundary_ratio <= 1.0:
        raise ValueError("max_boundary_ratio must be in (0, 1]")
    if distance_clip <= 0.0:
        raise ValueError("distance_clip must be positive")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")

    batch_size, channels, height, width = logits.shape
    if height < 3 or width < 3:
        return logits.float().sum() * 0.0

    # Compute probabilities in float32 even under autocast. A one-channel
    # sigmoid head is represented as implicit background + foreground so the
    # same semantic-boundary formulation works for every dataset.
    logits_float = logits.float()
    if channels == 1:
        foreground = torch.sigmoid(logits_float[:, 0])
        probabilities = torch.stack((1.0 - foreground, foreground), dim=1)
    elif channels >= 2:
        probabilities = torch.softmax(logits_float, dim=1)
    else:
        raise ValueError("ABL requires at least one logit channel")

    # The discrete boundary locations and target directions are intentionally
    # non-differentiable. Gradients are applied only in Phase II at PDB points.
    with torch.no_grad():
        detached_probabilities = probabilities.detach()
        labels_cpu = target.detach().to(device="cpu").numpy()
        predicted_boundaries = []
        gt_distances = []
        target_directions = []
        usable_samples = []

        # Use the maximum KL change to the right/down as the paper's PDB score;
        # an adaptive per-image top-k selects no more than about 1% of pixels.
        for batch_index in range(batch_size):
            sample_probs = detached_probabilities[batch_index]
            center = sample_probs.clamp_min(eps)
            center_log = center.log()
            boundary_score = sample_probs.new_zeros((height, width))

            for dy, dx in ((1, 0), (0, 1)):
                neighbor = _shift_probabilities(sample_probs, dy, dx).clamp_min(eps)
                valid = _shift_map(
                    sample_probs.new_ones((height, width)), dy, dx, 0.0
                ) > 0.5
                kl = (
                    center * (center_log - neighbor.log())
                ).sum(dim=0).clamp_min(0.0)
                boundary_score = torch.maximum(
                    boundary_score, torch.where(valid, kl, torch.zeros_like(kl))
                )

            # Border pixels have incomplete 8-neighborhoods. Excluding them
            # keeps every selected PDB point's eight candidate directions valid.
            interior_score = boundary_score[1:-1, 1:-1].reshape(-1)
            candidate_count = max(
                1, int(math.floor(height * width * max_boundary_ratio))
            )
            candidate_count = min(candidate_count, interior_score.numel())
            selected_scores, selected_indices = torch.topk(
                interior_score, k=candidate_count, largest=True, sorted=False
            )
            pdb = torch.zeros_like(boundary_score, dtype=torch.bool)
            valid_selected = selected_scores > eps
            if valid_selected.any():
                interior_width = width - 2
                flat_indices = selected_indices[valid_selected]
                ys = torch.div(flat_indices, interior_width, rounding_mode="floor") + 1
                xs = torch.remainder(flat_indices, interior_width) + 1
                pdb[ys, xs] = True

            gt_boundary_np, distance_np = _ground_truth_boundary_and_distance(
                labels_cpu[batch_index]
            )
            if not gt_boundary_np.any() or not pdb.any():
                predicted_boundaries.append(pdb)
                gt_distances.append(sample_probs.new_zeros((height, width)))
                target_directions.append(
                    torch.zeros((height, width), dtype=torch.long,
                                device=logits.device)
                )
                usable_samples.append(False)
                continue

            distance = torch.as_tensor(
                distance_np, dtype=torch.float32, device=logits.device
            )
            # For each pixel, choose the neighboring location with the smallest
            # distance to a GT boundary: this is the active movement direction.
            neighbor_distances = torch.stack([
                _shift_map(distance, dy, dx, float("inf"))
                for dy, dx in _DIRECTIONS_8
            ], dim=0)
            direction = neighbor_distances.argmin(dim=0)

            predicted_boundaries.append(pdb)
            gt_distances.append(distance)
            target_directions.append(direction)
            usable_samples.append(True)

    sample_losses = []
    for batch_index in range(batch_size):
        if not usable_samples[batch_index]:
            sample_losses.append(logits_float[batch_index].sum() * 0.0)
            continue

        distance = gt_distances[batch_index]
        # Points already on the GT boundary have zero distance and are omitted,
        # as specified in the paper.
        active = predicted_boundaries[batch_index] & (distance > 0.0)
        coordinates = torch.nonzero(active, as_tuple=False)
        if coordinates.numel() == 0:
            sample_losses.append(logits_float[batch_index].sum() * 0.0)
            continue

        ys = coordinates[:, 0]
        xs = coordinates[:, 1]
        sample_probs = probabilities[batch_index]
        center = sample_probs[:, ys, xs].transpose(0, 1).clamp_min(eps)
        center_log = center.log()

        # For conflict suppression, neighboring distributions are detached:
        # each PDB point receives gradients, but its neighbors do not receive
        # contradictory gradients through this point's direction comparison.
        neighbor_kls = []
        for dy, dx in _DIRECTIONS_8:
            neighbor = sample_probs[
                :, ys + dy, xs + dx
            ].transpose(0, 1).detach().clamp_min(eps)
            kl = (center * (center_log - neighbor.log())).sum(dim=1)
            neighbor_kls.append(kl)
        predicted_direction_log_prob = F.log_softmax(
            torch.stack(neighbor_kls, dim=1), dim=1
        )

        direction_index = target_directions[batch_index][ys, xs]
        smoothed_target = predicted_direction_log_prob.new_full(
            predicted_direction_log_prob.shape,
            label_smoothing / 7.0,
        )
        smoothed_target.scatter_(
            1, direction_index.unsqueeze(1), 1.0 - label_smoothing
        )
        direction_cross_entropy = -(
            smoothed_target * predicted_direction_log_prob
        ).sum(dim=1)

        # Normalize the distance weight to [0,1] and cap it at theta=20.
        distance_weight = distance[ys, xs].clamp(max=distance_clip) / distance_clip
        sample_losses.append(
            (direction_cross_entropy * distance_weight).sum()
            / float(coordinates.shape[0])
        )

    return torch.stack(sample_losses).mean()
