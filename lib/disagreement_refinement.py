"""EMCAD final-logit refinement routed by disagreement between p1 and p2."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EMCADDisagreementRefiner(nn.Module):
    """Residual head for dense or tile-routed refinement of EMCAD logits."""

    # 创新点：定义仅使用 EMCAD 的 p1/p2 分歧路由图块的增量细化器；时间：20260923
    def __init__(self, num_classes, hidden_channels=32, tile_size=16, tile_ratio=0.25):
        """Create a zero-initialized residual head; logits use [B,K,H,W]."""
        super().__init__()
        if tile_size < 1:
            raise ValueError("tile_size must be >= 1")
        if not 0.0 < tile_ratio <= 1.0:
            raise ValueError("tile_ratio must be in (0, 1]")
        self.num_classes = int(num_classes)
        self.tile_size = int(tile_size)
        self.tile_ratio = float(tile_ratio)
        # The head sees the two EMCAD outputs as [B,2K,H,W] and predicts K residual logits.
        self.refine_head = nn.Sequential(
            nn.Conv2d(2 * self.num_classes, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, self.num_classes, kernel_size=1),
        )
        # Zero initialization makes every newly added mode start exactly at the EMCAD output.
        nn.init.zeros_(self.refine_head[-1].weight)
        nn.init.zeros_(self.refine_head[-1].bias)

    # 创新点：按 EMCAD p1/p2 类别预测分歧选择候选图块，并仅计算所选块残差；时间：20260923
    def forward(self, p1, p2, mode):
        """Return refined p1 [B,K,H,W] and the selected-tile fraction."""
        if p1.ndim != 4 or p2.ndim != 4:
            raise ValueError("p1 and p2 must have shape [B,K,H,W]")
        if p1.shape != p2.shape or p1.shape[1] != self.num_classes:
            raise ValueError("p1 and p2 must have matching [B,K,H,W] shapes")
        if mode not in {"dense", "uniform", "disagreement"}:
            raise ValueError("mode must be dense, uniform, or disagreement")

        batch_size, _, height, width = p1.shape
        features = torch.cat((p1, p2), dim=1)
        if mode == "dense":
            return p1 + self.refine_head(features), 1.0

        # Non-overlapping tiles cover odd image sizes by padding only on the lower/right edges.
        tile = self.tile_size
        rows = math.ceil(height / tile)
        cols = math.ceil(width / tile)
        pad_h, pad_w = rows * tile - height, cols * tile - width
        disagreement = (p1.argmax(dim=1) != p2.argmax(dim=1)).to(p1.dtype).unsqueeze(1)
        disagreement = F.pad(disagreement, (0, pad_w, 0, pad_h))
        tile_scores = F.avg_pool2d(disagreement, kernel_size=tile, stride=tile).flatten(1)
        tile_count = rows * cols
        selected_count = max(1, min(tile_count, math.ceil(tile_count * self.tile_ratio)))
        if mode == "uniform":
            # Uniform routing is a compute-matched control for the p1/p2 disagreement signal.
            selected = torch.linspace(
                0, tile_count - 1, steps=selected_count, device=p1.device
            ).round().long().expand(batch_size, -1)
        else:
            # A tiny index offset resolves equal scores reproducibly without changing real rankings.
            tie_break = torch.arange(tile_count, device=p1.device, dtype=p1.dtype)
            tie_break = tie_break / max(tile_count, 1) * 1e-7
            selected = torch.topk(tile_scores + tie_break, selected_count, dim=1).indices

        # Extract each chosen tile with a one-pixel halo for the 3x3 local convolution.
        padded = F.pad(features, (1, 1 + pad_w, 1, 1 + pad_h))
        windows = padded.unfold(2, tile + 2, tile).unfold(3, tile + 2, tile)
        residual = torch.zeros_like(p1)
        for batch_index in range(batch_size):
            flat_index = selected[batch_index]
            tile_rows = torch.div(flat_index, cols, rounding_mode="floor")
            tile_cols = flat_index.remainder(cols)
            # unfold 的布局为 [C, rows, cols, tile+2, tile+2]；索引后转为卷积要求的 [N,C,H,W]。
            chosen = windows[batch_index][:, tile_rows, tile_cols].permute(1, 0, 2, 3).contiguous()
            # Conv output includes halo; only the center tile is written back into EMCAD p1.
            tile_residual = self.refine_head(chosen)[..., 1:-1, 1:-1]
            for chosen_index, tile_index in enumerate(flat_index.tolist()):
                row = tile_index // cols
                col = tile_index % cols
                y0, x0 = row * tile, col * tile
                y1, x1 = min(y0 + tile, height), min(x0 + tile, width)
                residual[batch_index, :, y0:y1, x0:x1] = tile_residual[
                    chosen_index, :, : y1 - y0, : x1 - x0
                ]
        active_fraction = selected_count / tile_count
        return p1 + residual, active_fraction
