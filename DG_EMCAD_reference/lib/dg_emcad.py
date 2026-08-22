"""Reference implementation of Disagreement-Guided Adaptive EMCAD.

This file lives outside the user's repository. Copy it to ``lib/dg_emcad.py``
only after reviewing the integration guide.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.decoders import CAB, EUCB, LGAG, SAB, act_layer, channel_shuffle, gcd
from lib.networks import EMCADNet


def _probabilities(logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return an explicit class distribution for binary or multiclass logits."""
    if logits.shape[1] == 1:
        foreground = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
        return torch.cat((1.0 - foreground, foreground), dim=1)
    return torch.softmax(logits, dim=1).clamp_min(eps)


def _normalized_entropy(probability: torch.Tensor) -> torch.Tensor:
    class_count = probability.shape[1]
    entropy = -(probability * probability.log()).sum(dim=1, keepdim=True)
    return entropy / torch.log(probability.new_tensor(float(class_count)))


def _normalized_js(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    mixture = 0.5 * (first + second)
    first_kl = (first * (first.log() - mixture.log())).sum(dim=1, keepdim=True)
    second_kl = (second * (second.log() - mixture.log())).sum(dim=1, keepdim=True)
    return 0.5 * (first_kl + second_kl) / torch.log(first.new_tensor(2.0))


class DisagreementGuidedMSCB(nn.Module):
    """MSCB whose 1/3/5 depth-wise branches are mixed per pixel.

    ``router_mode`` supports the required ablations:
      equal: fixed equal weights (mathematically recovers branch summation)
      global: one learned image-independent weight per branch
      feature: weights predicted from the current feature only
      disagreement: weights predicted from feature + entropy/adjacent-scale JS
    """

    VALID_MODES = {"equal", "global", "feature", "disagreement"}

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_classes: int,
        kernel_sizes: Sequence[int] = (1, 3, 5),
        expansion_factor: int = 2,
        activation: str = "relu6",
        router_mode: str = "disagreement",
        disagreement_lambda: float = 1.0,
        router_temperature: float = 1.0,
        router_hidden: int = 32,
    ) -> None:
        super().__init__()
        if router_mode not in self.VALID_MODES:
            raise ValueError("Unknown router_mode: {}".format(router_mode))
        if router_temperature <= 0:
            raise ValueError("router_temperature must be positive")
        if not kernel_sizes:
            raise ValueError("kernel_sizes must not be empty")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
        self.branch_count = len(self.kernel_sizes)
        self.router_mode = router_mode
        self.disagreement_lambda = float(disagreement_lambda)
        self.router_temperature = float(router_temperature)
        expanded_channels = int(in_channels * expansion_factor)

        self.pconv1 = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            act_layer(activation, inplace=True),
        )
        self.dwconvs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        expanded_channels,
                        expanded_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size // 2,
                        groups=expanded_channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(expanded_channels),
                    act_layer(activation, inplace=True),
                )
                for kernel_size in self.kernel_sizes
            ]
        )
        self.pconv2 = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )

        # q_i in U_i = H(q_i) + lambda * JS(q_i, up(q_{i+1})).
        self.routing_prediction = nn.Conv2d(in_channels, num_classes, 1)

        if router_mode == "global":
            self.global_router_logits = nn.Parameter(torch.zeros(self.branch_count))
        elif router_mode in {"feature", "disagreement"}:
            hidden = max(8, min(int(router_hidden), in_channels))
            self.feature_projection = nn.Sequential(
                nn.Conv2d(in_channels, hidden, 1, bias=False),
                nn.GroupNorm(1, hidden),
                nn.ReLU(inplace=True),
            )
            router_channels = hidden + (1 if router_mode == "disagreement" else 0)
            self.router = nn.Conv2d(router_channels, self.branch_count, 1)

        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _uncertainty(
        self,
        current_logits: torch.Tensor,
        deeper_logits: Optional[torch.Tensor],
    ) -> torch.Tensor:
        current_probability = _probabilities(current_logits)
        entropy = _normalized_entropy(current_probability)
        if deeper_logits is None:
            return entropy
        deeper_logits = F.interpolate(
            deeper_logits,
            size=current_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        deeper_probability = _probabilities(deeper_logits)
        disagreement = _normalized_js(current_probability, deeper_probability)
        return (entropy + self.disagreement_lambda * disagreement) / (
            1.0 + self.disagreement_lambda
        )

    def _routing_weights(
        self,
        feature: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, height, width = feature.shape
        if self.router_mode == "equal":
            return feature.new_full(
                (batch, self.branch_count, height, width),
                1.0 / self.branch_count,
            )
        if self.router_mode == "global":
            weights = torch.softmax(
                self.global_router_logits / self.router_temperature, dim=0
            )
            return weights.view(1, -1, 1, 1).expand(batch, -1, height, width)

        router_feature = self.feature_projection(feature)
        if self.router_mode == "disagreement":
            router_feature = torch.cat((router_feature, uncertainty), dim=1)
        router_logits = self.router(router_feature)
        return torch.softmax(router_logits / self.router_temperature, dim=1)

    def forward(
        self,
        feature: torch.Tensor,
        deeper_logits: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        routing_logits = self.routing_prediction(feature)
        uncertainty = self._uncertainty(routing_logits, deeper_logits)
        weights = self._routing_weights(feature, uncertainty)

        expanded = self.pconv1(feature)
        branch_outputs = [branch(expanded) for branch in self.dwconvs]
        mixed = sum(
            weights[:, index : index + 1] * branch_output
            for index, branch_output in enumerate(branch_outputs)
        )
        # Multiplication by K makes equal routing exactly match the original sum.
        mixed = mixed * float(self.branch_count)
        mixed = channel_shuffle(mixed, gcd(mixed.shape[1], self.out_channels))
        output = self.shortcut(feature) + self.pconv2(mixed)
        return output, routing_logits, uncertainty, weights


class DisagreementGuidedEMCAD(nn.Module):
    """Drop-in decoder that preserves EMCAD's [d4,d3,d2,d1] contract."""

    def __init__(
        self,
        channels: Sequence[int],
        num_classes: int,
        kernel_sizes: Sequence[int] = (1, 3, 5),
        expansion_factor: int = 2,
        lgag_ks: int = 3,
        activation: str = "relu6",
        router_mode: str = "disagreement",
        disagreement_lambda: float = 1.0,
        router_temperature: float = 1.0,
        router_hidden: int = 32,
    ) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError("channels must contain [C4,C3,C2,C1]")
        c4, c3, c2, c1 = channels

        common = dict(
            num_classes=num_classes,
            kernel_sizes=kernel_sizes,
            expansion_factor=expansion_factor,
            activation=activation,
            router_mode=router_mode,
            disagreement_lambda=disagreement_lambda,
            router_temperature=router_temperature,
            router_hidden=router_hidden,
        )
        self.mscb4 = DisagreementGuidedMSCB(c4, c4, **common)
        self.mscb3 = DisagreementGuidedMSCB(c3, c3, **common)
        self.mscb2 = DisagreementGuidedMSCB(c2, c2, **common)
        self.mscb1 = DisagreementGuidedMSCB(c1, c1, **common)

        self.eucb3 = EUCB(c4, c3, kernel_size=3, stride=1)
        self.eucb2 = EUCB(c3, c2, kernel_size=3, stride=1)
        self.eucb1 = EUCB(c2, c1, kernel_size=3, stride=1)
        self.lgag3 = LGAG(c3, c3, c3 // 2, kernel_size=lgag_ks, groups=c3 // 2)
        self.lgag2 = LGAG(c2, c2, c2 // 2, kernel_size=lgag_ks, groups=c2 // 2)
        self.lgag1 = LGAG(c1, c1, c1 // 2, kernel_size=lgag_ks, groups=c1 // 2)
        self.cab4, self.cab3, self.cab2, self.cab1 = CAB(c4), CAB(c3), CAB(c2), CAB(c1)
        self.sab = SAB()

    def _attention(self, feature: torch.Tensor, cab: nn.Module) -> torch.Tensor:
        feature = cab(feature) * feature
        return self.sab(feature) * feature

    def forward(
        self, deepest: torch.Tensor, skips: Sequence[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], Dict[str, List[torch.Tensor]]]:
        if len(skips) != 3:
            raise ValueError("skips must be [x3,x2,x1]")

        route_logits: List[torch.Tensor] = []
        uncertainty_maps: List[torch.Tensor] = []
        route_weights: List[torch.Tensor] = []

        d4, q4, u4, a4 = self.mscb4(self._attention(deepest, self.cab4), None)
        route_logits.append(q4); uncertainty_maps.append(u4); route_weights.append(a4)

        d3 = self.eucb3(d4)
        d3 = d3 + self.lgag3(g=d3, x=skips[0])
        d3, q3, u3, a3 = self.mscb3(self._attention(d3, self.cab3), q4)
        route_logits.append(q3); uncertainty_maps.append(u3); route_weights.append(a3)

        d2 = self.eucb2(d3)
        d2 = d2 + self.lgag2(g=d2, x=skips[1])
        d2, q2, u2, a2 = self.mscb2(self._attention(d2, self.cab2), q3)
        route_logits.append(q2); uncertainty_maps.append(u2); route_weights.append(a2)

        d1 = self.eucb1(d2)
        d1 = d1 + self.lgag1(g=d1, x=skips[2])
        d1, q1, u1, a1 = self.mscb1(self._attention(d1, self.cab1), q2)
        route_logits.append(q1); uncertainty_maps.append(u1); route_weights.append(a1)

        auxiliary = {
            "routing_logits": route_logits,
            "uncertainty": uncertainty_maps,
            "routing_weights": route_weights,
        }
        return [d4, d3, d2, d1], auxiliary


class DGEMCADNet(EMCADNet):
    """EMCADNet with only its decoder replaced by disagreement-guided EMCAD."""

    def __init__(
        self,
        *args,
        router_mode: str = "disagreement",
        disagreement_lambda: float = 1.0,
        router_temperature: float = 1.0,
        router_hidden: int = 32,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        num_classes = self.out_head1.out_channels
        channels = [
            self.out_head4.in_channels,
            self.out_head3.in_channels,
            self.out_head2.in_channels,
            self.out_head1.in_channels,
        ]
        self.decoder = DisagreementGuidedEMCAD(
            channels=channels,
            num_classes=num_classes,
            kernel_sizes=kwargs.get("kernel_sizes", (1, 3, 5)),
            expansion_factor=kwargs.get("expansion_factor", 2),
            lgag_ks=kwargs.get("lgag_ks", 3),
            activation=kwargs.get("activation", "relu6"),
            router_mode=router_mode,
            disagreement_lambda=disagreement_lambda,
            router_temperature=router_temperature,
            router_hidden=router_hidden,
        )

    def forward(
        self,
        image: torch.Tensor,
        mode: str = "test",
        return_aux: bool = False,
    ):
        del mode  # Kept for compatibility with the original public interface.
        input_size = image.shape[-2:]
        if image.shape[1] == 1:
            image = self.conv(image)
        x1, x2, x3, x4 = self.backbone(image)
        decoded, auxiliary = self.decoder(x4, (x3, x2, x1))

        logits = [
            self.out_head4(decoded[0]),
            self.out_head3(decoded[1]),
            self.out_head2(decoded[2]),
            self.out_head1(decoded[3]),
        ]
        logits = [
            F.interpolate(item, size=input_size, mode="bilinear", align_corners=False)
            for item in logits
        ]
        if return_aux:
            return {"logits": logits, "adaptive_aux": auxiliary}
        return logits
