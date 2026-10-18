"""在不复制 EMCAD 的前提下，按两个显式开关构造研究模型。"""

from torch import nn

from lib.full_resolution_detail import FullResolutionDetailBranch
from lib.networks import EMCADNet
from lib.semantic_gated_residual import SemanticGatedResidual


class DetailResearchWrapper(nn.Module):
    """保留四个 logits 契约，仅为最后一个 p1 加可独立消融的残差。"""

    def __init__(self, base_model, use_full_resolution_detail_branch=False, use_semantic_gated_residual=False):
        super().__init__()
        if not (use_full_resolution_detail_branch or use_semantic_gated_residual):
            raise ValueError("use EMCADNet directly when both modules are off")
        self.base_model = base_model
        num_classes = base_model.out_head1.out_channels
        d1_channels = base_model.out_head1.in_channels
        self.full_resolution_detail_branch = FullResolutionDetailBranch(num_classes) if use_full_resolution_detail_branch else None
        self.semantic_gated_residual = SemanticGatedResidual(d1_channels, num_classes) if use_semantic_gated_residual else None

    def forward(self, image, mode="test"):
        outputs, d1 = self.base_model(image, mode=mode, return_decoder_feature=True)
        if outputs[-1].shape[-2:] != image.shape[-2:]:
            raise ValueError("base p1 resolution differs from input; use a size supported by EMCAD")
        correction = None
        if self.full_resolution_detail_branch is not None:
            correction = self.full_resolution_detail_branch(image)
        if self.semantic_gated_residual is not None:
            part_b = self.semantic_gated_residual(image, d1)
            correction = part_b if correction is None else correction + part_b
        # p4/p3/p2 和训练损失的组合规则保持不变。
        return list(outputs[:-1]) + [outputs[-1] + correction]


def build_research_model(use_full_resolution_detail_branch=0, use_semantic_gated_residual=0, **base_kwargs):
    """0/0 直接返回原始 EMCADNet，参数名和 forward 路径完全一致。"""
    if use_full_resolution_detail_branch not in (0, 1) or use_semantic_gated_residual not in (0, 1):
        raise ValueError("use_full_resolution_detail_branch and use_semantic_gated_residual must be 0 or 1")
    base_model = EMCADNet(**base_kwargs)
    if use_full_resolution_detail_branch == 0 and use_semantic_gated_residual == 0:
        return base_model
    return DetailResearchWrapper(base_model, bool(use_full_resolution_detail_branch), bool(use_semantic_gated_residual))
