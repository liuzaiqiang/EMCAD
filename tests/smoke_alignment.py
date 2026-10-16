"""Quick CPU/GPU independent shape, backward, and ablation-switch check.

Run from the repository root: python tests/smoke_alignment.py
Requires the same PyTorch/timm environment as the training scripts; no dataset or pretrained weights.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lib.decoders import EMCAD, SkipSpatialAlignment
from lib.networks import EMCADNet


def main():
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    channels = [64, 32, 16, 8]
    for alignment in (0, 1):
        for consistency in (0, 1):
            decoder = EMCAD(channels=channels, expansion_factor=2, activation='relu6',
                            use_skip_alignment=alignment,
                            use_bidirectional_consistency=consistency)
            bottom = torch.randn(2, 64, 2, 2, requires_grad=True)
            skips = [torch.randn(2, 32, 4, 4), torch.randn(2, 16, 8, 8),
                     torch.randn(2, 8, 16, 16)]
            features, auxiliary = decoder(bottom, skips, return_consistency=True)
            assert [f.shape[-2:] for f in features] == [(2, 2), (4, 4), (8, 8), (16, 16)]
            assert torch.isfinite(auxiliary).all()
            (sum(x.mean() for x in features) + 0.05 * auxiliary.mean()).backward()
            assert bottom.grad is not None and torch.isfinite(bottom.grad).all()
            if not consistency:
                assert auxiliary.item() == 0
            if not alignment:
                assert decoder.skip_aligners is None
            print('alignment={} consistency={} passed'.format(alignment, consistency))

    aligner = SkipSpatialAlignment(8)
    decoder = torch.randn(2, 8, 8, 8, requires_grad=True)
    skip = torch.randn(2, 8, 8, 8)
    aligned, _, flow = aligner(decoder, skip)
    assert flow.abs().max().item() == 0
    assert torch.allclose(aligned, skip, atol=1e-6)
    aligned.square().mean().backward()
    assert aligner.offset[-1].weight.grad.abs().sum().item() > 0

    # Whole-network smoke checks both single foreground logit and multiclass logits.
    for classes in (1, 9):
        model = EMCADNet(num_classes=classes, encoder='pvt_v2_b0', pretrain=False,
                         activation='relu6', use_skip_alignment=True,
                         use_bidirectional_consistency=True).eval()
        with torch.no_grad():
            outputs = model(torch.randn(1, 1, 64, 64))
        assert len(outputs) == 4
        assert all(output.shape == (1, classes, 64, 64) for output in outputs)
        print('classes={} full network passed'.format(classes))


if __name__ == '__main__':
    main()
