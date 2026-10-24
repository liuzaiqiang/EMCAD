"""类别条件尺度融合的独立前向、反向与参数恢复。"""
import io
import unittest

import torch

from research_emcad.class_conditional_scale_fusion import ClassConditionalScaleFusion


class ModuleTests(unittest.TestCase):
    def test_forward_backward_and_restore(self):
        for size in (8, 12):
            features = [torch.randn(2, c, size * factor, size * factor, requires_grad=True)
                        for c, factor in zip((32, 16, 8), (1, 2, 4))]
            logits = [torch.randn(2, 4, *x.shape[-2:], requires_grad=True) for x in features]
            module = ClassConditionalScaleFusion((32, 16, 8), 4, 12)
            for training in (True, False):
                module.train(training)
                output = module(features, logits)
                self.assertEqual(output.shape, logits[-1].shape)
                self.assertTrue(torch.isfinite(output).all().item())
                output.mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all().item()
                                    for p in module.parameters()))
            buffer = io.BytesIO()
            torch.save(module.state_dict(), buffer)
            buffer.seek(0)
            module.load_state_dict(torch.load(buffer, map_location='cpu'), strict=True)

    def test_binary_output(self):
        features = [torch.randn(1, c, s, s) for c, s in ((32, 8), (16, 16), (8, 32))]
        logits = [torch.randn(1, 1, *x.shape[-2:]) for x in features]
        module = ClassConditionalScaleFusion((32, 16, 8), 1)
        self.assertEqual(module(features, logits).shape, logits[-1].shape)


if __name__ == '__main__':
    unittest.main()
