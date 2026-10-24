"""在已安装项目依赖的环境中运行：python -m unittest discover research_emcad/tests。"""
import io
import unittest

import torch

from research_emcad.class_prototype_query import ClassPrototypeQuery
from research_emcad.class_conditional_scale_fusion import ClassConditionalScaleFusion


class ModuleTests(unittest.TestCase):
    def test_independent_forward_backward_and_restore(self):
        for size in (8, 12):
            features = [torch.randn(2, c, size * factor, size * factor, requires_grad=True)
                        for c, factor in zip((32, 16, 8), (1, 2, 4))]
            logits = [torch.randn(2, 4, *x.shape[-2:], requires_grad=True) for x in features]
            prototype_query = ClassPrototypeQuery((32, 16, 8), 4, 12)
            scale_fusion = ClassConditionalScaleFusion((32, 16, 8), 4, 12)
            for training in (True, False):
                prototype_query.train(training)
                scale_fusion.train(training)
                correction, prototype = prototype_query(features)
                output = scale_fusion(features, logits, prototype)
                self.assertEqual(correction.shape, logits[-1].shape)
                self.assertEqual(output.shape, logits[-1].shape)
                self.assertEqual(prototype.shape, (2, 4, 12))
                self.assertTrue(torch.isfinite(output).all().item())
                fallback_output = scale_fusion(features, logits)
                (output.mean() + fallback_output.mean() + correction.mean()).backward()
                for module in (prototype_query, scale_fusion):
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all().item()
                                        for p in module.parameters()))
            buffer = io.BytesIO()
            torch.save(scale_fusion.state_dict(), buffer)
            buffer.seek(0)
            scale_fusion.load_state_dict(torch.load(buffer, map_location='cpu'))

    def test_scale_fusion_without_prototypes(self):
        features = [torch.randn(1, c, s, s) for c, s in ((32, 8), (16, 16), (8, 32))]
        logits = [torch.randn(1, 1, *x.shape[-2:]) for x in features]
        module = ClassConditionalScaleFusion((32, 16, 8), 1)
        self.assertEqual(module(features, logits).shape, logits[-1].shape)


if __name__ == '__main__':
    unittest.main()
