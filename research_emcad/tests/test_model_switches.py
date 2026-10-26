"""开关关闭时维持原始读出，开启时仅改写最高分辨率输出。"""
import io
import unittest

import torch
from torch.nn import functional as F

from lib.networks import EMCADNet


class ModelSwitchTests(unittest.TestCase):
    def test_covariance_switch_preserves_baseline_and_reaches_final_head(self):
        torch.manual_seed(619)
        baseline = EMCADNet(num_classes=1, encoder='pvt_v2_b0', pretrain=False).eval()
        self.assertIsNone(baseline.decoder.low_rank_covariance4)
        self.assertIsNone(baseline.decoder.low_rank_covariance3)
        torch.manual_seed(619)
        covariance = EMCADNet(num_classes=1, encoder='pvt_v2_b0', pretrain=False,
                              use_low_rank_covariance=True).eval()
        base_state, cov_state = baseline.state_dict(), covariance.state_dict()
        self.assertEqual(len(set(cov_state) - set(base_state)), 4)
        self.assertTrue(all(torch.equal(value, cov_state[key]) for key, value in base_state.items()))
        image = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            self.assertTrue(all(torch.equal(a, b) for a, b in
                                zip(baseline(image), covariance(image))))
        covariance.train()
        covariance(image.repeat(2, 1, 1, 1))[-1].square().mean().backward()
        for module in (covariance.decoder.low_rank_covariance4,
                       covariance.decoder.low_rank_covariance3):
            self.assertIsNotNone(module.expand.grad)
            self.assertGreater(module.expand.grad.abs().sum().item(), 0)

    def test_covariance_joint_starts_from_scale_only(self):
        torch.manual_seed(701)
        scale_only = EMCADNet(num_classes=4, encoder='pvt_v2_b0', pretrain=False,
                              use_class_conditional_scale_fusion=True).eval()
        torch.manual_seed(701)
        joint = EMCADNet(num_classes=4, encoder='pvt_v2_b0', pretrain=False,
                         use_class_conditional_scale_fusion=True,
                         use_low_rank_covariance=True).eval()
        self.assertTrue(all(torch.equal(value, joint.state_dict()[key])
                            for key, value in scale_only.state_dict().items()))
        image = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            self.assertTrue(all(torch.equal(a, b) for a, b in
                                zip(scale_only(image), joint(image))))

    def test_off_matches_original_heads(self):
        torch.manual_seed(123)
        model = EMCADNet(num_classes=4, encoder='pvt_v2_b0', pretrain=False,
                         use_class_conditional_scale_fusion=False).eval()
        self.assertIsNone(model.class_conditional_scale_fusion)
        self.assertFalse(any(n.startswith('class_conditional_scale_fusion.')
                             for n in model.state_dict()))
        image = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            actual = model(image)
            x1, x2, x3, x4 = model.backbone(image)
            decoded = model.decoder(x4, [x3, x2, x1])
            expected = [F.interpolate(head(feature), scale_factor=scale, mode='bilinear')
                        for head, feature, scale in zip(
                            (model.out_head4, model.out_head3, model.out_head2, model.out_head1),
                            decoded, (32, 16, 8, 4))]
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected)))
        self.assertTrue(all(x.shape == (1, 4, 128, 128) for x in actual))

    def test_on_preserves_other_heads_and_restores_checkpoint(self):
        torch.manual_seed(123)
        base = EMCADNet(num_classes=1, encoder='pvt_v2_b0', pretrain=False).eval()
        torch.manual_seed(123)
        model = EMCADNet(num_classes=1, encoder='pvt_v2_b0', pretrain=False,
                         use_class_conditional_scale_fusion=True).eval()
        self.assertEqual(set(model.state_dict()) - set(base.state_dict()),
                         {f'class_conditional_scale_fusion.{key}'
                          for key in model.class_conditional_scale_fusion.state_dict()})
        image = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            reference, actual = base(image), model(image)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(reference[:3], actual[:3])))
        self.assertFalse(torch.equal(reference[-1], actual[-1]))
        model.train()
        model(image.repeat(2, 1, 1, 1))[-1].square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in model.class_conditional_scale_fusion.parameters()))
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        model.load_state_dict(torch.load(buffer, map_location='cpu'), strict=True)


if __name__ == '__main__':
    unittest.main()
