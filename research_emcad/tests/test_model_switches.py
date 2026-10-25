"""开关关闭时维持原始读出，开启时仅改写最高分辨率输出。"""
import io
import unittest

import torch
from torch.nn import functional as F

from lib.networks import EMCADNet


class ModelSwitchTests(unittest.TestCase):
    def test_kernel_selector_is_independent_and_starts_from_baseline(self):
        image = torch.randn(1, 3, 64, 64)
        for num_classes in (1, 4):
            torch.manual_seed(321)
            base = EMCADNet(num_classes=num_classes, encoder='pvt_v2_b0', pretrain=False).eval()
            torch.manual_seed(321)
            selected = EMCADNet(num_classes=num_classes, encoder='pvt_v2_b0', pretrain=False,
                                use_spatial_kernel_selection=True).eval()
            added = set(selected.state_dict()) - set(base.state_dict())
            self.assertEqual(len(added), 8)
            self.assertTrue(all('spatial_kernel_selection' in key for key in added))
            self.assertTrue(all(torch.equal(base.state_dict()[key], selected.state_dict()[key])
                                for key in base.state_dict()))
            with torch.no_grad():
                original, initial = base(image), selected(image)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(original, initial)))
            selected.train()
            selected(image.repeat(2, 1, 1, 1))[-1].square().mean().backward()
            selectors = [block[0].spatial_kernel_selection for block in
                         (selected.decoder.mscb4, selected.decoder.mscb3,
                          selected.decoder.mscb2, selected.decoder.mscb1)]
            self.assertTrue(all(p.grad is not None for selector in selectors
                                for p in selector.parameters()))
            self.assertTrue(any(selector.weight.grad.abs().sum().item() > 0
                                for selector in selectors))

    def test_joint_switch_preserves_existing_scale_only_initialization(self):
        torch.manual_seed(456)
        scale_only = EMCADNet(num_classes=4, encoder='pvt_v2_b0', pretrain=False,
                              use_class_conditional_scale_fusion=True).eval()
        torch.manual_seed(456)
        joint = EMCADNet(num_classes=4, encoder='pvt_v2_b0', pretrain=False,
                         use_class_conditional_scale_fusion=True,
                         use_spatial_kernel_selection=True).eval()
        self.assertTrue(all(torch.equal(scale_only.state_dict()[key], joint.state_dict()[key])
                            for key in scale_only.state_dict()))
        image = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(scale_only(image), joint(image))))

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
