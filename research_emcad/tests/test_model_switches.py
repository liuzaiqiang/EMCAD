import io
import unittest
import torch

from lib.networks import EMCADNet


class ModelSwitchTests(unittest.TestCase):
    def test_four_combinations_and_baseline_state(self):
        image = torch.randn(1, 3, 128, 128)
        for prototype in (False, True):
            for fusion in (False, True):
                torch.manual_seed(123)
                model = EMCADNet(num_classes=4, encoder='pvt_v2_b0', pretrain=False,
                                 use_class_prototype_query=prototype,
                                 use_class_conditional_scale_fusion=fusion).eval()
                with torch.no_grad():
                    outputs = model(image)
                self.assertEqual(len(outputs), 4)
                self.assertTrue(all(x.shape == (1, 4, 128, 128) for x in outputs))
                names = list(model.state_dict())
                self.assertEqual(any(n.startswith('class_prototype_query.') for n in names), prototype)
                self.assertEqual(any(n.startswith('class_conditional_scale_fusion.') for n in names), fusion)
                if not prototype and not fusion:
                    torch.manual_seed(123)
                    original = EMCADNet(num_classes=4, encoder='pvt_v2_b0', pretrain=False).eval()
                    original.load_state_dict(model.state_dict(), strict=True)
                    with torch.no_grad():
                        reference = original(image)
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(outputs, reference)))

    def test_optimizer_and_checkpoint(self):
        model = EMCADNet(num_classes=1, encoder='pvt_v2_b0', pretrain=False,
                         use_class_prototype_query=True,
                         use_class_conditional_scale_fusion=True).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        output = model(torch.randn(2, 3, 128, 128))[-1]
        output.square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in model.class_prototype_query.parameters()))
        self.assertTrue(all(p.grad is not None for p in model.class_conditional_scale_fusion.parameters()))
        optimizer.step()
        state = io.BytesIO()
        torch.save(model.state_dict(), state)
        state.seek(0)
        model.load_state_dict(torch.load(state, map_location='cpu'), strict=True)


if __name__ == '__main__':
    unittest.main()
