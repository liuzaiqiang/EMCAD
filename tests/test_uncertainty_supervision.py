import unittest

try:
    import torch
except ImportError:  # The repository's training environment provides PyTorch.
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for this test")
class UncertaintyScaleWeighterTest(unittest.TestCase):
    def test_weights_are_bounded_and_sum_to_four(self):
        from utils.uncertainty_supervision import UncertaintyScaleWeighter

        outputs = [torch.randn(2, 1, 16, 16) for _ in range(4)]
        weights = UncertaintyScaleWeighter().update(outputs)

        self.assertEqual(tuple(weights.shape), (4,))
        self.assertTrue(torch.all(weights >= 0.8))
        self.assertTrue(torch.all(weights <= 1.6))
        self.assertAlmostEqual(float(weights.sum()), 4.0, places=5)

    def test_update_does_not_attach_prediction_graph(self):
        from utils.uncertainty_supervision import UncertaintyScaleWeighter

        outputs = [torch.randn(1, 2, 8, 8, requires_grad=True) for _ in range(4)]
        weights = UncertaintyScaleWeighter().update(outputs)

        self.assertFalse(weights.requires_grad)


if __name__ == "__main__":
    unittest.main()
