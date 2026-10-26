import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for decoder tests")
class LORFAblationTest(unittest.TestCase):
    def _decoder(self, caa, lorf):
        from lib.decoders import EMCAD

        return EMCAD(
            channels=[64, 32, 32, 32], kernel_sizes=[1, 3, 5],
            expansion_factor=2, caa_mode="caa" if caa else "off",
            use_lorf=lorf,
        ).eval()

    def _features(self):
        return torch.randn(1, 64, 2, 2), [
            torch.randn(1, 32, 4, 4),
            torch.randn(1, 32, 8, 8),
            torch.randn(1, 32, 16, 16),
        ]

    def test_all_four_switch_combinations(self):
        for caa in (False, True):
            for lorf in (False, True):
                with self.subTest(caa=caa, lorf=lorf):
                    decoder = self._decoder(caa, lorf)
                    keys = tuple(decoder.state_dict())
                    self.assertEqual(any(key.startswith("lorf1.") for key in keys), lorf)
                    self.assertEqual(
                        any("anti_alias_kernel" in key for key in keys), caa
                    )
                    with torch.no_grad():
                        outputs = decoder(*self._features())
                    self.assertEqual([tuple(x.shape) for x in outputs], [
                        (1, 64, 2, 2), (1, 32, 4, 4),
                        (1, 32, 8, 8), (1, 32, 16, 16),
                    ])

    def test_lorf_starts_as_identity_but_can_change_skip(self):
        torch.manual_seed(123)
        baseline = self._decoder(False, False)
        torch.manual_seed(123)
        enhanced = self._decoder(False, True)
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(value, enhanced.state_dict()[name], rtol=0, atol=0)
        x, skips = self._features()
        with torch.no_grad():
            base_outputs = baseline(x, skips)
            initial_outputs = enhanced(x, skips)
        for base, initial in zip(base_outputs, initial_outputs):
            torch.testing.assert_close(base, initial, rtol=0, atol=0)

        with torch.no_grad():
            enhanced.lorf1.project.weight[:, 0].fill_(0.5)
            modified = enhanced(x, skips)[-1]
        self.assertGreater((modified - base_outputs[-1]).abs().max().item(), 0)


if __name__ == "__main__":
    unittest.main()
