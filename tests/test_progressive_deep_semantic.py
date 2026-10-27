"""Decoder ablation checks; run with a PyTorch environment."""

import sys
import types
import unittest

import torch


try:
    import timm  # noqa: F401
except ImportError:
    # The decoder uses only these two timm initialization helpers.
    timm = types.ModuleType("timm")
    models = types.ModuleType("timm.models")
    layers = types.ModuleType("timm.models.layers")
    helpers = types.ModuleType("timm.models.helpers")
    layers.trunc_normal_tf_ = torch.nn.init.trunc_normal_

    def named_apply(fn, module, name="", depth_first=True, include_root=False):
        for child_name, child in module.named_children():
            named_apply(fn, child, child_name, depth_first, True)
        if include_root:
            fn(module, name)
        return module

    helpers.named_apply = named_apply
    timm.models = models
    models.layers = layers
    models.helpers = helpers
    sys.modules.update({
        "timm": timm, "timm.models": models,
        "timm.models.layers": layers, "timm.models.helpers": helpers,
    })

from lib.decoders import EMCAD


class ProgressiveSemanticAblationTest(unittest.TestCase):
    def test_disabled_has_legacy_state_and_enabled_is_independent_of_caa(self):
        torch.manual_seed(17)
        channels = [64, 32, 16, 16]
        x = torch.randn(1, 64, 2, 2)
        skips = [torch.randn(1, 32, 4, 4),
                 torch.randn(1, 16, 8, 8),
                 torch.randn(1, 16, 16, 16)]
        for caa_mode in ("off", "caa"):
            with self.subTest(caa_mode=caa_mode):
                baseline = EMCAD(channels=channels, expansion_factor=2,
                                 caa_mode=caa_mode, use_progressive_deep_semantic=False).eval()
                enhanced = EMCAD(channels=channels, expansion_factor=2,
                                 caa_mode=caa_mode, use_progressive_deep_semantic=True).eval()
                self.assertFalse(any("progressive_deep_semantic" in key
                                     for key in baseline.state_dict()))
                missing, unexpected = enhanced.load_state_dict(baseline.state_dict(), strict=False)
                self.assertFalse(unexpected)
                self.assertTrue(missing)
                self.assertTrue(all(key.startswith("progressive_deep_semantic.")
                                    for key in missing))
                with torch.no_grad():
                    enhanced.progressive_deep_semantic.strength.zero_()
                    original = baseline(x, skips)
                    identity = enhanced(x, skips)
                    for expected, actual in zip(original, identity):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    enhanced.progressive_deep_semantic.strength.fill_(0.1)
                    modified = enhanced(x, skips)
                    self.assertEqual([tuple(t.shape) for t in original],
                                     [tuple(t.shape) for t in modified])

    def test_semantic_path_receives_gradients(self):
        model = EMCAD(channels=[64, 32, 16, 16], expansion_factor=2,
                      use_progressive_deep_semantic=True).eval()
        x = torch.randn(1, 64, 2, 2)
        skips = [torch.randn(1, 32, 4, 4), torch.randn(1, 16, 8, 8),
                 torch.randn(1, 16, 16, 16)]
        model(x, skips)[-1].mean().backward()
        self.assertIsNotNone(model.progressive_deep_semantic.compress.weight.grad)
        self.assertIsNotNone(model.progressive_deep_semantic.strength.grad)

    def test_supported_encoder_channel_layouts(self):
        for channels in ([256, 160, 64, 32], [512, 320, 128, 64],
                         [512, 256, 128, 64], [2048, 1024, 512, 256]):
            with self.subTest(channels=channels):
                module = EMCAD(channels=channels, expansion_factor=2,
                               use_progressive_deep_semantic=True).eval()
                x = torch.randn(1, channels[0], 2, 2)
                skips = [torch.randn(1, channel, 2 ** (index + 2), 2 ** (index + 2))
                         for index, channel in enumerate(channels[1:])]
                with torch.no_grad():
                    outputs = module(x, skips)
                self.assertEqual([tuple(output.shape) for output in outputs],
                                 [(1, channel, 2 ** (index + 1), 2 ** (index + 1))
                                  for index, channel in enumerate(channels)])


if __name__ == "__main__":
    unittest.main()
