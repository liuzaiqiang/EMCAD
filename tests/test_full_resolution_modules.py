"""有 PyTorch 的环境中运行：python -m unittest tests.test_full_resolution_modules。"""

import unittest

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None
if torch is not None:
    from lib.full_resolution_detail import FullResolutionDetailBranch
    from lib.networks import EMCADNet
    from lib.research_model import DetailResearchWrapper, build_research_model
    from lib.semantic_gated_residual import SemanticGatedResidual


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class FullResolutionModuleTests(unittest.TestCase):
    def test_component_shapes_and_initial_identity(self):
        for channels, classes in ((1, 9), (3, 1), (1, 4)):
            image = torch.randn(2, channels, 64, 96)
            d1 = torch.randn(2, 32, 16, 24)
            a = FullResolutionDetailBranch(classes)
            b = SemanticGatedResidual(32, classes)
            self.assertEqual(a(image).shape, (2, classes, 64, 96))
            self.assertEqual(b(image, d1).shape, (2, classes, 64, 96))
            self.assertTrue(torch.count_nonzero(a(image)) == 0)
            self.assertTrue(torch.count_nonzero(b(image, d1)) == 0)

    def test_invalid_spatial_contract(self):
        with self.assertRaises(ValueError):
            SemanticGatedResidual(32, 1)(torch.randn(1, 3, 64, 64),
                                          torch.randn(1, 32, 15, 16))

    def test_both_components_receive_gradients(self):
        image = torch.randn(2, 3, 64, 64)
        d1 = torch.randn(2, 32, 16, 16)
        a = FullResolutionDetailBranch(1)
        b = SemanticGatedResidual(32, 1)
        # 零初始化使第一次纯输出为零；用非零目标检查更新路径。
        loss = (a(image) + b(image, d1) - 1).square().mean()
        loss.backward()
        self.assertIsNotNone(a.to_logits.weight.grad)
        self.assertIsNotNone(b.detail_projection.weight.grad)
        self.assertGreater(a.to_logits.weight.grad.abs().sum().item(), 0)
        self.assertGreater(b.detail_projection.weight.grad.abs().sum().item(), 0)

    def test_all_switch_states_preserve_four_output_contract(self):
        image = torch.randn(1, 1, 64, 64)
        torch.manual_seed(11)
        baseline = build_research_model(use_full_resolution_detail_branch=0, use_semantic_gated_residual=0, num_classes=4,
                                        encoder="pvt_v2_b0", pretrain=False)
        baseline.eval()
        with torch.no_grad():
            reference = baseline(image)
        self.assertEqual(len(reference), 4)
        self.assertNotIsInstance(baseline, DetailResearchWrapper)
        self.assertIsInstance(baseline, EMCADNet)
        self.assertTrue(all(not key.startswith("base_model.") for key in baseline.state_dict()))
        for use_full_resolution_detail_branch, use_semantic_gated_residual in ((1, 0), (0, 1), (1, 1)):
            torch.manual_seed(11)
            model = build_research_model(use_full_resolution_detail_branch=use_full_resolution_detail_branch, use_semantic_gated_residual=use_semantic_gated_residual,
                                         num_classes=4, encoder="pvt_v2_b0", pretrain=False)
            model.eval()
            with torch.no_grad():
                outputs = model(image)
            self.assertEqual(len(outputs), 4)
            for actual, original in zip(outputs, reference):
                self.assertEqual(actual.shape, original.shape)
                self.assertTrue(torch.equal(actual, original))

    def test_switch_validation(self):
        with self.assertRaises(ValueError):
            build_research_model(use_full_resolution_detail_branch=2, use_semantic_gated_residual=0, pretrain=False)


if __name__ == "__main__":
    unittest.main()
