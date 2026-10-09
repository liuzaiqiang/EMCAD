import ast
import itertools
from pathlib import Path
from types import SimpleNamespace
import unittest

from utils.supervision_weights import supervision_group_weights


ROOT = Path(__file__).resolve().parents[1]


def load_functions(relative_path, names, namespace):
    """Execute the actual loss functions without importing model/GPU dependencies."""
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(nodes) != len(names):
        raise AssertionError("Missing loss function in " + relative_path)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), relative_path, "exec"), namespace)
    return namespace


class ScalarTarget:
    def new_tensor(self, value, **kwargs):
        return float(value)

    def long(self):
        return self

    def __getitem__(self, key):
        return self


class MutationLossRegressionTest(unittest.TestCase):
    """Check production loss aggregation against an independent scalar reference."""

    def groups(self, supervision):
        singles = [[i] for i in range(4)]
        if supervision == "mutation":
            return [list(g) for k in range(1, 5) for g in itertools.combinations(range(4), k)]
        if supervision == "paper":
            return singles + [[0, 1, 2, 3]]
        return singles

    def test_group_weight_budget_and_bounds(self):
        # Highly unequal weights expose wrong group indices and incorrect normalization.
        weights = [0.8, 0.8, 0.8, 1.6]
        for mode, count in (("mutation", 15), ("deep_supervision", 4), ("paper", 5)):
            groups = self.groups(mode)
            values = supervision_group_weights(weights, groups, mode, 4)
            self.assertEqual(len(values), count)
            self.assertAlmostEqual(sum(values), count)
            self.assertTrue(all(0.8 <= value <= 1.6 for value in values))
        empty_groups = [[]] + self.groups("mutation")
        self.assertEqual(supervision_group_weights(weights, empty_groups, "mutation", 4)[0], 0.0)

    def test_invalid_weighting_is_rejected(self):
        with self.assertRaises(ValueError):
            supervision_group_weights([1] * 4, [[3]], "last_layer", 4)
        with self.assertRaises(ValueError):
            supervision_group_weights([1] * 3, [[0]], "mutation", 4)

    def test_binary_and_acdc_preserve_groups_and_disabled_loss(self):
        outputs = [0.2, -0.7, 1.1, 0.5]
        target = ScalarTarget()
        base = {"itertools": itertools, "supervision_group_weights": supervision_group_weights}
        binary = load_functions("utils/polyp_utils.py", {"supervised_structure_loss"},
                                dict(base, structure_loss=lambda x, _: x * x + 1))
        acdc = load_functions("utils/acdc_utils.py", {"_supervision_groups", "supervised_loss"},
                              dict(base, torch=SimpleNamespace(float32="float32")))
        for mode in ("mutation", "deep_supervision", "paper"):
            groups = self.groups(mode)
            for weights in (None, [1.0] * 4, [0.8, 0.9, 1.1, 1.2]):
                expected = sum(
                    (sum(outputs[i] for i in group) ** 2 + 1)
                    * (sum(weights[i] for i in group) / len(group) if weights else 1)
                    for group in groups
                )
                result = binary["supervised_structure_loss"](outputs, target, mode, weights)
                self.assertAlmostEqual(result, expected)
                if mode != "paper":
                    # Identical synthetic CE/Dice isolates aggregation while invoking real code.
                    loss_fn = lambda x, _: x * x + 1
                    result = acdc["supervised_loss"](outputs, target, mode, loss_fn, loss_fn, weights)
                    self.assertAlmostEqual(result, expected)

    def test_synapse_actual_loss_loop_handles_empty_mutation_subset(self):
        tree = ast.parse((ROOT / "trainer.py").read_text(encoding="utf-8"))
        loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)
                 and ast.unparse(node.target) == "(group_index, s)"
                 and ast.unparse(node.iter) == "enumerate(ss)"]
        self.assertEqual(len(loops), 1)
        code = compile(ast.Module(body=loops, type_ignores=[]), "trainer.py", "exec")
        outputs = [0.2, -0.7, 1.1, 0.5]
        groups = [[]] + self.groups("mutation")
        for weights in (None, [1.0] * 4, [0.8, 0.9, 1.1, 1.2]):
            context = dict(P=outputs, ss=groups, loss=0.0, w_ce=0.3, w_dice=0.7,
                           label_batch=ScalarTarget(), scale_weights=weights,
                           group_weights=supervision_group_weights(weights, groups, "mutation", 4),
                           ce_loss=lambda x, _: x * x + 1,
                           dice_loss=lambda x, _, **kwargs: x * x + 1)
            exec(code, context)
            expected = sum((sum(outputs[i] for i in group) ** 2 + 1)
                           * (sum(weights[i] for i in group) / len(group) if weights else 1)
                           for group in groups if group)
            self.assertAlmostEqual(context["loss"], expected)

try:
    import torch
except ImportError:  # The repository's training environment provides PyTorch.
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for this test")
class UncertaintyScaleWeighterTest(unittest.TestCase):
    def test_mutation_weighted_loss_backpropagates_to_all_heads(self):
        from utils.uncertainty_supervision import UncertaintyScaleWeighter

        outputs = [torch.randn(2, 1, 8, 8, requires_grad=True) for _ in range(4)]
        groups = [list(g) for k in range(1, 5) for g in itertools.combinations(range(4), k)]
        weights = UncertaintyScaleWeighter().update(outputs)
        mapped = supervision_group_weights(weights, groups, "mutation", 4)
        self.assertAlmostEqual(float(torch.stack(mapped).sum()), 15.0, places=4)
        self.assertTrue(all(not weight.requires_grad for weight in mapped))
        # Use production binary aggregation with real structure loss and tensors.
        namespace = load_functions("utils/polyp_utils.py",
                                   {"structure_loss", "supervised_structure_loss"},
                                   dict(torch=torch, F=torch.nn.functional, itertools=itertools,
                                        supervision_group_weights=supervision_group_weights))
        mask = torch.randint(0, 2, (2, 1, 8, 8)).float()
        loss = namespace["supervised_structure_loss"](outputs, mask, "mutation", weights)
        loss.backward()
        for output in outputs:
            self.assertIsNotNone(output.grad)
            self.assertTrue(torch.isfinite(output.grad).all())
            self.assertGreater(float(output.grad.abs().sum()), 0)

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
