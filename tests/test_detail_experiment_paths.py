"""四种消融配置必须落到不同目录，且同配置并发运行不覆盖。"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from lib.experiment_paths import make_experiment_dir


class ExperimentPathTests(unittest.TestCase):
    def test_four_variants_and_collision_avoidance(self):
        with tempfile.TemporaryDirectory() as root:
            created = {}
            for a, b, label in ((0, 0, "baseline"), (1, 0, "full_resolution_detail"),
                                (0, 1, "semantic_gated_residual"), (1, 1, "detail_and_semantic_gate")):
                args = SimpleNamespace(output_dir=root, dataset_name="ClinicDB",
                                       use_full_resolution_detail_branch=a, use_semantic_gated_residual=b, seed=2222)
                path = Path(make_experiment_dir(args, "Polyp"))
                self.assertTrue(path.is_dir())
                self.assertIn("variant_" + label, path.parts)
                created[label] = path
            self.assertEqual(len(set(created.values())), 4)
            same = Path(make_experiment_dir(SimpleNamespace(
                output_dir=root, dataset_name="ClinicDB", use_full_resolution_detail_branch=0,
                use_semantic_gated_residual=0, seed=2222), "Polyp"))
            self.assertNotEqual(same, created["baseline"])


if __name__ == "__main__":
    unittest.main()
