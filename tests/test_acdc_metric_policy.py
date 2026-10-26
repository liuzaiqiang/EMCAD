import math
import unittest

import numpy as np

from utils.acdc_utils import calculate_metric_percase


class ACDCMetricPolicyTest(unittest.TestCase):
    def test_false_positive_on_empty_ground_truth(self):
        empty = np.zeros((2, 8, 8), dtype=np.uint8)
        false_positive = empty.copy()
        false_positive[0, 1, 1] = 1
        result = calculate_metric_percase(false_positive, empty)
        self.assertEqual(result['dice'], 1.0)
        self.assertEqual(result['jaccard'], 1.0)
        self.assertTrue(math.isnan(result['hd95']))

    def test_both_empty(self):
        empty = np.zeros((2, 8, 8), dtype=np.uint8)
        result = calculate_metric_percase(empty, empty)
        self.assertEqual(result['dice'], 1.0)
        self.assertEqual(result['jaccard'], 1.0)
        self.assertEqual(result['hd95'], 0.0)

    def test_missed_foreground(self):
        empty = np.zeros((2, 8, 8), dtype=np.uint8)
        foreground = empty.copy()
        foreground[0, 1, 1] = 1
        result = calculate_metric_percase(empty, foreground)
        self.assertEqual(result['dice'], 0.0)
        self.assertTrue(math.isnan(result['hd95']))


if __name__ == '__main__':
    unittest.main()
