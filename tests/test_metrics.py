"""Metrics + aggregator gates."""
from __future__ import annotations

import unittest

from evaluation.metrics import bootstrap_ci, plateau_detect, wilson


class MetricTests(unittest.TestCase):
    def test_wilson_brackets_rate(self) -> None:
        lo, hi = wilson(9, 12)
        self.assertLess(lo, 0.75)
        self.assertGreater(hi, 0.75)

    def test_bootstrap_brackets_mean(self) -> None:
        lo, hi = bootstrap_ci([1.0] * 6 + [0.0] * 6, iters=200, seed=0)
        self.assertLessEqual(lo, 0.5)
        self.assertGreaterEqual(hi, 0.5)

    def test_plateau_flat_vs_improving(self) -> None:
        self.assertTrue(plateau_detect([0.9, 0.91, 0.905, 0.9]))
        self.assertFalse(plateau_detect([1.2, 1.0, 0.8, 0.6]))


if __name__ == "__main__":
    unittest.main()
