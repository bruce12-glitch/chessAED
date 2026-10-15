"""Phase A diagnostics gates."""
from __future__ import annotations

import unittest

from evaluation.diagnostics import (bad_minimum_event, entropy_slope, escaped,
                                    repeat_rate)


class DiagnosticsTests(unittest.TestCase):
    def test_repeat_rate(self) -> None:
        self.assertAlmostEqual(repeat_rate([7, 7, 7, 3]), 0.75)
        self.assertAlmostEqual(repeat_rate([1, 2, 3, 4]), 0.25)
        self.assertEqual(repeat_rate([]), 0.0)

    def test_entropy_slope(self) -> None:
        self.assertAlmostEqual(entropy_slope([2.0, 2.0, 2.0]), 0.0)
        self.assertLess(abs(entropy_slope([3.0, 2.0, 1.0])), 2.0)

    def test_event_fires_on_plateau(self) -> None:
        self.assertTrue(bad_minimum_event(
            [0.10, 0.09, 0.08], [2.5, 2.51, 2.50],
            [0.30, 0.25, 0.20], [0.6, 0.7, 0.8]))

    def test_event_quiet_when_strong(self) -> None:
        self.assertFalse(bad_minimum_event(
            [0.10, 0.09, 0.08], [2.5, 2.51, 2.50],
            [0.30, 0.60, 0.20], [0.6, 0.7, 0.8]))

    def test_event_quiet_when_diverse(self) -> None:
        self.assertFalse(bad_minimum_event(
            [0.10, 0.09, 0.08], [2.5, 2.51, 2.50],
            [0.30, 0.25, 0.20], [0.6, 0.2, 0.8]))

    def test_escape_needs_hold(self) -> None:
        self.assertTrue(escaped(0.30, [0.45, 0.46]))
        self.assertFalse(escaped(0.30, [0.45, 0.40]))
        self.assertFalse(escaped(0.30, [0.45]))


if __name__ == "__main__":
    unittest.main()
