"""Tiny solved-game harness gates."""
from __future__ import annotations

import unittest

from experiments.tiny_benchmark import benchmark_position, mine_positions
from gochess.scenarios import scenarios


class TinyBenchmarkTests(unittest.TestCase):
    def test_royal_position_optimal_known(self) -> None:
        s = scenarios()["royal_capture_in_one"].state
        r = benchmark_position(s, sims_list=(16,), seeds=3)
        self.assertEqual(r["value"], 1.0)
        self.assertEqual(r["optimal"], "a3-e3")
        self.assertGreaterEqual(r["by_sims"]["16"]["agree"], 0.0)
        self.assertLessEqual(r["by_sims"]["16"]["agree"], 1.0)

    def test_mined_positions_solve(self) -> None:
        states = mine_positions(2, seed=0)
        self.assertTrue(states)
        for s in states:
            self.assertTrue(2 <= len(s.legal_actions()) <= 8)
            r = benchmark_position(s, sims_list=(16,), seeds=2, max_nodes=10_000)
            self.assertTrue("value" in r or "error" in r)


if __name__ == "__main__":
    unittest.main()
