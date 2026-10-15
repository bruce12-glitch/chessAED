from __future__ import annotations

import unittest

from gochess.evaluate import play_game, wilson_interval
from gochess.scenarios import scenarios
from gochess.search import RandomAgent, UCTAgent, solve_exact


class SearchTests(unittest.TestCase):
    def test_exact_solver_finds_royal_capture(self) -> None:
        state = scenarios()["royal_capture_in_one"].state
        result = solve_exact(state)
        self.assertEqual(result.value, 1.0)
        self.assertEqual(state.format_action(result.best_action), "a3-e3")

    def test_uct_finds_royal_capture(self) -> None:
        state = scenarios()["royal_capture_in_one"].state
        action = UCTAgent(simulations=64, seed=7).select_action(state)
        self.assertEqual(state.format_action(action), "a3-e3")

    def test_seeded_game_terminates(self) -> None:
        state = scenarios()["initial_5x5"].state
        terminal, plies = play_game(state, RandomAgent(seed=1), RandomAgent(seed=2))
        self.assertTrue(terminal.is_terminal())
        self.assertLessEqual(plies, state.rules.max_plies)

    def test_wilson_interval_contains_observed_rate(self) -> None:
        low, high = wilson_interval(9, 12)
        self.assertLess(low, 0.75)
        self.assertGreater(high, 0.75)


if __name__ == "__main__":
    unittest.main()
