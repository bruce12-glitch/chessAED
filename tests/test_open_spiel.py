from __future__ import annotations

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("pyspiel"), "OpenSpiel is not installed")
class OpenSpielAdapterTests(unittest.TestCase):
    def test_registration_and_conformance(self) -> None:
        import gochess.open_spiel_game  # noqa: F401
        import pyspiel

        game = pyspiel.load_game(
            "python_gochess(size=5,stone_supply=4,max_plies=30,komi=1.5)"
        )
        self.assertEqual(game.num_distinct_actions(), 651)
        self.assertEqual(game.observation_tensor_shape(), [11, 5, 5])
        pyspiel.random_sim_test(
            game,
            num_sims=25,
            serialize=False,
            verbose=False,
        )


if __name__ == "__main__":
    unittest.main()
