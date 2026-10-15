"""Hard validation gates: action masks, MCTS invariants, reproducibility."""
from __future__ import annotations

import unittest

from gochess.core import GoChessState
from gochess.masks import legal_mask, masked_policy
from gochess.scenarios import scenarios
from gochess.search import UCTAgent


class MaskTests(unittest.TestCase):
    def test_mask_matches_legal_ids(self) -> None:
        for name in ("initial_5x5", "mixed_midgame", "encirclement_choice"):
            state = scenarios()[name].state
            mask = legal_mask(state)
            self.assertEqual(len(mask), state.num_distinct_actions)
            self.assertEqual(sum(mask), len(state.legal_action_ids()))
            for aid in state.legal_action_ids():
                self.assertEqual(mask[aid], 1)

    def test_terminal_mask_empty(self) -> None:
        from gochess.core import Action

        state = scenarios()["royal_capture_in_one"].state.apply(Action.move(10, 14))
        self.assertTrue(state.is_terminal())
        self.assertEqual(sum(legal_mask(state)), 0)
        self.assertEqual(masked_policy(state, [0.0] * state.num_distinct_actions), [])

    def test_masked_policy_zero_on_illegal(self) -> None:
        state = scenarios()["mixed_midgame"].state
        n = state.num_distinct_actions
        probs = masked_policy(state, [0.5] * n)
        self.assertAlmostEqual(sum(probs), 1.0)
        mask = legal_mask(state)
        for p, m in zip(probs, mask):
            if m == 0:
                self.assertEqual(p, 0.0)


class MCTSInvariantTests(unittest.TestCase):
    def test_visit_counts_sum_to_simulations(self) -> None:
        state = scenarios()["mixed_midgame"].state
        sims = 32
        root = UCTAgent(simulations=sims, seed=0).search(state)
        self.assertEqual(root.visits, sims)
        self.assertEqual(sum(c.visits for c in root.children), sims)

    def test_search_does_not_mutate_state(self) -> None:
        state = scenarios()["mixed_midgame"].state
        before = (state.board_signature(), state.current_player, state.reserves, state.ply)
        UCTAgent(simulations=16, seed=1).search(state)
        after = (state.board_signature(), state.current_player, state.reserves, state.ply)
        self.assertEqual(before, after)

    def test_same_seed_same_action(self) -> None:
        state = scenarios()["mixed_midgame"].state
        a = UCTAgent(simulations=32, seed=7).select_action(state)
        b = UCTAgent(simulations=32, seed=7).select_action(state)
        self.assertEqual(a, b)

    def test_checkpoint_reload_same_prediction(self) -> None:
        import torch

        from gochess.nn import GoChessNet, obs_to_tensor

        net = GoChessNet()
        net.eval()
        state = scenarios()["mixed_midgame"].state
        x = obs_to_tensor(state).unsqueeze(0)
        with torch.no_grad():
            p0, v0 = net(x)
        sd = {k: v.clone() for k, v in net.state_dict().items()}
        net2 = GoChessNet()
        net2.load_state_dict(sd)
        net2.eval()
        with torch.no_grad():
            p1, v1 = net2(x)
        self.assertTrue(bool((p0 - p1).abs().max() < 1e-6))
        self.assertTrue(bool((v0 - v1).abs().max() < 1e-6))


if __name__ == "__main__":
    unittest.main()
