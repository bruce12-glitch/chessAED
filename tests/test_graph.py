"""Graph builder + action-index mapping (torch-free)."""
from __future__ import annotations

import random
import unittest

import numpy as np

from gochess import graph as G
from gochess.core import Action, ActionKind, GoChessState
from gochess.scenarios import scenarios


def _states():
    out = [s.state for s in scenarios().values()]
    rng = random.Random(7)
    s = GoChessState.initial()
    for _ in range(14):                       # a short seeded random walk
        if s.is_terminal():
            break
        s = s.apply(rng.choice(s.legal_actions()))
        out.append(s)
    return out


class GraphTests(unittest.TestCase):
    def test_deterministic_and_fresh_state_equal(self) -> None:
        for s in _states():
            a, b = G.pack_graph(s), G.pack_graph(s)
            self.assertTrue(np.array_equal(a, b))
        s1 = GoChessState.initial()
        s2 = GoChessState.initial()
        self.assertTrue(np.array_equal(G.pack_graph(s1), G.pack_graph(s2)))

    def test_shapes_and_roundtrip(self) -> None:
        for s in _states():
            p = G.pack_graph(s)
            self.assertEqual(p.shape, (25, G.packed_dim(5)))
            self.assertEqual(p.dtype, np.float32)
            x, e = G.unpack_graph(p, 5)
            x0, e0 = G.build_graph(s)
            self.assertTrue(np.array_equal(x, x0))
            self.assertTrue(np.array_equal(e, e0))

    def test_node_features_consistent_with_board(self) -> None:
        s = scenarios()["mixed_midgame"].state
        x, _ = G.build_graph(s)
        self.assertTrue(np.all(x[:, :5].sum(1) == 1.0))       # empty xor one piece type
        for i, p in enumerate(s.board):
            self.assertEqual(x[i, 0] == 1.0, int(p) == 0)
            if int(p) != 0:
                self.assertEqual(x[i, 5], 1.0 if int(p) > 0 else 0.0)
                self.assertEqual(x[i, 6], 1.0 if int(p) < 0 else 0.0)
        self.assertTrue(np.all(x[:, 7] == (1.0 if s.current_player == 0 else 0.0)))

    def test_edges_match_legal_actions(self) -> None:
        for s in _states():
            _, e = G.build_graph(s)
            moves = {(a.source, a.target) for a in s.legal_actions() if a.kind is ActionKind.MOVE}
            places = {a.target for a in s.legal_actions() if a.kind is ActionKind.PLACE}
            self.assertEqual({(i, j) for i, j in zip(*np.nonzero(e[1]))}, moves)
            self.assertEqual({int(i) for i, j in zip(*np.nonzero(e[3]))}, places)
            self.assertTrue(np.array_equal(e[1] * e[2], e[2]))      # capture subset of move
            for i, j in zip(*np.nonzero(e[2])):
                self.assertNotEqual(int(s.board[j]), 0)
                self.assertNotEqual(np.sign(int(s.board[j])), np.sign(int(s.board[i])))
        adj = G.build_graph(GoChessState.initial())[1][0]
        self.assertTrue(np.array_equal(adj, adj.T))
        self.assertEqual(int(adj.sum()), 2 * 2 * 5 * 4)              # 40 undirected -> 80 directed

    def test_action_mask_and_index_mapping(self) -> None:
        for s in _states():
            mask = G.action_mask(s)
            self.assertEqual(mask.shape, (G.num_actions(5),))
            ids = s.legal_action_ids()
            self.assertEqual(len(ids), len(set(ids)))                # injective
            self.assertEqual(int(mask.sum()), len(ids))
            for a in s.legal_actions():
                i = G.action_to_index(a, 5)
                self.assertEqual(mask[i], 1.0)
                self.assertEqual(G.index_to_action(i, 5), a)         # bijective round trip
            if not s.is_terminal():
                self.assertEqual(mask[G.num_actions(5) - 1], 1.0)    # pass always legal

    def test_terminal_state_has_empty_mask_and_edges(self) -> None:
        s = scenarios()["royal_capture_in_one"].state
        term = next((s.apply(a) for a in s.legal_actions() if s.apply(a).is_terminal()), None)
        self.assertIsNotNone(term)
        self.assertEqual(float(G.action_mask(term).sum()), 0.0)
        self.assertEqual(float(G.build_graph(term)[1][1:].sum()), 0.0)

    def test_policy_layout_matches_action_encode(self) -> None:
        lay = G.graph_policy_index_layout(5)
        self.assertEqual(lay["place"], (0, 25))
        self.assertEqual(Action.place(24).encode(5), lay["place"][1] - 1)
        self.assertEqual(Action.move(0, 0).encode(5), lay["move"][0])
        self.assertEqual(Action.move(24, 24).encode(5), lay["move"][1] - 1)
        self.assertEqual(Action.pass_turn().encode(5), lay["pass"][0])
        self.assertEqual(lay["pass"][1], G.num_actions(5))


if __name__ == "__main__":
    unittest.main()
