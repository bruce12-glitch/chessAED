from __future__ import annotations

import unittest

from gochess.core import Action, GoChessState, IllegalAction, Piece, Rules
from gochess.scenarios import scenarios


class ActionTests(unittest.TestCase):
    def test_action_encoding_round_trip(self) -> None:
        for action in (Action.place(7), Action.move(3, 22), Action.pass_turn()):
            self.assertEqual(Action.decode(action.encode(5), 5), action)

    def test_action_space_is_fixed(self) -> None:
        self.assertEqual(Action.action_space_size(5), 651)


class RuleTests(unittest.TestCase):
    def test_initial_position_is_symmetric_and_legal(self) -> None:
        state = GoChessState.initial(Rules(max_plies=20))
        self.assertEqual(state.board.count(Piece.P0_KING), 1)
        self.assertEqual(state.board.count(Piece.P1_KING), 1)
        self.assertGreater(len(state.legal_actions()), 1)

    def test_royal_capture_ends_game(self) -> None:
        state = scenarios()["royal_capture_in_one"].state
        terminal = state.apply(Action.move(10, 14))
        self.assertTrue(terminal.is_terminal())
        self.assertEqual(terminal.terminal_reason, "king_capture")
        self.assertEqual(terminal.returns(), (1.0, -1.0))

    def test_go_enclosure_captures_connected_enemy_group(self) -> None:
        state = GoChessState.from_ascii(
            (
                "K....",
                ".X...",
                "XooX.",
                ".XX..",
                "....k",
            ),
            current_player=0,
            reserves=(1, 0),
            max_plies=10,
            komi=0,
        )
        after = state.apply(Action.place(7))
        self.assertEqual(after.board[11], Piece.EMPTY)
        self.assertEqual(after.board[12], Piece.EMPTY)

    def test_suicide_is_illegal(self) -> None:
        state = GoChessState.from_ascii(
            (
                "K....",
                "..o..",
                ".o.o.",
                "..o..",
                "....k",
            ),
            current_player=0,
            reserves=(1, 0),
            max_plies=10,
        )
        with self.assertRaises(IllegalAction):
            state.apply(Action.place(12))

    def test_knight_jumps_and_rook_does_not(self) -> None:
        state = GoChessState.from_ascii(
            (
                "K....",
                ".....",
                "RNX..",
                ".....",
                "....k",
            ),
            current_player=0,
            reserves=(0, 0),
        )
        actions = set(state.legal_actions())
        self.assertIn(Action.move(11, 2), actions)
        self.assertNotIn(Action.move(10, 13), actions)

    def test_two_passes_end_and_score(self) -> None:
        state = GoChessState.initial(Rules(max_plies=20))
        state = state.apply(Action.pass_turn()).apply(Action.pass_turn())
        self.assertTrue(state.is_terminal())
        self.assertEqual(state.terminal_reason, "two_passes")

    def test_observation_shape(self) -> None:
        self.assertEqual(len(GoChessState.initial().observation_tensor(0)), 11 * 5 * 5)

    def test_no_actions_after_terminal(self) -> None:
        state = scenarios()["royal_capture_in_one"].state.apply(Action.move(10, 14))
        self.assertEqual(state.legal_actions(), ())
        with self.assertRaises(IllegalAction):
            state.apply(Action.pass_turn())


if __name__ == "__main__":
    unittest.main()
