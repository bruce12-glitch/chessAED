"""Versioned benchmark positions for GoChess."""

from __future__ import annotations

from dataclasses import dataclass

from gochess.core import GoChessState


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    description: str
    state: GoChessState


def scenarios() -> dict[str, Scenario]:
    return {
        "initial_5x5": Scenario(
            name="initial_5x5",
            description="Symmetric default opening with six stones per player.",
            state=GoChessState.initial(),
        ),
        "royal_capture_in_one": Scenario(
            name="royal_capture_in_one",
            description="Rule check: player 0 has a unique immediate royal capture.",
            state=GoChessState.from_ascii(
                (
                    "K....",
                    ".....",
                    "R...k",
                    ".....",
                    "....r",
                ),
                current_player=0,
                reserves=(0, 0),
                max_plies=1,
                komi=0.0,
            ),
        ),
        "encirclement_choice": Scenario(
            name="encirclement_choice",
            description="Player 0 can complete an enclosure or pursue material.",
            state=GoChessState.from_ascii(
                (
                    "K....",
                    ".XoX.",
                    ".Xko.",
                    "..X..",
                    "N...R",
                ),
                current_player=0,
                reserves=(2, 1),
                max_plies=8,
                komi=0.5,
            ),
        ),
        "mixed_midgame": Scenario(
            name="mixed_midgame",
            description="A reproducible complex start for budget-sensitivity studies.",
            state=GoChessState.from_ascii(
                (
                    "R.XK.",
                    ".X.oN",
                    "..o..",
                    "n.oX.",
                    ".k..r",
                ),
                current_player=0,
                reserves=(2, 2),
                max_plies=24,
                komi=1.5,
            ),
        ),
    }
