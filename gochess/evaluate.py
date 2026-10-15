"""Seeded match evaluation and machine-readable result writing."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Callable

from gochess.core import GoChessState
from gochess.search import Agent


@dataclass(frozen=True, slots=True)
class GameRecord:
    game: int
    seed: int
    player0: str
    player1: str
    return0: float
    plies: int
    terminal_reason: str
    elapsed_seconds: float


def play_game(
    state: GoChessState,
    player0: Agent,
    player1: Agent,
) -> tuple[GoChessState, int]:
    current = state
    plies = 0
    while not current.is_terminal():
        agent = player0 if current.current_player == 0 else player1
        current = current.apply(agent.select_action(current))
        plies += 1
    return current, plies


def run_match(
    *,
    initial_state: GoChessState,
    games: int,
    player0_name: str,
    player1_name: str,
    player0_factory: Callable[[int], Agent],
    player1_factory: Callable[[int], Agent],
    seed: int,
) -> list[GameRecord]:
    records = []
    for game in range(games):
        game_seed = seed + game
        started = time.perf_counter()
        terminal, plies = play_game(
            initial_state,
            player0_factory(game_seed * 2),
            player1_factory(game_seed * 2 + 1),
        )
        records.append(
            GameRecord(
                game=game,
                seed=game_seed,
                player0=player0_name,
                player1=player1_name,
                return0=terminal.returns()[0],
                plies=plies,
                terminal_reason=terminal.terminal_reason or "move_limit",
                elapsed_seconds=time.perf_counter() - started,
            )
        )
    return records


def write_records(path: Path, records: list[GameRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GameRecord.__dataclass_fields__)
        writer.writeheader()
        writer.writerows(
            {field: getattr(record, field) for field in GameRecord.__dataclass_fields__}
            for record in records
        )


def summarize(records: list[GameRecord]) -> dict[str, float]:
    if not records:
        raise ValueError("cannot summarize an empty match")
    wins = sum(record.return0 > 0 for record in records)
    draws = sum(record.return0 == 0 for record in records)
    return {
        "games": float(len(records)),
        "player0_win_rate": wins / len(records),
        "draw_rate": draws / len(records),
        "mean_return0": sum(record.return0 for record in records) / len(records),
        "mean_plies": sum(record.plies for record in records) / len(records),
        "elapsed_seconds": sum(record.elapsed_seconds for record in records),
    }


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("require 0 <= successes <= trials and trials > 0")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / trials
            + z * z / (4 * trials * trials)
        )
        / denominator
    )
    return center - radius, center + radius
