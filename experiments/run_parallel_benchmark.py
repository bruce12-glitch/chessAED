"""Parallel smoke benchmark for Kaggle (CPU-parallel, GPU-ready stub).

Code Set 1 has no neural net, so GPU does nothing yet.
Speedup comes from multiprocessing across games.
Run as module: python -m experiments.run_parallel_benchmark
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

from gochess.evaluate import GameRecord, play_game, summarize, wilson_interval, write_records
from gochess.scenarios import scenarios
from gochess.search import RandomAgent, UCTAgent


def _play_one(args) -> GameRecord:
    import time
    game, seed, uct_sims, uct_first = args
    initial = scenarios()["initial_5x5"].state
    game_seed = seed + game
    started = time.perf_counter()
    if uct_first:
        p0 = UCTAgent(simulations=uct_sims, rollout_depth=40, seed=game_seed * 2)
        p1 = RandomAgent(seed=game_seed * 2 + 1)
        p0_name, p1_name = f"uct_{uct_sims}", "random"
    else:
        p0 = RandomAgent(seed=game_seed * 2)
        p1 = UCTAgent(simulations=uct_sims, rollout_depth=40, seed=game_seed * 2 + 1)
        p0_name, p1_name = "random", f"uct_{uct_sims}"
    terminal, plies = play_game(initial, p0, p1)
    return GameRecord(
        game=game, seed=game_seed, player0=p0_name, player1=p1_name,
        return0=terminal.returns()[0], plies=plies,
        terminal_reason=terminal.terminal_reason or "move_limit",
        elapsed_seconds=time.perf_counter() - started,
    )


def run_seating(games: int, uct_sims: int, seed: int, uct_first: bool, workers: int):
    tasks = [(g, seed, uct_sims, uct_first) for g in range(games)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_play_one, tasks))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--uct-simulations", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--output", type=Path, default=Path("results/smoke"))
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)

    records = run_seating(a.games, a.uct_simulations, a.seed, True, a.workers)
    swapped = run_seating(a.games, a.uct_simulations, a.seed + 10_000, False, a.workers)
    write_records(a.output / "games.csv", records + swapped)

    uct_wins = sum(r.return0 > 0 for r in records) + sum(r.return0 < 0 for r in swapped)
    total = len(records) + len(swapped)
    lo, hi = wilson_interval(uct_wins, total)
    summary = {
        "protocol": {"games_per_seating": a.games, "uct_simulations": a.uct_simulations,
                      "seed": a.seed, "workers": a.workers},
        "uct_as_player0": summarize(records),
        "random_as_player0": summarize(swapped),
        "uct_overall": {"wins": uct_wins, "games": total, "win_rate": uct_wins / total,
                         "wilson_95_low": lo, "wilson_95_high": hi},
        "note": "CPU-parallel. GPU unused in Code Set 1 (no neural net).",
    }
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
