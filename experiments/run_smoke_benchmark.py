"""Run the first empirical validation of the GoChess rules/search stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gochess.evaluate import run_match, summarize, wilson_interval, write_records
from gochess.scenarios import scenarios
from gochess.search import RandomAgent, UCTAgent, solve_exact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=12)
    parser.add_argument("--uct-simulations", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--output", type=Path, default=Path("results/smoke"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    initial = scenarios()["initial_5x5"].state
    records = run_match(
        initial_state=initial,
        games=args.games,
        player0_name=f"uct_{args.uct_simulations}",
        player1_name="random",
        player0_factory=lambda seed: UCTAgent(
            simulations=args.uct_simulations,
            rollout_depth=40,
            seed=seed,
        ),
        player1_factory=lambda seed: RandomAgent(seed=seed),
        seed=args.seed,
    )
    swapped = run_match(
        initial_state=initial,
        games=args.games,
        player0_name="random",
        player1_name=f"uct_{args.uct_simulations}",
        player0_factory=lambda seed: RandomAgent(seed=seed),
        player1_factory=lambda seed: UCTAgent(
            simulations=args.uct_simulations,
            rollout_depth=40,
            seed=seed,
        ),
        seed=args.seed + 10_000,
    )
    write_records(args.output / "games.csv", records + swapped)

    puzzle = scenarios()["royal_capture_in_one"].state
    solved = solve_exact(puzzle)
    uct_wins = sum(record.return0 > 0 for record in records) + sum(
        record.return0 < 0 for record in swapped
    )
    total_games = len(records) + len(swapped)
    interval = wilson_interval(uct_wins, total_games)
    summary = {
        "protocol": {
            "games_per_seating": args.games,
            "uct_simulations": args.uct_simulations,
            "seed": args.seed,
        },
        "uct_as_player0": summarize(records),
        "random_as_player0": summarize(swapped),
        "uct_overall": {
            "wins": uct_wins,
            "games": total_games,
            "win_rate": uct_wins / total_games,
            "wilson_95_low": interval[0],
            "wilson_95_high": interval[1],
        },
        "royal_capture_solver": {
            "value": solved.value,
            "best_action": puzzle.format_action(solved.best_action) if solved.best_action else None,
            "nodes": solved.nodes,
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
