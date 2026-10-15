"""Command-line utilities for inspecting and evaluating GoChess."""

from __future__ import annotations

import argparse

from gochess.scenarios import scenarios
from gochess.search import UCTAgent, solve_exact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gochess")
    subparsers = parser.add_subparsers(dest="command", required=True)

    show = subparsers.add_parser("show", help="print a benchmark position")
    show.add_argument("scenario", choices=sorted(scenarios()))

    solve = subparsers.add_parser("solve", help="exactly solve a short-horizon position")
    solve.add_argument("scenario", choices=sorted(scenarios()))
    solve.add_argument("--max-nodes", type=int, default=250_000)

    search = subparsers.add_parser("search", help="run UCT from a benchmark position")
    search.add_argument("scenario", choices=sorted(scenarios()))
    search.add_argument("--simulations", type=int, default=128)
    search.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scenario = scenarios()[args.scenario]
    state = scenario.state
    if args.command == "show":
        print(state)
        print(f"player={state.current_player} reserves={state.reserves} ply={state.ply}")
        return 0
    if args.command == "solve":
        result = solve_exact(state, max_nodes=args.max_nodes)
        action = state.format_action(result.best_action) if result.best_action else "none"
        print(f"value={result.value:+.1f} best_action={action} nodes={result.nodes}")
        return 0
    agent = UCTAgent(simulations=args.simulations, seed=args.seed)
    statistics = agent.action_statistics(state)
    ranked = sorted(statistics.items(), key=lambda item: item[1][0], reverse=True)
    for action, (visits, value) in ranked[:10]:
        print(f"{state.format_action(action):12s} visits={visits:4d} value_p0={value:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
