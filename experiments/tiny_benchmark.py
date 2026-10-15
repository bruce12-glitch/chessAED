"""Tiny solved-game benchmark: exact optima vs limited search.

Mines near-terminal states (few legal actions) from seeded random play,
solves each exactly, then measures whether shallow UCT repeatedly picks
suboptimal moves. A position where UCT-16 stabilises off-optimum across
seeds is a search-level bad minimum with a proven optimum.
Usage: python -m experiments.tiny_benchmark --positions 8 --seeds 5
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from gochess.core import GoChessState
from gochess.search import UCTAgent, solve_exact


def mine_positions(count: int, seed: int) -> list[GoChessState]:
    """Sample dense random boards; keep near-terminal ones (2..8 legal actions).

    from_ascii does not enforce liberties, so dense fills are cheap to propose;
    legality is decided by legal_actions(). Deterministic in seed.
    """
    import string

    from gochess.core import Rules
    rng = random.Random(seed)
    chars = "XoKRNkrn"
    found: dict[bytes, GoChessState] = {}
    guard = 0
    while len(found) < count and guard < count * 400:
        guard += 1
        rows = []
        for _ in range(5):
            row = "".join("." if rng.random() < 0.12 else rng.choice(chars)
                          for _ in range(5))
            rows.append(row)
        # exactly one king per side
        rows = [r.replace("K", "X").replace("k", "o") for r in rows]
        cells = [(r, c) for r in range(5) for c in range(5)]
        (kr, kc), (qr, qc) = rng.sample(cells, 2)
        grid = [list(r) for r in rows]
        grid[kr][kc] = "K"
        grid[qr][qc] = "k"
        try:
            s = GoChessState.from_ascii(
                tuple("".join(r) for r in grid),
                current_player=rng.randrange(2),
                reserves=(rng.randrange(3), rng.randrange(3)),
                max_plies=100, komi=0.0)
        except ValueError:
            continue
        n = len(s.legal_actions())
        if 2 <= n <= 8:
            found.setdefault(s.board_signature() + bytes([s.current_player]), s)
    return list(found.values())[:count]


def benchmark_position(state: GoChessState, sims_list: tuple[int, ...] = (16, 64),
                       seeds: int = 5, max_nodes: int = 30_000) -> dict:
    try:
        solved = solve_exact(state, max_nodes=max_nodes)
    except Exception as e:  # noqa: BLE001 - budget/depth: unsolved is data
        return {"error": f"{type(e).__name__}: {e}"}
    opt = solved.best_action
    out = {"value": solved.value, "nodes": solved.nodes,
           "optimal": state.format_action(opt) if opt else None, "by_sims": {}}
    for sims in sims_list:
        agree, vals = 0, []
        for sd in range(seeds):
            a = UCTAgent(simulations=sims, seed=1000 + sd).select_action(state)
            agree += int(a == opt)
            vals.append(state.apply(a).heuristic_value())
        out["by_sims"][str(sims)] = {"agree": agree / seeds,
                                     "mean_heuristic": sum(vals) / len(vals)}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-nodes", type=int, default=30_000)
    ap.add_argument("--out", type=Path, default=Path("results/tiny/benchmark.json"))
    a = ap.parse_args(argv)
    states = mine_positions(a.positions, a.seed)
    rows = []
    for i, s in enumerate(states):
        r = benchmark_position(s, seeds=a.seeds, max_nodes=a.max_nodes)
        r.update({"position": i, "legal": len(s.legal_actions()),
                  "ascii": s.to_ascii()})
        rows.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "ascii"}))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    sub = [r for r in rows if "by_sims" in r]
    n = sum(1 for r in sub if r["by_sims"]["16"]["agree"] < 1.0)
    print(f"solved={len(sub)}/{len(rows)} suboptimal_at_16={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
