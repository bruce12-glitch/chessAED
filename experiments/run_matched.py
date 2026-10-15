"""Matched AlphaZero vs AED comparison across seeds (CPU).

Runs baseline + AED with identical games/sims/epochs/seeds,
evals both vs UCT-128 from mixed_midgame, reports Wilson + bootstrap CI + paired sign test.

Run: python -m experiments.run_matched --seeds 5 --games 6 --uct-simulations 16 --epochs 6 --eval-games 8
"""
from __future__ import annotations
import argparse, json, random, subprocess, sys
from pathlib import Path

import torch

from gochess.nn import GoChessNet
from gochess.scenarios import scenarios
from gochess.search import UCTAgent
from gochess.evaluate import play_game
from experiments.eval_net import NetAgent


def run_cmd(args: list[str]) -> None:
    r = subprocess.run([sys.executable, "-m"] + args, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed: {args}")


def eval_ckpt(ckpt: Path, net_sims: int, uct_sims: int, games: int, seed: int) -> list[float]:
    net = GoChessNet()
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    net.eval()
    agent = NetAgent(net, sims=net_sims, seed=seed)
    state = scenarios()["mixed_midgame"].state
    outs: list[float] = []
    for g in range(games):
        gs = seed + g
        t, _ = play_game(state, agent, UCTAgent(simulations=uct_sims, seed=gs * 2 + 1))
        outs.append(1.0 if t.returns()[0] > 0 else 0.0)
        t2, _ = play_game(state, UCTAgent(simulations=uct_sims, seed=gs * 2), agent)
        outs.append(1.0 if t2.returns()[0] > 0 else 0.0)
    return outs


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    r = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return c - r, c + r


def bootstrap_ci(x: list[float], iters: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(x)
    means = []
    for _ in range(iters):
        s = sum(x[rng.randrange(n)] for _ in range(n)) / n
        means.append(s)
    means.sort()
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--uct-simulations", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--eval-games", type=int, default=8)
    ap.add_argument("--output", type=Path, default=Path("results/matched"))
    ap.add_argument("--aed-args", type=str, default="--tau 0.01 --beta-l 2.0 --beta-c 0.5 --Bmax 0.5")
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in range(a.seeds):
        base_out = a.output / f"base_s{s}"
        aed_out = a.output / f"aed_s{s}"
        run_cmd(["experiments.train_alphazero", "--games", str(a.games),
                 "--uct-simulations", str(a.uct_simulations), "--epochs", str(a.epochs),
                 "--seed", str(1000 + s), "--output", str(base_out)])
        run_cmd(["experiments.train_aed", "--games", str(a.games),
                 "--uct-simulations", str(a.uct_simulations), "--epochs", str(a.epochs),
                 "--seed", str(1000 + s), "--output", str(aed_out)] + a.aed_args.split())
        base_ckpt = base_out / f"ckpt_{a.epochs - 1}.pt"
        aed_ckpt = aed_out / f"ckpt_{a.epochs - 1}.pt"
        base_w = eval_ckpt(base_ckpt, 32, 128, a.eval_games, 5000 + s)
        aed_w = eval_ckpt(aed_ckpt, 32, 128, a.eval_games, 5000 + s)
        rows.append({"seed": s,
                     "base_winrate": sum(base_w) / len(base_w),
                     "aed_winrate": sum(aed_w) / len(aed_w),
                     "base_wins": base_w, "aed_wins": aed_w})

    base_all = [v for r in rows for v in r["base_wins"]]
    aed_all = [v for r in rows for v in r["aed_wins"]]
    base_wr = sum(base_all) / len(base_all)
    aed_wr = sum(aed_all) / len(aed_all)
    summary = {
        "config": vars(a) | {"output": str(a.output)},
        "per_seed": [{k: v for k, v in r.items() if k != "base_wins" and k != "aed_wins"} for r in rows],
        "base": {"winrate": base_wr, "wilson": wilson(sum(base_all), len(base_all)),
                 "bootstrap95": bootstrap_ci(base_all)},
        "aed": {"winrate": aed_wr, "wilson": wilson(sum(aed_all), len(aed_all)),
                "bootstrap95": bootstrap_ci(aed_all)},
        "paired_seed_wins": sum(1 for r in rows if r["aed_winrate"] > r["base_winrate"]),
        "seeds": a.seeds,
    }
    (a.output / "matched.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
