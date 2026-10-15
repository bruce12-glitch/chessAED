"""Fixed-protocol evaluation. Never decides quality, only records games.

Temp 0, no Dirichlet noise, equal seating, fixed reference.
One row per game with config hash.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from training.device import resolve_device

from experiments.eval_net import NetAgent
from gochess.representation import load_model_from_checkpoint
from gochess.scenarios import scenarios
from gochess.search import UCTAgent
from gochess.evaluate import play_game


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", type=str, default="baseline_v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--ckpt-step", type=int, default=0)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--net-sims", type=int, default=64)
    ap.add_argument("--eval-sims", type=int, default=400)
    ap.add_argument("--scenario", type=str, default="mixed_midgame")
    ap.add_argument("--output", type=Path, default=Path("results/raw/baseline_v1"))
    ap.add_argument("--tag", type=str, default="",
                    help="file prefix for non-periodic evals, e.g. final_uct800")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    a = ap.parse_args()
    device = resolve_device(a.device)
    if a.manifest is not None:
        from experiments import manifest as MF
        MF.copy_into(a.output, a.manifest)

    net, ckpt = load_model_from_checkpoint(a.checkpoint)
    net.to(device)
    config_hash = ckpt.get("config", {}).get("config_hash", "unknown")

    out = a.output / f"seed_{a.seed:03d}"
    out.mkdir(parents=True, exist_ok=True)
    state = scenarios()[a.scenario].state
    rows = []
    for g in range(a.games):
        for seating in (0, 1):
            agent = NetAgent(net, sims=a.net_sims, seed=a.seed + g, device=a.device)
            ref = UCTAgent(simulations=a.eval_sims, seed=a.seed + g * 2 + seating)
            if seating == 0:
                t, plies = play_game(state, agent, ref)
                p0, p1 = "net", f"uct{a.eval_sims}"
            else:
                t, plies = play_game(state, ref, agent)
                p0, p1 = f"uct{a.eval_sims}", "net"
            r0 = t.returns()[0]
            rows.append({"experiment_id": a.experiment, "seed": a.seed,
                         "checkpoint_step": a.ckpt_step,
                         "game_id": f"{g}_{seating}", "player0_agent": p0,
                         "player1_agent": p1, "seating": seating,
                         "winner": 0 if r0 > 0 else (1 if r0 < 0 else -1),
                         "return_p0": r0,
                         "terminal_reason": t.terminal_reason or "move_limit",
                         "plies": plies, "train_mcts_sims": a.net_sims,
                         "eval_mcts_sims": a.eval_sims, "config_hash": config_hash,
                         "arch": ckpt.get("config", {}).get("arch", "cnn")})
    name = f"{a.tag}_step{a.ckpt_step}.csv" if a.tag else f"eval_step{a.ckpt_step}.csv"
    with (out / name).open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    wins = sum(1 for r in rows if (r["return_p0"] > 0) == (r["player0_agent"] == "net"))
    print(json.dumps({"winrate_net": wins / len(rows), "games": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
