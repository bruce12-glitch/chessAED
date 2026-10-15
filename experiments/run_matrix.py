"""Matched E1-E4 matrix: same env/seeds/compute, only AED varies.

Modes: baseline (run_train), fixed-noise (run_aed --mode fixed-noise),
aed (run_aed --mode detector), random (run_aed --mode random).
Each cell: train (with periodic eval) -> final eval -> aggregate.
Writes results/raw/<mode>_v1/seed_### + results/tables/matrix.md.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

TRAINER_MODE = {"fixed-noise": "fixed-noise", "aed": "detector", "random": "random"}
EXP_NAME = {"baseline": "baseline_v1", "fixed-noise": "noise_v1",
            "aed": "aed_v1", "random": "random_v1"}


def cmd(args: list[str]) -> None:
    root = Path(__file__).resolve().parent.parent
    r = subprocess.run([sys.executable, "-m"] + args, cwd=root)
    if r.returncode != 0:
        raise RuntimeError(f"failed: {args}")


def plan_cells(modes: list[str], seeds: int) -> list[tuple[str, str, int]]:
    """Return (matrix_mode, experiment, seed) cells. Pure, unit-tested."""
    cells = []
    for mode in modes:
        if mode not in EXP_NAME:
            raise ValueError(f"unknown mode: {mode}")
        for s in range(seeds):
            cells.append((mode, EXP_NAME[mode], s))
    return cells


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", type=str, default="baseline,fixed-noise,aed,random")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--sims", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--buffer", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--eval-games", type=int, default=4)
    ap.add_argument("--eval-sims", type=int, default=128)
    ap.add_argument("--eval-scenario", type=str, default="mixed_midgame")
    ap.add_argument("--raw", type=Path, default=Path("results/raw"))
    a = ap.parse_args()

    common = ["--games", str(a.games), "--uct-simulations", str(a.sims),
              "--epochs", str(a.epochs), "--buffer", str(a.buffer),
              "--batch", str(a.batch), "--eval-every", str(a.eval_every),
              "--eval-games", str(a.eval_games), "--eval-sims", str(a.eval_sims),
              "--eval-scenario", a.eval_scenario]
    for mode, exp, s in plan_cells([m.strip() for m in a.modes.split(",")], a.seeds):
        if mode == "baseline":
            cmd(["experiments.run_train", "--experiment", exp, "--seed", str(s),
                 "--output", str(a.raw / exp)] + common)
        else:
            cmd(["experiments.run_aed", "--experiment", exp, "--seed", str(s),
                 "--mode", TRAINER_MODE[mode],
                 "--output", str(a.raw / exp)] + common)
        ckpt = a.raw / exp / f"seed_{s:03d}" / f"ckpt_{a.epochs - 1}.pt"
        cmd(["experiments.run_evaluate", "--experiment", exp, "--seed", str(s),
             "--checkpoint", str(ckpt), "--ckpt-step", str(a.epochs - 1),
             "--games", str(a.eval_games), "--net-sims", str(a.sims),
             "--eval-sims", str(a.eval_sims), "--scenario", a.eval_scenario,
             "--output", str(a.raw / exp)])
    cmd(["experiments.aggregate_results", "--raw", str(a.raw),
         "--output", "results/tables/matrix.md"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
