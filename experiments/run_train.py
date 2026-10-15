"""Baseline-only trainer. No AED, no GNN, no reward changes.

Loss: (z-v)^2 - pi^T log p + L2 via optimizer weight_decay.
Writes per-iter full checkpoint + seed manifest + metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from evaluation.diagnostics import repeat_rate
from experiments.train_alphazero import policy_tv
from gochess.nn import GoChessNet
from training.checkpointing import git_hash, save_checkpoint
from training.replay_buffer import ReplayBuffer
from training.self_play import collect, probe_disagreement

PROBE_STATES = ["mixed_midgame", "encirclement_choice"]


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", type=str, default="baseline_v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--uct-simulations", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--buffer", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--output", type=Path, default=Path("results/raw/baseline_v1"))
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--eval-games", type=int, default=16)
    ap.add_argument("--eval-sims", type=int, default=400)
    ap.add_argument("--eval-scenario", type=str, default="mixed_midgame")
    a = ap.parse_args()

    config = {"experiment": a.experiment, "seed": a.seed, "games": a.games,
              "train_mcts_sims": a.uct_simulations, "epochs": a.epochs,
              "buffer": a.buffer, "batch": a.batch, "lr": a.lr, "l2": a.l2,
              "net": "resnet_4x64", "git_commit": git_hash(),
              "eval_every": a.eval_every, "eval_games": a.eval_games,
              "eval_sims": a.eval_sims, "eval_scenario": a.eval_scenario}
    config["config_hash"] = config_hash({k: v for k, v in config.items() if k != "config_hash"})
    out = a.output / f"seed_{a.seed:03d}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    net = GoChessNet(channels=64, blocks=4)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=a.l2)
    buf = ReplayBuffer(capacity=a.buffer)
    prev_pm = None
    metrics = []
    root = Path(__file__).resolve().parent.parent
    all_openings: list[int | None] = []
    for it in range(a.epochs):
        X, PI, Z, ent, openings = collect(a.games, a.uct_simulations, a.seed + it * 1000)
        all_openings.extend(openings)
        for i in range(len(X)):
            buf.add(X[i], PI[i], float(Z[i]), {"iter": it, "seed": a.seed})
        xb, pib, zb = buf.sample(min(len(buf), 512), seed=a.seed + it)
        ds = DataLoader(TensorDataset(xb, pib, zb), batch_size=a.batch, shuffle=True,
                        generator=torch.Generator().manual_seed(a.seed + it))
        net.train()
        for bxb, bpib, bzb in ds:
            logits, v = net(bxb)
            loss = F.mse_loss(v, bzb) + F.cross_entropy(logits, bpib)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            logits, v = net(X)
            vloss = F.mse_loss(v, Z).item()
            acc = (logits.argmax(1) == PI.argmax(1)).float().mean().item()
            pm = F.softmax(logits, 1).mean(0)
            tv = policy_tv(pm.unsqueeze(0), prev_pm.unsqueeze(0)) if prev_pm is not None else 0.0
            prev_pm = pm.clone()
        row = {"iter": it, "vloss": vloss, "acc": acc, "tv": tv,
               "entropy": ent, "buffer": len(buf),
               "repeat": repeat_rate(all_openings), "openings": openings}
        if a.eval_every and (it % a.eval_every == 0 or it == a.epochs - 1):
            row["disagreement"] = probe_disagreement(
                PROBE_STATES, a.uct_simulations, a.eval_sims, a.seed + it)
        metrics.append(row)
        print(row, flush=True)
        save_checkpoint(out / f"ckpt_{it}.pt", net, opt, it, config)
        if a.eval_every and (it % a.eval_every == 0 or it == a.epochs - 1):
            r = subprocess.run(
                [sys.executable, "-m", "experiments.run_evaluate",
                 "--experiment", a.experiment, "--seed", str(a.seed),
                 "--checkpoint", str(out / f"ckpt_{it}.pt"),
                 "--ckpt-step", str(it), "--games", str(a.eval_games),
                 "--net-sims", str(a.uct_simulations),
                 "--eval-sims", str(a.eval_sims),
                 "--scenario", a.eval_scenario,
                 "--output", str(a.output)], cwd=root)
            if r.returncode != 0:
                raise RuntimeError("periodic eval failed")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
