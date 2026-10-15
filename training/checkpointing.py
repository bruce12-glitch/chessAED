"""Full checkpointing: weights + optimizer + RNG states + config + git hash."""
from __future__ import annotations

import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def save_checkpoint(path: Path, net: torch.nn.Module, opt: torch.optim.Optimizer,
                     iteration: int, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": net.state_dict(),
        "optimizer": opt.state_dict(),
        "iteration": iteration,
        "config": dict(config),
        "git_commit": git_hash(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }, path)


def load_checkpoint(path: Path, net: torch.nn.Module,
                     opt: torch.optim.Optimizer | None = None) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["model"])
    if opt is not None and "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
    return ckpt
