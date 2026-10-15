"""Atomic, exact resume: model/optimizer via ckpt_N.pt, everything else here."""
from __future__ import annotations

import os
import random
import re
from pathlib import Path

import numpy as np
import torch


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def export_controller(ctl) -> dict:
    d = ctl.detector
    return {"b": d._b, "rng": d._rng.getstate() if d._rng is not None else None,
            "remaining": ctl.state.remaining, "prev_vloss": ctl.prev_vloss}


def import_controller(ctl, s: dict) -> None:
    ctl.detector._b = s["b"]
    if s["rng"] is not None:
        ctl.detector._rng = random.Random()
        ctl.detector._rng.setstate(s["rng"])
    ctl.state.remaining = s["remaining"]
    ctl.prev_vloss = s["prev_vloss"]


def save_train_state(path: Path, *, it: int, buf, ctl, prev_pm, metrics: list,
                     all_openings: list, config_hash: str) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({
        "iter": it, "config_hash": config_hash,
        "buffer": {"obs": buf.obs, "pi": buf.pi, "z": buf.z, "meta": buf.meta},
        "controller": export_controller(ctl), "prev_pm": prev_pm,
        "metrics": metrics, "all_openings": all_openings,
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": (torch.cuda.get_rng_state_all()
                               if torch.cuda.is_available() else None)},
    }, tmp)
    os.replace(tmp, path)


def load_train_state(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def restore_rng(rng: dict) -> None:
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["torch_cuda"])


def latest_resumable(out: Path) -> int | None:
    found = []
    for p in out.glob("train_state_*.pt"):
        m = re.fullmatch(r"train_state_(\d+)\.pt", p.name)
        if m and (out / f"ckpt_{m.group(1)}.pt").exists():
            found.append(int(m.group(1)))
    return max(found) if found else None


def prune_older(out: Path, keep: int) -> None:
    for p in out.glob("train_state_*.pt"):
        m = re.fullmatch(r"train_state_(\d+)\.pt", p.name)
        if m and int(m.group(1)) != keep:
            p.unlink()
