"""Eval metrics from raw artifacts. No training, no plotting backend."""
from __future__ import annotations

import csv
import math
import random
from pathlib import Path


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - r, c + r


def bootstrap_ci(x: list[float], iters: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(x)
    means = sorted(sum(x[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


def read_eval_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as h:
        return list(csv.DictReader(h))


def net_winrate(rows: list[dict]) -> float:
    wins = sum(1 for r in rows if (float(r["return_p0"]) > 0) == (r["player0_agent"] == "net"))
    return wins / len(rows) if rows else 0.0


def plateau_detect(vloss: list[float], window: int = 3, tol: float = 0.05) -> bool:
    """Flat and not improving over trailing window."""
    if len(vloss) < window + 1:
        return False
    tail = vloss[-window:]
    return max(tail) - min(tail) < tol and tail[-1] >= min(vloss) - tol
