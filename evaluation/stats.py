"""Paired statistics over seeds (numpy only)."""
from __future__ import annotations

import itertools

import numpy as np


def paired_bootstrap_ci(diffs, n_boot: int = 10000, seed: int = 0,
                        alpha: float = 0.05) -> tuple[float, float]:
    d = np.asarray(diffs, dtype=float)
    if d.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_permutation_pvalue(diffs, n_perm: int = 100000, seed: int = 0) -> float:
    """Two-sided sign-flip test of mean(diff)=0. Exact when 2^n <= n_perm."""
    d = np.asarray(diffs, dtype=float)
    n = d.size
    if n < 2:
        return float("nan")
    obs = abs(d.mean())
    tol = 1e-12
    if 2 ** n <= n_perm:
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=n)))
        stat = np.abs((signs * d).mean(axis=1))
        return float((stat >= obs - tol).mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n_perm, n))
    stat = np.abs((signs * d).mean(axis=1))
    return float(((stat >= obs - tol).sum() + 1) / (n_perm + 1))


def holm(pvals: list[float]) -> list[float]:
    idx = sorted(range(len(pvals)), key=lambda i: (np.isnan(pvals[i]), pvals[i]))
    out = [float("nan")] * len(pvals)
    running, m = 0.0, len(pvals)
    for rank, i in enumerate(idx):
        if np.isnan(pvals[i]):
            continue
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        out[i] = running
    return out


def bootstrap_mean_ci(x, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05):
    x = np.asarray([v for v in x if not np.isnan(v)], dtype=float)
    if x.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
