"""WeakStable / Escape exactly as specified for experiment_manifest_v1.

WeakStable(t) = [TV_t < tv_t] and [|dH_t| < eps_H] and [W_t < wr_t], true for
`consecutive` evaluations in a row. dH_t is the entropy change per iteration
between consecutive evaluations. Escape = exists t' > t0 with
W_t' - W_t0 >= escape_pp and the same holding at the next evaluation too
(any two consecutive evaluations, not only the final two).
"""
from __future__ import annotations

TOL = 1e-9


def entropy_gradients(iters: list[int], ents: list[float]) -> list[float | None]:
    out: list[float | None] = [None]
    for k in range(1, len(iters)):
        d = iters[k] - iters[k - 1]
        out.append((ents[k] - ents[k - 1]) / d if d else None)
    return out


def weak_stable_flags(tv, dH, wr, *, tv_t=0.15, eps_h=0.05, wr_t=0.40) -> list[bool]:
    flags = []
    for a, g, w in zip(tv, dH, wr):
        flags.append(a is not None and g is not None and w is not None
                     and a < tv_t and abs(g) < eps_h and w < wr_t)
    return flags


def weak_stable_event(flags: list[bool], consecutive: int = 3) -> int | None:
    """Index of the evaluation completing the first run of `consecutive` True flags."""
    run = 0
    for k, f in enumerate(flags):
        run = run + 1 if f else 0
        if run >= consecutive:
            return k
    return None


def escape_index(wr: list[float], t0: int, *, escape_pp=0.15, hold=2) -> int | None:
    """First t' > t0 where the gain over wr[t0] holds for `hold` consecutive evals."""
    base = wr[t0]
    for k in range(t0 + 1, len(wr) - hold + 1):
        if all(wr[k + j] - base >= escape_pp - TOL for j in range(hold)):
            return k
    return None


def analyse_series(iters, tv, ents, wr, *, tv_t=0.15, eps_h=0.05, wr_t=0.40,
                   consecutive=3, escape_pp=0.15, hold=2) -> dict:
    flags = weak_stable_flags(tv, entropy_gradients(iters, ents), wr,
                              tv_t=tv_t, eps_h=eps_h, wr_t=wr_t)
    t0 = weak_stable_event(flags, consecutive)
    esc = escape_index(wr, t0, escape_pp=escape_pp, hold=hold) if t0 is not None else None
    return {"weak_stable_idx": t0, "weak_stable_iter": iters[t0] if t0 is not None else None,
            "escape_idx": esc, "escape_iter": iters[esc] if esc is not None else None,
            "escaped": esc is not None, "flags": flags}
