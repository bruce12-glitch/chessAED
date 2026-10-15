"""AED diagnostics from training observables. No side effects."""
from __future__ import annotations

import math


def mean_entropy(PIs) -> float:
    ents = []
    for pi in PIs:
        e = -sum(float(p) * math.log(float(p) + 1e-12) for p in pi if float(p) > 0)
        ents.append(e)
    return sum(ents) / max(1, len(ents))


def argmax_diversity(PIs) -> float:
    """Fraction of unique argmax actions. 1 = diverse, ~0 = repeating."""
    if not PIs:
        return 0.0
    best = []
    for pi in PIs:
        m = max(range(len(pi)), key=lambda i: float(pi[i]))
        best.append(m)
    return len(set(best)) / len(best)


def learning_progress(prev_vloss: float | None, vloss: float) -> float:
    return (prev_vloss - vloss) if prev_vloss is not None else 0.0


def leverage(tv: float, tv_scale: float = 0.15) -> float:
    """Policy-movement proxy for causal leverage in [0, 1].

    A frozen policy (tv -> 0) exerts no detectable control, so leverage is
    low and the detector may fire; a changing policy (tv >= tv_scale) scores
    full leverage and suppresses boredom. True empowerment I(A;S') is
    unmeasured here; TV drift is the observable proxy (see protocol_v2.md).
    """
    return min(1.0, max(0.0, tv / tv_scale))


def utility(lt: float, ct: float, beta_l: float, beta_c: float,
            drop: str = "none") -> float:
    if drop == "value":
        lt = 0.0
    if drop == "kl":
        ct = 1.0  # TV term neutralized -> Ct pinned high
    return beta_l * lt + beta_c * ct
