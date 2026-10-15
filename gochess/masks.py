"""Legal-action masks over the fixed action index. Additive only.

Wraps gochess.core.Action encode/legal_action_ids; no rule logic duplicated.
"""
from __future__ import annotations

from gochess.core import GoChessState


def legal_mask(state: GoChessState) -> list[int]:
    mask = [0] * state.num_distinct_actions
    for aid in state.legal_action_ids():
        mask[aid] = 1
    return mask


def masked_policy(state: GoChessState, logits: list[float]) -> list[float]:
    """Zero illegal logits, renormalize. Empty list if terminal."""
    import math

    mask = legal_mask(state)
    if sum(mask) == 0:
        return []
    exps = [math.exp(l) * m for l, m in zip(logits, mask)]
    total = sum(exps)
    return [e / total for e in exps]
