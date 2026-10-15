"""Phase A diagnostics + frozen event/escape rules (protocol v2).

Thresholds live here and in protocol_v2.md. Change = new protocol version.
"""
from __future__ import annotations

TV_T = 0.15
ENT_SLOPE_T = 0.05
WR_T = 0.40
REPEAT_T = 0.50
WINDOW = 3
ESCAPE_PP = 0.15
HOLD = 2


def entropy_slope(ents: list[float]) -> float:
    if len(ents) < 2:
        return 0.0
    return (ents[-1] - ents[0]) / (len(ents) - 1)


def repeat_rate(openings: list[int | None]) -> float:
    moves = [o for o in openings if o is not None]
    if not moves:
        return 0.0
    return max(moves.count(m) for m in set(moves)) / len(moves)


def bad_minimum_event(tv: list[float], ents: list[float], winrates: list[float],
                      repeats: list[float]) -> bool:
    """All four conditions for WINDOW consecutive evals."""
    if len(tv) < WINDOW or len(winrates) < WINDOW or len(ents) < WINDOW or len(repeats) < WINDOW:
        return False
    if any(v is None for v in tv[-WINDOW:] + winrates[-WINDOW:]):
        return False
    if any(v >= TV_T for v in tv[-WINDOW:]):
        return False
    if any(w >= WR_T for w in winrates[-WINDOW:]):
        return False
    if abs(entropy_slope(ents[-WINDOW:])) >= ENT_SLOPE_T:
        return False
    if any(r <= REPEAT_T for r in repeats[-WINDOW:]):
        return False
    return True


def escaped(event_baseline_wr: float, recent_wrs: list[float]) -> bool:
    """+15 percentage points held for HOLD consecutive evals."""
    if len(recent_wrs) < HOLD:
        return False
    return all(w - event_baseline_wr >= ESCAPE_PP for w in recent_wrs[-HOLD:])
