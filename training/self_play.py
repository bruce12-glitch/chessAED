"""Self-play data collection with openings + probe disagreement.

Wraps frozen gochess search; no rule logic here.
Opening = first action id of each game (argmax of root visits).
"""
from __future__ import annotations

import math
import random

import torch

from gochess.nn import obs_to_tensor
from gochess.scenarios import scenarios
from gochess.search import UCTAgent


def collect(games: int, uct_sims: int, seed: int,
            start: str = "initial_5x5"):
    X, PI, Z, entropy, openings = [], [], [], [], []
    state0 = scenarios()[start].state
    for g in range(games):
        s = state0
        traj = []
        teacher = UCTAgent(simulations=uct_sims, seed=seed + g)
        first_move = None
        while not s.is_terminal():
            root = teacher.search(s)
            visits: dict[int, int] = {}
            for child in root.children:
                if child.action is not None:
                    visits[child.action.encode(s.size)] = child.visits
            total = sum(visits.values()) or 1
            pi = [0.0] * s.num_distinct_actions
            for aid in s.legal_action_ids():
                pi[aid] = visits.get(aid, 0) / total
            ent = -sum(p * math.log(p + 1e-12) for p in pi if p > 0)
            entropy.append(ent)
            traj.append((s, pi))
            best = max(root.children, key=lambda c: c.visits)
            assert best.action is not None
            if first_move is None:
                first_move = best.action.encode(s.size)
            s = s.apply(best.action)
        openings.append(first_move)
        z = s.returns()[0]
        for st, pi in traj:
            X.append(obs_to_tensor(st))
            PI.append(torch.tensor(pi, dtype=torch.float32))
            Z.append(z if st.current_player == 0 else -z)
    return (torch.stack(X), torch.stack(PI),
            torch.tensor(Z, dtype=torch.float32),
            sum(entropy) / max(1, len(entropy)), openings)


def probe_disagreement(probe_names: list[str], shallow_sims: int,
                       deep_sims: int, seed: int) -> float:
    """Fraction of probes where shallow and deep search agree on top action."""
    agree, total = 0, 0
    for name in probe_names:
        s = scenarios()[name].state
        a = UCTAgent(simulations=shallow_sims, seed=seed).select_action(s)
        b = UCTAgent(simulations=deep_sims, seed=seed + 1).select_action(s)
        agree += int(a == b)
        total += 1
    return agree / max(1, total)
