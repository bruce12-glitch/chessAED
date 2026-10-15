"""Fixed probe positions: frozen reference actions and net-vs-reference disagreement."""
from __future__ import annotations

from gochess.scenarios import scenarios
from gochess.search import UCTAgent


def compute_probe_references(names: list[str], sims: int, seed: int) -> dict[str, int]:
    out = {}
    for i, name in enumerate(names):
        s = scenarios()[name].state
        out[name] = UCTAgent(simulations=sims, seed=seed + i).select_action(s).encode(s.size)
    return out


def net_probe_disagreement(net, reference: dict[str, int], net_sims: int, seed: int) -> dict:
    """Fraction of probes where NetAgent search picks a different action than the
    frozen reference, plus mean masked-policy mass on the reference action."""
    from experiments.eval_net import NetAgent
    from gochess.representation import policy_probs
    dis, mass = [], []
    for i, (name, ref_id) in enumerate(sorted(reference.items())):
        s = scenarios()[name].state
        a = NetAgent(net, sims=net_sims, seed=seed + i).select_action(s)
        dis.append(int(a.encode(s.size) != ref_id))
        mass.append(float(policy_probs(net, s)[ref_id]))
    n = max(1, len(dis))
    return {"disagreement": sum(dis) / n, "ref_policy_mass": sum(mass) / n}
