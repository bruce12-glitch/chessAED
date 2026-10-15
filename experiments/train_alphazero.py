"""Code Set 2 trainer: distill UCT-128 into net, log plateau signals.

Run: python -m experiments.train_alphazero --games 20 --uct-simulations 64 --epochs 3
Logs: value loss, policy acc, policy TV vs prev checkpoint, root entropy.
No AED yet. This creates the baseline that Code Set 3 will compare against.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from gochess.core import GoChessState
from gochess.nn import GoChessNet, obs_to_tensor
from gochess.scenarios import scenarios
from gochess.search import UCTAgent


def collect(games: int, uct_sims: int, seed: int):
    X, PI, Z, entropy = [], [], [], []
    state0 = scenarios()["initial_5x5"].state
    for g in range(games):
        s = state0
        traj = []
        teacher = UCTAgent(simulations=uct_sims, seed=seed + g)
        while not s.is_terminal():
            root = teacher.search(s)
            import math, random
            rng = random.Random(seed + g * 1000 + len(traj))
            visits = {}
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
            s = s.apply(best.action)
        z = s.returns()[0]
        for st, pi in traj:
            X.append(obs_to_tensor(st))
            PI.append(torch.tensor(pi, dtype=torch.float32))
            Z.append(z if st.current_player == 0 else -z)
    return torch.stack(X), torch.stack(PI), torch.tensor(Z, dtype=torch.float32), sum(entropy) / max(1, len(entropy))


def policy_tv(p: torch.Tensor, q: torch.Tensor) -> float:
    return 0.5 * (p - q).abs().sum(1).mean().item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--uct-simulations", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, default=Path("results/az_baseline"))
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    net = GoChessNet()
    prev_pi = None
    log = []
    for it in range(a.epochs):
        X, PI, Z, ent = collect(a.games, a.uct_simulations, a.seed + it * 1000)
        ds = DataLoader(TensorDataset(X, PI, Z), batch_size=64, shuffle=True)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        net.train()
        for xb, pib, zb in ds:
            logits, v = net(xb)
            loss = F.cross_entropy(logits, pib) + F.mse_loss(v, zb)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            logits, v = net(X)
            vloss = F.mse_loss(v, Z).item()
            acc = (logits.argmax(1) == PI.argmax(1)).float().mean().item()
            pm = F.softmax(logits, 1).mean(0)
            tv = policy_tv(pm.unsqueeze(0), prev_pi.unsqueeze(0)) if prev_pi is not None else 0.0
            prev_pi = pm.clone()
        row = {"iter": it, "vloss": vloss, "policy_acc": acc, "policy_tv_vs_prev": tv, "root_entropy": ent}
        log.append(row); print(row)
        torch.save(net.state_dict(), a.output / f"ckpt_{it}.pt")
    (a.output / "plateau.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
