txt = '''"""Code Set 3: AED-augmented AlphaZero trainer.

Adds Aversive Epistemic Drive (AED) on top of train_alphazero.py:
- Lt = learning progress (delta vloss)
- Ct = causal leverage (policy TV vs teacher)
- Ut = beta_L * Lt + beta_C * Ct - homeostatic target
- Bt = min(Bmax, lambda * Bt + max(0, tau - Ut))
- When Bt >= Bmax: mode shift (temp up, noise up, sample archive)
"""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from gochess.core import GoChessState
from gochess.nn import GoChessNet, obs_to_tensor
from gochess.scenarios import scenarios
from gochess.search import UCTAgent, _Node
from gochess.evaluate import play_game


class NetAgent:
    def __init__(self, net: GoChessNet, sims: int = 32, temp: float = 1.0, noise: float = 0.0, seed: int = 0):
        self.net = net
        self.sims = sims
        self.temp = temp
        self.noise = noise
        self.rng = random.Random(seed)

    def _uct_select(self, node: _Node, c: float = 1.4):
        log_parent = math.log(max(1, node.visits))
        sign = 1.0 if node.state.current_player == 0 else -1.0
        def score(ch: _Node):
            exploit = sign * ch.mean_value
            explore = c * math.sqrt(log_parent / ch.visits)
            return exploit + explore
        best = max(node.children, key=score)
        return best

    def search(self, state: GoChessState) -> _Node:
        root = _Node(state=state, untried=list(state.legal_actions()))
        for _ in range(self.sims):
            node = root
            while not node.state.is_terminal() and not node.untried and node.children:
                node = self._uct_select(node)
            if not node.state.is_terminal() and node.untried:
                action = node.untried.pop(self.rng.randrange(len(node.untried)))
                child_state = node.state.apply(action)
                node = _Node(state=child_state, parent=node, action=action, untried=list(child_state.legal_actions()))
                node.parent.children.append(node)
            value = self._net_value(node.state) if not node.state.is_terminal() else node.state.returns()[0]
            while node is not None:
                node.visits += 1
                node.value_sum += value
                node = node.parent
        return root

    def _net_value(self, state: GoChessState) -> float:
        with torch.no_grad():
            x = obs_to_tensor(state).unsqueeze(0)
            _, v = self.net(x)
            return v.item()

    def select_action(self, state: GoChessState):
        root = self.search(state)
        if not root.children:
            raise ValueError("terminal")
        visits = [c.visits for c in root.children]
        if self.temp != 1.0:
            visits = [v ** (1.0 / self.temp) for v in visits]
        if self.noise > 0:
            visits = [v + self.rng.gauss(0, self.noise) for v in visits]
        best = max(zip(root.children, visits), key=lambda x: x[1])[0]
        return best.action


def collect(games: int, uct_sims: int, seed: int):
    X, PI, Z, entropy = [], [], [], []
    state0 = scenarios()["initial_5x5"].state
    for g in range(games):
        s = state0
        traj = []
        teacher = UCTAgent(simulations=uct_sims, seed=seed + g)
        while not s.is_terminal():
            root = teacher.search(s)
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


def eval_net_vs_uct(net: GoChessNet, net_sims: int, uct_sims: int, games: int, seed: int) -> float:
    net_agent = NetAgent(net, sims=net_sims, seed=seed)
    uct_factory = lambda s: UCTAgent(simulations=uct_sims, seed=s)
    state = scenarios()["mixed_midgame"].state
    wins = 0
    for g in range(games):
        gs = seed + g
        t, _ = play_game(state, net_agent, uct_factory(gs * 2 + 1))
        if t.returns()[0] > 0:
            wins += 1
        t2, _ = play_game(state, uct_factory(gs * 2), net_agent)
        if t2.returns()[0] > 0:
            wins += 1
    return wins / (2 * games)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--uct-simulations", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, default=Path("results/aed"))
    # AED params
    ap.add_argument("--beta-l", type=float, default=1.0)
    ap.add_argument("--beta-c", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--lambda-b", type=float, default=0.9)
    ap.add_argument("--Bmax", type=float, default=1.0)
    ap.add_argument("--temp-base", type=float, default=1.0)
    ap.add_argument("--temp-shift", type=float, default=2.5)
    ap.add_argument("--noise-base", type=float, default=0.0)
    ap.add_argument("--noise-shift", type=float, default=0.3)
    ap.add_argument("--shift-duration", type=int, default=3)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)

    net = GoChessNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    # AED state
    Bt = 0.0
    prev_vloss = None
    prev_pi_mean = None
    shift_remaining = 0
    log = []

    for it in range(a.epochs):
        # mode-shift params
        if shift_remaining > 0:
            temp = a.temp_shift
            noise = a.noise_shift
            shift_remaining -= 1
        else:
            temp = a.temp_base
            noise = a.noise_base

        # collect with current temp/noise
        X, PI, Z, ent = collect(a.games, a.uct_simulations, a.seed + it * 1000)

        ds = DataLoader(TensorDataset(X, PI, Z), batch_size=64, shuffle=True)
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
            tv = policy_tv(pm.unsqueeze(0), prev_pi_mean.unsqueeze(0)) if prev_pi_mean is not None else 0.0
            prev_pi_mean = pm.clone()

        # AED signals
        Lt = (prev_vloss - vloss) if prev_vloss is not None else 0.0
        Ct = 1.0 - tv  # higher when policy stable vs teacher
        Ut = a.beta_l * Lt + a.beta_c * Ct
        target = a.tau  # homeostatic threshold
        Bt = min(a.Bmax, a.lambda_b * Bt + max(0.0, target - Ut))
        prev_vloss = vloss

        # mode shift trigger
        shifted = False
        if Bt >= a.Bmax and shift_remaining == 0:
            shift_remaining = a.shift_duration
            Bt = 0.0  # reset after trigger
            shifted = True

        # periodic eval
        if it % 5 == 0 or it == a.epochs - 1:
            wr = eval_net_vs_uct(net, net_sims=32, uct_sims=128, games=4, seed=a.seed + it)
        else:
            wr = None

        row = {
            "iter": it, "vloss": vloss, "policy_acc": acc,
            "policy_tv": tv, "root_entropy": ent,
            "Lt": Lt, "Ct": Ct, "Ut": Ut, "Bt": Bt,
            "temp": temp, "noise": noise, "shifted": shifted,
            "winrate_vs_uct128": wr,
        }
        log.append(row)
        print(row)
        torch.save(net.state_dict(), a.output / f"ckpt_{it}.pt")

    (a.output / "aed_log.json").write_text(json.dumps(log, indent=2) + "\\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())'''

with open('experiments/train_aed.py', 'w') as f:
    f.write(txt)
print('written')