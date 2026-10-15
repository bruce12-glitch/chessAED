"""Evaluate trained net + MCTS vs UCT teacher from mixed_midgame."""
from __future__ import annotations
import argparse
import math
from pathlib import Path

import torch

from training.device import resolve_device

from gochess.core import GoChessState
from gochess.nn import GoChessNet, obs_to_tensor
from gochess.representation import encode_for
from gochess.scenarios import scenarios
from gochess.search import UCTAgent, _Node
from gochess.evaluate import play_game, summarize


class NetAgent:
    def __init__(self, net: GoChessNet, sims: int = 32, seed: int = 0,
                 device: str = "auto"):
        self.device = resolve_device(device)
        self.net = net.to(self.device)
        self.sims = sims
        self.rng = __import__("random").Random(seed)

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
            # rollout with net value
            value = self._net_value(node.state) if not node.state.is_terminal() else node.state.returns()[0]
            while node is not None:
                node.visits += 1
                node.value_sum += value
                node = node.parent
        return root

    def _net_value(self, state: GoChessState) -> float:
        with torch.no_grad():
            x = encode_for(self.net, state).unsqueeze(0).to(self.device)
            _, v = self.net(x)
            return v.item()

    def select_action(self, state: GoChessState):
        root = self.search(state)
        if not root.children:
            raise ValueError("terminal")
        best = max(root.children, key=lambda c: c.visits)
        return best.action


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--net-sims", type=int, default=32)
    ap.add_argument("--uct-sims", type=int, default=128)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    a = ap.parse_args()

    device = resolve_device(a.device)
    net = GoChessNet()
    net.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    net.to(device)
    net.eval()

    state = scenarios()["mixed_midgame"].state
    net_agent = NetAgent(net, sims=a.net_sims, seed=a.seed, device=a.device)
    uct_agent = lambda s: UCTAgent(simulations=a.uct_sims, seed=s)

    records = []
    for g in range(a.games):
        gs = a.seed + g
        t, _ = play_game(state, net_agent, uct_agent(gs * 2 + 1))
        records.append(t.returns()[0])
        t2, _ = play_game(state, uct_agent(gs * 2), net_agent)
        records.append(t2.returns()[0])

    wins = sum(r > 0 for r in records)
    draws = sum(r == 0 for r in records)
    losses = sum(r < 0 for r in records)
    print(f"Net({a.net_sims}) vs UCT({a.uct_sims}) from mixed_midgame: W={wins} D={draws} L={losses} winrate={wins/len(records):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())