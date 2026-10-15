"""GNN + representation tests. Require torch; skipped where it is absent."""
from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError:           # sandbox without torch
    torch = None

from gochess.core import GoChessState
from gochess.scenarios import scenarios

needs_torch = unittest.skipUnless(torch is not None, "torch not installed")


def _states(k: int = 10):
    out = [s.state for s in scenarios().values()]
    rng, s = random.Random(3), GoChessState.initial()
    for _ in range(k):
        s = s.apply(rng.choice(s.legal_actions()))
        out.append(s)
    return out


@needs_torch
class GNNTests(unittest.TestCase):
    def setUp(self) -> None:
        from gochess.representation import build_model
        torch.manual_seed(0)
        self.gnn = build_model("gnn", hidden=32, layers=2).eval()
        self.cnn = build_model("cnn", channels=16, blocks=1).eval()

    def test_output_shapes_match_cnn(self) -> None:
        from gochess.representation import encode_for
        for net in (self.gnn, self.cnn):
            x = torch.stack([encode_for(net, s) for s in _states(3)])
            logits, v = net(x)
            self.assertEqual(tuple(logits.shape), (x.shape[0], 651))
            self.assertEqual(tuple(v.shape), (x.shape[0],))
            self.assertTrue(bool(((v > -1) & (v < 1)).all()))
            self.assertTrue(bool(torch.isfinite(logits).all()))

    def test_batch_equals_single(self) -> None:
        from gochess.representation import encode_for
        xs = torch.stack([encode_for(self.gnn, s) for s in _states(4)])
        with torch.no_grad():
            lb, vb = self.gnn(xs)
            for i in range(xs.shape[0]):
                l1, v1 = self.gnn(xs[i:i + 1])
                self.assertTrue(torch.allclose(lb[i], l1[0], atol=1e-5))
                self.assertTrue(torch.allclose(vb[i], v1[0], atol=1e-5))

    def test_policy_probs_only_over_legal_moves(self) -> None:
        from gochess.representation import policy_probs
        for net in (self.gnn, self.cnn):
            for s in _states():
                p = policy_probs(net, s)
                legal = set(s.legal_action_ids())
                if s.is_terminal():
                    self.assertEqual(float(p.sum()), 0.0)
                    continue
                self.assertAlmostEqual(float(p.sum()), 1.0, places=5)
                for i in range(len(p)):
                    if i not in legal:
                        self.assertEqual(float(p[i]), 0.0)
                self.assertTrue(all(float(p[i]) > 0 for i in legal))

    def test_place_and_move_logits_land_in_encode_layout(self) -> None:
        """Perturbing the place head bias shifts exactly the place block."""
        from gochess.representation import encode_for
        x = encode_for(self.gnn, scenarios()["mixed_midgame"].state).unsqueeze(0)
        with torch.no_grad():
            base, _ = self.gnn(x)
            self.gnn.place_head.bias += 1.0
            moved, _ = self.gnn(x)
            self.gnn.place_head.bias -= 1.0
        diff = (moved - base).abs()[0]
        self.assertTrue(bool((diff[:25] > 0.5).all()))
        self.assertTrue(bool((diff[25:] < 1e-6).all()))

    def test_checkpoint_reload_reproducible(self) -> None:
        from gochess.representation import (build_model, encode_for,
                                            load_model_from_checkpoint, model_kwargs)
        from training.checkpointing import save_checkpoint
        for arch, kw in (("gnn", {"hidden": 32, "layers": 2}),
                         ("cnn", {"channels": 16, "blocks": 1})):
            net = build_model(arch, **kw)
            opt = torch.optim.Adam(net.parameters())
            with tempfile.TemporaryDirectory() as d:
                path = Path(d) / "c.pt"
                save_checkpoint(path, net, opt, 3,
                                {"arch": arch, "model_kwargs": model_kwargs(arch, kw)})
                net2, ck = load_model_from_checkpoint(path)
            self.assertEqual(ck["iteration"], 3)
            x = torch.stack([encode_for(net, s) for s in _states(3)])
            net.eval()
            with torch.no_grad():
                a, b = net(x), net2(x)
            self.assertTrue(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))

    def test_same_seed_same_init(self) -> None:
        from gochess.representation import build_model
        torch.manual_seed(5); a = build_model("gnn", hidden=16, layers=2)
        torch.manual_seed(5); b = build_model("gnn", hidden=16, layers=2)
        for p, q in zip(a.parameters(), b.parameters()):
            self.assertTrue(torch.equal(p, q))

    def test_mcts_compat_returns_legal_action(self) -> None:
        from experiments.eval_net import NetAgent
        for net in (self.gnn, self.cnn):
            for name in ("mixed_midgame", "encirclement_choice"):
                s = scenarios()[name].state
                a = NetAgent(net, sims=6, seed=1).select_action(s)
                self.assertIn(a, s.legal_actions())

    def test_trains_one_step_and_loss_decreases_on_fixed_batch(self) -> None:
        import torch.nn.functional as F
        from gochess.representation import encode_for
        ss = _states(8)
        x = torch.stack([encode_for(self.gnn, s) for s in ss])
        pi = torch.zeros(len(ss), 651)
        for i, s in enumerate(ss):
            ids = s.legal_action_ids()
            for j in ids:
                pi[i, j] = 1.0 / len(ids)
        z = torch.tensor([0.5, -0.5] * (len(ss) // 2))
        self.gnn.train()
        opt = torch.optim.Adam(self.gnn.parameters(), lr=1e-2)
        losses = []
        for _ in range(30):
            lg, v = self.gnn(x)
            loss = F.mse_loss(v, z) + F.cross_entropy(lg, pi)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0])


class RepresentationTorchFree(unittest.TestCase):
    def test_arch_validation(self) -> None:
        from gochess.representation import check_arch, model_kwargs
        with self.assertRaises(ValueError):
            check_arch("transformer")
        self.assertEqual(model_kwargs("gnn", {"hidden": 8})["layers"], 4)


if __name__ == "__main__":
    unittest.main()
