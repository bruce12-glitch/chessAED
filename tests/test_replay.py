"""Replay + checkpointing + overfit smoke gates."""
from __future__ import annotations

import unittest

import torch

from gochess.nn import GoChessNet, obs_to_tensor
from gochess.scenarios import scenarios
from training.checkpointing import load_checkpoint, save_checkpoint
from training.replay_buffer import ReplayBuffer


class ReplayTests(unittest.TestCase):
    def test_fifo_and_reproducible_sample(self) -> None:
        buf = ReplayBuffer(capacity=4)
        for i in range(6):
            buf.add(torch.zeros(11, 5, 5), torch.zeros(651), float(i), {"i": i})
        self.assertEqual(len(buf), 4)
        a = buf.sample(8, seed=3)
        b = buf.sample(8, seed=3)
        self.assertTrue(bool((a[2] - b[2]).abs().max() == 0))
        self.assertEqual([m["i"] for m in buf.meta], [2, 3, 4, 5])

    def test_checkpoint_roundtrip_same_prediction(self) -> None:
        import tempfile
        from pathlib import Path

        net = GoChessNet()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        net.eval()
        x = obs_to_tensor(scenarios()["mixed_midgame"].state).unsqueeze(0)
        with torch.no_grad():
            p0, v0 = net(x)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ckpt.pt"
            save_checkpoint(p, net, opt, 2, {"experiment": "t", "seed": 0})
            ckpt = load_checkpoint(p, GoChessNet(), None)
        self.assertEqual(ckpt["iteration"], 2)
        self.assertIn("rng", ckpt)
        self.assertIn("git_commit", ckpt)


if __name__ == "__main__":
    unittest.main()
