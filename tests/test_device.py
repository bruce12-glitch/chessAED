"""Device helper gates."""
from __future__ import annotations

import unittest

import torch

from training.device import optimizer_to_device, resolve_device, seed_all


class DeviceTests(unittest.TestCase):
    def test_resolve_cpu(self) -> None:
        self.assertEqual(resolve_device("cpu"), torch.device("cpu"))

    def test_resolve_auto_matches_availability(self) -> None:
        want = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.assertEqual(resolve_device("auto"), want)

    def test_resolve_cuda_without_cuda_raises(self) -> None:
        if torch.cuda.is_available():
            self.assertEqual(resolve_device("cuda"), torch.device("cuda"))
        else:
            with self.assertRaises(RuntimeError):
                resolve_device("cuda")

    def test_resolve_unknown_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_device("tpu")

    def test_seed_all_reproducible_on_cpu(self) -> None:
        seed_all(0, torch.device("cpu"))
        a = torch.randn(8)
        seed_all(0, torch.device("cpu"))
        b = torch.randn(8)
        self.assertTrue(bool((a == b).all()))

    def test_optimizer_state_moves_to_device(self) -> None:
        import torch.nn as nn

        net = nn.Linear(4, 2)
        opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
        x = torch.randn(3, 4)
        opt.zero_grad()
        net(x).sum().backward()
        opt.step()  # creates momentum buffers
        self.assertTrue(any(torch.is_tensor(v) for s in opt.state.values() for v in s.values()))
        optimizer_to_device(opt, torch.device("cpu"))
        for s in opt.state.values():
            for v in s.values():
                if torch.is_tensor(v):
                    self.assertEqual(v.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
