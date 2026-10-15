"""Fixed-size FIFO replay with metadata. Uniform sampling only."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch


@dataclass
class ReplayBuffer:
    capacity: int
    obs: list = field(default_factory=list)
    pi: list = field(default_factory=list)
    z: list = field(default_factory=list)
    meta: list = field(default_factory=list)

    def add(self, ob: torch.Tensor, pi: torch.Tensor, z: float, meta: dict) -> None:
        self.obs.append(ob)
        self.pi.append(pi)
        self.z.append(float(z))
        self.meta.append(dict(meta))
        while len(self.obs) > self.capacity:
            self.obs.pop(0)
            self.pi.pop(0)
            self.z.pop(0)
            self.meta.pop(0)

    def __len__(self) -> int:
        return len(self.obs)

    def sample(self, n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = random.Random(seed)
        idx = [rng.randrange(len(self.obs)) for _ in range(n)]
        return (torch.stack([self.obs[i] for i in idx]),
                torch.stack([self.pi[i] for i in idx]),
                torch.tensor([self.z[i] for i in idx], dtype=torch.float32))
