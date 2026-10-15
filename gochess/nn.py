"""Minimal torch net for GoChess 5x5 Code Set 2 (CPU)."""
from __future__ import annotations
import torch
import torch.nn as nn


class GoChessNet(nn.Module):
    arch = "cnn"

    def __init__(self, size: int = 5, channels: int = 64, blocks: int = 2):
        super().__init__()
        self.size = size
        self.action_space = size * size + (size * size) ** 2 + 1
        self.stem = nn.Sequential(nn.Conv2d(11, channels, 3, padding=1), nn.ReLU())
        self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 8, 1), nn.ReLU(), nn.Flatten(),
            nn.Linear(8 * size * size, self.action_space),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 4, 1), nn.ReLU(), nn.Flatten(),
            nn.Linear(4 * size * size, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        h = self.tower(self.stem(x))
        return self.policy_head(h), self.value_head(h).squeeze(-1)


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return torch.relu(x + self.c2(torch.relu(self.c1(x))))


def obs_to_tensor(state) -> torch.Tensor:
    import numpy as np
    v = state.observation_tensor(0)
    return torch.tensor(np.array(v, dtype="float32").reshape(11, state.size, state.size))
