"""Boredom accumulator with optional random-trigger control mode."""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class AEDDetector:
    tau: float = 0.01
    lam: float = 0.9
    bmax: float = 0.5
    mode: str = "detector"  # detector | random | off
    random_p: float = 0.2
    _b: float = 0.0
    _rng: random.Random | None = None

    def reset(self, seed: int = 0) -> None:
        self._b = 0.0
        self._rng = random.Random(seed)

    def update(self, ut: float) -> tuple[float, bool]:
        if self.mode == "off":
            return 0.0, False
        if self.mode == "random":
            if self._rng is None:
                self._rng = random.Random(0)
            return 0.0, self._rng.random() < self.random_p
        self._b = min(self.bmax, self.lam * self._b + max(0.0, self.tau - ut))
        if self._b >= self.bmax:
            self._b = 0.0
            return self._b, True
        return self._b, False
