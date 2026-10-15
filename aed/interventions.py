"""Single intervention: root temp + noise for a fixed duration. Nothing else."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Intervention:
    temp_base: float = 1.0
    temp_shift: float = 2.5
    noise_base: float = 0.0
    noise_shift: float = 0.3
    duration: int = 3


@dataclass
class InterventionState:
    remaining: int = 0

    def params(self, iv: Intervention) -> tuple[float, float]:
        if self.remaining > 0:
            return iv.temp_shift, iv.noise_shift
        return iv.temp_base, iv.noise_base

    def trigger(self, iv: Intervention) -> None:
        self.remaining = iv.duration

    def step(self) -> None:
        if self.remaining > 0:
            self.remaining -= 1
