"""AED controller: signals -> detector -> one intervention. Diagnostics-only supported."""
from __future__ import annotations

from dataclasses import dataclass

from aed.detector import AEDDetector
from aed.interventions import Intervention, InterventionState
from aed.signals import learning_progress, leverage, utility


@dataclass
class ControllerConfig:
    beta_l: float = 2.0
    beta_c: float = 0.5
    tau: float = 0.01
    lam: float = 0.9
    bmax: float = 0.5
    mode: str = "detector"  # detector | random | off | diagnostics
    drop: str = "none"
    random_p: float = 0.2


class AEDController:
    def __init__(self, config: ControllerConfig,
                 intervention: Intervention | None = None, seed: int = 0):
        self.config = config
        self.detector = AEDDetector(tau=config.tau, lam=config.lam,
                                    bmax=config.bmax, mode=config.mode if config.mode != "diagnostics" else "off",
                                    random_p=config.random_p)
        self.detector.reset(seed)
        self.iv = intervention or Intervention()
        self.state = InterventionState()
        self.prev_vloss: float | None = None

    def step(self, vloss: float, tv: float) -> dict:
        lt = learning_progress(self.prev_vloss, vloss)
        ct = leverage(tv)
        ut = utility(lt, ct, self.config.beta_l, self.config.beta_c, self.config.drop)
        temp, noise = self.state.params(self.iv)
        bt, fired = (0.0, False) if self.config.mode == "diagnostics" else self.detector.update(ut)
        if fired:
            self.state.trigger(self.iv)
        self.state.step()
        self.prev_vloss = vloss
        return {"Lt": lt, "Ct": ct, "Ut": ut, "Bt": bt, "temp": temp,
                "noise": noise, "shifted": fired}
