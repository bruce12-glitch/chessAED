"""AED unit gates: detector, intervention, controller."""
from __future__ import annotations

import unittest

from aed.controller import AEDController, ControllerConfig
from aed.detector import AEDDetector
from aed.interventions import Intervention, InterventionState
from aed.signals import leverage, utility


class DetectorTests(unittest.TestCase):
    def test_fires_on_sustained_low_utility(self) -> None:
        d = AEDDetector(tau=0.5, lam=1.0, bmax=1.0)
        d.reset(0)
        fired = [d.update(-1.0)[1] for _ in range(3)]
        self.assertTrue(any(fired))

    def test_quiet_on_high_utility(self) -> None:
        d = AEDDetector(tau=0.01, lam=0.9, bmax=0.5)
        d.reset(0)
        self.assertFalse(any(d.update(5.0)[1] for _ in range(10)))

    def test_random_mode_fires_sometimes(self) -> None:
        d = AEDDetector(mode="random", random_p=0.5)
        d.reset(0)
        hits = sum(d.update(0.0)[1] for _ in range(50))
        self.assertGreater(hits, 5)
        self.assertLess(hits, 45)


class InterventionTests(unittest.TestCase):
    def test_shift_then_base(self) -> None:
        iv = Intervention(temp_shift=2.5, noise_shift=0.3, duration=2)
        st = InterventionState()
        self.assertEqual(st.params(iv), (1.0, 0.0))
        st.trigger(iv)
        self.assertEqual(st.params(iv), (2.5, 0.3))
        st.step()
        st.step()
        self.assertEqual(st.params(iv), (1.0, 0.0))


class ControllerTests(unittest.TestCase):
    def test_diagnostics_never_fires(self) -> None:
        c = AEDController(ControllerConfig(mode="diagnostics"), seed=0)
        for _ in range(10):
            d = c.step(1.0, 0.9)
        self.assertFalse(d["shifted"])
        self.assertEqual((d["temp"], d["noise"]), (1.0, 0.0))

    def test_drop_value_changes_utility(self) -> None:
        self.assertNotEqual(utility(1.0, 0.5, 2.0, 0.5, "none"),
                            utility(1.0, 0.5, 2.0, 0.5, "value"))
        self.assertGreater(leverage(0.0), leverage(0.9))


if __name__ == "__main__":
    unittest.main()
