"""Main-matrix planning gates (manifest-native)."""
from __future__ import annotations

import unittest

from experiments import manifest as MF
from experiments.run_matrix import DEFAULT_ARMS, plan


class MainMatrixPlanTests(unittest.TestCase):
    def test_four_cnn_arms_default(self) -> None:
        self.assertEqual(DEFAULT_ARMS.split(","),
                         ["cnn_baseline", "cnn_fixed_noise",
                          "cnn_random_trigger", "cnn_aed"])

    def test_plan_pairs_seeds_per_arm(self) -> None:
        m = MF.load()
        cells = plan(m, DEFAULT_ARMS.split(","), [0, 1])
        self.assertEqual(len(cells), 8)
        self.assertEqual([s for _, s in cells], [0, 1] * 4)

    def test_gnn_arm_rejected_from_main_matrix(self) -> None:
        m = MF.load()
        with self.assertRaises(MF.ManifestError):
            plan(m, ["cnn_baseline", "gnn_baseline"], [0])

    def test_unknown_arm_rejected(self) -> None:
        m = MF.load()
        with self.assertRaises(MF.ManifestError):
            plan(m, ["nope"], [0])

    def test_main_arms_differ_only_in_mode(self) -> None:
        m = MF.load()
        MF.assert_arm_parity(m, DEFAULT_ARMS.split(","))
        ds = {MF.args_to_dict(MF.trainer_args(m, a, 0, "x"))["mode"]
              for a in DEFAULT_ARMS.split(",")}
        self.assertEqual(ds, {"off", "fixed-noise", "random", "detector"})


if __name__ == "__main__":
    unittest.main()
