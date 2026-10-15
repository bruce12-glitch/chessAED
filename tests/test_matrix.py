"""Matrix planning + escape-reporting gates (synthetic histories)."""
from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.aggregate_results import seed_escape
from experiments.run_matrix import EXP_NAME, TRAINER_MODE, plan_cells


class MatrixPlanTests(unittest.TestCase):
    def test_modes_map_to_single_variable(self) -> None:
        cells = plan_cells(["baseline", "fixed-noise", "aed", "random"], 2)
        self.assertEqual(len(cells), 8)
        exps = {c[1] for c in cells}
        self.assertEqual(exps, {"baseline_v1", "noise_v1", "aed_v1", "random_v1"})
        self.assertEqual(TRAINER_MODE["aed"], "detector")

    def test_unknown_mode_rejected(self) -> None:
        with self.assertRaises(ValueError):
            plan_cells(["gnn"], 1)

    def test_escape_reported_on_synthetic_plateau_then_recovery(self) -> None:
        with TemporaryDirectory() as d:
            seed = Path(d) / "seed_000"
            seed.mkdir()
            metrics = [{"iter": i, "tv": tv, "entropy": 2.5, "openings": [7, 7, 7, 7]}
                       for i, tv in enumerate([0.3, 0.2, 0.1, 0.09, 0.08, 0.08, 0.08])]
            (seed / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            wrs = [0.30, 0.25, 0.20, 0.20, 0.20, 0.45, 0.50]
            for i, w in enumerate(wrs):
                wins = round(w * 4)
                with (seed / f"eval_step{i}.csv").open("w", newline="") as h:
                    wri = csv.DictWriter(h, fieldnames=["return_p0", "player0_agent"])
                    wri.writeheader()
                    for g in range(4):
                        wri.writerow({"return_p0": 1.0 if g < wins else -1.0,
                                      "player0_agent": "net"})
            got = seed_escape(seed)
            self.assertIsNotNone(got["event_iter"])
            self.assertTrue(got["escaped"])

    def test_no_event_when_strong(self) -> None:
        with TemporaryDirectory() as d:
            seed = Path(d) / "seed_000"
            seed.mkdir()
            metrics = [{"iter": i, "tv": 0.1, "entropy": 2.5, "openings": [7, 7, 7, 7]}
                       for i in range(4)]
            (seed / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            for i in range(4):
                with (seed / f"eval_step{i}.csv").open("w", newline="") as h:
                    wri = csv.DictWriter(h, fieldnames=["return_p0", "player0_agent"])
                    wri.writeheader()
                    for g in range(4):
                        wri.writerow({"return_p0": 1.0, "player0_agent": "net"})
            got = seed_escape(seed)
            self.assertIsNone(got["event_iter"])
            self.assertFalse(got["escaped"])


if __name__ == "__main__":
    unittest.main()
