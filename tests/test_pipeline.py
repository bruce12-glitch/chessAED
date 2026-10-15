"""Torch-free tests: events, stats, manifest, parity, preflight helpers, gate, analysis."""
from __future__ import annotations

import csv
import json
import math
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from evaluation import events as E
from evaluation import stats as S
from experiments import analyze as AN
from experiments import manifest as MF
from experiments import run_preflight as PF
from experiments import run_representation_ablation as AB


class EventTests(unittest.TestCase):
    def test_weak_stable_then_escape(self) -> None:
        it = [0, 2, 4, 6, 8, 10]
        r = E.analyse_series(it, [0.0] * 6, [2.0] * 6, [0.3, 0.3, 0.3, 0.3, 0.5, 0.5])
        self.assertEqual(r["weak_stable_idx"], 3)              # flags at idx 1,2,3 (idx 0 has no dH)
        self.assertEqual(r["escape_idx"], 4)
        self.assertTrue(r["escaped"])

    def test_float_boundary_exactly_15pp_counts(self) -> None:
        self.assertEqual(E.escape_index([0.2, 0.35, 0.35], 0), 1)

    def test_escape_any_two_consecutive_not_just_last_two(self) -> None:
        self.assertEqual(E.escape_index([0.3, 0.5, 0.5, 0.3, 0.3], 0), 1)

    def test_single_eval_gain_is_not_escape(self) -> None:
        self.assertIsNone(E.escape_index([0.3, 0.5, 0.3, 0.3], 0))

    def test_conditions_each_required(self) -> None:
        it = [0, 1, 2, 3]
        base = dict(tv=[0.0] * 4, ents=[2.0] * 4, wr=[0.3] * 4)
        self.assertIsNotNone(E.analyse_series(it, base["tv"], base["ents"], base["wr"])["weak_stable_idx"])
        self.assertIsNone(E.analyse_series(it, [0.0, 0.2, 0.2, 0.2], base["ents"], base["wr"])["weak_stable_idx"])
        self.assertIsNone(E.analyse_series(it, base["tv"], [2.0, 2.2, 2.4, 2.6], base["wr"])["weak_stable_idx"])
        self.assertIsNone(E.analyse_series(it, base["tv"], base["ents"], [0.3, 0.45, 0.45, 0.45])["weak_stable_idx"])


class StatsTests(unittest.TestCase):
    def test_permutation_exact(self) -> None:
        self.assertAlmostEqual(S.paired_permutation_pvalue([0.1] * 10), 2 / 1024)
        self.assertEqual(S.paired_permutation_pvalue([0.1, -0.1]), 1.0)
        self.assertTrue(math.isnan(S.paired_permutation_pvalue([0.1])))

    def test_permutation_monte_carlo_close_to_exact(self) -> None:
        d = [0.1, 0.2, -0.05, 0.15, 0.12, 0.08, 0.2, 0.1, 0.05, 0.11, 0.09, -0.02, 0.14, 0.2, 0.1, 0.1, 0.07, 0.1, 0.12, 0.09]
        exact = S.paired_permutation_pvalue(d, n_perm=2 ** 20)
        mc = S.paired_permutation_pvalue(d, n_perm=2000, seed=1)
        self.assertLess(abs(exact - mc), 0.01)

    def test_bootstrap_and_holm(self) -> None:
        lo, hi = S.paired_bootstrap_ci([0.1, 0.2, 0.3, 0.2], 2000, 0)
        self.assertLessEqual(lo, 0.2); self.assertGreaterEqual(hi, 0.2)
        self.assertTrue(math.isnan(S.paired_bootstrap_ci([0.1])[0]))
        self.assertEqual([round(x, 3) for x in S.holm([0.01, 0.04, 0.03])], [0.03, 0.06, 0.06])


class ManifestTests(unittest.TestCase):
    def _copy(self, d: Path) -> Path:
        p = d / "m.yaml"
        shutil.copyfile(MF.DEFAULT_PATH, p)
        return p

    def test_lock_verify_tamper_and_relock(self) -> None:
        with TemporaryDirectory() as d:
            p = self._copy(Path(d))
            m2 = MF.load(p)
            m2["lock"] = {"locked": False, "manifest_hash": None}
            m2["scenario"]["digest"] = None
            m2["code"] = {"git_commit": None, "file_sha256": None}
            m2["evaluation"]["probes"]["reference"] = None
            for a in m2["models"]:
                m2["models"][a]["param_count"] = None
            import yaml as _yaml
            p.write_text(_yaml.safe_dump(m2, sort_keys=False), encoding="utf-8")
            m = MF.lock(p, param_counts={"cnn": 111, "gnn": 222}, probe_reference={"mixed_midgame": 5})
            MF.verify(MF.load(p))
            self.assertEqual(m["models"]["gnn"]["param_count"], 222)
            with self.assertRaises(MF.ManifestError):
                MF.lock(p, param_counts={}, probe_reference={})
            txt = p.read_text()
            cur = f"tau: {m['aed']['tau']}"
            assert cur in txt, "tamper target must match locked tau"
            p.write_text(txt.replace(cur, "tau: 999.25", 1))
            with self.assertRaises(MF.ManifestError):
                MF.verify(MF.load(p))

    def test_unlocked_refused_when_required(self) -> None:
        m = MF.clone(MF.load())
        m["lock"] = {"locked": False, "manifest_hash": None}
        with self.assertRaises(MF.ManifestError):
            MF.verify(m, require_locked=True)

    def test_copy_into_preserves_source_filename(self) -> None:
        with TemporaryDirectory() as d:
            p = self._copy(Path(d))
            out = Path(d) / "out"
            dst = MF.copy_into(out, p)
            self.assertEqual(dst.name, p.name)

    def test_copy_into_idempotent_and_refuses_mixing(self) -> None:
        with TemporaryDirectory() as d:
            p = self._copy(Path(d))
            out = Path(d) / "out"
            MF.copy_into(out, p); MF.copy_into(out, p)
            p.write_text(p.read_text() + "\n# edit\n")
            with self.assertRaises(MF.ManifestError):
                MF.copy_into(out, p)

    def test_arm_parity_and_detection(self) -> None:
        m = MF.load()
        shared = MF.assert_arm_parity(m)
        self.assertEqual(shared["games"], "16")
        real = MF.trainer_args

        def skewed(mm, arm, seed, out, *a, **k):
            args = real(mm, arm, seed, out, *a, **k)
            if arm == "gnn_aed":
                args[args.index("--games") + 1] = "17"
            return args
        with mock.patch.object(MF, "trainer_args", skewed):
            with self.assertRaises(MF.ManifestError):
                MF.assert_arm_parity(m)

    def test_arms_differ_only_in_arch_and_mode(self) -> None:
        m = MF.load()
        a = MF.args_to_dict(MF.trainer_args(m, "cnn_baseline", 0, "x"))
        b = MF.args_to_dict(MF.trainer_args(m, "gnn_aed", 0, "x"))
        diff = {k for k in a if a[k] != b[k]}
        self.assertEqual(diff, {"experiment", "arch", "model-kwargs", "mode"})

    def test_cli_must_match_manifest(self) -> None:
        m = MF.load()
        got = MF.args_to_dict(MF.trainer_args(m, "cnn_aed", 0, "x"))
        MF.check_cli_matches(m, "cnn_aed", 0, got)
        got["tau"] = "0.5"
        with self.assertRaises(MF.ManifestError):
            MF.check_cli_matches(m, "cnn_aed", 0, got)


class PreflightHelperTests(unittest.TestCase):
    def test_budget_parity_detects_extra_compute(self) -> None:
        row = {"games": 1, "sims_per_move": 4, "grad_steps": 2, "teacher_sims": 40}
        a = {"x": [dict(row)] * 3, "y": [dict(row)] * 3}
        ok, _ = PF.budget_parity(a, {"x": 2, "y": 2}, {"x": {8}, "y": {8}})
        self.assertTrue(ok)
        b = {"x": [dict(row)] * 3, "y": [dict(row, grad_steps=3)] * 3}
        self.assertFalse(PF.budget_parity(b, {"x": 2, "y": 2}, {"x": {8}, "y": {8}})[0])
        self.assertFalse(PF.budget_parity(a, {"x": 2, "y": 4}, {"x": {8}, "y": {8}})[0])
        self.assertFalse(PF.budget_parity(a, {"x": 2, "y": 2}, {"x": {8}, "y": {16}})[0])
        c = {"x": [dict(row)] * 3, "y": [dict(row, teacher_sims=99)] * 3}     # length differences allowed
        self.assertTrue(PF.budget_parity(c, {"x": 2, "y": 2}, {"x": {8}, "y": {8}})[0])

    def test_helpers(self) -> None:
        self.assertEqual(PF.missing_columns([{"a": 1}], ["a", "b"]), ["b"])
        self.assertTrue(PF.runs_identical([{"vloss": 1.0}], [{"vloss": 1.0}]))
        self.assertFalse(PF.runs_identical([{"vloss": 1.0}], [{"vloss": 1.0000001}]))
        args = PF.tiny_args(MF.load(), "gnn_baseline", 0, Path("o"))
        d = MF.args_to_dict(args)
        self.assertNotIn("manifest", d)
        self.assertEqual((d["games"], d["arch"], d["mode"]), ("1", "gnn", "off"))

    def test_aed_can_fire_probe(self) -> None:
        m = MF.clone(MF.load())
        m["aed"].update(tau=0.01, bmax=0.5, lam=0.9)
        self.assertEqual(PF.aed_can_fire(m)["fires"], 0)
        m["aed"]["tau"] = 0.6
        self.assertGreater(PF.aed_can_fire(m)["fires"], 0)


def _write_shard(root: Path, arm: str, seed: int, *, mode: str, wr_eval, f400, f800,
                 complete=True, shifted_at=None) -> None:
    sd = root / arm / f"seed_{seed:03d}"
    sd.mkdir(parents=True)
    (sd / "config.json").write_text(json.dumps(
        {"arch": "cnn", "mode": mode, "param_count": 123, "bmax": 0.5}))
    rows = [{"iter": i, "vloss": 1.0 - 0.01 * i, "tv": 0.02, "entropy": 2.0, "repeat": 0.6,
             "Bt": 0.1 * i if mode == "detector" else 0.0, "shifted": i == shifted_at,
             "disagreement": 0.5, "openings": [1]} for i in range(6)]
    (sd / "metrics.json").write_text(json.dumps(rows))

    def games(p: Path, k: int, n: int) -> None:
        with p.open("w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=["player0_agent", "return_p0"])
            w.writeheader()
            for g in range(n):
                win = g < k
                net_p0 = g % 2 == 0
                w.writerow({"player0_agent": "net" if net_p0 else "uct400",
                            "return_p0": (1.0 if win else -1.0) * (1 if net_p0 else -1)})
    for step, k in zip(range(6), wr_eval):
        games(sd / f"eval_step{step}.csv", k, 10)
    if complete:
        games(sd / "final_uct400_step5.csv", f400, 10)
        games(sd / "final_uct800_step5.csv", f800, 10)
        (sd / "DONE").write_text("ok")


class GateTests(unittest.TestCase):
    def _rows(self, n=20, v0=1.0, v1=0.8, bad=False):
        rows = [{"vloss": v0 + (v1 - v0) * i / (n - 1), "tv": 0.1, "entropy": 2.0} for i in range(n)]
        if bad:
            rows[5]["vloss"] = float("nan")
        return rows

    def test_gate_logic(self) -> None:
        m = MF.load()
        ok = AB.evaluate_gate(m, {0: self._rows(), 1: self._rows()}, [0, 1], {0: 3, 1: 3})
        self.assertTrue(ok["passed"])
        for series, evals in (({0: self._rows(bad=True), 1: self._rows()}, {0: 3, 1: 3}),
                              ({0: self._rows(v1=1.2), 1: self._rows()}, {0: 3, 1: 3}),
                              ({0: self._rows(n=5), 1: self._rows()}, {0: 3, 1: 3}),
                              ({0: self._rows(), 1: self._rows()}, {0: 3, 1: 0})):
            self.assertFalse(AB.evaluate_gate(m, series, [0, 1], evals)["passed"])

    def test_gnn_aed_blocked_until_gate_passes(self) -> None:
        with TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                AB.main(["--allow-unlocked", "--arms", "gnn_aed", "--dry-run", "--raw", d])
            raw = Path(d)
            for s in (0, 1, 2):
                sd = raw / "gnn_baseline" / f"seed_{s:03d}"
                sd.mkdir(parents=True)
                (sd / "metrics.json").write_text(json.dumps(self._rows()))
                (sd / "eval_step0.csv").write_text("a\n1\n")
            self.assertEqual(AB.main(["--allow-unlocked", "--check-gate", "gnn_baseline", "--raw", d]), 0)
            self.assertEqual(AB.main(["--allow-unlocked", "--arms", "gnn_aed", "--dry-run", "--raw", d]), 0)


class AnalysisTests(unittest.TestCase):
    def _build(self, root: Path) -> None:
        for s in range(4):
            _write_shard(root, "cnn_baseline", s, mode="off", wr_eval=(3, 3, 3, 3, 3, 3), f400=3 + s % 2, f800=2)
            _write_shard(root, "cnn_aed", s, mode="detector", wr_eval=(3, 3, 3, 3, 6, 6),
                         f400=5 + s % 2, f800=4, shifted_at=3)

    def test_outputs_and_values(self) -> None:
        with TemporaryDirectory() as d:
            raw, out = Path(d) / "raw", Path(d) / "out"
            self._build(raw)
            AN.main(["--raw", str(raw), "--out", str(out), "--allow-unlocked",
                     "--n-perm", "5000", "--n-boot", "500"])
            for f in ("per_seed.csv", "paired_differences.csv", "paired_summary.csv",
                      "primary_result.json", "escape_summary.csv", "final_eval.csv", "curves.csv",
                      "triggers.csv", "learning_curves.png", "learning_curves.pdf",
                      "aed_triggers.png", "aed_triggers.pdf", "final_eval.png", "final_eval.pdf",
                      "experiment_manifest_v1.yaml", "analysis_summary.json"):
                self.assertTrue((out / f).exists(), f)
            per = list(csv.DictReader((out / "per_seed.csv").open()))
            self.assertEqual(len(per), 8)                           # per-seed rows, not means
            r = next(x for x in per if x["arm"] == "cnn_aed" and x["seed"] == "1")
            self.assertAlmostEqual(float(r["final_winrate_uct400"]), 0.6)
            self.assertEqual(r["n_triggers"], "1")
            self.assertEqual(r["first_trigger_iter"], "3")
            summ = list(csv.DictReader((out / "paired_summary.csv").open()))
            p = next(x for x in summ if x["kind"] == "primary" and x["metric"] == "final_winrate_uct400")
            self.assertEqual(p["n_pairs"], "4")
            self.assertAlmostEqual(float(p["mean_diff"]), 0.2)       # (0.5-0.3)/(0.6-0.4)
            self.assertAlmostEqual(float(p["p_perm"]), 2 / 16)       # exact, n=4, all positive
            prim = json.loads((out / "primary_result.json").read_text())
            self.assertFalse(prim["reject_null"])
            esc = {(x["arm"], x["event"]): x for x in csv.DictReader((out / "escape_summary.csv").open())}
            self.assertEqual(esc[("cnn_aed", "escape")]["k"], "4")   # 0.3 -> 0.6 held
            self.assertEqual(esc[("cnn_baseline", "escape")]["k"], "0")
            self.assertEqual(esc[("cnn_baseline", "weak_stable")]["k"], "4")   # stuck but never escapes

    def test_partial_shard_handled_and_complete_required_by_default(self) -> None:
        with TemporaryDirectory() as d:
            raw, out = Path(d) / "raw", Path(d) / "out"
            self._build(raw)
            _write_shard(raw, "cnn_aed", 9, mode="detector", wr_eval=(3, 3, 3, 3, 3, 3),
                         f400=0, f800=0, complete=False)
            with self.assertRaises(SystemExit):
                AN.main(["--raw", str(raw), "--out", str(out), "--allow-unlocked"])
            AN.main(["--raw", str(raw), "--out", str(out), "--allow-unlocked",
                     "--allow-incomplete", "--n-perm", "200", "--n-boot", "200"])
            per = list(csv.DictReader((out / "per_seed.csv").open()))
            part = next(x for x in per if x["seed"] == "9")
            self.assertEqual(part["complete"], "False")
            self.assertEqual(part["final_winrate_uct400"].lower(), "nan")
            summ = list(csv.DictReader((out / "paired_summary.csv").open()))
            p = next(x for x in summ if x["kind"] == "primary" and x["metric"] == "final_winrate_uct400")
            self.assertEqual(p["n_pairs"], "4")                      # seed 9 has no control pair


if __name__ == "__main__":
    unittest.main()
