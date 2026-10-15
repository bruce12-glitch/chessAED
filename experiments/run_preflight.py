"""Single preflight command. Exit 0 only if every check passes.

  python -m experiments.run_preflight [--allow-unlocked] [--out results/preflight]

Runs tiny but real trainings (same code path as the full runs) and checks
manifest integrity, arm parity, determinism, resume-after-kill, CSV schemas,
aggregator on partial+complete shards, AED firing ability, device/worker info.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from experiments import manifest as MF

ROOT = Path(__file__).resolve().parent.parent
TINY = {"games": "1", "uct-simulations": "4", "epochs": "3", "buffer": "64",
        "sample": "16", "batch": "8", "eval-every": "1", "eval-games": "1",
        "eval-sims": "8", "device": "cpu"}
CONTROL_ARMS = ["cnn_baseline", "cnn_fixed_noise", "cnn_random_trigger", "cnn_aed"]
EXTRA_ARMS = ["gnn_baseline"]
METRIC_COLS = ["iter", "vloss", "acc", "tv", "entropy", "Lt", "Ct", "Ut", "Bt", "temp",
               "noise", "shifted", "repeat", "games", "positions", "grad_steps",
               "sims_per_move", "teacher_sims", "openings"]
EVAL_COLS = ["experiment_id", "seed", "checkpoint_step", "game_id", "player0_agent",
             "player1_agent", "seating", "winner", "return_p0", "terminal_reason", "plies",
             "train_mcts_sims", "eval_mcts_sims", "config_hash"]
RUN_INFO_KEYS = ["arch", "param_count", "device", "cuda_available", "cuda_devices",
                 "cuda_visible_devices", "mcts_workers", "torch_threads", "cpu_count"]
BUDGET_KEYS = ["games", "sims_per_move", "grad_steps"]


# ---- pure helpers (unit-tested without torch) -------------------------------
def dict_to_args(d: dict[str, str]) -> list[str]:
    return [x for k, v in d.items() for x in (f"--{k}", v)]


def tiny_args(m: dict, arm: str, seed: int, out: Path, **over: str) -> list[str]:
    d = MF.args_to_dict(MF.trainer_args(m, arm, seed, out))
    d.pop("manifest")
    d.update(TINY)
    d.update(over)
    return dict_to_args(d)


def missing_columns(rows: list[dict], required: list[str]) -> list[str]:
    have = set(rows[0]) if rows else set()
    return [c for c in required if c not in have]


def runs_identical(a: list[dict], b: list[dict],
                   keys=("vloss", "acc", "tv", "entropy", "Bt", "openings", "positions")) -> bool:
    return len(a) == len(b) and all(
        all(x.get(k) == y.get(k) for k in keys) for x, y in zip(a, b))


def budget_parity(per_arm: dict[str, list[dict]], eval_games: dict[str, int],
                  eval_sims: dict[str, set]) -> tuple[bool, dict]:
    """Games/iteration, sims per move, gradient steps, #iterations, eval games and
    eval sims must be identical for every arm. Total teacher sims is informational
    (game length legitimately differs between arms)."""
    sig = {a: (len(r), tuple(tuple(x[k] for k in BUDGET_KEYS) for x in r))
           for a, r in per_arm.items()}
    ok = len(set(sig.values())) == 1 and len(set(eval_games.values())) == 1 \
        and len({frozenset(v) for v in eval_sims.values()}) == 1
    tot = {a: sum(x["teacher_sims"] for x in r) for a, r in per_arm.items()}
    return ok, {"iterations": {a: s[0] for a, s in sig.items()},
                "eval_games": eval_games, "eval_sims": {a: sorted(v) for a, v in eval_sims.items()},
                "total_teacher_sims_informational": tot}


def aed_can_fire(m: dict, n: int = 200) -> dict:
    """Frozen detector config on the weak-stable regime (flat vloss, tiny TV)."""
    from aed.controller import AEDController, ControllerConfig
    from aed.interventions import Intervention
    ae = m["aed"]
    c = AEDController(ControllerConfig(beta_l=ae["beta_l"], beta_c=ae["beta_c"], tau=ae["tau"],
                                       lam=ae["lam"], bmax=ae["bmax"], drop=ae["drop"],
                                       mode="detector"), Intervention(**ae["intervention"]), 0)
    fires, peak, ut = 0, 0.0, 0.0
    for _ in range(n):
        d = c.step(1.0, 0.02)
        fires += int(d["shifted"]); peak = max(peak, d["Bt"]); ut = d["Ut"]
    return {"fires": fires, "peak_Bt": peak, "Ut_in_stuck_regime": ut,
            "tau": ae["tau"], "bmax": ae["bmax"], "lam": ae["lam"]}


# ---- orchestration ------------------------------------------------------------
def _run(args: list[str], timeout: int = 1800) -> tuple[int, str]:
    r = subprocess.run([sys.executable, "-m"] + args, cwd=ROOT, capture_output=True,
                       text=True, timeout=timeout)
    return r.returncode, (r.stderr or "")[-600:]


def _metrics(sd: Path) -> list[dict]:
    return json.loads((sd / "metrics.json").read_text())


def _eval_rows(sd: Path) -> list[dict]:
    rows = []
    for p in sorted(sd.glob("eval_step*.csv")):
        rows += list(csv.DictReader(p.open()))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=MF.DEFAULT_PATH)
    ap.add_argument("--allow-unlocked", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results/preflight"))
    a = ap.parse_args(argv)
    checks: list[dict] = []

    def rec(name: str, ok: bool, detail=None) -> bool:
        checks.append({"name": name, "passed": bool(ok), "detail": detail})
        print(("PASS " if ok else "FAIL ") + name, flush=True)
        return ok

    if a.out.exists():
        shutil.rmtree(a.out)
    a.out.mkdir(parents=True)
    m = MF.load(a.manifest)
    try:
        MF.verify(m, require_locked=not a.allow_unlocked, check_code=not a.allow_unlocked)
        rec("manifest_verified", True, {"locked": m["lock"]["locked"]})
    except MF.ManifestError as e:
        rec("manifest_verified", False, str(e))
    try:
        MF.assert_arm_parity(m)
        rec("manifest_arm_parity", True, MF.expected_budget(m))
    except MF.ManifestError as e:
        rec("manifest_arm_parity", False, str(e))
    rec("manifest_copied_to_output", MF.copy_into(a.out, a.manifest).exists())
    fire = aed_can_fire(m)
    rec("aed_detector_can_fire_in_weak_stable_regime", fire["fires"] > 0, fire)

    raw = a.out / "raw"
    seed = 0
    t0 = time.time()
    for arm in CONTROL_ARMS + EXTRA_ARMS:
        code, err = _run(["experiments.run_aed"] + tiny_args(m, arm, seed, raw / arm))
        sd = raw / arm / f"seed_{seed:03d}"
        rec(f"arm_runs_and_saves[{arm}]",
            code == 0 and (sd / "DONE").exists() and (sd / "ckpt_2.pt").exists()
            and (sd / "metrics.json").exists() and (sd / "eval_step2.csv").exists(),
            err if code else None)
    per_arm, eg, es = {}, {}, {}
    for arm in CONTROL_ARMS + EXTRA_ARMS:
        sd = raw / arm / f"seed_{seed:03d}"
        if not (sd / "metrics.json").exists():
            continue
        per_arm[arm] = _metrics(sd)
        rows = _eval_rows(sd)
        eg[arm] = len(rows)
        es[arm] = {int(r["eval_mcts_sims"]) for r in rows}
        miss_m = missing_columns(per_arm[arm], METRIC_COLS)
        miss_e = missing_columns(rows, EVAL_COLS)
        info_p = sd / "run_info.json"
        info = json.loads(info_p.read_text()) if info_p.exists() else {}
        miss_i = [k for k in RUN_INFO_KEYS if k not in info]
        rec(f"csv_and_json_columns[{arm}]", not (miss_m or miss_e or miss_i),
            {"metrics": miss_m, "eval": miss_e, "run_info": miss_i})
    if len(per_arm) > 1:
        ok, detail = budget_parity(per_arm, eg, es)
        rec("no_arm_gets_more_games_epochs_sims", ok, detail)
    cnn_info = raw / "cnn_baseline" / f"seed_{seed:03d}" / "run_info.json"
    gnn_info = raw / "gnn_baseline" / f"seed_{seed:03d}" / "run_info.json"
    if cnn_info.exists() and gnn_info.exists():
        ci, gi = json.loads(cnn_info.read_text()), json.loads(gnn_info.read_text())
        rec("device_and_worker_info_recorded",
            ci["mcts_workers"] >= 1 and "cuda_visible_devices" in ci,
            {"cnn_params": ci["param_count"], "gnn_params": gi["param_count"],
             "device": ci["device"], "cuda_devices": ci["cuda_devices"],
             "cuda_visible_devices": ci["cuda_visible_devices"],
             "mcts_workers": ci["mcts_workers"], "cpu_count": ci["cpu_count"]})

    # determinism: same seed twice, CNN and GNN
    for arm in ("cnn_baseline", "gnn_baseline"):
        code, err = _run(["experiments.run_aed"] + tiny_args(m, arm, seed, raw / f"{arm}_repeat"))
        ref = raw / arm / f"seed_{seed:03d}"
        rep = raw / f"{arm}_repeat" / f"seed_{seed:03d}"
        rec(f"same_seed_same_result[{arm}]",
            code == 0 and ref.exists() and runs_identical(_metrics(ref), _metrics(rep)),
            err if code else None)

    # interrupted run resumes from checkpoint and matches the uninterrupted reference
    for arm in ("cnn_aed", "gnn_baseline"):
        out = raw / f"{arm}_interrupted"
        sd = out / f"seed_{seed:03d}"
        cmd = [sys.executable, "-m", "experiments.run_aed"] + tiny_args(m, arm, seed, out)
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 900
        while time.time() < deadline and p.poll() is None and not (sd / "train_state_0.pt").exists():
            time.sleep(0.1)
        killed = p.poll() is None
        p.kill()
        p.wait()
        code, err = _run(["experiments.run_aed"] + tiny_args(m, arm, seed, out) + ["--resume"])
        ref = raw / arm / f"seed_{seed:03d}"
        same = False
        if code == 0 and (sd / "DONE").exists() and ref.exists():
            import torch
            a_, b_ = (torch.load(p_ / "ckpt_2.pt", weights_only=False)["model"] for p_ in (ref, sd))
            same = runs_identical(_metrics(ref), _metrics(sd)) and all(
                torch.equal(a_[k], b_[k]) for k in a_)
        rec(f"interrupt_then_resume_matches_reference[{arm}]", killed and same,
            {"was_interrupted": killed, "resume_rc": code, "err": err if code else None})

    # final-eval path (UCT-800 tag) writes a valid, separately named CSV
    ck = raw / "cnn_baseline" / f"seed_{seed:03d}" / "ckpt_2.pt"
    code, err = _run(["experiments.run_evaluate", "--experiment", "cnn_baseline", "--seed", "0",
                      "--checkpoint", str(ck), "--ckpt-step", "2", "--games", "1",
                      "--net-sims", "4", "--eval-sims", "8", "--tag", "final_uct800",
                      "--output", str(raw / "cnn_baseline"), "--manifest", str(a.manifest)])
    f = raw / "cnn_baseline" / f"seed_{seed:03d}" / "final_uct800_step2.csv"
    rec("final_eval_tagged_csv", code == 0 and f.exists()
        and not missing_columns(list(csv.DictReader(f.open())), EVAL_COLS), err if code else None)

    # aggregator accepts a partial shard and complete shards
    part = raw / "cnn_baseline_partial" / f"seed_{seed:03d}"
    shutil.copytree(raw / "cnn_baseline" / f"seed_{seed:03d}", part)
    (part / "DONE").unlink()
    rows = _metrics(part)[:2]
    (part / "metrics.json").write_text(json.dumps(rows))
    for p_ in list(part.glob("eval_step2.csv")) + list(part.glob("final_*")):
        p_.unlink()
    code, err = _run(["experiments.analyze", "--raw", str(raw), "--manifest", str(a.manifest),
                      "--out", str(a.out / "analysis"), "--allow-incomplete",
                      "--allow-unlocked", "--n-perm", "200", "--n-boot", "200"])
    rec("aggregator_accepts_partial_and_complete_shards",
        code == 0 and (a.out / "analysis" / "per_seed.csv").exists(), err if code else None)

    report = {"passed": all(c["passed"] for c in checks), "checks": checks,
              "seconds": round(time.time() - t0, 1), "env": {
                  "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "cpu_count": os.cpu_count()}}
    (a.out / "preflight_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"preflight {'PASSED' if report['passed'] else 'FAILED'} -> {a.out / 'preflight_report.json'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
