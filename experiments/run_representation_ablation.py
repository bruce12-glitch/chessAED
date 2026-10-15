"""Representation ablation: CNN vs GNN (vs GNN+AED, gated). Manifest-driven.

Every arm runs through experiments.run_aed with arguments derived from the
manifest, so only arch / aed_mode can differ. gnn_aed refuses to run until the
gnn_baseline gate has passed (`--check-gate gnn_baseline`).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

from experiments import manifest as MF

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARMS = "cnn_baseline,gnn_baseline"


def plan(m: dict, arms: list[str], seeds: list[int]) -> list[tuple[str, int]]:
    for a in arms:
        MF.arm(m, a)
    return [(a, s) for a in arms for s in seeds]          # arm-major, seeds paired


def evaluate_gate(m: dict, series: dict[int, list[dict]], seeds: list[int],
                  eval_counts: dict[int, int]) -> dict:
    """Pure gate check. series[seed] = metrics rows; eval_counts[seed] = #eval csvs."""
    crit = m["gates"]["gnn_baseline"]
    iters = m["training"]["iterations"]
    res: dict[str, dict] = {}
    for s in seeds:
        rows = series.get(s, [])
        ok = {}
        ok["all_iterations"] = len(rows) == iters
        ok["finite"] = bool(rows) and all(
            math.isfinite(r[k]) for r in rows for k in ("vloss", "tv", "entropy"))
        ok["vloss_improved"] = len(rows) > 1 and rows[-1]["vloss"] < rows[0]["vloss"]
        ok["eval_rows"] = eval_counts.get(s, 0) > 0
        need = {"all_iterations": crit["require_all_iterations"],
                "finite": crit["require_finite"],
                "vloss_improved": crit["require_vloss_improvement"],
                "eval_rows": crit["require_eval_rows"]}
        res[str(s)] = {"checks": ok, "pass": all(ok[k] for k, req in need.items() if req)}
    return {"passed": all(v["pass"] for v in res.values()), "per_seed": res}


def check_gate(m: dict, raw: Path, gate: str, seeds: list[int]) -> dict:
    arm_dir = raw / gate
    series, evals = {}, {}
    for s in seeds:
        sd = arm_dir / f"seed_{s:03d}"
        p = sd / "metrics.json"
        series[s] = json.loads(p.read_text()) if p.exists() else []
        evals[s] = len(list(sd.glob("eval_step*.csv")))
    out = evaluate_gate(m, series, seeds, evals)
    out.update({"gate": gate, "manifest_hash": m["lock"]["manifest_hash"], "seeds": seeds})
    (raw / "gates").mkdir(parents=True, exist_ok=True)
    (raw / "gates" / f"{gate}.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def gate_ok(m: dict, raw: Path, gate: str) -> bool:
    p = raw / "gates" / f"{gate}.json"
    if not p.exists():
        return False
    g = json.loads(p.read_text())
    return bool(g["passed"]) and g["manifest_hash"] == m["lock"]["manifest_hash"]


def _run(args: list[str]) -> None:
    r = subprocess.run([sys.executable, "-m"] + args, cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError(f"failed: {args[:3]}")


def final_evals(m: dict, arm: str, seed: int, raw: Path, mpath: Path,
                device: str = "auto") -> None:
    last = m["training"]["iterations"] - 1
    ck = raw / arm / f"seed_{seed:03d}" / f"ckpt_{last}.pt"
    for key in ("uct400", "uct800"):
        f = m["evaluation"]["final"][key]
        csv_p = raw / arm / f"seed_{seed:03d}" / f"final_{key}_step{last}.csv"
        if csv_p.exists():
            continue
        _run(["experiments.run_evaluate", "--experiment", arm, "--seed", str(seed),
              "--checkpoint", str(ck), "--ckpt-step", str(last),
              "--games", str(f["games_per_seat"]), "--net-sims", str(m["evaluation"]["net_sims"]),
              "--eval-sims", str(f["ref_sims"]), "--scenario", m["scenario"]["eval_scenario"],
              "--device", device,
              "--tag", f"final_{key}", "--output", str(raw / arm), "--manifest", str(mpath)])


def write_costs(m: dict, arms: list[str], raw: Path) -> None:
    from experiments.throughput import measure_throughput
    rows = []
    for arm in arms:
        a = MF.arm(m, arm)["arch"]
        t = measure_throughput(a, m["models"][a]["kwargs"], size=m["scenario"]["rules"]["size"])
        (raw / arm).mkdir(parents=True, exist_ok=True)
        (raw / arm / "throughput.json").write_text(json.dumps(t, indent=2) + "\n")
        rows.append({"arm": arm, **{k: v for k, v in t.items() if k != "kwargs"}})
    with (raw / "representation_costs.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=MF.DEFAULT_PATH)
    ap.add_argument("--allow-unlocked", action="store_true")
    ap.add_argument("--arms", default=DEFAULT_ARMS)
    ap.add_argument("--seed-set", default="ablation")
    ap.add_argument("--raw", type=Path, default=Path("results/raw/ablation"))
    ap.add_argument("--check-gate", default="")
    ap.add_argument("--skip-final-eval", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--device", type=str, default=None,
                    help="auto|cpu|cuda; default follows manifest hardware.device")
    a = ap.parse_args(argv)
    m = MF.load(a.manifest)
    MF.verify(m, require_locked=not a.allow_unlocked, check_code=not a.allow_unlocked)
    seeds = m["seeds"][a.seed_set]
    if a.check_gate:
        g = check_gate(m, a.raw, a.check_gate, seeds)
        print(json.dumps({"gate": a.check_gate, "passed": g["passed"]}))
        return 0 if g["passed"] else 2
    arms = [x.strip() for x in a.arms.split(",")]
    MF.assert_arm_parity(m, arms)
    for arm in arms:
        need = MF.arm(m, arm).get("requires_gate")
        if need and not gate_ok(m, a.raw, need):
            raise SystemExit(f"{arm} blocked: gate '{need}' has not passed "
                             f"(run --check-gate {need} after the baseline finishes)")
    device = a.device or m["hardware"].get("device", "auto")
    cells = plan(m, arms, seeds)
    if a.dry_run:
        print(json.dumps({"cells": cells, "budget": MF.expected_budget(m)}, indent=2))
        return 0
    MF.copy_into(a.raw, a.manifest)
    write_costs(m, arms, a.raw)
    for arm, seed in cells:
        args = MF.trainer_args(m, arm, seed, a.raw / arm, a.manifest) + ["--device", device]
        if a.allow_unlocked:
            args.append("--allow-unlocked")
        _run(["experiments.run_aed"] + args)
        if not a.skip_final_eval:
            final_evals(m, arm, seed, a.raw, a.manifest, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
