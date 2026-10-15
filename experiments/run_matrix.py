"""Main CNN AED matrix: four matched arms over paired seeds. Manifest-driven.

Every arm runs through experiments.run_aed with arguments derived from the
manifest, so only the AED mode can differ (arch is cnn for all four).
Final UCT-400/800 evals run per seed unless --skip-final-eval.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from experiments import manifest as MF
from experiments.run_representation_ablation import final_evals

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARMS = "cnn_baseline,cnn_fixed_noise,cnn_random_trigger,cnn_aed"


def plan(m: dict, arms: list[str], seeds: list[int]) -> list[tuple[str, int]]:
    for a in arms:
        MF.arm(m, a)
        if MF.arm(m, a)["arch"] != "cnn":
            raise MF.ManifestError(f"main matrix is CNN-only; {a} uses "
                                   f"{MF.arm(m, a)['arch']}")
    return [(a, s) for a in arms for s in seeds]          # arm-major, seeds paired


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=MF.DEFAULT_PATH)
    ap.add_argument("--allow-unlocked", action="store_true")
    ap.add_argument("--arms", default=DEFAULT_ARMS)
    ap.add_argument("--seed-set", default="main")
    ap.add_argument("--raw", type=Path, default=Path("results/raw/main"))
    ap.add_argument("--skip-final-eval", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    m = MF.load(a.manifest)
    MF.verify(m, require_locked=not a.allow_unlocked, check_code=not a.allow_unlocked)
    seeds = m["seeds"][a.seed_set]
    arms = [x.strip() for x in a.arms.split(",")]
    MF.assert_arm_parity(m, arms)
    cells = plan(m, arms, seeds)
    if a.dry_run:
        print(json.dumps({"cells": cells, "budget": MF.expected_budget(m)}, indent=2))
        return 0
    MF.copy_into(a.raw, a.manifest)
    for arm, seed in cells:
        args = MF.trainer_args(m, arm, seed, a.raw / arm, a.manifest) + ["--resume"]
        if a.allow_unlocked:
            args.append("--allow-unlocked")
        r = subprocess.run([sys.executable, "-m", "experiments.run_aed"] + args, cwd=ROOT)
        if r.returncode != 0:
            raise RuntimeError(f"cell failed: {arm} seed {seed}")
        if not a.skip_final_eval:
            final_evals(m, arm, seed, a.raw, a.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
