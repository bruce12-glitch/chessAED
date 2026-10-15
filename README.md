# GoChess AED Research Testbed

GoChess is a deterministic, two-player, zero-sum, perfect-information game for
studying stable suboptimal self-play policies. 5x5 board, king/rook/knight per
side, six stones in reserve. Full rules in `docs/VALIDATION.md`.

## Scientific claim

A behavioral bad minimum must satisfy all of: (1) self-play strength and
value loss plateau over a declared window; (2) policy change and root
exploration small; (3) an independent stronger reference wins significantly
more often from the same serialized positions; (4) repeats across seeds and
not explained by more simulations at eval.

AED is evaluated as a training controller accumulating when learning progress
and action leverage stay below threshold, triggering temp/noise mode shift.
Compared at matched env steps, updates, and sims.

## Quick start

    python -m unittest discover -s tests -v
    python -m gochess.cli show mixed_midgame
    python -m gochess.cli solve royal_capture_in_one
    python -m experiments.run_smoke_benchmark --games 12 --uct-simulations 48

## Empirical results (Kaggle CPU + 2xT4, Oct 2026)

All runs use fixed seeds, `results/` artifacts versioned as Kaggle Datasets.
Code at this commit; checkpoints excluded from git (see Kaggle).

| # | Experiment | Config | Result |
|---|------------|--------|--------|
| 1 | Rules/search validation | UCT-32 vs random, 6+6 games, seed 20261009 | 9/12 wins, 0.75, Wilson 95% [0.468,0.911], plies 22.5 vs 27.7; solver `a3-e3 nodes=12`; 15/15 tests incl. OpenSpiel |
| 2 | Budget sensitivity, `mixed_midgame` | UCT-16 vs UCT-128, same start | 128 wins 8/8; decision flips 16->`stone@d3`, 128->`a1-a4`; 16 sims uniform visits=1, 128 concentrates `a1-a4 visits=23 value=+0.565` |
| 3 | Reference ordering | UCT-128 vs greedy, same start | UCT-128 wins 4/4; UCT-128 vs UCT-128 splits. No trap: stronger search wins |
| 4 | `encirclement_choice` UCT-512 | 512 sims | Flat top, 6 moves tied `visits=26 value=+1.0`. Exact solver infeasible beyond 1-ply (KeyboardInterrupt, expected) |
| 5 | Learned baseline plateau | 8 games / 16 sims / 4 epochs | vloss 0.94,0.96,0.82,1.01; acc 0.004,0.0,0.04,0.016; TV 0.0,0.22,0.40,0.23; entropy ~2.5 |
| 6 | Net-32 vs UCT-128, `mixed_midgame` | 8+8 games | W=6 D=0 L=10, winrate 0.38. Stagnation reference |
| 7 | Matched baseline vs AED, tiny config | 3 seeds, 4 games / 8 sims / 3 epochs | base 0.583 Wilson [0.388,0.755], AED 0.50 [0.314,0.686], paired wins 0/3. Null: config too small to learn (vloss ~1.0, acc 0%) |
| 8 | AED activation | tau 0.05 Bmax 1.0 vs tau 0.01 Bmax 0.5 | Default Bt=0.0 never triggers. Tuned Bt 0.0->0.49->0.44, shift at iter 5 to temp 2.5 noise 0.3. Mechanism works, no sustained winrate gain |
| 9 | 2xT4 CUDA sweeps | 128ch/4 blocks, 16 games / 16 sims / 10 epochs, seeds 0+100 | Both on `device=cuda`, 108MiB/GPU. Iter 0-3 vloss ~0.93-1.02, acc <=3.8%. Plateau regime, eval pending |

Boundary: deterministic perfect-info, CPU MCTS dominates wall time, GPUs idle during collection (111MiB held, 0% in snapshots). No noisy-TV test; corridor Theorems 1,6,7,10 untested here. No escape claim: CIs overlap, N small, single position.

## Experiment stages

1. Validate rules/serialization/search baselines.
2. OpenSpiel registration + API tests.
3. AlphaZero baselines across seeds with checkpoints.
4. Mine plateau positions with reference win-rate gap.
5. AED with preregistered threshold/saturation/mode-shift ablations.
6. AlphaZero vs AlphaZero+AED, bootstrap CIs + paired seed tests.

## Representation ablation + locked-manifest workflow

    python -m unittest discover -s tests -v                 # GNN tests activate when torch is installed
    python -m experiments.run_preflight --allow-unlocked    # tiny real runs; must PASS before locking
    python -m experiments.make_manifest lock                # freezes params, hashes, probe references
    python -m experiments.run_representation_ablation --arms cnn_baseline,gnn_baseline --seed-set ablation
    python -m experiments.run_representation_ablation --check-gate gnn_baseline   # unlocks gnn_aed
    python -m experiments.analyze --raw results/raw/ablation --out results/analysis/ablation

`experiments/configs/experiment_manifest_v1.yaml` is the single source of truth: trainers launched
with `--manifest` refuse any CLI value that differs, and every runner copies it into its output dir.
All arms run through `experiments.run_aed` (baseline = `--mode off`); only `arch`/`aed_mode` differ.
