# GoChess AED - full results log (Oct 2026)

All outputs below were produced on Kaggle CPU (+2xT4 where noted) from
https://github.com/bruce12-glitch/chessAED and pasted verbatim
(git `fatal: not a git repository` lines trimmed - harmless, means
checkpoint recorded `git_commit: unknown` on dataset copies).

## E0 - rules/search validation

- 15/15 unittest OK including OpenSpiel adapter (Kaggle; local skips OpenSpiel).
- Smoke `initial_5x5`, UCT-32 vs random, 6+6 games, seed 20261009:

```json
{
  "uct_as_player0": {"games": 6.0, "player0_win_rate": 0.833, "mean_plies": 22.5},
  "random_as_player0": {"games": 6.0, "player0_win_rate": 0.333, "mean_plies": 27.67},
  "uct_overall": {"wins": 9, "games": 12, "win_rate": 0.75,
                  "wilson_95_low": 0.468, "wilson_95_high": 0.911},
  "royal_capture_solver": {"value": 1.0, "best_action": "a3-e3", "nodes": 12}
}
```

- Gate tests 7/7 (masks, visit-sum, immutability, same-seed, ckpt reload).
- Replay tests 2/2, metrics tests 3/3, AED unit tests 6/6.

## Budget sensitivity - mixed_midgame

- UCT-16: uniform `visits=1` all children. UCT-128: `a1-a4 visits=23
  value=+0.565` top, then `stone@c5 12 +0.500`.
- select_action flips: 16 -> `stone@d3`, 128 -> `a1-a4`.
- Match from same start, 4 seeds each seating: UCT-128 beats UCT-16 8/8.

## Reference ordering - mixed_midgame

- UCT-128 vs UCT-128: 1.0, 1.0, -1.0, 1.0 (first-player noise).
- Greedy vs UCT-128: -1.0 x4. UCT dominates greedy 4/4.

## encirclement_choice - UCT-512

Flat top, 6 moves tied `visits=26 value=+1.0`. Exact solver infeasible
beyond 1-ply (KeyboardInterrupt, expected).

## Learned baseline plateau (az_baseline2: 8 games / 16 sims / 4 epochs)

```json
[
  {"iter": 0, "vloss": 0.941, "policy_acc": 0.0044, "policy_tv_vs_prev": 0.0,  "root_entropy": 2.59},
  {"iter": 1, "vloss": 0.956, "policy_acc": 0.0,    "policy_tv_vs_prev": 0.222, "root_entropy": 2.61},
  {"iter": 2, "vloss": 0.824, "policy_acc": 0.040,  "policy_tv_vs_prev": 0.401, "root_entropy": 2.41},
  {"iter": 3, "vloss": 1.009, "policy_acc": 0.016,  "policy_tv_vs_prev": 0.232, "root_entropy": 2.60}
]
```

- Net-32 vs UCT-128 from mixed_midgame: W=6 D=0 L=10, winrate 0.38.

## AED activation (aed_test2: tau 0.01, beta-l 2.0, beta-c 0.5, Bmax 0.5)

- Default tau 0.05 / Bmax 1.0: Bt=0.0 all iters, never triggers.
- Tuned: Bt 0.0 -> 0.0 -> 0.0 -> 0.49 -> 0.44, shift at iter 5 to
  temp 2.5 / noise 0.3. Mechanism works, no sustained winrate gain
  (0.625 / 0.5 / 0.625 / 0.375 / 0.75 oscillating).

## Matched null (3 seeds, 4 games / 8 sims / 3 epochs)

- base 0.583 Wilson [0.388, 0.755] bootstrap [0.375, 0.792].
- AED 0.500 Wilson [0.314, 0.686] bootstrap [0.292, 0.708].
- Paired wins 0/3. Config too small to learn (vloss ~1.0, acc 0%).

## 2xT4 CUDA sweeps (128ch/4 blocks, 16 games / 16 sims)

- Both GPUs active (108MiB each, P0). Iter 0-3 vloss 0.93-1.02,
  acc <= 3.8%. Plateau regime, eval pending at time of writing.

## E1 - baseline_v1 (6 epochs, 8 games / 64 sims train, 16 games vs UCT-400 ref)

| seed | final_vloss | games | winrate | wilson | bootstrap95 |
|---|---|---|---|---|---|
| seed_000 | 0.999 | 16 | 0.375 | [0.185, 0.614] | [0.125, 0.625] |
| seed_001 | 1.023 | 16 | 0.062 | [0.011, 0.283] | [0.0, 0.188] |
| seed_002 | 1.021 | 16 | 0.375 | [0.185, 0.614] | [0.125, 0.625] |
| seed_003 | 1.008 | 16 | 0.375 | [0.185, 0.614] | [0.125, 0.625] |
| seed_004 | 0.997 | 16 | 0.312 | [0.142, 0.556] | [0.062, 0.562] |

5/5 seeds weak (all <= 0.375), acc <= 0.047 throughout, TV decaying
0.62 -> ~0.10. Machine-readable copy: results/tables/matrix.md.
Per-seed ckpt/config/metrics/eval CSVs: Kaggle results tgz (see below).

## Artifact locations

- GitHub (code + small results):
  https://github.com/bruce12-glitch/chessAED
  - results/smoke/games.csv + summary.json
  - results/tables/matrix.md
  - docs/VALIDATION.md, docs/RESULTS_LOG.md (this file)
- Kaggle (large results, must Save Version first):
  - /kaggle/working/gochess-aed/results/ -> gochess-aed-results.tgz
  - contains results/raw/baseline_v1/seed_00*/{ckpt_5.pt, config.json,
    metrics.json, eval_step5.csv} + results/tables/matrix.md
