# Code Set 1 Validation

Validation date: 2026-10-09

## Automated checks

- 15 unit and integration tests passed.
- Python bytecode compilation passed for gochess, experiments, and tests.
- OpenSpiel 2.0.2 loaded python_gochess with 651 actions and an 11x5x5
  observation tensor.
- OpenSpiel random_sim_test passed for 25 seeded simulations.
- The exact solver found the forced royal capture a3-e3 with value +1 in
  12 visited nodes.

## Smoke simulation

Protocol:

- Position: initial_5x5
- UCT simulations per move: 32
- Random rollout depth: 40
- Games: 12 total, six in each seating
- Root seed: 20261009

Observed result:

- UCT wins: 9/12
- UCT win rate: 0.75
- Wilson 95% interval: [0.468, 0.911]
- Mean game length with UCT as player 0: 22.5 plies
- Mean game length with random as player 0: 27.7 plies

The machine-readable trajectories and aggregate are in results/smoke.

## Evidence boundary

This smoke run validates executable rules, search signal, deterministic seeding,
OpenSpiel compatibility, and result logging. It does not demonstrate an
AlphaZero bad minimum, compare AED, establish game balance, or identify an
optimal full-game strategy. Those claims require neural self-play across many
seeds, fixed compute budgets, checkpoint evaluation, and an independently
strong reference policy.
