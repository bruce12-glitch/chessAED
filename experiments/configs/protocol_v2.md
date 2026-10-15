# Protocol v2 (frozen) - Phase A gate + E2 entry criteria

Inherits all of protocol_v1.md. Additions:

## Bad-minimum event (all four, 3 consecutive evals)

- TV < 0.15 (policy drift vs previous ckpt)
- |entropy slope| < 0.05 nats/iter over window
- ref winrate vs UCT-400 < 0.40
- opening repeat rate > 0.50 (same argmax first move in >50% of games)

## Escape (percentage points, not relative)

- ref winrate rises >= 15 percentage points (e.g. 0.30 -> 0.45)
  from event baseline and holds for 2 consecutive evals.

## Phase A gate

- Long baseline: same baseline_v1 config, 12 iters min (20 if flat),
  eval every 2 iters vs UCT-400, 16 games/seed, 1 seed then 3 seeds.
- Pass = event fires on all 3 seeds with no recovery by iter 12.
- Any recovery = stop, retune difficulty, new protocol version.
- No 10-seed AED spend before pass.

## Phase B entry (frozen on pass)

- Four arms E1/E2/E3/E4, 10 seeds, identical games/sims/net/epochs/evaluator.
- Minimum acceptable if compute breaks: E1+E2+E3.
- UCT-400 main evaluator; UCT-800 spot-check on final ckpts only.
