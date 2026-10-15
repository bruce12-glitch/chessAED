# Protocol v1 (frozen)

Board: 5x5 GoChess hybrid, rules hash = git commit of gochess/core.py.
Encoding: 11x5x5 observation_tensor(player=0), action map 651 (place/move/pass).
Terminals: royal capture, two passes, move limit; area + rook/knight bonus + komi scoring.
Net: resnet_4x64 policy_value_net v1 (param count logged at train start).
Train MCTS: 64 sims, UCT c=sqrt(2), rollout depth 80, Dirichlet off, temp schedule fixed.
Eval MCTS: 400 sims reference, temp 0, no noise, equal seating.
Replay: size + sampling rule fixed per config; metadata logged per iter.
Optimizer: Adam lr 1e-3, batch 64, steps fixed per config.
Seeds: 10 final (3 debug). Manifest per seed: python/numpy/torch-cpu/cuda/env/MCTS/replay/eval.
Opponents: UCT-400 from mixed_midgame + encirclement_choice, games/seating fixed.
AED: signals entropy/KL/repeat/dValue; threshold/duration/action per aed_v1.yaml.
Rule: any change -> new config ID. Never compare across IDs except one variable.
