# Deprecated entry points (kept working, do not use for new runs)

| Legacy | Use instead |
|---|---|
| experiments/train_alphazero.py | experiments/run_train.py |
| experiments/train_aed.py | experiments/run_aed.py |
| experiments/train_cuda.py | experiments/run_train.py (CPU) - CUDA port pending |
| experiments/eval_net.py | experiments/run_evaluate.py |
| experiments/run_smoke_benchmark.py | experiments/run_train.py + run_evaluate.py smoke |
| experiments/run_parallel_benchmark.py | experiments/run_matrix.py (1-seed smoke) |
| experiments/run_matched.py | experiments/run_matrix.py |

Legacy scripts still run; new dha results must come from run_train /
run_aed / run_evaluate / run_matrix + aggregate_results only.
