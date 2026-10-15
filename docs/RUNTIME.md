
## Runtime dependencies and Kaggle

Install with `pip install -e .`, which pulls `numpy`, `pyyaml`, and
`matplotlib`. PyTorch is **not** pinned: Kaggle notebooks already provide a
CUDA-enabled torch build, and pinning would force a multi-GB download.
On a bare machine install torch separately following pytorch.org for your
CUDA version, then verify with:

    python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

Every training/evaluation runner accepts `--device auto|cpu|cuda`
(`auto` uses CUDA when available, else CPU). Determinism checks in
`run_preflight` pin `--device cpu` because cuDNN kernels are not
bit-reproducible across devices. Matrix and ablation runners forward one
`--device` value to every arm so budget parity includes device placement.
OpenSpiel remains optional: `pip install -e '.[openspiel]'`.
