"""Verify source-v2 requirements. Exit nonzero on any failure."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

checks = {
    "run_aed device arg": '"--device", type=str' in (ROOT / "experiments/run_aed.py").read_text(),
    "run_aed seed_all": "seed_all(a.seed" in (ROOT / "experiments/run_aed.py").read_text(),
    "run_train device arg": "--device" in (ROOT / "experiments/run_train.py").read_text(),
    "eval_net device": "device: str =" in (ROOT / "experiments/eval_net.py").read_text(),
    "run_evaluate device": "--device" in (ROOT / "experiments/run_evaluate.py").read_text(),
    "matrix device": "--device" in (ROOT / "experiments/run_matrix.py").read_text(),
    "ablation device": "--device" in (ROOT / "experiments/run_representation_ablation.py").read_text(),
    "preflight cpu pin": '"device": "cpu"' in (ROOT / "experiments/run_preflight.py").read_text(),
    "resume cuda": "torch_cuda" in (ROOT / "training/resume.py").read_text(),
    "manifest trainer device": '"--device"' in (ROOT / "experiments/manifest.py").read_text(),
    "manifest copy name": "Path(path).name" in (ROOT / "experiments/manifest.py").read_text(),
    "manifest default v2": "manifest_v2" in (ROOT / "experiments/manifest.py").read_text(),
    "pyproject deps": "pyyaml" in (ROOT / "pyproject.toml").read_text(),
    "RUNTIME doc": (ROOT / "docs/RUNTIME.md").exists(),
    "device helper": (ROOT / "training/device.py").exists(),
    "v1 preserved draft": (
        yaml.safe_load((ROOT / "experiments/configs/experiment_manifest_v1.yaml").read_text())["lock"]["locked"]
        is False
    ),
}
v2 = yaml.safe_load((ROOT / "experiments/configs/experiment_manifest_v2.yaml").read_text())
checks["v2 locked"] = v2["lock"]["locked"] is True
checks["v2 tau 0.15"] = v2["aed"]["tau"] == 0.15

fails = [k for k, v in checks.items() if not v]
for k, v in checks.items():
    print(("PASS " if v else "FAIL "), k)
print("failures:", fails)
raise SystemExit(1 if fails else 0)
