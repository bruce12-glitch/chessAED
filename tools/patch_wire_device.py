"""Wire --device through matrix/ablation/preflight (companion to patch_source_v2.py)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def patch(path: str, old: str, new: str) -> None:
    p = ROOT / path
    text = p.read_text(encoding="utf-8")
    assert text.count(old) == 1, (path, text.count(old))
    p.write_text(text.replace(old, new), encoding="utf-8")
    print("patched", path)


# ---------------- run_matrix.py ----------------
patch("experiments/run_matrix.py",
      '    ap.add_argument("--dry-run", action="store_true")\n    a = ap.parse_args(argv)',
      '    ap.add_argument("--dry-run", action="store_true")\n'
      '    ap.add_argument("--device", type=str, default=None,\n'
      '                    help="auto|cpu|cuda; default follows manifest hardware.device")\n'
      "    a = ap.parse_args(argv)")
patch("experiments/run_matrix.py",
      "    cells = plan(m, arms, seeds)\n",
      '    device = a.device or m["hardware"].get("device", "auto")\n'
      "    cells = plan(m, arms, seeds)\n")
patch("experiments/run_matrix.py",
      '        args = MF.trainer_args(m, arm, seed, a.raw / arm, a.manifest) + ["--resume"]',
      '        args = MF.trainer_args(m, arm, seed, a.raw / arm, a.manifest) + ["--resume", "--device", device]')
patch("experiments/run_matrix.py",
      "            final_evals(m, arm, seed, a.raw, a.manifest)",
      "            final_evals(m, arm, seed, a.raw, a.manifest, device)")

# ---------------- run_representation_ablation.py ----------------
patch("experiments/run_representation_ablation.py",
      "def final_evals(m: dict, arm: str, seed: int, raw: Path, mpath: Path) -> None:",
      'def final_evals(m: dict, arm: str, seed: int, raw: Path, mpath: Path,\n'
      '                device: str = "auto") -> None:')
patch("experiments/run_representation_ablation.py",
      '"--tag", f"final_{key}", "--output", str(raw / arm), "--manifest", str(mpath)])',
      '"--device", device,\n'
      '              "--tag", f"final_{key}", "--output", str(raw / arm), "--manifest", str(mpath)])')
patch("experiments/run_representation_ablation.py",
      '    ap.add_argument("--dry-run", action="store_true")\n    a = ap.parse_args(argv)',
      '    ap.add_argument("--dry-run", action="store_true")\n'
      '    ap.add_argument("--device", type=str, default=None,\n'
      '                    help="auto|cpu|cuda; default follows manifest hardware.device")\n'
      "    a = ap.parse_args(argv)")
patch("experiments/run_representation_ablation.py",
      "    cells = plan(m, arms, seeds)\n",
      '    device = a.device or m["hardware"].get("device", "auto")\n'
      "    cells = plan(m, arms, seeds)\n")
patch("experiments/run_representation_ablation.py",
      '        args = MF.trainer_args(m, arm, seed, a.raw / arm, a.manifest) + ["--resume"]',
      '        args = MF.trainer_args(m, arm, seed, a.raw / arm, a.manifest) + ["--device", device]')
patch("experiments/run_representation_ablation.py",
      "            final_evals(m, arm, seed, a.raw, a.manifest)",
      "            final_evals(m, arm, seed, a.raw, a.manifest, device)")

# ---------------- run_preflight.py (pin CPU for determinism) ----------------
patch("experiments/run_preflight.py",
      '"eval-sims": "8"}',
      '"eval-sims": "8", "device": "cpu"}')

print("all wiring patches applied")
