"""Assign device in run_aed main (companion to patch_source_v2.py)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def patch(path: str, old: str, new: str) -> None:
    p = ROOT / path
    text = p.read_text(encoding="utf-8")
    assert text.count(old) == 1, (path, text.count(old))
    p.write_text(text.replace(old, new), encoding="utf-8")
    print("patched", path)


patch("experiments/run_aed.py",
      "    kw = model_kwargs(a.arch, kw)\n",
      "    kw = model_kwargs(a.arch, kw)\n    device = resolve_device(a.device)\n")
