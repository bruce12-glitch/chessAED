"""Deps + README torch docs (companion patch)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def patch(path: str, old: str, new: str) -> None:
    p = ROOT / path
    text = p.read_text(encoding="utf-8")
    assert text.count(old) == 1, (path, text.count(old))
    p.write_text(text.replace(old, new), encoding="utf-8")
    print("patched", path)


patch("pyproject.toml",
      "dependencies = []",
      'dependencies = ["numpy", "pyyaml", "matplotlib"]')

print("deps done (torch documented in README, provided by Kaggle)")
