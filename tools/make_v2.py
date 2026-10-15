"""Create experiment_manifest_v2.yaml from v1 bytes (v1 file untouched)."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
v1 = ROOT / "experiments" / "configs" / "experiment_manifest_v1.yaml"
v2 = ROOT / "experiments" / "configs" / "experiment_manifest_v2.yaml"
assert v1.exists()
m = yaml.safe_load(v1.read_text(encoding="utf-8"))
m["manifest_version"] = 2
m["name"] = "experiment_manifest_v2"
v2.write_text(yaml.safe_dump(m, sort_keys=False), encoding="utf-8")
print("wrote", v2)
