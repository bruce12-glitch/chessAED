"""python -m experiments.make_manifest {lock|verify|show-budget}"""
from __future__ import annotations

import argparse
import json

from experiments import manifest as MF


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["lock", "verify", "show-budget"])
    ap.add_argument("--path", default=str(MF.DEFAULT_PATH))
    a = ap.parse_args(argv)
    if a.cmd == "lock":
        m = MF.lock(a.path)
        print(json.dumps({"locked": True, "hash": m["lock"]["manifest_hash"],
                          "params": {k: v["param_count"] for k, v in m["models"].items()}}))
    elif a.cmd == "verify":
        MF.verify(MF.load(a.path))
        print("manifest verified")
    else:
        m = MF.load(a.path)
        MF.assert_arm_parity(m)
        print(json.dumps(MF.expected_budget(m), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
