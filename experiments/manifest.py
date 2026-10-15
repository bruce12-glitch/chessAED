"""Locked experiment manifest: load, lock, verify, copy, derive trainer args."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "experiments" / "configs" / "experiment_manifest_v2.yaml"
MANIFEST_NAME = "experiment_manifest_v1.yaml"
CODE_FILES = [
    "gochess/core.py", "gochess/search.py", "gochess/scenarios.py", "gochess/nn.py",
    "gochess/gnn.py", "gochess/graph.py", "gochess/representation.py",
    "aed/controller.py", "aed/detector.py", "aed/interventions.py", "aed/signals.py",
    "training/self_play.py", "training/replay_buffer.py", "training/resume.py",
    "experiments/run_aed.py", "experiments/run_evaluate.py", "experiments/eval_net.py",
    "evaluation/events.py", "evaluation/diagnostics.py", "evaluation/probes.py",
]


class ManifestError(RuntimeError):
    pass


def load(path: Path | str = DEFAULT_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def content_hash(m: dict) -> str:
    body = {k: v for k, v in m.items() if k != "lock"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def file_hashes(files=CODE_FILES, root: Path = ROOT) -> dict[str, str]:
    out = {}
    for f in files:
        p = root / f
        if p.exists():                       # CRLF-normalised so checkouts agree
            out[f] = hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return out


def scenario_digest() -> str:
    from gochess.scenarios import scenarios
    parts = []
    for name, sc in sorted(scenarios().items()):
        s = sc.state
        parts.append([name, s.to_ascii(), s.current_player, list(s.reserves),
                      s.rules.max_plies, s.rules.komi, s.rules.size])
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()


def _git() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def compute_param_counts(m: dict) -> dict[str, int]:
    from gochess.representation import build_model, count_params
    return {a: count_params(build_model(a, size=m["scenario"]["rules"]["size"],
                                        **m["models"][a]["kwargs"]))
            for a in m["models"]}


def compute_probe_reference(m: dict) -> dict[str, int]:
    from evaluation.probes import compute_probe_references
    p = m["evaluation"]["probes"]
    return compute_probe_references(p["positions"], p["reference_sims"], p["reference_seed"])


def lock(path: Path | str = DEFAULT_PATH, *, param_counts: dict | None = None,
         probe_reference: dict | None = None) -> dict:
    m = load(path)
    if m["lock"]["locked"]:
        raise ManifestError("already locked; create experiment_manifest_v2 instead")
    m["scenario"]["digest"] = scenario_digest()
    m["code"] = {"git_commit": _git(), "file_sha256": file_hashes()}
    pc = param_counts if param_counts is not None else compute_param_counts(m)
    for a, n in pc.items():
        m["models"][a]["param_count"] = int(n)
    m["evaluation"]["probes"]["reference"] = (
        probe_reference if probe_reference is not None else compute_probe_reference(m))
    m["lock"] = {"locked": True, "manifest_hash": content_hash(m)}
    Path(path).write_text(yaml.safe_dump(m, sort_keys=False), encoding="utf-8")
    return m


def verify(m: dict, *, require_locked: bool = True, check_code: bool = True) -> None:
    if not m["lock"]["locked"]:
        if require_locked:
            raise ManifestError("manifest is not locked (python -m experiments.make_manifest lock)")
        return
    if content_hash(m) != m["lock"]["manifest_hash"]:
        raise ManifestError("manifest content changed after lock")
    if m["scenario"]["digest"] != scenario_digest():
        raise ManifestError("scenario definitions differ from locked digest")
    if check_code:
        now = file_hashes()
        bad = [f for f, h in m["code"]["file_sha256"].items() if now.get(f) != h]
        if bad:
            raise ManifestError(f"code changed after lock: {bad}")


def copy_into(out_dir: Path | str, path: Path | str = DEFAULT_PATH) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / Path(path).name
    if dst.exists() and dst.read_bytes() != Path(path).read_bytes():
        raise ManifestError(f"{dst} exists with different content; refusing to mix manifests")
    shutil.copyfile(path, dst)
    return dst


def arm(m: dict, name: str) -> dict:
    if name not in m["arms"]:
        raise ManifestError(f"unknown arm {name!r}; have {sorted(m['arms'])}")
    return m["arms"][name]


def trainer_args(m: dict, arm_name: str, seed: int, output: Path | str,
                 manifest_path: Path | str = DEFAULT_PATH) -> list[str]:
    """CLI args for experiments.run_aed. Every arm gets the same budget args."""
    a, t, e, ae = arm(m, arm_name), m["training"], m["evaluation"], m["aed"]
    iv = ae["intervention"]
    kw = m["models"][a["arch"]]["kwargs"]
    return [
        "--experiment", arm_name, "--seed", str(seed), "--output", str(output),
        "--arch", a["arch"], "--model-kwargs", json.dumps(kw, sort_keys=True),
        "--mode", a["aed_mode"], "--drop", ae["drop"],
        "--games", str(t["games_per_iter"]), "--uct-simulations", str(t["train_sims"]),
        "--epochs", str(t["iterations"]), "--buffer", str(t["buffer"]),
        "--sample", str(t["sample_per_iter"]), "--passes", str(t["passes_per_iter"]),
        "--batch", str(t["batch"]), "--lr", str(t["lr"]), "--l2", str(t["l2"]),
        "--tau", str(ae["tau"]), "--bmax", str(ae["bmax"]), "--lam", str(ae["lam"]),
        "--beta-l", str(ae["beta_l"]), "--beta-c", str(ae["beta_c"]),
        "--random-p", str(ae["random_p"]),
        "--temp-base", str(iv["temp_base"]), "--temp-shift", str(iv["temp_shift"]),
        "--noise-base", str(iv["noise_base"]), "--noise-shift", str(iv["noise_shift"]),
        "--duration", str(iv["duration"]),
        "--eval-every", str(e["periodic"]["every"]),
        "--eval-games", str(e["periodic"]["games_per_seat"]),
        "--eval-sims", str(e["periodic"]["ref_sims"]),
        "--eval-scenario", m["scenario"]["eval_scenario"],
        "--device", str(m["hardware"].get("device", "auto")),
        "--manifest", str(manifest_path),
    ]


def expected_budget(m: dict) -> dict:
    """Per-seed compute that must be identical for every arm."""
    t, e = m["training"], m["evaluation"]
    n_eval = len(range(0, t["iterations"], e["periodic"]["every"])) + (
        0 if (t["iterations"] - 1) % e["periodic"]["every"] == 0 else 1)
    return {"iterations": t["iterations"], "games_per_iter": t["games_per_iter"],
            "train_sims": t["train_sims"], "sample_per_iter": t["sample_per_iter"],
            "passes_per_iter": t["passes_per_iter"], "batch": t["batch"],
            "periodic_evals": n_eval, "periodic_games": 2 * e["periodic"]["games_per_seat"],
            "net_sims": e["net_sims"], "ref_sims": e["periodic"]["ref_sims"]}


def args_to_dict(argv: list[str]) -> dict[str, str]:
    d, i = {}, 0
    while i < len(argv):
        d[argv[i].lstrip("-")] = argv[i + 1]
        i += 2
    return d


def _same(got: str | None, want: str) -> bool:
    if got is None:
        return False
    try:
        return float(got) == float(want)
    except ValueError:
        pass
    try:
        return json.loads(got) == json.loads(want)
    except (ValueError, TypeError):
        return got == want


def check_cli_matches(m: dict, arm_name: str, seed: int, got: dict[str, str]) -> None:
    """Direct trainer invocation with --manifest must agree with the manifest."""
    want = args_to_dict(trainer_args(m, arm_name, seed, "x"))
    skip = {"output", "manifest", "experiment", "seed"}
    bad = {k: (got.get(k), v) for k, v in want.items()
           if k not in skip and not _same(got.get(k), v)}
    if bad:
        raise ManifestError(f"CLI differs from manifest (got, want): {bad}")


def clone(m: dict) -> dict:
    return copy.deepcopy(m)


ARM_VARIABLES = {"experiment", "output", "arch", "model-kwargs", "mode"}


def assert_arm_parity(m: dict, arms: list[str] | None = None, seed: int = 0) -> dict[str, str]:
    """Every arm must get identical budget/optimizer/AED/eval args; only the
    architecture and AED mode may differ. Returns the shared args."""
    arms = arms or list(m["arms"])
    per = {a: args_to_dict(trainer_args(m, a, seed, "x")) for a in arms}
    shared = {k: v for k, v in per[arms[0]].items() if k not in ARM_VARIABLES}
    for a in arms[1:]:
        other = {k: v for k, v in per[a].items() if k not in ARM_VARIABLES}
        if other != shared:
            diff = {k: (shared.get(k), other.get(k)) for k in set(shared) | set(other)
                    if shared.get(k) != other.get(k)}
            raise ManifestError(f"arm {a} differs from {arms[0]} beyond arch/mode: {diff}")
    return shared
