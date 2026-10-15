"""AED training: baseline loop + controller-driven data collection.

Collection samples from visit distribution with controller temp/noise
(diagnostics mode: temp=1, noise=0, argmax equivalent).
Checkpoints/manifests identical format to run_train.py.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from aed.controller import AEDController, ControllerConfig
from aed.interventions import Intervention
from experiments.train_alphazero import policy_tv
from evaluation.diagnostics import repeat_rate
from gochess.core import Action
from gochess.representation import (build_model, count_params, encode_state,
                                    model_kwargs)
from gochess.scenarios import scenarios
from gochess.search import UCTAgent
from training import resume as R
from training.device import optimizer_to_device, resolve_device, seed_all
from training.checkpointing import git_hash, save_checkpoint
from training.replay_buffer import ReplayBuffer


def collect_soft(games: int, uct_sims: int, seed: int, temp: float, noise: float,
                 arch: str = "cnn"):
    X, PI, Z, entropy, openings = [], [], [], [], []
    state0 = scenarios()["initial_5x5"].state
    for g in range(games):
        s = state0
        rng = random.Random(seed + g)
        traj = []
        teacher = UCTAgent(simulations=uct_sims, seed=seed + g)
        first_move = None
        while not s.is_terminal():
            root = teacher.search(s)
            visits = {}
            for child in root.children:
                if child.action is not None:
                    visits[child.action.encode(s.size)] = child.visits
            total = sum(visits.values()) or 1
            pi = [0.0] * s.num_distinct_actions
            for aid in s.legal_action_ids():
                pi[aid] = visits.get(aid, 0) / total
            ent = -sum(p * math.log(p + 1e-12) for p in pi if p > 0)
            entropy.append(ent)
            traj.append((s, pi))
            weights = [max(1e-9, visits.get(aid, 0)) ** (1.0 / temp)
                       + (rng.gauss(0, noise) if noise > 0 else 0.0)
                       for aid in s.legal_action_ids()]
            weights = [max(0.0, w) for w in weights]
            tot = sum(weights) or 1.0
            r = rng.random() * tot
            acc = 0.0
            chosen = s.legal_action_ids()[-1]
            for aid, w in zip(s.legal_action_ids(), weights):
                acc += w
                if r <= acc:
                    chosen = aid
                    break
            if first_move is None:
                first_move = chosen
            s = s.apply(Action.decode(chosen, s.size))
        openings.append(first_move)
        z = s.returns()[0]
        for st, pi in traj:
            X.append(encode_state(arch, st))
            PI.append(torch.tensor(pi, dtype=torch.float32))
            Z.append(z if st.current_player == 0 else -z)
    return (torch.stack(X), torch.stack(PI), torch.tensor(Z, dtype=torch.float32),
            sum(entropy) / max(1, len(entropy)), openings)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", type=str, default="aed_v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--uct-simulations", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--buffer", type=int, default=4096)
    ap.add_argument("--sample", type=int, default=512)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--arch", type=str, default="cnn", choices=["cnn", "gnn"])
    ap.add_argument("--model-kwargs", type=str, default="")
    ap.add_argument("--mode", type=str, default="detector",
                    choices=["detector", "random", "off", "diagnostics", "fixed-noise"])
    ap.add_argument("--drop", type=str, default="none", choices=["none", "value", "kl"])
    ap.add_argument("--tau", type=float, default=0.15)
    ap.add_argument("--bmax", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=0.9)
    ap.add_argument("--beta-l", type=float, default=2.0)
    ap.add_argument("--beta-c", type=float, default=0.5)
    ap.add_argument("--random-p", type=float, default=0.2)
    ap.add_argument("--temp-base", type=float, default=1.0)
    ap.add_argument("--temp-shift", type=float, default=2.5)
    ap.add_argument("--noise-base", type=float, default=0.0)
    ap.add_argument("--noise-shift", type=float, default=0.3)
    ap.add_argument("--duration", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--eval-games", type=int, default=16)
    ap.add_argument("--eval-sims", type=int, default=400)
    ap.add_argument("--eval-scenario", type=str, default="mixed_midgame")
    ap.add_argument("--output", type=Path, default=Path("results/raw/aed_v1"))
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--allow-unlocked", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    kw = json.loads(a.model_kwargs) if a.model_kwargs else {}
    kw = model_kwargs(a.arch, kw)
    device = resolve_device(a.device)
    manifest = None
    if a.manifest is not None:
        from experiments import manifest as MF
        manifest = MF.load(a.manifest)
        MF.verify(manifest, require_locked=not a.allow_unlocked,
                  check_code=not a.allow_unlocked)
        got = {k.replace("_", "-"): str(v) for k, v in vars(a).items()}
        got["model-kwargs"] = json.dumps(kw, sort_keys=True)
        MF.check_cli_matches(manifest, a.experiment, a.seed, got)
        MF.copy_into(a.output, a.manifest)

    iv = Intervention(temp_base=a.temp_base, temp_shift=a.temp_shift,
                      noise_base=a.noise_base, noise_shift=a.noise_shift,
                      duration=a.duration)
    device = resolve_device(a.device)
    seed_all(a.seed, device)
    net = build_model(a.arch, **kw).to(device)
    n_params = count_params(net)
    config = {"experiment": a.experiment, "seed": a.seed, "games": a.games,
              "train_mcts_sims": a.uct_simulations, "epochs": a.epochs,
              "buffer": a.buffer, "sample": a.sample, "passes": a.passes,
              "batch": a.batch, "lr": a.lr, "l2": a.l2,
              "mode": a.mode, "drop": a.drop, "tau": a.tau, "bmax": a.bmax,
              "lam": a.lam, "beta_l": a.beta_l, "beta_c": a.beta_c,
              "random_p": a.random_p, "intervention": dataclasses.asdict(iv),
              "arch": a.arch, "model_kwargs": kw, "param_count": n_params,
              "net": f"{a.arch}_{'x'.join(str(v) for v in kw.values())}",
              "git_commit": git_hash(),
              "manifest_hash": manifest["lock"]["manifest_hash"] if manifest else None,
              "eval_every": a.eval_every, "eval_games": a.eval_games,
              "eval_sims": a.eval_sims, "eval_scenario": a.eval_scenario}
    config["config_hash"] = hashlib.sha256(json.dumps(
        {k: v for k, v in config.items() if k != "git_commit"},
        sort_keys=True).encode()).hexdigest()[:12]
    config["device"] = str(device)
    out = a.output / f"seed_{a.seed:03d}"
    out.mkdir(parents=True, exist_ok=True)
    if a.resume and (out / "DONE").exists():
        print("already complete; nothing to do", flush=True)
        return 0
    R.atomic_write_text(out / "config.json", json.dumps(config, indent=2) + "\n")
    R.atomic_write_text(out / "run_info.json", json.dumps({
        "arch": a.arch, "param_count": n_params, "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "mcts_workers": 1, "torch_threads": torch.get_num_threads(),
        "cpu_count": os.cpu_count(), "python": platform.python_version(),
        "torch": torch.__version__, "platform": platform.platform()}, indent=2) + "\n")

    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=a.l2)
    ctl = AEDController(ControllerConfig(
        beta_l=a.beta_l, beta_c=a.beta_c, tau=a.tau, lam=a.lam, bmax=a.bmax,
        mode=a.mode, drop=a.drop, random_p=a.random_p), iv, seed=a.seed)
    buf = ReplayBuffer(capacity=a.buffer)
    prev_pm = None
    metrics: list = []
    all_openings: list = []
    start_it = 0
    if a.resume:
        last = R.latest_resumable(out)
        if last is not None:
            st = R.load_train_state(out / f"train_state_{last}.pt")
            if st["config_hash"] != config["config_hash"]:
                raise RuntimeError("resume refused: config differs from interrupted run")
            ck = torch.load(out / f"ckpt_{last}.pt", map_location="cpu", weights_only=False)
            net.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            optimizer_to_device(opt, device)
            b = st["buffer"]
            buf.obs, buf.pi, buf.z, buf.meta = b["obs"], b["pi"], b["z"], b["meta"]
            R.import_controller(ctl, st["controller"])
            prev_pm, metrics, all_openings = st["prev_pm"], st["metrics"], st["all_openings"]
            if torch.is_tensor(prev_pm):
                prev_pm = prev_pm.to(device)
            R.restore_rng(st["rng"])
            start_it = last + 1
            print(f"resumed after iter {last}", flush=True)

    reference = None
    if manifest is not None:
        reference = manifest["evaluation"]["probes"].get("reference")
    root = Path(__file__).resolve().parent.parent
    for it in range(start_it, a.epochs):
        if a.mode == "fixed-noise":
            temp, noise, diag = iv.temp_shift, iv.noise_shift, {
                "Lt": 0.0, "Ct": 0.0, "Ut": 0.0, "Bt": 0.0, "temp": iv.temp_shift,
                "noise": iv.noise_shift, "shifted": False}
        else:
            temp, noise = ctl.state.params(ctl.iv)
            diag = None
        X, PI, Z, ent, openings = collect_soft(
            a.games, a.uct_simulations, a.seed + it * 1000, temp, noise, a.arch)
        all_openings.extend(openings)
        for i in range(len(X)):
            buf.add(X[i].clone(), PI[i].clone(), float(Z[i]), {"iter": it, "seed": a.seed})
        xb, pib, zb = buf.sample(min(len(buf), a.sample), seed=a.seed + it)
        net.train()
        steps = 0
        for ps in range(a.passes):
            ds = DataLoader(TensorDataset(xb, pib, zb), batch_size=a.batch, shuffle=True,
                            generator=torch.Generator().manual_seed(a.seed + it + 7919 * ps))
            for bxb, bpib, bzb in ds:
                bxb, bpib, bzb = bxb.to(device), bpib.to(device), bzb.to(device)
                logits, v = net(bxb)
                loss = F.mse_loss(v, bzb) + F.cross_entropy(logits, bpib)
                opt.zero_grad()
                loss.backward()
                opt.step()
                steps += 1
        net.eval()
        Xd, PId, Zd = X.to(device), PI.to(device), Z.to(device)
        with torch.no_grad():
            logits, v = net(Xd)
            vloss = F.mse_loss(v, Zd).item()
            acc = (logits.argmax(1) == PId.argmax(1)).float().mean().item()
            pm = F.softmax(logits, 1).mean(0)
            tv = policy_tv(pm.unsqueeze(0), prev_pm.unsqueeze(0)) if prev_pm is not None else 0.0
            prev_pm = pm.clone()
        budget = {"games": a.games, "positions": len(X), "grad_steps": steps,
                  "sims_per_move": a.uct_simulations,
                  "teacher_sims": len(X) * a.uct_simulations}
        if a.mode == "fixed-noise":
            row = {"iter": it, "vloss": vloss, "acc": acc, "tv": tv, "entropy": ent,
                   "buffer": len(buf), "openings": openings, **diag}
        else:
            d = ctl.step(vloss, tv)
            row = {"iter": it, "vloss": vloss, "acc": acc, "tv": tv, "entropy": ent,
                   "buffer": len(buf), "openings": openings,
                   **{k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in d.items()}}
        row["repeat"] = repeat_rate(all_openings)
        row.update(budget)
        metrics.append(row)
        print(row, flush=True)
        save_checkpoint(out / f"ckpt_{it}.pt", net, opt, it, config)
        if a.eval_every and (it % a.eval_every == 0 or it == a.epochs - 1):
            if reference:
                from evaluation.probes import net_probe_disagreement
                row.update(net_probe_disagreement(net, reference, a.uct_simulations,
                                                  a.seed + it))
            r = subprocess.run(
                [sys.executable, "-m", "experiments.run_evaluate",
                 "--experiment", a.experiment, "--seed", str(a.seed),
                 "--checkpoint", str(out / f"ckpt_{it}.pt"),
                 "--ckpt-step", str(it), "--games", str(a.eval_games),
                 "--net-sims", str(a.uct_simulations),
                 "--eval-sims", str(a.eval_sims),
                 "--scenario", a.eval_scenario,
                 "--device", a.device,
                 "--output", str(a.output)], cwd=root)
            if r.returncode != 0:
                raise RuntimeError("periodic eval failed")
        R.atomic_write_text(out / "metrics.json", json.dumps(metrics, indent=2) + "\n")
        R.save_train_state(out / f"train_state_{it}.pt", it=it, buf=buf, ctl=ctl,
                           prev_pm=prev_pm, metrics=metrics, all_openings=all_openings,
                           config_hash=config["config_hash"])
        R.prune_older(out, it)
    R.atomic_write_text(out / "metrics.json", json.dumps(metrics, indent=2) + "\n")
    (out / "DONE").write_text("ok\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
