"""Source-v2 patch: --device auto|cpu|cuda across runners + manifest v2 groundwork.

Every replacement asserts a unique anchor; aborts loudly otherwise.
Run: python tools/patch_source_v2.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def patch(path: str, old: str, new: str, expect: int = 1) -> None:
    p = ROOT / path
    text = p.read_text(encoding="utf-8")
    hits = text.count(old)
    assert hits == expect, (path, hits, old[:70])
    p.write_text(text.replace(old, new), encoding="utf-8")
    print("patched", path)


def patch_line(path: str, needle: str, insert: str) -> None:
    """Insert a line before the unique line containing needle (same indent)."""
    p = ROOT / path
    lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
    hits = [i for i, l in enumerate(lines) if needle in l]
    assert len(hits) == 1, (path, needle, len(hits))
    i = hits[0]
    indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
    lines.insert(i, indent + insert + "\n")
    p.write_text("".join(lines), encoding="utf-8")
    print("patched-line", path)


# ---------------- run_aed.py ----------------
patch("experiments/run_aed.py",
      "from training import resume as R\n",
      "from training import resume as R\nfrom training.device import optimizer_to_device, resolve_device, seed_all\n")
patch("experiments/run_aed.py",
      '    ap.add_argument("--resume", action="store_true")\n',
      '    ap.add_argument("--resume", action="store_true")\n'
      '    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")\n')
patch("experiments/run_aed.py",
      "    random.seed(a.seed)\n    np.random.seed(a.seed)\n    torch.manual_seed(a.seed)\n"
      "    net = build_model(a.arch, **kw)\n",
      "    device = resolve_device(a.device)\n    seed_all(a.seed, device)\n"
      "    net = build_model(a.arch, **kw).to(device)\n")
patch("experiments/run_aed.py",
      '"arch": a.arch, "param_count": n_params, "device": "cpu",',
      '"arch": a.arch, "param_count": n_params, "device": str(device),')
patch("experiments/run_aed.py",
      '            opt.load_state_dict(ck["optimizer"])',
      '            opt.load_state_dict(ck["optimizer"])\n'
      "            optimizer_to_device(opt, device)")
patch("experiments/run_aed.py",
      '            prev_pm, metrics, all_openings = st["prev_pm"], st["metrics"], st["all_openings"]',
      '            prev_pm, metrics, all_openings = st["prev_pm"], st["metrics"], st["all_openings"]\n'
      "            if torch.is_tensor(prev_pm):\n"
      "                prev_pm = prev_pm.to(device)")
patch("experiments/run_aed.py",
      "            for bxb, bpib, bzb in ds:\n                logits, v = net(bxb)",
      "            for bxb, bpib, bzb in ds:\n"
      "                bxb, bpib, bzb = bxb.to(device), bpib.to(device), bzb.to(device)\n"
      "                logits, v = net(bxb)")
patch("experiments/run_aed.py",
      "        net.eval()\n        with torch.no_grad():\n"
      "            logits, v = net(X)\n"
      "            vloss = F.mse_loss(v, Z).item()\n"
      "            acc = (logits.argmax(1) == PI.argmax(1)).float().mean().item()",
      "        net.eval()\n"
      "        Xd, PId, Zd = X.to(device), PI.to(device), Z.to(device)\n"
      "        with torch.no_grad():\n"
      "            logits, v = net(Xd)\n"
      "            vloss = F.mse_loss(v, Zd).item()\n"
      "            acc = (logits.argmax(1) == PId.argmax(1)).float().mean().item()")
patch("experiments/run_aed.py",
      '    out = a.output / f"seed_{a.seed:03d}"',
      '    config["device"] = str(device)\n'
      '    out = a.output / f"seed_{a.seed:03d}"')
patch_line("experiments/run_aed.py", '"--output", str(a.output)], cwd=root)', '"--device", a.device,')

# ---------------- run_train.py ----------------
patch("experiments/run_train.py",
      "from training.self_play import collect, probe_disagreement\n",
      "from training.self_play import collect, probe_disagreement\n"
      "from training.device import resolve_device, seed_all\n")
patch("experiments/run_train.py",
      '    ap.add_argument("--eval-scenario", type=str, default="mixed_midgame")\n'
      "    a = ap.parse_args()",
      '    ap.add_argument("--eval-scenario", type=str, default="mixed_midgame")\n'
      '    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")\n'
      "    a = ap.parse_args()\n"
      "    device = resolve_device(a.device)")
patch("experiments/run_train.py",
      "    random.seed(a.seed)\n    np.random.seed(a.seed)\n    torch.manual_seed(a.seed)\n"
      "\n"
      "    net = GoChessNet(channels=64, blocks=4)",
      "    seed_all(a.seed, device)\n"
      "    net = GoChessNet(channels=64, blocks=4).to(device)")
patch("experiments/run_train.py",
      "        for bxb, bpib, bzb in ds:\n            logits, v = net(bxb)",
      "        for bxb, bpib, bzb in ds:\n"
      "            bxb, bpib, bzb = bxb.to(device), bpib.to(device), bzb.to(device)\n"
      "            logits, v = net(bxb)")
patch("experiments/run_train.py",
      "        net.eval()\n        with torch.no_grad():\n"
      "            logits, v = net(X)\n"
      "            vloss = F.mse_loss(v, Z).item()\n"
      "            acc = (logits.argmax(1) == PI.argmax(1)).float().mean().item()",
      "        net.eval()\n"
      "        Xd, PId, Zd = X.to(device), PI.to(device), Z.to(device)\n"
      "        with torch.no_grad():\n"
      "            logits, v = net(Xd)\n"
      "            vloss = F.mse_loss(v, Zd).item()\n"
      "            acc = (logits.argmax(1) == PId.argmax(1)).float().mean().item()")
patch("experiments/run_train.py",
      '    out = a.output / f"seed_{a.seed:03d}"',
      '    config["device"] = str(device)\n'
      '    out = a.output / f"seed_{a.seed:03d}"')
patch_line("experiments/run_train.py", '"--output", str(a.output)], cwd=root)', '"--device", a.device,')

# ---------------- eval_net.py ----------------
patch("experiments/eval_net.py",
      "import torch\n\nfrom gochess.core import GoChessState",
      "import torch\n\nfrom training.device import resolve_device\n\nfrom gochess.core import GoChessState")
patch("experiments/eval_net.py",
      "    def __init__(self, net: GoChessNet, sims: int = 32, seed: int = 0):\n"
      "        self.net = net\n"
      "        self.sims = sims\n"
      "        self.rng = __import__(\"random\").Random(seed)",
      "    def __init__(self, net: GoChessNet, sims: int = 32, seed: int = 0,\n"
      "                 device: str = \"auto\"):\n"
      "        self.device = resolve_device(device)\n"
      "        self.net = net.to(self.device)\n"
      "        self.sims = sims\n"
      "        self.rng = __import__(\"random\").Random(seed)")
patch("experiments/eval_net.py",
      "    def _net_value(self, state: GoChessState) -> float:\n"
      "        with torch.no_grad():\n"
      "            x = encode_for(self.net, state).unsqueeze(0)",
      "    def _net_value(self, state: GoChessState) -> float:\n"
      "        with torch.no_grad():\n"
      "            x = encode_for(self.net, state).unsqueeze(0).to(self.device)")
patch("experiments/eval_net.py",
      '    ap.add_argument("--seed", type=int, default=42)\n    a = ap.parse_args()',
      '    ap.add_argument("--seed", type=int, default=42)\n'
      '    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")\n'
      "    a = ap.parse_args()")
patch("experiments/eval_net.py",
      '    net = GoChessNet()\n    net.load_state_dict(torch.load(a.ckpt, map_location="cpu"))\n    net.eval()',
      '    device = resolve_device(a.device)\n'
      '    net = GoChessNet()\n    net.load_state_dict(torch.load(a.ckpt, map_location="cpu"))\n    net.to(device)\n    net.eval()')
patch("experiments/eval_net.py",
      "    net_agent = NetAgent(net, sims=a.net_sims, seed=a.seed)",
      '    net_agent = NetAgent(net, sims=a.net_sims, seed=a.seed, device=a.device)')

# ---------------- run_evaluate.py ----------------
patch("experiments/run_evaluate.py",
      "import torch\n\nfrom experiments.eval_net import NetAgent",
      "import torch\n\nfrom training.device import resolve_device\n\nfrom experiments.eval_net import NetAgent")
patch("experiments/run_evaluate.py",
      '    ap.add_argument("--manifest", type=Path, default=None)\n    a = ap.parse_args()',
      '    ap.add_argument("--manifest", type=Path, default=None)\n'
      '    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")\n'
      "    a = ap.parse_args()\n"
      "    device = resolve_device(a.device)")
patch("experiments/run_evaluate.py",
      "    net, ckpt = load_model_from_checkpoint(a.checkpoint)",
      "    net, ckpt = load_model_from_checkpoint(a.checkpoint)\n    net.to(device)")
patch("experiments/run_evaluate.py",
      "            agent = NetAgent(net, sims=a.net_sims, seed=a.seed + g)",
      '            agent = NetAgent(net, sims=a.net_sims, seed=a.seed + g, device=a.device)')

# ---------------- representation.py ----------------
patch("gochess/representation.py",
      "    with torch.no_grad():\n"
      "        logits, _ = net(encode_for(net, state).unsqueeze(0))\n"
      "        mask = torch.from_numpy(action_mask(state)).unsqueeze(0)\n"
      "        return masked_softmax(logits, mask)[0]",
      "    with torch.no_grad():\n"
      "        dev = next(net.parameters()).device\n"
      "        logits, _ = net(encode_for(net, state).unsqueeze(0).to(dev))\n"
      "        mask = torch.from_numpy(action_mask(state)).unsqueeze(0).to(dev)\n"
      "        return masked_softmax(logits, mask)[0].cpu()")

# ---------------- resume.py ----------------
patch("training/resume.py",
      '        "rng": {"python": random.getstate(), "numpy": np.random.get_state(),\n'
      '                "torch": torch.get_rng_state()},',
      '        "rng": {"python": random.getstate(), "numpy": np.random.get_state(),\n'
      '                "torch": torch.get_rng_state(),\n'
      '                "torch_cuda": (torch.cuda.get_rng_state_all()\n'
      '                               if torch.cuda.is_available() else None)},')
patch("training/resume.py",
      "    random.setstate(rng[\"python\"])\n"
      "    np.random.set_state(rng[\"numpy\"])\n"
      '    torch.set_rng_state(rng["torch"])',
      "    random.setstate(rng[\"python\"])\n"
      "    np.random.set_state(rng[\"numpy\"])\n"
      '    torch.set_rng_state(rng["torch"])\n'
      '    if rng.get("torch_cuda") is not None and torch.cuda.is_available():\n'
      "        torch.cuda.set_rng_state_all(rng[\"torch_cuda\"])")

# ---------------- manifest.py ----------------
patch("experiments/manifest.py",
      '        "--eval-scenario", m["scenario"]["eval_scenario"],',
      '        "--eval-scenario", m["scenario"]["eval_scenario"],\n'
      '        "--device", str(m["hardware"].get("device", "auto")),')
patch("experiments/manifest.py",
      '    dst = out_dir / MANIFEST_NAME',
      '    dst = out_dir / Path(path).name')
patch("experiments/manifest.py",
      'DEFAULT_PATH = ROOT / "experiments" / "configs" / "experiment_manifest_v1.yaml"',
      'DEFAULT_PATH = ROOT / "experiments" / "configs" / "experiment_manifest_v2.yaml"')

print("all patches applied")
