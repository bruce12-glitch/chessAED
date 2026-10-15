"""Architecture registry: one state encoder + model factory per representation.

All arms share the 651-way action layout, so loss/TV/accuracy code is unchanged.
Checkpoints store {"arch", "model_kwargs"} in config so they reload standalone.
"""
from __future__ import annotations

ARCHS = ("cnn", "gnn")
DEFAULT_MODEL_KWARGS = {"cnn": {"channels": 64, "blocks": 4},
                        "gnn": {"hidden": 128, "layers": 4}}


def check_arch(arch: str) -> str:
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}; expected one of {ARCHS}")
    return arch


def model_kwargs(arch: str, overrides: dict | None = None) -> dict:
    kw = dict(DEFAULT_MODEL_KWARGS[check_arch(arch)])
    kw.update(overrides or {})
    return kw


def build_model(arch: str, size: int = 5, **overrides):
    kw = model_kwargs(arch, overrides)
    if arch == "cnn":
        from gochess.nn import GoChessNet
        return GoChessNet(size=size, **kw)
    from gochess.gnn import GoChessGNN
    return GoChessGNN(size=size, **kw)


def encode_state(arch: str, state):
    """Per-state input tensor with a fixed shape (stackable in the replay buffer)."""
    import torch
    if check_arch(arch) == "cnn":
        from gochess.nn import obs_to_tensor
        return obs_to_tensor(state)
    from gochess.graph import pack_graph
    return torch.from_numpy(pack_graph(state))


def net_arch(net) -> str:
    return getattr(net, "arch", "cnn")


def encode_for(net, state):
    return encode_state(net_arch(net), state)


def count_params(net) -> int:
    return sum(p.numel() for p in net.parameters())


def masked_softmax(logits, mask):
    """Softmax over legal actions only; rows with no legal action give zeros."""
    import torch
    legal = mask > 0
    neg = torch.finfo(logits.dtype).min
    p = torch.softmax(logits.masked_fill(~legal, neg), dim=-1) * legal
    return p / p.sum(-1, keepdim=True).clamp_min(1e-12)


def policy_probs(net, state):
    """[A] probabilities, nonzero only on legal actions (zeros if terminal)."""
    import torch
    from gochess.graph import action_mask
    with torch.no_grad():
        dev = next(net.parameters()).device
        logits, _ = net(encode_for(net, state).unsqueeze(0).to(dev))
        mask = torch.from_numpy(action_mask(state)).unsqueeze(0).to(dev)
        return masked_softmax(logits, mask)[0].cpu()


def load_model_from_checkpoint(path):
    """Rebuild the right architecture from checkpoint config and load weights."""
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    arch = cfg.get("arch", "cnn")
    net = build_model(arch, **cfg.get("model_kwargs", model_kwargs(arch)))
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net, ckpt
