"""Architecture cost: parameters, encode/inference speed, NetAgent search speed."""
from __future__ import annotations

import time

import torch

from experiments.eval_net import NetAgent
from gochess.representation import build_model, count_params, encode_for
from gochess.scenarios import scenarios


def measure_throughput(arch: str, kwargs: dict, *, size: int = 5, repeats: int = 100,
                       batch: int = 64, search_sims: int = 64,
                       probe: str = "mixed_midgame", seed: int = 0) -> dict:
    torch.manual_seed(seed)
    net = build_model(arch, size=size, **kwargs).eval()
    s = scenarios()[probe].state
    t = time.perf_counter()
    for _ in range(repeats):
        x = encode_for(net, s)
    encode_ms = (time.perf_counter() - t) / repeats * 1e3
    x1 = x.unsqueeze(0)
    with torch.no_grad():
        net(x1)
        t = time.perf_counter()
        for _ in range(repeats):
            net(x1)
        forward_ms = (time.perf_counter() - t) / repeats * 1e3
        xb = x.unsqueeze(0).repeat(batch, *([1] * x.dim()))
        net(xb)
        t = time.perf_counter()
        for _ in range(20):
            net(xb)
        batch_states_per_s = batch * 20 / (time.perf_counter() - t)
        agent = NetAgent(net, sims=search_sims, seed=seed)
        agent.search(s)
        t = time.perf_counter()
        for _ in range(3):
            agent.search(s)
        search_sims_per_s = 3 * search_sims / (time.perf_counter() - t)
    return {"arch": arch, "kwargs": kwargs, "param_count": count_params(net),
            "encode_ms": encode_ms, "forward_ms_single": forward_ms,
            "forward_states_per_s_batch": batch_states_per_s,
            "search_sims_per_s": search_sims_per_s, "search_sims": search_sims,
            "probe": probe, "torch_threads": torch.get_num_threads()}
