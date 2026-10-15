"""Dense relational GNN policy-value net (no torch_geometric; 25-node graphs).

Same I/O contract as GoChessNet: forward(x) -> (policy_logits[B, A], value[B]),
value in (-1, 1), logits laid out as Action.encode(): place [0,N), move
[N, N+N*N) as source*N+target, pass last. Input is the packed graph from
gochess.graph.pack_graph, shape [B, N, F + R*N].
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from gochess.graph import NUM_NODE_FEATURES, NUM_REL, num_actions, num_nodes


def unpack_batch(x: torch.Tensor, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """[B,N,F+R*N] -> node feats [B,N,F], relations [B,R,N,N] with rel[b,r,i,j]: i->j."""
    n = num_nodes(size)
    feats = x[..., :NUM_NODE_FEATURES]
    rel = x[..., NUM_NODE_FEATURES:].reshape(x.shape[0], n, NUM_REL, n).permute(0, 2, 1, 3)
    return feats, rel


class RelLayer(nn.Module):
    """h_j <- LN(h_j + ReLU(W_self h_j + sum_r mean_{i->j in r} W_r h_i + b))."""

    def __init__(self, d: int, num_rel: int = NUM_REL):
        super().__init__()
        self.d, self.num_rel = d, num_rel
        self.self_lin = nn.Linear(d, d, bias=False)
        self.rel_lin = nn.Linear(d, num_rel * d, bias=False)
        self.bias = nn.Parameter(torch.zeros(d))
        self.norm = nn.LayerNorm(d)

    def forward(self, h: torch.Tensor, rel_norm: torch.Tensor) -> torch.Tensor:
        b, n, _ = h.shape
        m = self.rel_lin(h).view(b, n, self.num_rel, self.d).permute(0, 2, 1, 3)  # [B,R,N,d]
        agg = torch.matmul(rel_norm, m).sum(1)                                    # [B,N,d]
        return self.norm(h + torch.relu(self.self_lin(h) + agg + self.bias))


class GoChessGNN(nn.Module):
    arch = "gnn"

    def __init__(self, size: int = 5, hidden: int = 128, layers: int = 4):
        super().__init__()
        self.size, self.hidden, self.layers = size, hidden, layers
        self.action_space = num_actions(size)
        self.inp = nn.Linear(NUM_NODE_FEATURES, hidden)
        self.convs = nn.ModuleList([RelLayer(hidden) for _ in range(layers)])
        self.place_head = nn.Linear(hidden, 1)
        self.src_proj = nn.Linear(hidden, hidden)
        self.dst_proj = nn.Linear(hidden, hidden)
        self.pair_rel = nn.Linear(NUM_REL, 1)
        self.pass_head = nn.Linear(hidden, 1)
        self.value_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1), nn.Tanh())

    def forward(self, x: torch.Tensor):
        feats, rel = unpack_batch(x, self.size)
        b, n = feats.shape[0], feats.shape[1]
        a_t = rel.transpose(-1, -2)                                   # message j <- i
        deg_in = rel.sum(-2).clamp_min(1.0).unsqueeze(-1)             # [B,R,N,1]
        rel_norm = a_t / deg_in
        h = torch.relu(self.inp(feats))
        for conv in self.convs:
            h = conv(h, rel_norm)
        place = self.place_head(h).squeeze(-1)                                        # [B,N]
        pair = torch.einsum("bnd,bmd->bnm", self.src_proj(h), self.dst_proj(h))
        pair = pair / math.sqrt(self.hidden) + self.pair_rel(rel.permute(0, 2, 3, 1)).squeeze(-1)
        pass_logit = self.pass_head(h.mean(1))                                        # [B,1]
        logits = torch.cat([place, pair.reshape(b, n * n), pass_logit], dim=1)
        value = self.value_head(torch.cat([h.mean(1), h.max(1).values], dim=1)).squeeze(-1)
        return logits, value
