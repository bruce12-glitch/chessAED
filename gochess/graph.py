"""Deterministic, torch-free graph encoding of GoChessState.

Dense relational graph over board squares (N = size*size nodes):
  node features F=10: empty, stone, king, rook, knight, owner_p0, owner_p1,
                      side_to_move_is_p0, row/(size-1), col/(size-1)
  relations R=4 (edges[r, i, j] = 1 means i -> j):
    0 adjacent     orthogonal board adjacency (static, symmetric)
    1 legal_move   legal MOVE source -> target for the side to move
    2 capture      subset of legal_move whose target holds an enemy piece
    3 legal_place  self-loop on every square where a stone may legally be placed
Information parity with the CNN planes: piece type/owner/side-to-move only
(no reserves, ply or pass counters). Legality edges come from the rules engine.

pack_graph() flattens to a fixed [N, F + R*N] array so replay buffers and
TensorDataset need no changes. Action index space is Action.encode():
place t -> t, move s->t -> N + s*N + t, pass -> N + N*N.
"""
from __future__ import annotations

import numpy as np

from gochess.core import Action, ActionKind, GoChessState, Piece, owner

REL_NAMES = ("adjacent", "legal_move", "capture", "legal_place")
NUM_REL = len(REL_NAMES)
NODE_FEATURES = ("empty", "stone", "king", "rook", "knight", "owner_p0",
                 "owner_p1", "stm_p0", "row", "col")
NUM_NODE_FEATURES = len(NODE_FEATURES)


def num_nodes(size: int) -> int:
    return size * size


def packed_dim(size: int) -> int:
    return NUM_NODE_FEATURES + NUM_REL * num_nodes(size)


def num_actions(size: int) -> int:
    return Action.action_space_size(size)


def _adjacency(size: int) -> np.ndarray:
    n = num_nodes(size)
    adj = np.zeros((n, n), dtype=np.float32)
    for r in range(size):
        for c in range(size):
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    adj[r * size + c, nr * size + nc] = 1.0
    return adj


def node_features(state: GoChessState) -> np.ndarray:
    size, n = state.size, num_nodes(state.size)
    x = np.zeros((n, NUM_NODE_FEATURES), dtype=np.float32)
    for i, piece in enumerate(state.board):
        row, col = divmod(i, size)
        if piece is Piece.EMPTY:
            x[i, 0] = 1.0
        else:
            x[i, abs(int(piece))] = 1.0          # 1 stone, 2 king, 3 rook, 4 knight
            x[i, 5 if owner(piece) == 0 else 6] = 1.0
        x[i, 7] = 1.0 if state.current_player == 0 else 0.0
        x[i, 8] = row / (size - 1)
        x[i, 9] = col / (size - 1)
    return x


def edge_tensor(state: GoChessState) -> np.ndarray:
    size, n = state.size, num_nodes(state.size)
    e = np.zeros((NUM_REL, n, n), dtype=np.float32)
    e[0] = _adjacency(size)
    for a in state.legal_actions():
        if a.kind is ActionKind.MOVE:
            e[1, a.source, a.target] = 1.0
            tgt = state.board[a.target]
            if tgt is not Piece.EMPTY and owner(tgt) != state.current_player:
                e[2, a.source, a.target] = 1.0
        elif a.kind is ActionKind.PLACE:
            e[3, a.target, a.target] = 1.0
    return e


def build_graph(state: GoChessState) -> tuple[np.ndarray, np.ndarray]:
    return node_features(state), edge_tensor(state)


def pack_graph(state: GoChessState) -> np.ndarray:
    x, e = build_graph(state)
    n = x.shape[0]
    rel = np.transpose(e, (1, 0, 2)).reshape(n, NUM_REL * n)   # [N, R*N]
    return np.concatenate([x, rel], axis=1).astype(np.float32)


def unpack_graph(packed: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    n = num_nodes(size)
    if packed.shape[-2:] != (n, packed_dim(size)):
        raise ValueError(f"bad packed shape {packed.shape} for size {size}")
    x = packed[..., :NUM_NODE_FEATURES]
    rel = packed[..., NUM_NODE_FEATURES:].reshape(*packed.shape[:-2], n, NUM_REL, n)
    return x, np.swapaxes(rel, -3, -2)                          # [..., R, N, N]


def action_mask(state: GoChessState) -> np.ndarray:
    """1.0 at every legal action index, 0.0 elsewhere (all zero if terminal)."""
    m = np.zeros(num_actions(state.size), dtype=np.float32)
    for aid in state.legal_action_ids():
        m[aid] = 1.0
    return m


def action_to_index(action: Action, size: int) -> int:
    return action.encode(size)


def index_to_action(index: int, size: int) -> Action:
    return Action.decode(index, size)


def graph_policy_index_layout(size: int) -> dict[str, tuple[int, int]]:
    """[start, stop) of each policy-logit block; the GNN head emits this layout."""
    n = num_nodes(size)
    return {"place": (0, n), "move": (n, n + n * n), "pass": (n + n * n, n + n * n + 1)}
