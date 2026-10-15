"""GPU trainer: same data as train_alphazero, training on CUDA (uses 1 GPU).

Run: CUDA_VISIBLE_DEVICES=0 python -m experiments.train_cuda --games 16 --uct-simulations 16 --epochs 10 --channels 128 --blocks 4 --batch 256
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from gochess.nn import GoChessNet
from experiments.train_alphazero import collect


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--uct-simulations", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--output", type=Path, default=Path("results/cuda"))
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} gpus={torch.cuda.device_count()}")
    net = GoChessNet(channels=a.channels, blocks=a.blocks).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    log = []
    for it in range(a.epochs):
        X, PI, Z, ent = collect(a.games, a.uct_simulations, a.seed + it * 1000)
        X, PI, Z = X.to(device), PI.to(device), Z.to(device)
        ds = DataLoader(TensorDataset(X, PI, Z), batch_size=a.batch, shuffle=True)
        net.train()
        for xb, pib, zb in ds:
            logits, v = net(xb)
            loss = F.cross_entropy(logits, pib) + F.mse_loss(v, zb)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            logits, v = net(X)
            vloss = F.mse_loss(v, Z).item()
            acc = (logits.argmax(1) == PI.argmax(1)).float().mean().item()
        row = {"iter": it, "vloss": vloss, "acc": acc, "ent": ent, "device": str(device)}
        log.append(row); print(row, flush=True)
        torch.save(net.state_dict(), a.output / f"ckpt_{it}.pt")
    (a.output / "cuda_log.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
