#!/usr/bin/env python3
"""AEGIS M5 — train the from-scratch CNN target discriminator.

Trains our own conv net (src/aegis/cnn/model.py) on synthetic patches to tell
the designated RED balloon from wrong-colour balloons, distractor shapes and
background. Runs on CPU in seconds.

Examples:
    python train_cnn.py                       # quick train, report accuracy
    python train_cnn.py --epochs 15 --n 2000 --save runs/discriminator.pt
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "src")


def main() -> None:
    p = argparse.ArgumentParser(description="AEGIS M5 — train target discriminator")
    p.add_argument("--n", type=int, default=1500, help="training patches")
    p.add_argument("--val", type=int, default=400, help="validation patches")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--save", default=None, help="path to save weights")
    args = p.parse_args()

    import torch
    from torch import nn, optim

    from aegis.cnn.model import build_net
    from aegis.cnn.patches import make_dataset

    Xtr, ytr = make_dataset(args.n, seed=1)
    Xva, yva = make_dataset(args.val, seed=99)
    Xtr, ytr = torch.tensor(Xtr), torch.tensor(ytr)
    Xva, yva = torch.tensor(Xva), torch.tensor(yva)

    net = build_net()
    opt = optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    def accuracy(X, y):
        net.eval()
        with torch.no_grad():
            return (net(X).argmax(1) == y).float().mean().item()

    print(f"training from-scratch CNN on {args.n} synthetic patches (CPU)...")
    n = len(Xtr)
    for epoch in range(1, args.epochs + 1):
        net.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            loss = loss_fn(net(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print(f"  epoch {epoch:2d}  loss {total/n:.4f}  "
              f"train_acc {accuracy(Xtr, ytr):.3f}  val_acc {accuracy(Xva, yva):.3f}")

    if args.save:
        import os
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        torch.save(net.state_dict(), args.save)
        print(f"saved -> {args.save}")


if __name__ == "__main__":
    main()
