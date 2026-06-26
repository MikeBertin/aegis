#!/usr/bin/env python3
"""AEGIS M5 — train the discriminator with ZERO autograd.

Same CNN as train_cnn.py, but every gradient comes from our own backward pass
(aegis/cnn/autograd.py) and the optimiser is our own Adam — no PyTorch anywhere.
Pure NumPy, CPU.

    python train_scratch.py
    python train_scratch.py --epochs 20 --n 1200
"""

from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, "src")


def main() -> None:
    p = argparse.ArgumentParser(description="train the discriminator with hand-written backprop")
    p.add_argument("--n", type=int, default=900)
    p.add_argument("--val", type=int, default=300)
    p.add_argument("--epochs", type=int, default=14)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    args = p.parse_args()

    import numpy as np

    from aegis.cnn.autograd import Adam, ScratchNet
    from aegis.cnn.patches import make_dataset

    Xtr, ytr = make_dataset(args.n, seed=1)
    Xva, yva = make_dataset(args.val, seed=99)
    net = ScratchNet(seed=0)
    opt = Adam(net.p, lr=args.lr)

    def acc(X, y):
        return float((net.predict(X) == y).mean())

    print(f"training with hand-written backprop on {args.n} patches (pure NumPy)...")
    t0 = time.time()
    n = len(Xtr)
    for epoch in range(1, args.epochs + 1):
        perm = np.random.permutation(n)
        total = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            loss, grads = net.loss_and_grad(Xtr[idx], ytr[idx])
            opt.step(grads)
            total += loss * len(idx)
        print(f"  epoch {epoch:2d}  loss {total/n:.4f}  "
              f"train_acc {acc(Xtr, ytr):.3f}  val_acc {acc(Xva, yva):.3f}")
    print(f"done in {time.time()-t0:.1f}s — trained with zero autograd")


if __name__ == "__main__":
    main()
