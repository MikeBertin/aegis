"""Model tests — the from-scratch NumPy forward must match PyTorch, and the net
must actually learn the target. Skipped automatically if torch isn't installed."""

import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

torch = pytest.importorskip("torch")  # skip the whole module without torch

from aegis.cnn.discriminator import is_valid_target  # noqa: E402
from aegis.cnn.model import build_net, forward_numpy  # noqa: E402
from aegis.cnn.patches import make_dataset  # noqa: E402


def test_numpy_forward_matches_pytorch():
    """The hand-written NumPy forward reproduces PyTorch's output — the proof
    that our from-scratch conv/pool/linear are faithful to the real thing."""
    torch.manual_seed(0)
    net = build_net()
    net.eval()
    x = np.random.RandomState(0).rand(3, 32, 32).astype(np.float32)

    with torch.no_grad():
        ref = net(torch.tensor(x[None], dtype=torch.float32))[0].numpy()
    ours = forward_numpy(net, x)

    assert ours.shape == ref.shape == (2,)
    assert np.allclose(ours, ref, atol=1e-4), f"{ours} vs {ref}"


def test_net_learns_to_discriminate_the_target():
    """A few epochs on synthetic patches should push val accuracy well above
    chance — the from-scratch CNN genuinely learns 'designated red balloon'."""
    torch.manual_seed(0)
    from torch import nn, optim

    Xtr, ytr = make_dataset(600, seed=1)
    Xva, yva = make_dataset(200, seed=2)
    Xtr, ytr = torch.tensor(Xtr), torch.tensor(ytr)
    Xva, yva = torch.tensor(Xva), torch.tensor(yva)

    net = build_net()
    opt = optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(8):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss_fn(net(Xtr[idx]), ytr[idx]).backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        acc = (net(Xva).argmax(1) == yva).float().mean().item()
    assert acc > 0.8, f"val accuracy only {acc:.2f}"


def test_decision_threshold_pure():
    assert is_valid_target(0.7, 0.6) and not is_valid_target(0.5, 0.6)
