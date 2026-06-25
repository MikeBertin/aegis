"""The target-discriminator CNN — our own architecture, trained from scratch.

A small conv net we define ourselves (not a library detector): two conv blocks
then two fully-connected layers. PyTorch handles autograd/training; :func:`forward_numpy`
re-runs the *same* forward pass using the hand-written primitives in
:mod:`aegis.cnn.conv`, and a test asserts the two match bit-for-bit-ish — proving
the from-scratch implementation is faithful.

Input: (3, 32, 32). Two 2× pools take 32→16→8, so the flattened size is 16·8·8.
"""

from __future__ import annotations

from .conv import conv2d, linear, max_pool2d, relu

PATCH = 32
NUM_CLASSES = 2


def build_net(num_classes: int = NUM_CLASSES):
    """Our architecture as an nn.Sequential (lazy torch import)."""
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 32 -> 16
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16 -> 8
        nn.Flatten(),
        nn.Linear(16 * 8 * 8, 32), nn.ReLU(),
        nn.Linear(32, num_classes),
    )


def forward_numpy(net, x):
    """Re-implement ``net``'s forward pass in pure NumPy from its weights.

    ``net`` is a :func:`build_net` module; ``x`` is a single (3,32,32) array.
    Returns the logits. Should equal ``net(x[None])[0]`` up to float error.
    """
    import numpy as np

    def params(layer):
        return (layer.weight.detach().cpu().numpy().astype(np.float64),
                layer.bias.detach().cpu().numpy().astype(np.float64))

    w0, b0 = params(net[0])      # conv1
    w3, b3 = params(net[3])      # conv2
    w7, b7 = params(net[7])      # fc1
    w9, b9 = params(net[9])      # fc2

    a = max_pool2d(relu(conv2d(x.astype(np.float64), w0, b0, padding=1)))
    a = max_pool2d(relu(conv2d(a, w3, b3, padding=1)))
    a = a.reshape(-1)            # flatten (C,H,W) row-major == torch Flatten
    a = relu(linear(a, w7, b7))
    return linear(a, w9, b9)
