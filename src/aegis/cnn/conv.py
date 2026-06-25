"""CNN primitives from first principles — pure NumPy.

The forward operations of a convolutional net, written out by hand so the
mechanics are explicit (and unit-tested against values you can compute on
paper). The PyTorch model in ``model.py`` is the trainable/deployable version;
:func:`forward` here re-implements *its* forward pass from the trained weights,
and a test asserts the two agree — the "we actually understand the CNN, not
just call it" proof.

All tensors are single-sample, channel-first: ``(C, H, W)``. No batching, no
autograd — just the arithmetic.
"""

from __future__ import annotations

import numpy as np


def conv2d(x, weight, bias=None, stride: int = 1, padding: int = 0):
    """2D cross-correlation (what every framework calls "convolution").

    x:      (C_in, H, W)
    weight: (C_out, C_in, kH, kW)
    returns (C_out, H_out, W_out)

    Production kernels use im2col/Winograd/FFT; this explicit triple loop is the
    same computation, written for clarity.
    """
    c_in, h, w = x.shape
    c_out, c_in_w, kh, kw = weight.shape
    assert c_in == c_in_w, f"channel mismatch: {c_in} vs {c_in_w}"

    if padding:
        x = np.pad(x, ((0, 0), (padding, padding), (padding, padding)))
    _, hp, wp = x.shape
    h_out = (hp - kh) // stride + 1
    w_out = (wp - kw) // stride + 1

    out = np.zeros((c_out, h_out, w_out), dtype=np.float64)
    for co in range(c_out):
        for i in range(h_out):
            si = i * stride
            for j in range(w_out):
                sj = j * stride
                region = x[:, si:si + kh, sj:sj + kw]
                out[co, i, j] = np.sum(region * weight[co])
        if bias is not None:
            out[co] += bias[co]
    return out


def relu(x):
    return np.maximum(0.0, x)


def max_pool2d(x, k: int = 2, stride: int = 2):
    """Channel-wise max pooling. x: (C, H, W) -> (C, H_out, W_out)."""
    c, h, w = x.shape
    h_out = (h - k) // stride + 1
    w_out = (w - k) // stride + 1
    out = np.zeros((c, h_out, w_out), dtype=x.dtype)
    for ch in range(c):
        for i in range(h_out):
            for j in range(w_out):
                out[ch, i, j] = x[ch, i * stride:i * stride + k,
                                  j * stride:j * stride + k].max()
    return out


def linear(x, weight, bias):
    """Fully-connected layer. x: (in,), weight: (out, in), bias: (out,)."""
    return weight @ x + bias


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()
