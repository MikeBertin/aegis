"""Backpropagation from scratch — train the CNN with zero autograd.

`conv.py` gives the forward pass; this gives the *backward* pass — the gradients
of every layer (conv, max-pool, ReLU, linear, softmax-cross-entropy) plus an
Adam optimiser, so the discriminator can be trained with gradients we compute
ourselves. Correctness is established by gradient-checking against finite
differences (`test_cnn_autograd.py`).

Conv uses **im2col** (the standard real implementation — built here) so training
is fast; a test asserts it equals the readable loop version in `conv.py`.
Batched, channel-first: ``(N, C, H, W)``.
"""

from __future__ import annotations

import numpy as np


# --- im2col / col2im -----------------------------------------------------

def im2col(x, kh, kw, stride, pad):
    n, c, h, w = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    oh = (h + 2 * pad - kh) // stride + 1
    ow = (w + 2 * pad - kw) // stride + 1
    cols = np.zeros((n, c, kh, kw, oh, ow))
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j, :, :] = xp[:, :, i:i + stride * oh:stride, j:j + stride * ow:stride]
    cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(n * oh * ow, -1)
    return cols, (oh, ow)


def col2im(cols, x_shape, kh, kw, stride, pad, oh, ow):
    n, c, h, w = x_shape
    xp = np.zeros((n, c, h + 2 * pad, w + 2 * pad))
    cols = cols.reshape(n, oh, ow, c, kh, kw).transpose(0, 3, 4, 5, 1, 2)
    for i in range(kh):
        for j in range(kw):
            xp[:, :, i:i + stride * oh:stride, j:j + stride * ow:stride] += cols[:, :, i, j, :, :]
    return xp[:, :, pad:pad + h, pad:pad + w] if pad else xp


# --- layers: forward returns (out, cache); backward returns grads --------

def conv_forward(x, w, b, stride=1, pad=0):
    n, c, h, wd = x.shape
    f, _, kh, kw = w.shape
    cols, (oh, ow) = im2col(x, kh, kw, stride, pad)
    out = (cols @ w.reshape(f, -1).T + b).reshape(n, oh, ow, f).transpose(0, 3, 1, 2)
    return out, (x.shape, w, stride, pad, cols, oh, ow)


def conv_backward(dout, cache):
    x_shape, w, stride, pad, cols, oh, ow = cache
    f, c, kh, kw = w.shape
    dout2 = dout.transpose(0, 2, 3, 1).reshape(-1, f)
    db = dout2.sum(0)
    dw = (dout2.T @ cols).reshape(w.shape)
    dx = col2im(dout2 @ w.reshape(f, -1), x_shape, kh, kw, stride, pad, oh, ow)
    return dx, dw, db


def relu_forward(x):
    return np.maximum(0, x), x


def relu_backward(dout, x):
    return dout * (x > 0)


def maxpool_forward(x, k=2, stride=2):
    n, c, h, w = x.shape
    assert h % k == 0 and w % k == 0 and stride == k, "this pool assumes tiling"
    xr = x.reshape(n, c, h // k, k, w // k, k)
    out = xr.max(axis=(3, 5))
    mask = xr == out[:, :, :, None, :, None]
    return out, (mask, x.shape, k)


def maxpool_backward(dout, cache):
    mask, shape, k = cache
    n, c, h, w = shape
    dexp = dout[:, :, :, None, :, None]
    # split gradient across ties so it sums correctly.
    dx = mask * dexp / mask.sum(axis=(3, 5), keepdims=True)
    return dx.reshape(shape)


def linear_forward(x, w, b):
    return x @ w.T + b, (x, w)


def linear_backward(dout, cache):
    x, w = cache
    return dout @ w, dout.T @ x, dout.sum(0)


def softmax_cross_entropy(logits, y):
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(1, keepdims=True)
    n = logits.shape[0]
    loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
    dlogits = p.copy()
    dlogits[np.arange(n), y] -= 1
    dlogits /= n
    return loss, dlogits


# --- the network (our architecture, hand-rolled forward+backward) --------

class ScratchNet:
    """Same architecture as cnn.model.build_net, trained with our own gradients."""

    def __init__(self, num_classes: int = 2, seed: int = 0) -> None:
        rng = np.random.RandomState(seed)

        def he(shape, fan_in):
            return rng.randn(*shape) * np.sqrt(2.0 / fan_in)

        self.p = {
            "conv1_w": he((8, 3, 3, 3), 3 * 9), "conv1_b": np.zeros(8),
            "conv2_w": he((16, 8, 3, 3), 8 * 9), "conv2_b": np.zeros(16),
            "fc1_w": he((32, 16 * 8 * 8), 16 * 8 * 8), "fc1_b": np.zeros(32),
            "fc2_w": he((num_classes, 32), 32), "fc2_b": np.zeros(num_classes),
        }

    def forward(self, X):
        p, cache = self.p, {}
        a1, cache["c1"] = conv_forward(X, p["conv1_w"], p["conv1_b"], pad=1)
        r1, cache["r1"] = relu_forward(a1)
        m1, cache["m1"] = maxpool_forward(r1)
        a2, cache["c2"] = conv_forward(m1, p["conv2_w"], p["conv2_b"], pad=1)
        r2, cache["r2"] = relu_forward(a2)
        m2, cache["m2"] = maxpool_forward(r2)
        cache["flat_shape"] = m2.shape
        flat = m2.reshape(m2.shape[0], -1)
        f1, cache["f1"] = linear_forward(flat, p["fc1_w"], p["fc1_b"])
        h, cache["rh"] = relu_forward(f1)
        logits, cache["f2"] = linear_forward(h, p["fc2_w"], p["fc2_b"])
        self._cache = cache
        return logits

    def backward(self, dlogits):
        c, g = self._cache, {}
        dh, g["fc2_w"], g["fc2_b"] = linear_backward(dlogits, c["f2"])
        df1 = relu_backward(dh, c["rh"])
        dflat, g["fc1_w"], g["fc1_b"] = linear_backward(df1, c["f1"])
        dm2 = dflat.reshape(c["flat_shape"])
        dr2 = maxpool_backward(dm2, c["m2"])
        da2 = relu_backward(dr2, c["r2"])
        dm1, g["conv2_w"], g["conv2_b"] = conv_backward(da2, c["c2"])
        dr1 = maxpool_backward(dm1, c["m1"])
        da1 = relu_backward(dr1, c["r1"])
        _, g["conv1_w"], g["conv1_b"] = conv_backward(da1, c["c1"])
        return g

    def loss_and_grad(self, X, y):
        logits = self.forward(X)
        loss, dlogits = softmax_cross_entropy(logits, y)
        return loss, self.backward(dlogits)

    def predict(self, X):
        return self.forward(X).argmax(1)


# --- optimiser (from scratch) --------------------------------------------

class Adam:
    def __init__(self, params: dict, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p = params
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads: dict) -> None:
        self.t += 1
        for k in self.p:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
