"""Tests for from-scratch backprop (pure NumPy).

The headline test gradient-checks every layer against finite differences — the
proof our hand-derived gradients are correct. No torch needed.
"""

import sys

import numpy as np

sys.path.insert(0, "src")

from aegis.cnn import conv as loopconv  # noqa: E402
from aegis.cnn.autograd import (  # noqa: E402
    Adam,
    ScratchNet,
    conv_backward,
    conv_forward,
    linear_backward,
    linear_forward,
    maxpool_backward,
    maxpool_forward,
    relu_backward,
    relu_forward,
    softmax_cross_entropy,
)


def _num_grad(f, x, h=1e-5):
    """Numerical gradient of scalar f(x) wrt array x."""
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        i = it.multi_index
        old = x[i]
        x[i] = old + h; fp = f()
        x[i] = old - h; fm = f()
        x[i] = old
        g[i] = (fp - fm) / (2 * h)
        it.iternext()
    return g


# --- im2col conv equals the readable loop conv ---

def test_im2col_conv_matches_loop_reference():
    rng = np.random.RandomState(0)
    x = rng.randn(1, 3, 8, 8)
    w = rng.randn(4, 3, 3, 3)
    b = rng.randn(4)
    fast, _ = conv_forward(x, w, b, stride=1, pad=1)
    ref = loopconv.conv2d(x[0], w, b, stride=1, padding=1)  # single-sample loop
    assert np.allclose(fast[0], ref, atol=1e-9)


# --- per-layer gradient checks ---

def test_linear_gradient():
    rng = np.random.RandomState(1)
    x, w, b = rng.randn(4, 5), rng.randn(3, 5), rng.randn(3)
    dout = rng.randn(4, 3)
    out, cache = linear_forward(x, w, b)
    dx, dw, db = linear_backward(dout, cache)
    assert np.allclose(dx, _num_grad(lambda: (linear_forward(x, w, b)[0] * dout).sum(), x), atol=1e-5)
    assert np.allclose(dw, _num_grad(lambda: (linear_forward(x, w, b)[0] * dout).sum(), w), atol=1e-5)
    assert np.allclose(db, _num_grad(lambda: (linear_forward(x, w, b)[0] * dout).sum(), b), atol=1e-5)


def test_relu_gradient():
    rng = np.random.RandomState(2)
    x = rng.randn(3, 4)
    dout = rng.randn(3, 4)
    out, cache = relu_forward(x)
    dx = relu_backward(dout, cache)
    assert np.allclose(dx, _num_grad(lambda: (relu_forward(x)[0] * dout).sum(), x), atol=1e-6)


def test_maxpool_gradient():
    rng = np.random.RandomState(3)
    x = rng.randn(2, 3, 4, 4)
    dout = rng.randn(2, 3, 2, 2)
    out, cache = maxpool_forward(x)
    dx = maxpool_backward(dout, cache)
    assert np.allclose(dx, _num_grad(lambda: (maxpool_forward(x)[0] * dout).sum(), x), atol=1e-6)


def test_conv_gradient():
    rng = np.random.RandomState(4)
    x = rng.randn(2, 3, 6, 6)
    w = rng.randn(4, 3, 3, 3)
    b = rng.randn(4)
    dout = rng.randn(2, 4, 6, 6)
    out, cache = conv_forward(x, w, b, stride=1, pad=1)
    dx, dw, db = conv_backward(dout, cache)
    f = lambda: (conv_forward(x, w, b, stride=1, pad=1)[0] * dout).sum()
    assert np.allclose(dx, _num_grad(f, x), atol=1e-5)
    assert np.allclose(dw, _num_grad(f, w), atol=1e-5)
    assert np.allclose(db, _num_grad(f, b), atol=1e-5)


def test_softmax_cross_entropy_gradient():
    rng = np.random.RandomState(5)
    logits = rng.randn(4, 3)
    y = np.array([0, 2, 1, 0])
    loss, dlogits = softmax_cross_entropy(logits, y)
    num = _num_grad(lambda: softmax_cross_entropy(logits, y)[0], logits)
    assert np.allclose(dlogits, num, atol=1e-6)


# --- the whole net trains (loss decreases) with our own gradients ---

def test_scratchnet_loss_decreases_with_adam():
    rng = np.random.RandomState(0)
    X = rng.randn(16, 3, 32, 32)
    y = rng.randint(0, 2, size=16)
    net = ScratchNet(seed=0)
    opt = Adam(net.p, lr=1e-3)
    first, _ = net.loss_and_grad(X, y)
    for _ in range(15):
        loss, grads = net.loss_and_grad(X, y)
        opt.step(grads)
    assert loss < first  # our backprop + optimiser actually reduce the loss


def test_full_net_gradient_check_small():
    # End-to-end finite-difference check on one parameter slice.
    rng = np.random.RandomState(1)
    X = rng.randn(3, 3, 32, 32)
    y = np.array([0, 1, 0])
    net = ScratchNet(seed=2)
    _, grads = net.loss_and_grad(X, y)
    w = net.p["fc2_w"]

    def loss_only():
        return softmax_cross_entropy(net.forward(X), y)[0]

    num = _num_grad(loss_only, w)
    assert np.allclose(grads["fc2_w"], num, atol=1e-4)
