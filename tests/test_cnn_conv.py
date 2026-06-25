"""Tests for the from-scratch CNN primitives (pure NumPy, hand-computed)."""

import sys

import numpy as np

sys.path.insert(0, "src")

from aegis.cnn.conv import conv2d, linear, max_pool2d, relu, softmax  # noqa: E402


def test_conv2d_sum_of_ones_kernel():
    # 1 channel, 3x3 ramp; 2x2 kernel of ones, stride 1, no pad -> windowed sums.
    x = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=float)
    w = np.ones((1, 1, 2, 2))
    out = conv2d(x, w)
    assert out.shape == (1, 2, 2)
    # top-left window 1+2+4+5 = 12 ; etc.
    assert out[0].tolist() == [[12, 16], [24, 28]]


def test_conv2d_bias_and_padding_shape():
    x = np.ones((2, 4, 4))
    w = np.ones((3, 2, 3, 3))
    b = np.array([1.0, 2.0, 3.0])
    out = conv2d(x, w, b, stride=1, padding=1)
    assert out.shape == (3, 4, 4)               # padding preserves H,W
    # centre pixel sees full 3x3x2 ones = 18, plus bias.
    assert out[0, 2, 2] == 18 + 1


def test_conv2d_stride():
    x = np.ones((1, 4, 4))
    w = np.ones((1, 1, 2, 2))
    out = conv2d(x, w, stride=2)
    assert out.shape == (1, 2, 2) and (out == 4).all()


def test_relu():
    assert relu(np.array([-2.0, 0.0, 3.0])).tolist() == [0.0, 0.0, 3.0]


def test_max_pool2d():
    x = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]],
                 dtype=float)
    out = max_pool2d(x, k=2, stride=2)
    assert out[0].tolist() == [[6, 8], [14, 16]]


def test_linear():
    w = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    b = np.array([0.0, 1.0, -1.0])
    assert linear(np.array([3.0, 4.0]), w, b).tolist() == [3.0, 9.0, 6.0]


def test_softmax_sums_to_one_and_orders():
    s = softmax(np.array([1.0, 2.0, 3.0]))
    assert abs(s.sum() - 1.0) < 1e-9
    assert s[2] > s[1] > s[0]
