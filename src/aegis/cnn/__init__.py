"""From-scratch CNN target discriminator (M5).

Import-safe: only the pure NumPy primitives and the pure decision logic are
exported eagerly. The model + discriminator import torch lazily inside their
functions, so importing this package needs no torch.
"""

from .conv import conv2d, linear, max_pool2d, relu, softmax
from .discriminator import is_valid_target

__all__ = ["conv2d", "relu", "max_pool2d", "linear", "softmax", "is_valid_target"]
