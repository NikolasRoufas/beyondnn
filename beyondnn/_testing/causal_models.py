"""Fixed-weight ground-truth causal models for Phase-2 tests (internal, unstable).

Input ``x = (x0, x1)`` of shape (N, 2); output ``y`` of shape (N, 1). Weights are set
exactly (no randomness), so intervention effects are known analytically:

=============  ======================  ==================  =========================
Model          Definition              Intervention        Effect on y[0, 0]
=============  ======================  ==================  =========================
Additive       y = a(x) + b(x)         zero a              -x0
               a = x0, b = x1          constant a := c     c - x0
                                       patch a from x'     x'0 - x0
Gated          y = gate(x) * signal(x) zero gate           -x0 * x1  (output -> 0)
Redundant      y = p(x) + q(x),        zero p              -x0 (output stays x0:
               p = q = x0                                  no single path is necessary)
Interaction    y = a(x) * b(x)         zero a              -x0 * x1  (0 when x1 = 0)
=============  ======================  ==================  =========================
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["Additive", "Gated", "Interaction", "Redundant"]


def _pick(index: int) -> nn.Linear:
    """A bias-free linear map returning ``x[:, index]`` exactly."""
    layer = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        layer.weight.zero_()
        layer.weight[0, index] = 1.0
    return layer


class Additive(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = _pick(0)
        self.b = _pick(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.a(x) + self.b(x)
        return out


class Gated(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = _pick(0)
        self.signal = _pick(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.gate(x) * self.signal(x)
        return out


class Redundant(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p = _pick(0)
        self.q = _pick(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.p(x) + self.q(x)
        return out


class Interaction(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = _pick(0)
        self.b = _pick(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.a(x) * self.b(x)
        return out
