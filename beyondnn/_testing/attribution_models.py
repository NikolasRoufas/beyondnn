"""Fixed-weight ground-truth models for Phase-3 attribution tests (internal, unstable).

Input ``x = (x0, x1)`` of shape (1, 2); target ``y[0, 0]``. Expected attributions are
derived analytically in docs/PHASE_3_PLAN.md (before any experiment ran).

==========  ===================  ==============  ==============  =====================
Model       F                    gradient        input x grad    IG, zero baseline
==========  ===================  ==============  ==============  =====================
Linear      2 x0 + 3 x1          [2, 3]          [2 x0, 3 x1]    [2 x0, 3 x1]
Irrelevant  4 x0                 [4, 0]          [4 x0, 0]       [4 x0, 0]
Product     x0 x1                [x1, x0]        [x0 x1, x0 x1]  [x0 x1 / 2, x0 x1 / 2]
Saturating  tanh(x0) + x1        [sech^2 x0, 1]  [x0 sech^2 x0,  [tanh x0, x1]
                                                  x1]
==========  ===================  ==============  ==============  =====================

``Twice`` applies one scalar linear map (weight 2) twice: ``y = lin(lin(x))``; the
gradient at call 0's output is 2 and at call 1's output is 1. ``Aliased`` registers
one module under two paths.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["Aliased", "Irrelevant", "Linear", "Product", "Saturating", "Twice"]


def _linear(weights: list[float]) -> nn.Linear:
    layer = nn.Linear(len(weights), 1, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([weights]))
    return layer


class Linear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = _linear([2.0, 3.0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.lin(x)
        return out


class Irrelevant(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = _linear([4.0, 0.0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.lin(x)
        return out


class Product(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x[:, 0] * x[:, 1]).unsqueeze(1)


class Saturating(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.tanh(x[:, 0]) + x[:, 1]).unsqueeze(1)


class Twice(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = _linear([2.0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.lin(self.lin(x))
        return out


class Aliased(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = _linear([1.0, 1.0])
        self.alias = self.first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.first(x)
        return out
