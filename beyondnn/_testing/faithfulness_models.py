"""Fixed-weight ground-truth models for Phase-5 faithfulness tests (internal, unstable).

Input ``x`` of shape (1, d); target ``y[0, 0]``. The causal structure of each model is
known by construction and was written down in docs/PHASE_5_PLAN.md (§9) before any
method was evaluated on it.

================  ==============================  =========================================
Model             F                               Ground truth
================  ==============================  =========================================
Weighted8         8x0 + 4x1 + 2x2 + x3            importance 0 > 1 > 2 > 3; x4..x7 unused
ProxyDistractor   x0 + x1                         x2 unused (a declared proxy of y)
RedundantMax      max(x0, x1)                     redundant; each alone sufficient at x0 = x1
EqualSum4         x0 + x1 + x2 + x3               all relevant, none alone sufficient
ProbeReadable     h = (x0, x1); y = 2 h0          h1 readable (= x1) but causally unused
SaturatingPlus    tanh(4 x0) + 0.2 x1             x0 dominant at x0 = 3; its gradient ~ 0
================  ==============================  =========================================
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = [
    "EqualSum4",
    "ProbeReadable",
    "ProxyDistractor",
    "RedundantMax",
    "SaturatingPlus",
    "Weighted8",
    "proxy_dataset",
]


def _linear(weights: list[list[float]]) -> nn.Linear:
    layer = nn.Linear(len(weights[0]), len(weights), bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(weights))
    return layer


class Weighted8(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = _linear([[8.0, 4.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0]])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.lin(x)
        return out


class ProxyDistractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = _linear([[1.0, 1.0, 0.0]])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.lin(x)
        return out


def proxy_dataset() -> torch.Tensor:
    """The declared data distribution of ``ProxyDistractor``: a fixed grid with
    ``x2 = x0 + x1`` (x2 is a perfect proxy of y but causally unused)."""
    rows = [[a, b, a + b] for a in (0.0, 1.0, 2.0, 3.0) for b in (0.0, 1.0, 2.0)]
    return torch.tensor(rows)


class RedundantMax(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.maximum(x[:, 0], x[:, 1]).unsqueeze(1)


class EqualSum4(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = _linear([[1.0, 1.0, 1.0, 1.0]])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.lin(x)
        return out


class ProbeReadable(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = _linear([[1.0, 0.0], [0.0, 1.0]])
        self.readout = _linear([[2.0, 0.0]])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.readout(self.hidden(x))
        return out


class SaturatingPlus(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.tanh(4.0 * x[:, 0]) + 0.2 * x[:, 1]).unsqueeze(1)
