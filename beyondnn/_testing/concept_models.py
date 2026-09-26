"""Hand-built ground-truth models for Phase-6 concept validation (plan §21).

``ConceptToy`` maps x ∈ R^12 through a hidden module ``hidden`` (12 units, exact
formulas) and a readout ``readout`` (6 outputs):

    h0 = x0            h1 = x1 (feeds no output)
    h2 = x2 + x3       h3 = x2 - x3
    h4 = h5 = x4       (duplicate)
    h6 = x6            h7 = x7 + x8
    h8 = relu(x9)·[x10 > 0]    h9 = relu(x9)·[x10 <= 0]
    h10 = x10          h11 = x11

    y0 = 3·h0   y1 = h2 + h3   y2 = max(h4, h5)   y3 = h6   y4 = h7   y5 = h8 + h9

Every expected concept outcome in the plan is derived from these formulas.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["ConceptToy", "concept_inputs"]


class _Hidden(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0, x1, x2, x3, x4, _x5, x6, x7, x8, x9, x10, x11 = x.unbind(dim=1)
        pos = torch.relu(x9)
        h = [
            x0,
            x1,
            x2 + x3,
            x2 - x3,
            x4,
            x4,
            x6,
            x7 + x8,
            pos * (x10 > 0).to(x.dtype),
            pos * (x10 <= 0).to(x.dtype),
            x10,
            x11,
        ]
        return torch.stack(h, dim=1)


class _Readout(nn.Module):
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        y = [
            3.0 * h[:, 0],
            h[:, 2] + h[:, 3],
            torch.maximum(h[:, 4], h[:, 5]),
            h[:, 6],
            h[:, 7],
            h[:, 8] + h[:, 9],
        ]
        return torch.stack(y, dim=1)


class ConceptToy(nn.Module):
    """``scale`` (a persistent buffer, default 1) multiplies every output; changing it
    gives a different checkpoint (state digest) with the same structure."""

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.hidden = _Hidden()
        self.readout = _Readout()
        self.register_buffer("scale", torch.tensor(float(scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.scale
        assert isinstance(scale, torch.Tensor)
        out: torch.Tensor = self.readout(self.hidden(x)) * scale
        return out


def concept_inputs(n: int, *, seed: int, proxy_noise: float | None = None) -> torch.Tensor:
    """``n`` inputs x ~ N(0, I_12) from a local generator. With ``proxy_noise``,
    x6 = x5 + proxy_noise·ε (the correlated-proxy dataset E1 of plan §21)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 12, generator=g)
    if proxy_noise is not None:
        x[:, 6] = x[:, 5] + proxy_noise * torch.randn(n, generator=g)
    return x
