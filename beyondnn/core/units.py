"""Evidence units of a tensor site (ADR-034).

A *unit* is one index of the sub-grid spanned by declared ``unit_axes`` of a site
tensor (row-major over those axes); it covers every element whose indices along the
unit axes equal that grid index. The remaining ("within-unit") axes are replaced
together and, for scores, reduced by a declared reduction. Examples:

* tabular features ``(1, d)``: ``unit_axes=(1,)`` -> d units (one per feature);
* image pixels ``(1, C, H, W)``: ``unit_axes=(2, 3)`` -> H*W units (all channels of a
  spatial position); channels: ``unit_axes=(1,)`` -> C units (whole feature maps);
* token positions of embeddings ``(1, T, D)``: ``unit_axes=(1,)`` -> T units.

``unit_axes=None`` keeps the Phase-5 meaning: indices along the last axis. Axes are
explicit, non-negative, sorted and unique; nothing is inferred from the shape.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

__all__ = [
    "UNIT_REDUCTIONS",
    "UnitError",
    "check_axes",
    "unit_count",
    "unit_mask",
    "unit_values",
]

UNIT_REDUCTIONS = ("sum", "abs_sum", "l2")


class UnitError(ValueError):
    """Unit axes do not fit the tensor, or units are out of range."""


def check_axes(axes: Sequence[int] | None) -> tuple[int, ...] | None:
    """Validate declared unit axes (non-negative, sorted, unique, non-empty)."""
    if axes is None:
        return None
    out = tuple(axes)
    if not out or not all(isinstance(a, int) and not isinstance(a, bool) and a >= 0 for a in out):
        raise UnitError("unit_axes must be a non-empty sequence of non-negative ints")
    if list(out) != sorted(set(out)):
        raise UnitError("unit_axes must be sorted and unique")
    return out


def _axes(shape: Sequence[int], axes: tuple[int, ...] | None) -> tuple[int, ...]:
    if axes is None:
        if len(shape) == 0:
            raise UnitError("a scalar tensor has no units")
        return (len(shape) - 1,)
    if axes[-1] >= len(shape):
        raise UnitError(f"unit_axes {axes} do not fit a tensor of shape {tuple(shape)}")
    return axes


def unit_count(shape: Sequence[int], axes: tuple[int, ...] | None) -> int:
    return math.prod(shape[a] for a in _axes(shape, axes))


def unit_mask(
    shape: Sequence[int],
    axes: tuple[int, ...] | None,
    units: Sequence[int],
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Boolean tensor of ``shape``: True at every element of the given units."""
    ax = _axes(shape, axes)
    grid = [shape[a] for a in ax]
    n = math.prod(grid)
    if any(u < 0 or u >= n for u in units):
        raise UnitError(
            f"units {list(units)} are out of range for {n} units of shape {tuple(shape)}"
        )
    flat = torch.zeros(n, dtype=torch.bool, device=device)
    flat[list(units)] = True
    view = [1] * len(shape)
    for a, size in zip(ax, grid, strict=True):
        view[a] = size
    return flat.reshape(grid).reshape(view).expand(tuple(shape))


def unit_values(tensor: torch.Tensor, axes: tuple[int, ...] | None, how: str) -> torch.Tensor:
    """Per-unit values (row-major over the unit grid): ``how`` reduces the within-unit
    elements (``sum``, ``abs_sum`` or ``l2``), in float64."""
    if how not in UNIT_REDUCTIONS:
        raise UnitError(f"unit reduction must be one of {UNIT_REDUCTIONS}")
    ax = _axes(tuple(tensor.shape), axes)
    other = [d for d in range(tensor.dim()) if d not in ax]
    t = tensor.detach().double().permute(*ax, *other)
    t = t.reshape(unit_count(tuple(tensor.shape), ax), -1)
    if how == "sum":
        return t.sum(dim=1)
    if how == "abs_sum":
        return t.abs().sum(dim=1)
    return t.pow(2).sum(dim=1).sqrt()
