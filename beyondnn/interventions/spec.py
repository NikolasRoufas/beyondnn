"""Intervention builders (runtime descriptions turned into ``InterventionRecord``s)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from beyondnn.schema import InterventionOperation

__all__ = ["Intervention", "constant", "patch", "zero"]


@dataclass(frozen=True, slots=True, eq=False)
class Intervention:
    """What to replace: one tensor leaf (``output_path``) of call ``call_index`` of the
    module at ``site``'s output, and how. Build with :func:`zero`, :func:`constant`,
    :func:`patch`."""

    site: str
    operation: InterventionOperation
    output_path: str = ""
    call_index: int = 0
    constant: float | None = None
    tensor: torch.Tensor | None = None
    source_inputs: tuple[Any, ...] = ()
    source_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.site, str) or not self.site:
            raise ValueError("site must be a non-root module path (a string)")
        if (
            isinstance(self.call_index, bool)
            or not isinstance(self.call_index, int)
            or self.call_index < 0
        ):
            raise ValueError("call_index must be an int >= 0")


def zero(site: str, *, output_path: str = "", call_index: int = 0) -> Intervention:
    """Zero ablation: replace the leaf with ``zeros_like`` (not a neutral baseline)."""
    return Intervention(site, InterventionOperation.ZERO, output_path, call_index)


def constant(
    site: str, value: float | torch.Tensor, *, output_path: str = "", call_index: int = 0
) -> Intervention:
    """Replace the leaf with a constant: a finite scalar (filled into the leaf's shape)
    or a tensor of exactly the leaf's shape and dtype (no broadcasting)."""
    if isinstance(value, torch.Tensor):
        return Intervention(
            site,
            InterventionOperation.CONSTANT,
            output_path,
            call_index,
            tensor=value.detach().to("cpu").clone(memory_format=torch.contiguous_format),
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("a scalar constant must be a finite real number")
    return Intervention(
        site, InterventionOperation.CONSTANT, output_path, call_index, constant=float(value)
    )


def patch(
    site: str,
    *source_inputs: Any,
    output_path: str = "",
    call_index: int = 0,
    source_kwargs: dict[str, Any] | None = None,
) -> Intervention:
    """Activation patching: replace the leaf with the activation the same site/leaf/call
    produced on ``source_inputs``. The source execution runs as its own pass in the same
    experiment, so the patched value is a provenance-bearing, retained MEASURED record."""
    if not source_inputs:
        raise ValueError("patch() needs the source inputs whose activation is patched in")
    return Intervention(
        site,
        InterventionOperation.PATCH,
        output_path,
        call_index,
        source_inputs=tuple(source_inputs),
        source_kwargs=dict(source_kwargs or {}),
    )
