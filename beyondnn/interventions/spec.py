"""Intervention builders (runtime descriptions turned into ``InterventionRecord``s)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import torch

from beyondnn.core.units import check_axes
from beyondnn.schema import InterventionOperation

__all__ = [
    "Intervention",
    "constant",
    "constant_input",
    "direction",
    "patch",
    "zero",
    "zero_input",
]

_INPUT_LEAF = re.compile(r"^args\[(0|[1-9][0-9]*)\]$")


@dataclass(frozen=True, slots=True, eq=False)
class Intervention:
    """What to replace: one tensor leaf (``output_path``) of call ``call_index`` of the
    module at ``site``'s output (or, with ``site=""``, positional model input
    ``output_path="args[i]"``), and how. ``units`` restricts the replacement to indices
    along the leaf's last dimension (``retain=True``: replace every other unit). Build
    with :func:`zero`, :func:`constant`, :func:`patch`, :func:`zero_input`,
    :func:`constant_input`."""

    site: str
    operation: InterventionOperation
    output_path: str = ""
    call_index: int = 0
    constant: float | None = None
    tensor: torch.Tensor | None = None
    source_inputs: tuple[Any, ...] = ()
    source_kwargs: dict[str, Any] = field(default_factory=dict)
    units: tuple[int, ...] | None = None
    retain: bool = False
    unit_axes: tuple[int, ...] | None = None
    direction: torch.Tensor | None = None
    direction_axis: int | None = None

    @property
    def on_input(self) -> bool:
        return self.site == ""

    def __post_init__(self) -> None:
        if not isinstance(self.site, str):
            raise ValueError("site must be a module path (a string)")
        if self.site == "" and not _INPUT_LEAF.match(self.output_path):
            raise ValueError("site '' is a model input: use zero_input()/constant_input()")
        if self.units is not None:
            units = tuple(self.units)
            if not units or not all(
                isinstance(u, int) and not isinstance(u, bool) and u >= 0 for u in units
            ):
                raise ValueError("units must be a non-empty sequence of ints >= 0")
            if len(set(units)) != len(units):
                raise ValueError("units must be unique")
            object.__setattr__(self, "units", tuple(sorted(units)))
        if (
            self.retain
            and self.units is None
            and self.operation is not InterventionOperation.DIRECTION
        ):
            raise ValueError("retain=True needs units (the units to keep)")
        if self.unit_axes is not None:
            if self.units is None:
                raise ValueError("unit_axes needs units")
            object.__setattr__(self, "unit_axes", check_axes(self.unit_axes))
        if (
            isinstance(self.call_index, bool)
            or not isinstance(self.call_index, int)
            or self.call_index < 0
        ):
            raise ValueError("call_index must be an int >= 0")
        if self.operation is InterventionOperation.DIRECTION:
            if self.on_input or self.units is not None:
                raise ValueError("direction interventions act on whole module-output leaves")
            if self.direction is None or self.direction_axis is None:
                raise ValueError("a direction intervention needs a direction and an axis")
        elif self.direction is not None or self.direction_axis is not None:
            raise ValueError("only direction interventions take a direction")


def _module_site(site: str) -> str:
    if not isinstance(site, str) or not site:
        raise ValueError("site must be a non-root module path; for inputs use zero_input()")
    return site


def zero(
    site: str,
    *,
    output_path: str = "",
    call_index: int = 0,
    units: tuple[int, ...] | list[int] | None = None,
    retain: bool = False,
    unit_axes: tuple[int, ...] | None = None,
) -> Intervention:
    """Zero ablation: replace the leaf (or its ``units``; ``retain``: all other units) with
    zeros (not a neutral baseline)."""
    return Intervention(
        _module_site(site),
        InterventionOperation.ZERO,
        output_path,
        call_index,
        units=None if units is None else tuple(units),
        retain=retain,
        unit_axes=unit_axes,
    )


def constant(
    site: str,
    value: float | torch.Tensor,
    *,
    output_path: str = "",
    call_index: int = 0,
    units: tuple[int, ...] | list[int] | None = None,
    retain: bool = False,
    unit_axes: tuple[int, ...] | None = None,
) -> Intervention:
    """Replace the leaf (or its ``units``) with a constant: a finite scalar (filled into
    the leaf's shape) or a tensor of exactly the leaf's shape and dtype (no
    broadcasting); with ``units``, only those positions take the constant's values."""
    return _constant(_module_site(site), value, output_path, call_index, units, retain, unit_axes)


def _constant(
    site: str,
    value: float | torch.Tensor,
    output_path: str,
    call_index: int,
    units: tuple[int, ...] | list[int] | None,
    retain: bool,
    unit_axes: tuple[int, ...] | None = None,
) -> Intervention:
    unit_tuple = None if units is None else tuple(units)
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError("a constant tensor must be finite")
        return Intervention(
            site,
            InterventionOperation.CONSTANT,
            output_path,
            call_index,
            tensor=value.detach().to("cpu").clone(memory_format=torch.contiguous_format),
            units=unit_tuple,
            retain=retain,
            unit_axes=unit_axes,
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("a scalar constant must be a finite real number")
    return Intervention(
        site,
        InterventionOperation.CONSTANT,
        output_path,
        call_index,
        constant=float(value),
        units=unit_tuple,
        retain=retain,
        unit_axes=unit_axes,
    )


def _input_path(index: int) -> str:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("index must be an int >= 0")
    return f"args[{index}]"


def zero_input(
    index: int = 0,
    *,
    units: tuple[int, ...] | list[int] | None = None,
    retain: bool = False,
    unit_axes: tuple[int, ...] | None = None,
) -> Intervention:
    """Replace positional model input ``index`` (or its ``units``: along the last
    dimension, or over the declared ``unit_axes`` (ADR-034); ``retain``: all other
    units) with zeros: an intervention on the input."""
    return Intervention(
        "",
        InterventionOperation.ZERO,
        _input_path(index),
        units=None if units is None else tuple(units),
        retain=retain,
        unit_axes=unit_axes,
    )


def constant_input(
    value: float | torch.Tensor,
    index: int = 0,
    *,
    units: tuple[int, ...] | list[int] | None = None,
    retain: bool = False,
    unit_axes: tuple[int, ...] | None = None,
) -> Intervention:
    """Replace positional model input ``index`` (or its ``units``) with a constant scalar
    or an exact-shape tensor (e.g. caller-computed per-unit means)."""
    return _constant("", value, _input_path(index), 0, units, retain, unit_axes)


def patch(
    site: str,
    *source_inputs: Any,
    output_path: str = "",
    call_index: int = 0,
    source_kwargs: dict[str, Any] | None = None,
    units: tuple[int, ...] | list[int] | None = None,
    retain: bool = False,
    unit_axes: tuple[int, ...] | None = None,
) -> Intervention:
    """Activation patching: replace the leaf with the activation the same site/leaf/call
    produced on ``source_inputs``. The source execution runs as its own pass in the same
    experiment, so the patched value is a provenance-bearing, retained MEASURED record."""
    if not source_inputs:
        raise ValueError("patch() needs the source inputs whose activation is patched in")
    return Intervention(
        _module_site(site),
        InterventionOperation.PATCH,
        output_path,
        call_index,
        source_inputs=tuple(source_inputs),
        source_kwargs=dict(source_kwargs or {}),
        units=None if units is None else tuple(units),
        retain=retain,
        unit_axes=unit_axes,
    )


def direction(
    site: str,
    vector: torch.Tensor,
    *,
    axis: int,
    reference: torch.Tensor | None = None,
    retain: bool = False,
    output_path: str = "",
    call_index: int = 0,
) -> Intervention:
    """Intervene along one direction v of ``axis`` of a module-output leaf (ADR-040).

    ``retain=False`` (removal): the coordinate along v̂ = v/‖v‖ is replaced by the
    reference's, x' = x - <x - r, v̂> v̂. ``retain=True`` (retention): only that
    coordinate is kept and everything else comes from the reference,
    x' = r + <x - r, v̂> v̂. ``reference`` is a tensor of the leaf's exact shape
    (e.g. a training-mean activation) or ``None`` for zeros; it is part of the
    intervention's identity. Applied at every other position of the leaf."""
    if not isinstance(vector, torch.Tensor) or vector.dim() != 1 or not vector.is_floating_point():
        raise ValueError("a direction is a 1-D floating-point tensor")
    if not bool(torch.isfinite(vector).all()) or float(vector.norm()) == 0.0:
        raise ValueError("a direction must be finite and non-zero")
    if isinstance(axis, bool) or not isinstance(axis, int) or axis < 1:
        raise ValueError("axis must be a non-batch axis (int >= 1)")
    ref = None
    if reference is not None:
        if not isinstance(reference, torch.Tensor) or (
            reference.is_floating_point() and not bool(torch.isfinite(reference).all())
        ):
            raise ValueError("the reference must be a finite tensor")
        ref = reference.detach().to("cpu").clone(memory_format=torch.contiguous_format)
    return Intervention(
        _module_site(site),
        InterventionOperation.DIRECTION,
        output_path,
        call_index,
        tensor=ref,
        retain=retain,
        direction=vector.detach().to("cpu").clone(memory_format=torch.contiguous_format),
        direction_axis=axis,
    )
