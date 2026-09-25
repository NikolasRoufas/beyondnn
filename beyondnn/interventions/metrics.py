"""Scalar outcome metrics for intervention effects (Phase 2).

Built-ins are BeyondNN's own pure functions of the model output; their identity and
configuration are recorded as a :class:`~beyondnn.schema.MetricSpec`. Caller
metrics (:func:`custom`) are recorded by name only; their code is never
serialised or verified, and effects using them carry ``CUSTOM_METRIC_UNVERIFIED``.

``path`` selects a tensor leaf of the model's return value (``""`` = the value
itself; ``"[0]"``, ``'["logits"]'``, ...: the same grammar as trace leaf paths).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from beyondnn.core.tensors import walk
from beyondnn.schema import JsonMap, MetricSpec

__all__ = ["Metric", "custom", "difference", "get_leaf", "mean", "select"]

_NAME = re.compile(r"^[A-Za-z0-9_.\-]+$")


class MetricError(ValueError):
    """A metric could not be computed from the model output."""


def get_leaf(value: Any, path: str) -> torch.Tensor:
    """The tensor leaf of ``value`` at ``path`` (trace leaf-path grammar)."""
    for leaf in walk(value)[0]:
        if leaf.path == path:
            return leaf.tensor
    raise MetricError(f"no tensor leaf at path {path!r} in the model output")


@dataclass(frozen=True, slots=True, eq=False)
class Metric:
    """A scalar metric: ``metric(output) -> float``. Use the constructors below."""

    spec: MetricSpec
    function: Callable[[Any], float]

    def __call__(self, output: Any) -> float:
        try:
            with torch.no_grad():
                value = self.function(output)
        except MetricError:
            raise
        except Exception as exc:
            raise MetricError(f"metric {self.spec.name} failed: {exc}") from exc
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise MetricError(f"metric {self.spec.name} returned a non-scalar tensor")
            value = value.item()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MetricError(f"metric {self.spec.name} must return a real scalar")
        return float(value)


def _index(index: Sequence[int]) -> tuple[int, ...]:
    out = tuple(index)
    if not all(isinstance(i, int) and not isinstance(i, bool) for i in out):
        raise ValueError("index must be a sequence of ints")
    return out


def _element(t: torch.Tensor, index: tuple[int, ...]) -> float:
    selected = t.detach()[index]
    if selected.numel() != 1:
        raise MetricError(
            f"index {index} does not select a single element of shape {tuple(t.shape)}"
        )
    return float(selected.double().item())


def select(index: Sequence[int], *, path: str = "") -> Metric:
    """The element ``output[path][index]`` (e.g. one logit)."""
    idx = _index(index)
    spec = MetricSpec(
        name="select", builtin=True, params=JsonMap({"path": path, "index": list(idx)})
    )
    return Metric(spec, lambda out: _element(get_leaf(out, path), idx))


def difference(index_a: Sequence[int], index_b: Sequence[int], *, path: str = "") -> Metric:
    """``output[path][index_a] - output[path][index_b]`` (e.g. a logit difference)."""
    a, b = _index(index_a), _index(index_b)
    spec = MetricSpec(
        name="difference", builtin=True, params=JsonMap({"path": path, "a": list(a), "b": list(b)})
    )

    def fn(out: Any) -> float:
        leaf = get_leaf(out, path)
        return _element(leaf, a) - _element(leaf, b)

    return Metric(spec, fn)


def mean(*, path: str = "") -> Metric:
    """The mean of all elements of ``output[path]`` (computed in float64)."""
    spec = MetricSpec(name="mean", builtin=True, params=JsonMap({"path": path}))
    return Metric(spec, lambda out: float(get_leaf(out, path).detach().double().mean().item()))


def custom(name: str, function: Callable[[Any], float]) -> Metric:
    """A caller-supplied metric, recorded as ``custom:<name>``; never serialised or verified."""
    if not _NAME.match(name):
        raise ValueError(f"invalid metric name {name!r}")
    if not callable(function):
        raise TypeError("function must be callable")
    return Metric(MetricSpec(name=f"custom:{name}", builtin=False), function)
