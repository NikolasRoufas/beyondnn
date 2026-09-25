"""Scalar outcome metrics for intervention effects (Phase 2).

Built-ins are BeyondNN's own pure functions of the model output; their identity and
configuration are recorded as a :class:`~beyondnn.schema.MetricSpec`. Caller
metrics (:func:`custom`) are identified by name plus a caller-DECLARED
implementation revision and config (:class:`~beyondnn.schema.MetricDeclaration`);
their code is never serialised, inspected, or verified, and effects using them
carry ``CUSTOM_METRIC_UNVERIFIED``.

Built-ins are also differentiable (:meth:`Metric.tensor`), so the same metric
language defines Phase-3 attribution targets. A target is exactly one element;
anything else is refused, never summed. Caller metrics have no differentiable form
and cannot be attribution targets.

``path`` selects a tensor leaf of the model's return value (``""`` = the value
itself; ``"[0]"``, ``'["logits"]'``, ...: the same grammar as trace leaf paths).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from beyondnn.core.tensors import walk
from beyondnn.schema import JsonMap, MetricDeclaration, MetricSpec

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
    tensor_function: Callable[[Any], torch.Tensor] | None = None

    def tensor(self, output: Any) -> torch.Tensor:
        """The metric as a differentiable 0-d float64 tensor (built-ins only)."""
        if self.tensor_function is None:
            raise MetricError(
                f"metric {self.spec.name} has no differentiable form; caller metrics cannot "
                "be attribution targets in Phase 3"
            )
        try:
            value = self.tensor_function(output)
        except MetricError:
            raise
        except Exception as exc:
            raise MetricError(f"metric {self.spec.name} failed: {exc}") from exc
        if value.numel() != 1:
            raise MetricError(f"metric {self.spec.name} is not a single element; not summed")
        return value.reshape(())

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


def _element_t(t: torch.Tensor, index: tuple[int, ...]) -> torch.Tensor:
    if not t.is_floating_point():
        raise MetricError(f"cannot differentiate a {t.dtype} output")
    selected = t[index]
    if selected.numel() != 1:
        raise MetricError(
            f"index {index} does not select a single element of shape {tuple(t.shape)}"
        )
    return selected.double().reshape(())


def select(index: Sequence[int], *, path: str = "") -> Metric:
    """The element ``output[path][index]`` (e.g. one logit)."""
    idx = _index(index)
    spec = MetricSpec(
        name="select", builtin=True, params=JsonMap({"path": path, "index": list(idx)})
    )
    return Metric(
        spec,
        lambda out: _element(get_leaf(out, path), idx),
        lambda out: _element_t(get_leaf(out, path), idx),
    )


def difference(index_a: Sequence[int], index_b: Sequence[int], *, path: str = "") -> Metric:
    """``output[path][index_a] - output[path][index_b]`` (e.g. a logit difference)."""
    a, b = _index(index_a), _index(index_b)
    spec = MetricSpec(
        name="difference", builtin=True, params=JsonMap({"path": path, "a": list(a), "b": list(b)})
    )

    def fn(out: Any) -> float:
        leaf = get_leaf(out, path)
        return _element(leaf, a) - _element(leaf, b)

    def fn_t(out: Any) -> torch.Tensor:
        leaf = get_leaf(out, path)
        return _element_t(leaf, a) - _element_t(leaf, b)

    return Metric(spec, fn, fn_t)


def mean(*, path: str = "") -> Metric:
    """The mean of all elements of ``output[path]`` (computed in float64)."""
    spec = MetricSpec(name="mean", builtin=True, params=JsonMap({"path": path}))

    def fn_t(out: Any) -> torch.Tensor:
        leaf = get_leaf(out, path)
        if not leaf.is_floating_point():
            raise MetricError(f"cannot differentiate a {leaf.dtype} output")
        return leaf.double().mean()

    return Metric(
        spec, lambda out: float(get_leaf(out, path).detach().double().mean().item()), fn_t
    )


def custom(
    name: str,
    function: Callable[[Any], float],
    *,
    implementation_revision: str,
    config: Mapping[str, Any] | None = None,
) -> Metric:
    """A caller-supplied metric, recorded as ``custom:<name>`` with the DECLARED
    ``implementation_revision`` (required, e.g. ``"git:abc123"`` or ``"v2"``) and
    ``config``. The function is never serialised, inspected, or hashed: a declaration
    is the caller's statement, so change the revision whenever the function changes."""
    if not _NAME.match(name):
        raise ValueError(f"invalid metric name {name!r}")
    if not callable(function):
        raise TypeError("function must be callable")
    declaration = MetricDeclaration(
        implementation_revision=implementation_revision, config=JsonMap(dict(config or {}))
    )
    return Metric(
        MetricSpec(name=f"custom:{name}", builtin=False, declaration=declaration), function
    )
