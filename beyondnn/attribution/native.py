"""Attribution spaces and BeyondNN's native reference methods (Phase 3; ADR-030).

A *space* is the attributed tensor and a way to evaluate the scalar target (and its
gradient) at any replacement value of it:

* :class:`InputSpace`: positional model input ``i`` (floating point only);
* :class:`LayerSpace`: one tensor leaf of call ``k`` of one module's output. A
  temporary forward hook replaces exactly that call's leaf with the evaluation
  point, so the gradient is with respect to that call and no other.

Every evaluation works on a detached clone (``requires_grad`` set on the clone
only) and uses :func:`torch.autograd.grad`, so no caller tensor, parameter
``.grad``, or model state is touched. Native methods are small references for
validation; mature implementations come from Captum (:mod:`.captum`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from beyondnn.core.trace import TraceError
from beyondnn.interventions.metrics import Metric, get_leaf
from beyondnn.interventions.runner import _replace_leaf
from beyondnn.schema import BaselineKind

from .spec import At, Baseline, Method

__all__ = ["AttributionError", "DiscreteInputError", "InputSpace", "LayerSpace", "Space", "run"]


class AttributionError(TraceError):
    """An attribution could not be computed or recorded faithfully."""


class DiscreteInputError(AttributionError):
    """Gradients with respect to integer inputs (e.g. token ids) are undefined."""


class RepeatedCallError(AttributionError):
    """The attributed module call could not be targeted unambiguously."""


def _clone_leaf(value: torch.Tensor) -> torch.Tensor:
    return value.detach().clone().requires_grad_(True)


def _grad(value: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """d value / d point; zeros where the target does not depend on the point."""
    if not value.requires_grad:
        return torch.zeros_like(point)
    (grad,) = torch.autograd.grad(value, point, allow_unused=True)
    return torch.zeros_like(point) if grad is None else grad.detach()


class Space:
    """The attributed tensor ``x`` and the target as a function of it."""

    x: torch.Tensor

    def __init__(
        self,
        model: nn.Module,
        inputs: tuple[Any, ...],
        kwargs: dict[str, Any],
        metric: Metric,
        at: At,
    ) -> None:
        self.model = model
        self.inputs = inputs
        self.kwargs = kwargs
        self.metric = metric
        self.at = at

    def value_and_grad(self, z: torch.Tensor) -> tuple[float, torch.Tensor]:
        raise NotImplementedError

    def value(self, z: torch.Tensor) -> float:
        raise NotImplementedError

    def baseline_point(self, baseline: Baseline) -> torch.Tensor:
        raise NotImplementedError

    def _exact(self, tensor: torch.Tensor, like: torch.Tensor, what: str) -> torch.Tensor:
        if tuple(tensor.shape) != tuple(like.shape) or tensor.dtype != like.dtype:
            raise AttributionError(
                f"{what} of shape {tuple(tensor.shape)} / {tensor.dtype} does not match the "
                f"attributed tensor {tuple(like.shape)} / {like.dtype}; no broadcasting"
            )
        return tensor.to(like.device)


class InputSpace(Space):
    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        index = self.at.index
        if index >= len(self.inputs):
            raise AttributionError(f"the model was given {len(self.inputs)} positional inputs")
        x = self.inputs[index]
        if not isinstance(x, torch.Tensor):
            raise AttributionError(f"positional input {index} is not a tensor")
        if not x.is_floating_point():
            raise DiscreteInputError(
                f"positional input {index} is {x.dtype}: gradients with respect to discrete "
                "values (e.g. token ids) are undefined. Attribute to a representation instead, "
                "e.g. attribution.layer('token_embedding') (embedding dimensions per position)."
            )
        self.x = x.detach()

    def target(self, z: torch.Tensor) -> torch.Tensor:
        args = list(self.inputs)
        args[self.at.index] = z
        return self.metric.tensor(self.model(*args, **self.kwargs))

    def value_and_grad(self, z: torch.Tensor) -> tuple[float, torch.Tensor]:
        point = _clone_leaf(z)
        with torch.enable_grad():
            value = self.target(point)
            grad = _grad(value, point)
        return float(value.detach()), grad

    def value(self, z: torch.Tensor) -> float:
        with torch.no_grad():
            return float(self.target(z.detach()))

    def baseline_point(self, baseline: Baseline) -> torch.Tensor:
        if baseline.kind is BaselineKind.ZERO:
            return torch.zeros_like(self.x)
        if baseline.kind is BaselineKind.INPUT_TENSOR:
            raise AttributionError("input_baseline() is for layer attribution; use baseline()")
        assert baseline.tensor is not None
        return self._exact(baseline.tensor, self.x, "the baseline")


class LayerSpace(Space):
    """One leaf of call ``call_index`` of ``module``'s output, within one root pass."""

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self.module = self.model.get_submodule(self.at.module)
        self.calls, captured = self._capture(self.inputs)
        self.x = captured

    def _run(self, inputs: tuple[Any, ...], hook: Callable[[Any], Any]) -> tuple[int, Any]:
        """One forward with ``hook(output)`` applied to the target call; returns the
        number of calls of the module and the model output."""
        calls = 0

        def forward_hook(module: nn.Module, args: Any, output: Any) -> Any:
            nonlocal calls
            index = calls
            calls += 1
            return hook(output) if index == self.at.call_index else None

        handle = self.module.register_forward_hook(forward_hook)
        try:
            output = self.model(*inputs, **self.kwargs)
        finally:
            handle.remove()
        return calls, output

    def _leaf(self, output: Any) -> torch.Tensor:
        try:
            return get_leaf(output, self.at.output_path)
        except ValueError as exc:
            raise AttributionError(str(exc)) from None

    def _capture(self, inputs: tuple[Any, ...]) -> tuple[int, torch.Tensor]:
        seen: list[torch.Tensor] = []

        def capture(output: Any) -> None:
            seen.append(self._leaf(output).detach().clone())

        with torch.no_grad():
            calls, _ = self._run(inputs, capture)
        if not seen:
            raise RepeatedCallError(
                f"call {self.at.call_index} of {self.at.module!r} did not run "
                f"(the module ran {calls} time(s) in one forward)"
            )
        return calls, seen[0]

    def _replaced(self, z: torch.Tensor, grad: bool) -> tuple[float, torch.Tensor | None]:
        point = _clone_leaf(z) if grad else z.detach()

        def replace(output: Any) -> Any:
            leaf = self._leaf(output)
            if tuple(leaf.shape) != tuple(point.shape) or leaf.dtype != point.dtype:
                raise AttributionError("the layer output changed shape between passes")
            return _replace_leaf(output, self.at.output_path, point)

        with torch.enable_grad() if grad else torch.no_grad():
            calls, output = self._run(self.inputs, replace)
            if calls != self.calls:
                raise RepeatedCallError(
                    f"{self.at.module!r} ran {calls} times, not {self.calls}; the call "
                    "cannot be targeted consistently"
                )
            value = self.metric.tensor(output)
            g = _grad(value, point) if grad else None
        return float(value.detach()), g

    def value_and_grad(self, z: torch.Tensor) -> tuple[float, torch.Tensor]:
        value, grad = self._replaced(z, grad=True)
        assert grad is not None
        return value, grad

    def value(self, z: torch.Tensor) -> float:
        return self._replaced(z, grad=False)[0]

    def baseline_point(self, baseline: Baseline) -> torch.Tensor:
        if baseline.kind is BaselineKind.ZERO:
            return torch.zeros_like(self.x)
        assert baseline.tensor is not None
        if baseline.kind is BaselineKind.TENSOR:
            return self._exact(baseline.tensor, self.x, "the baseline")
        index = baseline.input_index
        assert index is not None
        if index >= len(self.inputs) or not isinstance(self.inputs[index], torch.Tensor):
            raise AttributionError(
                f"input_baseline replaces positional input {index}, "
                "which is not a tensor input of this call"
            )
        original = self.inputs[index]
        tensor = baseline.tensor
        if tuple(tensor.shape) != tuple(original.shape) or tensor.dtype != original.dtype:
            raise AttributionError(
                f"input baseline of shape {tuple(tensor.shape)} / {tensor.dtype} does not match "
                f"input {index} ({tuple(original.shape)} / {original.dtype}); no broadcasting"
            )
        args = list(self.inputs)
        args[index] = tensor.to(original.device)
        calls, h = self._capture(tuple(args))
        if calls != self.calls:
            raise RepeatedCallError("the layer runs a different number of times on the baseline")
        return h


def make_space(
    model: nn.Module, inputs: tuple[Any, ...], kwargs: dict[str, Any], metric: Metric, at: At
) -> Space:
    cls = LayerSpace if at.is_layer else InputSpace
    return cls(model, inputs, kwargs, metric, at)


def _nodes(rule: str, n: int) -> list[tuple[float, float]]:
    """(alpha, weight) pairs of an integration rule on [0, 1]."""
    if rule == "riemann_left":
        return [(k / n, 1 / n) for k in range(n)]
    if rule == "riemann_right":
        return [((k + 1) / n, 1 / n) for k in range(n)]
    if rule == "riemann_middle":
        return [((k + 0.5) / n, 1 / n) for k in range(n)]
    if rule == "trapezoid":
        w = 1 / (n - 1)
        return [(k / (n - 1), w / 2 if k in (0, n - 1) else w) for k in range(n)]
    raise AttributionError(f"unknown native rule {rule!r}")


class Outcome:
    """What a method computed: the attribution, the target at ``x`` as seen by the
    method's passes, and (IG) the baseline point's target value."""

    def __init__(
        self, attribution: torch.Tensor, value: float, baseline_value: float | None = None
    ) -> None:
        self.attribution = attribution
        self.value = value
        self.baseline_value = baseline_value


def integrate(space: Space, x_base: torch.Tensor, rule: str, n: int) -> torch.Tensor:
    """``(x - x') * sum_k w_k grad F(x' + alpha_k (x - x'))``, accumulated in float64."""
    x = space.x
    delta = x.double() - x_base.double()
    total = torch.zeros_like(delta)
    for alpha, weight in _nodes(rule, n):
        point = (x_base.double() + alpha * delta).to(x.dtype)
        _, grad = space.value_and_grad(point)
        total += weight * grad.double()
    return (delta * total).to(x.dtype)


def run(method: Method, space: Space) -> Outcome:
    """Run a native method on ``space``."""
    x = space.x
    if method.name == "gradient":
        value, grad = space.value_and_grad(x)
        return Outcome(grad, value)
    if method.name == "input_x_gradient":
        value, grad = space.value_and_grad(x)
        return Outcome((x.double() * grad.double()).to(x.dtype), value)
    assert method.baseline is not None
    x_base = space.baseline_point(method.baseline)
    attribution = integrate(space, x_base, method.rule, method.n_steps)
    return Outcome(attribution, space.value(x), space.value(x_base))
