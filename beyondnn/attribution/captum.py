"""Captum adapters (optional; ``pip install beyondnn[captum]``). Verified: Captum 0.9.x.

BeyondNN does not reimplement Captum. These adapters run Captum's
``IntegratedGradients``, ``Saliency``, ``InputXGradient`` and
``LayerIntegratedGradients`` and normalise the result into BeyondNN records with
the Captum version and every setting in the :class:`AttributionMethodSpec`.

Fixed settings, and why (see docs/PHASE_3_PLAN.md):

* ``internal_batch_size=1`` and a leading input dimension of exactly 1: Captum
  treats dim 0 as a batch and evaluates interpolation points as batch rows; with
  one row per forward the model sees exactly the tensors the native implementation
  gives it, whatever the model does across its batch dimension.
* ``Saliency(abs=False)``: Captum's default ``abs=True`` returns ``|gradient|``,
  which is a different quantity.
* Captum sets ``requires_grad`` on the tensors it is given, so it only ever
  receives detached clones.
* ``LayerIntegratedGradients`` hooks every call of its layer and takes its
  baseline in model-input space: it is refused when the layer runs more than once
  in a forward (the call could not be targeted), and only accepts
  :func:`~beyondnn.attribution.input_baseline`.

Rule names are Captum's own. ``riemann_trapezoid`` is not BeyondNN's ``trapezoid``:
Captum's step weights sum to (n-1)/n (documented, not "fixed").
"""

from __future__ import annotations

from typing import Any

import torch

from beyondnn.schema import AttributionMethodSpec, BaselineKind, JsonMap

from .native import AttributionError, InputSpace, LayerSpace, Outcome, RepeatedCallError, Space
from .spec import Baseline, Method, _check_steps

__all__ = [
    "SUPPORTED_CAPTUM",
    "CaptumUnavailableError",
    "input_x_gradient",
    "integrated_gradients",
    "saliency",
]

#: Captum major.minor versions this adapter was verified against.
SUPPORTED_CAPTUM = ("0.9",)


class CaptumUnavailableError(ImportError):
    """Captum is not installed, or its version is not one the adapter was verified with."""


def _captum_version() -> str:
    try:
        import captum
    except ImportError as exc:
        raise CaptumUnavailableError(
            "Captum is not installed; install the optional extra: pip install 'beyondnn[captum]'"
        ) from exc
    version = str(captum.__version__)
    if ".".join(version.split(".")[:2]) not in SUPPORTED_CAPTUM:
        raise CaptumUnavailableError(
            f"Captum {version} is not a verified version ({', '.join(SUPPORTED_CAPTUM)}.x); "
            "its conventions may differ, so the adapter refuses it"
        )
    return version


def _method(
    name: str, captum_class: str, params: dict[str, Any], baseline: Baseline | None = None
) -> Method:
    spec = AttributionMethodSpec(
        name=name,
        implementation="captum",
        implementation_version=_captum_version(),
        params=JsonMap({"captum_class": captum_class, **params}),
    )
    return Method(spec, baseline, captum_class)


def saliency() -> Method:
    """Captum ``Saliency`` with ``abs=False``: the raw gradient."""
    return _method("gradient", "Saliency", {"abs": False})


def input_x_gradient() -> Method:
    """Captum ``InputXGradient``."""
    return _method("input_x_gradient", "InputXGradient", {})


def integrated_gradients(
    *, baseline: Baseline, n_steps: int = 64, rule: str = "riemann_middle"
) -> Method:
    """Captum ``IntegratedGradients`` (input attribution) or ``LayerIntegratedGradients``
    (layer attribution, ``input_baseline`` only), ``multiply_by_inputs=True``,
    ``internal_batch_size=1``. ``rule`` is Captum's ``method`` argument."""
    if not isinstance(baseline, Baseline):
        raise TypeError("baseline must come from zero_baseline(), baseline() or input_baseline()")
    _check_steps(n_steps, rule, "captum")
    return _method(
        "integrated_gradients",
        "IntegratedGradients",
        {"n_steps": n_steps, "rule": rule, "internal_batch_size": 1, "multiply_by_inputs": True},
        baseline,
    )


def _batch_one(x: torch.Tensor, what: str) -> None:
    if x.dim() == 0 or x.shape[0] != 1:
        raise AttributionError(
            f"the Captum adapter needs {what} with leading (batch) dimension 1, got shape "
            f"{tuple(x.shape)}; Captum treats dim 0 as a batch"
        )


def run(method: Method, space: Space) -> Outcome:
    """Run a Captum method on ``space``; the result is checked for shape and dtype."""
    _captum_version()
    import captum.attr as ca

    if isinstance(space, LayerSpace):
        return _run_layer(method, space)
    assert isinstance(space, InputSpace)
    x = space.x
    _batch_one(x, "the attributed input")

    def forward(z: torch.Tensor) -> torch.Tensor:
        return space.target(z).reshape(1)

    point = x.detach().clone().requires_grad_(True)
    if method.captum_class == "Saliency":
        attribution = ca.Saliency(forward).attribute(point, target=None, abs=False)
        value = space.value(x)
        base_value = None
    elif method.captum_class == "InputXGradient":
        attribution = ca.InputXGradient(forward).attribute(point, target=None)
        value = space.value(x)
        base_value = None
    else:
        assert method.baseline is not None
        x_base = space.baseline_point(method.baseline)
        attribution = ca.IntegratedGradients(forward, multiply_by_inputs=True).attribute(
            point,
            baselines=x_base.detach().clone(),
            target=None,
            n_steps=method.n_steps,
            method=method.rule,
            internal_batch_size=1,
        )
        value, base_value = space.value(x), space.value(x_base)
    return Outcome(_checked(attribution, x), value, base_value)


def _run_layer(method: Method, space: LayerSpace) -> Outcome:
    import captum.attr as ca

    if method.captum_class != "IntegratedGradients":
        raise AttributionError("the Captum adapter supports layer attribution for IG only")
    if space.calls != 1:
        raise RepeatedCallError(
            f"{space.at.module!r} runs {space.calls} times per forward; Captum layer methods "
            "hook every call and cannot target one, so the attribution is refused"
        )
    if space.at.output_path != "":
        raise AttributionError("the Captum layer adapter supports a tensor output only")
    baseline = method.baseline
    if baseline is None or baseline.kind is not BaselineKind.INPUT_TENSOR:
        raise AttributionError(
            "Captum LayerIntegratedGradients takes its baseline in model-input space: use "
            "input_baseline(); a layer-space baseline would not be the same method"
        )
    x_base = space.baseline_point(baseline)  # validates the input baseline
    assert baseline.tensor is not None
    assert baseline.input_index is not None
    inputs = tuple(t.detach().clone() if isinstance(t, torch.Tensor) else t for t in space.inputs)
    for t in inputs:
        if isinstance(t, torch.Tensor):
            _batch_one(t, "every positional input")
    baselines = list(inputs)
    baselines[baseline.input_index] = baseline.tensor.to(space.inputs[baseline.input_index].device)

    def forward(*args: Any) -> torch.Tensor:
        return space.metric.tensor(space.model(*args, **space.kwargs)).reshape(1)

    lig = ca.LayerIntegratedGradients(forward, space.module, multiply_by_inputs=True)
    attribution = lig.attribute(
        inputs if len(inputs) > 1 else inputs[0],
        baselines=tuple(baselines) if len(baselines) > 1 else baselines[0],
        target=None,
        n_steps=method.n_steps,
        method=method.rule,
        internal_batch_size=1,
    )
    if not isinstance(attribution, torch.Tensor):
        raise AttributionError("Captum returned several layer attributions; refused")
    return Outcome(_checked(attribution, space.x), space.value(space.x), space.value(x_base))


def _checked(attribution: Any, x: torch.Tensor) -> torch.Tensor:
    if not isinstance(attribution, torch.Tensor) or tuple(attribution.shape) != tuple(x.shape):
        raise AttributionError("Captum returned an attribution that does not match the input")
    return attribution.detach().to(x.dtype)
