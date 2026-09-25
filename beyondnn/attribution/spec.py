"""Runtime attribution specifications: methods, baselines, attributed tensors, reductions.

Every scientifically relevant choice is explicit and becomes part of the record:
Integrated Gradients has no default baseline (``zero_baseline()`` must be asked
for), and nothing is reduced unless a :func:`reduce` is declared.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import torch

from beyondnn.provenance.fingerprint import tensor_bytes
from beyondnn.schema import (
    AttributionMethodSpec,
    BaselineKind,
    JsonMap,
    Site,
    SiteIO,
)
from beyondnn.schema.attribution import IG_RULES, REDUCTIONS

__all__ = [
    "NATIVE_VERSION",
    "At",
    "Baseline",
    "Method",
    "Reduce",
    "baseline",
    "gradient",
    "input",
    "input_baseline",
    "input_x_gradient",
    "integrated_gradients",
    "layer",
    "reduce",
    "zero_baseline",
]

#: Version of BeyondNN's native reference implementations.
NATIVE_VERSION = "1"


def tensor_digest(tensor: torch.Tensor) -> str:
    """``sha256:`` digest of a tensor's bytes (the retained-tensor digest)."""
    cpu = tensor.detach().to("cpu").contiguous()
    return "sha256:" + hashlib.sha256(tensor_bytes(cpu)).hexdigest()


# ------------------------------------------------------------------ baselines


@dataclass(frozen=True, slots=True, eq=False)
class Baseline:
    """A path-method baseline; build with :func:`zero_baseline`, :func:`baseline` or
    :func:`input_baseline`."""

    kind: BaselineKind
    tensor: torch.Tensor | None = None
    input_index: int | None = None

    def identity(self) -> dict[str, Any]:
        """What a claim test declares about the baseline (kind, digest, replaced input)."""
        out: dict[str, Any] = {"kind": self.kind.value}
        if self.tensor is not None:
            out["digest"] = tensor_digest(self.tensor)
        if self.input_index is not None:
            out["input_path"] = f"args[{self.input_index}]"
        return out


def _finite_tensor(t: Any, what: str) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{what} must be a torch.Tensor")
    if t.is_floating_point() and not bool(torch.isfinite(t).all()):
        raise ValueError(f"{what} must be finite")
    return t.detach().to("cpu").clone(memory_format=torch.contiguous_format)


def zero_baseline() -> Baseline:
    """All zeros in the attributed tensor's space. A choice, not a neutral value
    (``ATTRIBUTION_BASELINE_ASSUMPTION``)."""
    return Baseline(BaselineKind.ZERO)


def baseline(tensor: torch.Tensor) -> Baseline:
    """An explicit baseline in the attributed tensor's space; its shape and dtype must
    match the attributed tensor exactly (no broadcasting)."""
    return Baseline(BaselineKind.TENSOR, _finite_tensor(tensor, "baseline"))


def input_baseline(tensor: torch.Tensor, *, input_index: int = 0) -> Baseline:
    """Layer attribution only: a baseline for positional model input ``input_index``;
    the layer-space baseline is the layer's activation on it (Captum's
    ``LayerIntegratedGradients`` convention). May be an integer tensor (e.g. pad ids)."""
    if isinstance(input_index, bool) or not isinstance(input_index, int) or input_index < 0:
        raise ValueError("input_index must be an int >= 0")
    return Baseline(BaselineKind.INPUT_TENSOR, _finite_tensor(tensor, "baseline"), input_index)


# ------------------------------------------------------------------ methods


@dataclass(frozen=True, slots=True, eq=False)
class Method:
    """An attribution method with its full configuration. Use the constructors."""

    spec: AttributionMethodSpec
    baseline: Baseline | None = None
    captum_class: str | None = field(default=None)

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def is_captum(self) -> bool:
        return self.spec.implementation == "captum"

    @property
    def n_steps(self) -> int:
        n = self.spec.params.get("n_steps")
        assert isinstance(n, int)
        return n

    @property
    def rule(self) -> str:
        rule = self.spec.params.get("rule")
        assert isinstance(rule, str)
        return rule


def gradient() -> Method:
    """Raw gradient ``d target / d x`` (sensitivity; not credit, not causation)."""
    return Method(
        AttributionMethodSpec(
            name="gradient", implementation="beyondnn", implementation_version=NATIVE_VERSION
        )
    )


def input_x_gradient() -> Method:
    """``x * d target / d x``: its own method, not a more faithful gradient."""
    return Method(
        AttributionMethodSpec(
            name="input_x_gradient",
            implementation="beyondnn",
            implementation_version=NATIVE_VERSION,
        )
    )


def _check_steps(n_steps: int, rule: str, implementation: str) -> None:
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError("n_steps must be an int >= 1")
    if rule not in IG_RULES[implementation]:
        raise ValueError(
            f"rule {rule!r} is not a {implementation} rule; choose from "
            f"{sorted(IG_RULES[implementation])}"
        )


def integrated_gradients(
    *, baseline: Baseline, n_steps: int = 64, rule: str = "riemann_middle"
) -> Method:
    """Integrated Gradients, approximated with ``n_steps`` nodes of ``rule``
    (``riemann_left``/``riemann_right``/``riemann_middle``/``trapezoid``). The
    baseline is required; use :func:`zero_baseline` explicitly if that is the choice."""
    if not isinstance(baseline, Baseline):
        raise TypeError("baseline must come from zero_baseline(), baseline() or input_baseline()")
    _check_steps(n_steps, rule, "beyondnn")
    spec = AttributionMethodSpec(
        name="integrated_gradients",
        implementation="beyondnn",
        implementation_version=NATIVE_VERSION,
        params=JsonMap({"n_steps": n_steps, "rule": rule}),
    )
    return Method(spec, baseline)


# ------------------------------------------------------------------ attributed tensor


@dataclass(frozen=True, slots=True)
class At:
    """What is attributed: positional model input ``index``, or one leaf
    (``output_path``) of call ``call_index`` of the module at ``module``."""

    module: str = ""
    index: int = 0
    call_index: int = 0
    output_path: str = ""

    @property
    def is_layer(self) -> bool:
        return self.module != ""

    def site(self) -> Site:
        if self.is_layer:
            return Site(module=self.module, io=SiteIO.OUTPUT, output_path=self.output_path)
        return Site(module="", io=SiteIO.INPUT, output_path=f"args[{self.index}]")


def input(index: int = 0) -> At:
    """Attribute to positional model input ``index`` (a floating-point tensor)."""
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("index must be an int >= 0")
    return At(index=index)


def layer(module: str, *, call_index: int = 0, output_path: str = "") -> At:
    """Attribute to one tensor leaf of call ``call_index`` of a module's OUTPUT. The
    path must name exactly one module (no wildcards, no aliases)."""
    if not isinstance(module, str) or not module:
        raise ValueError("layer() needs a non-empty module path")
    if "*" in module:
        raise ValueError("layer() takes one exact module path, not a pattern")
    if isinstance(call_index, bool) or not isinstance(call_index, int) or call_index < 0:
        raise ValueError("call_index must be an int >= 0")
    return At(module=module, call_index=call_index, output_path=output_path)


# ------------------------------------------------------------------ reductions


@dataclass(frozen=True, slots=True)
class Reduce:
    """An explicit reduction of the attribution tensor (see :func:`reduce`)."""

    reduction: str
    dims: tuple[int, ...]


def reduce(reduction: str, dims: tuple[int, ...] | list[int]) -> Reduce:
    """Declare ``sum``/``abs_sum``/``l2`` over ``dims`` (negative dims allowed; they
    are recorded normalised). E.g. ``reduce("sum", (-1,))`` turns a (1, T, d)
    embedding attribution into a per-position score."""
    if reduction not in REDUCTIONS:
        raise ValueError(f"reduction must be one of {sorted(REDUCTIONS)}")
    out = tuple(dims)
    if not out or not all(isinstance(d, int) and not isinstance(d, bool) for d in out):
        raise ValueError("dims must be a non-empty sequence of ints")
    return Reduce(reduction, out)
