"""Attribution records (Phase 3; ADR-030).

Pure data. The runtime lives in :mod:`beyondnn.attribution`.

An attribution answers: *under method A (configuration C, baseline B), for scalar
target T on input X, what score was assigned to the elements of tensor S?* It does
not answer what caused the output. Its status is ATTRIBUTED by kind, never
supplied, never MEASURED and never INTERVENTIONAL (ADR-030).

* :class:`AttributionMethodSpec`: every scientifically relevant choice of the
  method (name, implementation and its version, parameters). Presentation-only
  settings are not part of it.
* :class:`AttributionBaseline`: the reference point of path methods (IG).
* :class:`AttributionRecord`: the raw attribution tensor (retained, same shape as
  the attributed tensor), the target, the attributed site/call/pass, the input
  sample, and method diagnostics.
* :class:`AttributionReduction`: an explicit reduction of one attribution over
  named dimensions (e.g. embedding dimensions to a per-token score). Nothing is
  reduced implicitly.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from ._canonical import EMPTY_JSON, JsonMap
from ._types import Value, require
from .base import BaseRecord, record_kind
from .interventions import MetricSpec
from .status import EvidenceStatus
from .values import Site, SiteIO, TensorRef

__all__ = [
    "ATTRIBUTION_METHODS",
    "IG_RULES",
    "REDUCTIONS",
    "AttributionBaseline",
    "AttributionMethodSpec",
    "AttributionRecord",
    "AttributionReduction",
    "BaselineKind",
]

_TOKEN_RE = re.compile(r"^\S+$")
_INPUT_LEAF_RE = re.compile(r"^args\[(0|[1-9][0-9]*)\]$")

#: The attribution methods Phase 3 implements (natively and through Captum).
ATTRIBUTION_METHODS = frozenset({"gradient", "input_x_gradient", "integrated_gradients"})
#: Integration rules by implementation (their node/weight conventions differ; see
#: docs/PHASE_3_PLAN.md: Captum's ``riemann_trapezoid`` weights sum to (n-1)/n).
IG_RULES = {
    "beyondnn": frozenset({"riemann_left", "riemann_right", "riemann_middle", "trapezoid"}),
    "captum": frozenset(
        {"riemann_left", "riemann_right", "riemann_middle", "riemann_trapezoid", "gausslegendre"}
    ),
}
REDUCTIONS = frozenset({"sum", "abs_sum", "l2"})
_IG_DIAGNOSTICS = frozenset({"baseline_target_value", "attribution_sum", "completeness_delta"})


def _retained(ref: TensorRef, what: str) -> None:
    require(
        ref.content_digest is not None and ref.storage_key is not None,
        f"{what} must be retained (content_digest and storage_key)",
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class AttributionMethodSpec(Value):
    """Identity and configuration of an attribution method.

    ``implementation`` is ``beyondnn`` (native reference implementation, version =
    BeyondNN's method version) or ``captum`` (version = the installed Captum
    version). Integrated Gradients requires ``n_steps`` (int >= 1) and a ``rule``
    valid for its implementation; the other methods take no integration params.
    """

    name: str
    implementation: str
    implementation_version: str
    params: JsonMap = EMPTY_JSON

    def _validate(self) -> None:
        require(self.name in ATTRIBUTION_METHODS, f"unknown attribution method {self.name!r}")
        require(self.implementation in IG_RULES, f"unknown implementation {self.implementation!r}")
        require(
            bool(_TOKEN_RE.match(self.implementation_version)),
            "implementation_version must be a non-empty token",
        )
        if self.name == "integrated_gradients":
            n = self.params.get("n_steps")
            require(
                isinstance(n, int) and not isinstance(n, bool) and n >= 1,
                "integrated_gradients requires params.n_steps >= 1",
            )
            rule = self.params.get("rule")
            require(
                rule in IG_RULES[self.implementation],
                f"integrated_gradients rule {rule!r} is not a {self.implementation} rule",
            )
            if rule == "trapezoid":
                require(isinstance(n, int) and n >= 2, "the trapezoid rule needs n_steps >= 2")
        else:
            require(
                "n_steps" not in self.params and "rule" not in self.params,
                f"{self.name} has no integration parameters",
            )


class BaselineKind(Enum):
    """ZERO / TENSOR: in the attributed tensor's space. INPUT_TENSOR (layer
    attribution only): a model-input baseline; the layer-space baseline is the
    layer's activation on that input (Captum LayerIntegratedGradients convention)."""

    ZERO = "zero"
    TENSOR = "tensor"
    INPUT_TENSOR = "input_tensor"


@dataclass(frozen=True, slots=True, kw_only=True)
class AttributionBaseline(Value):
    """A path method's baseline. Zero is a choice, not a neutral value.

    ``input_path`` (INPUT_TENSOR only) names the positional model input that the
    baseline tensor replaces to compute the layer-space baseline.
    """

    kind: BaselineKind
    value: TensorRef | None = None
    input_path: str | None = None

    def _validate(self) -> None:
        if self.kind is BaselineKind.ZERO:
            require(self.value is None, "a ZERO baseline has no tensor")
        else:
            require(self.value is not None, f"a {self.kind.value} baseline needs its tensor")
            assert self.value is not None
            _retained(self.value, "baseline tensors")
        if self.kind is BaselineKind.INPUT_TENSOR:
            require(
                self.input_path is not None and bool(_INPUT_LEAF_RE.match(self.input_path)),
                "an INPUT_TENSOR baseline names the positional input it replaces ('args[i]')",
            )
        else:
            require(self.input_path is None, "only INPUT_TENSOR baselines name an input")


@record_kind("attribution")
@dataclass(frozen=True, slots=True, kw_only=True)
class AttributionRecord(BaseRecord):
    """A raw attribution tensor for one scalar target, on one input (ATTRIBUTED).

    ``site`` is either a root input leaf (``module=""``, ``io=INPUT``,
    ``output_path="args[i]"``) or one leaf of a module OUTPUT; ``call_index`` is the
    module call within the reference pass ``pass_index`` (0 for inputs).
    ``sample_id`` identifies the input (not an estimand: attribution is not causal
    evidence). ``target_value`` is the target at the input. ``derived_from`` holds the
    reference pass's ``OutputRecord`` and the attributed ``InputRecord`` (input
    site) or ``ActivationRecord`` (module site).

    Integrated Gradients requires a baseline and exactly the diagnostics
    ``baseline_target_value``, ``attribution_sum`` and
    ``completeness_delta = attribution_sum - (target_value - baseline_target_value)``.
    Other methods have no baseline and no diagnostics.
    """

    STATUS: ClassVar[EvidenceStatus | None] = EvidenceStatus.ATTRIBUTED

    method: AttributionMethodSpec
    target: MetricSpec
    site: Site
    call_index: int = 0
    pass_index: int
    sample_id: str
    baseline: AttributionBaseline | None = None
    value: TensorRef
    target_value: float
    diagnostics: JsonMap = EMPTY_JSON

    def _validate(self) -> None:
        require(self.target.builtin, "Phase 3 attribution targets must be built-in metrics")
        require(self.call_index >= 0 and self.pass_index >= 0, "indices must be >= 0")
        require(bool(_TOKEN_RE.match(self.sample_id)), "sample_id must be a non-empty token")
        require(math.isfinite(self.target_value), "target_value must be finite")
        _retained(self.value, "attribution values")
        if self.site.module == "":
            require(
                self.site.io is SiteIO.INPUT and bool(_INPUT_LEAF_RE.match(self.site.output_path)),
                "a root attribution site is a positional input leaf (io=INPUT, 'args[i]')",
            )
            require(self.call_index == 0, "input attributions have call_index 0")
            require(
                self.baseline is None or self.baseline.kind is not BaselineKind.INPUT_TENSOR,
                "INPUT_TENSOR baselines are for layer attribution; use TENSOR for inputs",
            )
            attributed_kind = "input"
        else:
            require(self.site.io is SiteIO.OUTPUT, "module attribution is to module OUTPUTs only")
            attributed_kind = "activation"
        kinds = sorted(r.kind for r in self.derived_from)
        require(
            kinds == sorted(["output", attributed_kind]),
            f"an attribution derives from exactly the reference output and the attributed "
            f"{attributed_kind} record",
        )
        if self.method.name == "integrated_gradients":
            require(self.baseline is not None, "integrated_gradients requires a baseline")
            require(
                set(self.diagnostics) == _IG_DIAGNOSTICS,
                f"integrated_gradients diagnostics must be exactly {sorted(_IG_DIAGNOSTICS)}",
            )
            values: dict[str, float] = {}
            for key in _IG_DIAGNOSTICS:
                v = self.diagnostics[key]
                require(
                    isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v),
                    "diagnostics must be finite numbers",
                )
                assert isinstance(v, (int, float))
                values[key] = float(v)
            expected = float(values["attribution_sum"]) - (
                self.target_value - float(values["baseline_target_value"])
            )
            require(
                math.isclose(
                    float(values["completeness_delta"]), expected, rel_tol=1e-9, abs_tol=1e-9
                ),
                "completeness_delta must equal attribution_sum - (target - baseline target)",
            )
        else:
            require(self.baseline is None, f"{self.method.name} has no baseline")
            require(len(self.diagnostics) == 0, f"{self.method.name} has no diagnostics")


@record_kind("attribution_reduction")
@dataclass(frozen=True, slots=True, kw_only=True)
class AttributionReduction(BaseRecord):
    """An explicit reduction of one attribution tensor (ATTRIBUTED).

    ``reduction`` is ``sum``, ``abs_sum`` or ``l2`` over ``dims`` (sorted,
    non-negative, non-empty). ``scalar`` is set iff every dimension was reduced
    (``value.shape == ()``). ``derived_from`` is exactly the source attribution; the
    container checks that ``value.shape`` is its shape without ``dims``.
    """

    STATUS: ClassVar[EvidenceStatus | None] = EvidenceStatus.ATTRIBUTED

    reduction: str
    dims: tuple[int, ...]
    value: TensorRef
    scalar: float | None = None

    def _validate(self) -> None:
        require(self.reduction in REDUCTIONS, f"unknown reduction {self.reduction!r}")
        require(len(self.dims) > 0, "a reduction names at least one dimension")
        require(all(d >= 0 for d in self.dims), "dims must be non-negative (normalised)")
        require(list(self.dims) == sorted(set(self.dims)), "dims must be sorted and unique")
        _retained(self.value, "reduction values")
        require(
            (self.value.shape == ()) == (self.scalar is not None),
            "scalar is set iff every dimension was reduced",
        )
        if self.scalar is not None:
            require(math.isfinite(self.scalar), "scalar must be finite")
        require(
            len(self.derived_from) == 1 and self.derived_from[0].kind == "attribution",
            "a reduction derives from exactly one attribution",
        )
