"""Intervention specifications and intervention effects (Phase 2; ADR-028).

Pure data. The runtime lives in :mod:`beyondnn.interventions`.

* :class:`InterventionRecord` specifies *what was done*: which module output
  (``Site`` with ``io=OUTPUT`` and a tensor-leaf ``output_path``), which call of the
  module within the intervened pass, and the operation (ZERO / CONSTANT / PATCH)
  with its exact replacement. Its content-derived id is the ``intervention_id``
  recorded in ``ExecutionContext``. It is a specification, not evidence (no status).
* :class:`CausalEffect` records *what changed*: a scalar metric's value in the
  intervened pass minus its value in the paired baseline pass
  (``effect = intervention_value - baseline_value``, one sign convention for all
  metrics). Its status is derived, never supplied: INTERVENTIONAL for an exact
  comparison on one input (INSTANCE) or an exact summary over exactly the given
  inputs (FINITE_SAMPLE); ESTIMATED_CAUSAL only for a POPULATION estimand with a
  non-exact estimator. Activations recorded during an intervened pass stay
  MEASURED (ADR-017); only the effect is INTERVENTIONAL.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ._canonical import EMPTY_JSON, JsonMap
from ._types import Value, require
from .base import BaseRecord, RecordRef, record_kind, register_migration
from .errors import EvidenceRuleError
from .status import EstimandScope, EvidenceStatus
from .values import Estimand, Site, SiteIO, TargetSpec, TensorRef

__all__ = [
    "CausalEffect",
    "InterventionOperation",
    "InterventionRecord",
    "MetricDeclaration",
    "MetricSpec",
]

_TOKEN_RE = re.compile(r"^\S+$")


class InterventionOperation(Enum):
    ZERO = "zero"
    CONSTANT = "constant"
    PATCH = "patch"


@record_kind("intervention")
@dataclass(frozen=True, slots=True, kw_only=True)
class InterventionRecord(BaseRecord):
    """Replace one tensor leaf of one call of a module's output.

    ZERO: ``zeros_like`` (no other fields). CONSTANT: exactly one of ``constant``
    (a scalar, filled into the leaf's shape) or ``value`` (a tensor of exactly the
    leaf's shape and dtype). PATCH: ``value`` is the retained source activation and
    ``source`` references the source ``ActivationRecord`` in the same trace.
    Retained values carry a ``content_digest`` and a ``storage_key``.
    """

    site: Site
    call_index: int = 0
    operation: InterventionOperation
    constant: float | None = None
    value: TensorRef | None = None
    source: RecordRef | None = None

    def _validate(self) -> None:
        require(self.site.io is SiteIO.OUTPUT, "Phase 2 supports module OUTPUT interventions only")
        require(bool(self.site.module), "the root output cannot be intervened on")
        require(self.call_index >= 0, "InterventionRecord.call_index must be >= 0")
        op = self.operation
        if op is InterventionOperation.ZERO:
            require(
                self.constant is None and self.value is None and self.source is None,
                "ZERO takes no constant, value or source",
            )
        elif op is InterventionOperation.CONSTANT:
            require(
                (self.constant is None) != (self.value is None),
                "CONSTANT needs exactly one of constant or value",
            )
            require(self.source is None, "CONSTANT has no source (use PATCH for activations)")
            if self.constant is not None:
                require(math.isfinite(self.constant), "CONSTANT constant must be finite")
        else:
            require(
                self.value is not None and self.source is not None, "PATCH needs value and source"
            )
            require(self.constant is None, "PATCH has no scalar constant")
            assert self.source is not None
            require(self.source.kind == "activation", "PATCH source must be an ActivationRecord")
            require(self.source.status is EvidenceStatus.MEASURED, "PATCH source must be MEASURED")
        if self.value is not None:
            require(
                self.value.content_digest is not None and self.value.storage_key is not None,
                "intervention values must be retained (content_digest and storage_key)",
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class MetricDeclaration(Value):
    """Identity of a caller metric as DECLARED by the caller (like ``ModelDeclaration``).

    Recorded as stated and never verified: BeyondNN does not serialise, inspect,
    scrape, or hash the function. ``implementation_revision`` (e.g. ``"git:abc123"``,
    ``"v2"``) is required so two different functions sharing a human-readable name
    can be told apart; ``config`` holds the caller's declared parameters.
    """

    implementation_revision: str
    config: JsonMap = EMPTY_JSON

    def _validate(self) -> None:
        require(
            bool(_TOKEN_RE.match(self.implementation_revision)),
            "MetricDeclaration.implementation_revision must be a non-empty token",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class MetricSpec(Value):
    """Identity and configuration of a scalar outcome metric.

    ``builtin`` metrics are BeyondNN's own (pure, deterministic), identified by name
    and ``params``. Caller metrics are named ``custom:<name>``, have no ``params``,
    and must carry a caller :class:`MetricDeclaration` (declared revision and
    config): an undeclared caller metric has no scientific identity and is refused.
    Their code is never serialised, inspected, or verified.
    """

    name: str
    builtin: bool
    params: JsonMap = EMPTY_JSON
    declaration: MetricDeclaration | None = None

    def _validate(self) -> None:
        require(bool(self.name) and " " not in self.name, f"invalid metric name {self.name!r}")
        require(
            self.builtin != self.name.startswith("custom:"),
            "caller metrics are named 'custom:<name>'; built-ins are not",
        )
        if self.builtin:
            require(self.declaration is None, "built-in metrics are identified automatically")
        else:
            require(
                self.declaration is not None,
                "a caller metric needs a MetricDeclaration (declared implementation_revision)",
            )
            require(len(self.params) == 0, "caller metric parameters go in declaration.config")

    def target(self) -> TargetSpec:
        """The claim target this metric defines (one deterministic mapping; claims and
        evidence about different declared revisions never match)."""
        if self.declaration is None:
            return TargetSpec(metric=self.name, params=self.params)
        return TargetSpec(
            metric=self.name,
            params=JsonMap(
                {
                    "declared_implementation_revision": self.declaration.implementation_revision,
                    "declared_config": self.declaration.config.to_plain(),
                }
            ),
        )


@record_kind("causal_effect", version=2)
@dataclass(frozen=True, slots=True, kw_only=True)
class CausalEffect(BaseRecord):
    """A scalar intervention effect: ``effect = intervention_value - baseline_value``.

    INSTANCE: ``derived_from`` holds the paired baseline and intervention
    ``OutputRecord``s; the effect is exact. FINITE_SAMPLE: ``derived_from`` holds the
    ``n`` instance effects and the values are their means (aggregation ``mean``).

    Record version 2 added ``MetricSpec.declaration``. Version-1 payloads migrate
    with ``declaration = None``; a v1 effect on a caller metric therefore fails to
    load (its metric identity is ambiguous) rather than gaining an invented revision.
    """

    interventions: tuple[RecordRef, ...]
    metric: MetricSpec
    estimand: Estimand
    baseline_value: float
    intervention_value: float
    effect: float
    estimator: str = "exact"

    @property
    def status(self) -> EvidenceStatus | None:
        if self.estimand.scope is EstimandScope.POPULATION:
            return EvidenceStatus.ESTIMATED_CAUSAL
        return EvidenceStatus.INTERVENTIONAL

    def _validate(self) -> None:
        require(len(self.interventions) > 0, "CausalEffect needs at least one intervention")
        require(
            all(r.kind == "intervention" for r in self.interventions),
            "interventions must reference InterventionRecords",
        )
        for name in ("baseline_value", "intervention_value", "effect"):
            require(math.isfinite(getattr(self, name)), f"CausalEffect.{name} must be finite")
        scope = self.estimand.scope
        if scope is EstimandScope.POPULATION:
            if self.estimator == "exact":
                raise EvidenceRuleError("a POPULATION effect needs a (non-exact) estimator")
        else:
            require(self.estimator == "exact", "INSTANCE/FINITE_SAMPLE effects are exact")
        if scope is EstimandScope.INSTANCE:
            require(len(self.interventions) == 1, "an INSTANCE effect has exactly one intervention")
            require(
                self.effect == self.intervention_value - self.baseline_value,
                "effect must equal intervention_value - baseline_value",
            )
            require(
                sum(1 for r in self.derived_from if r.kind == "output") == 2,
                "an INSTANCE effect derives from the paired baseline and intervention outputs",
            )
        elif scope is EstimandScope.FINITE_SAMPLE:
            require(self.estimand.aggregation == "mean", "FINITE_SAMPLE effects aggregate by mean")
            parts = [r for r in self.derived_from if r.kind == "causal_effect"]
            require(
                len(parts) == self.estimand.n,
                "a FINITE_SAMPLE effect derives from exactly n instance effects",
            )
            require(
                math.isclose(
                    self.effect,
                    self.intervention_value - self.baseline_value,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                ),
                "effect must equal intervention_value - baseline_value",
            )


@register_migration("causal_effect", 1)
def _causal_effect_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    metric = data.get("metric")
    if not isinstance(metric, dict) or "declaration" in metric:
        raise ValueError("a causal_effect v1 payload has a metric without declaration")
    return data | {"metric": metric | {"declaration": None}}
