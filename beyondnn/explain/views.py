"""Immutable views over composed evidence (Phase 4; ADR-031).

Views hold references to the source traces' own record objects: nothing is copied
into a weaker duplicate, nothing is ranked, scored, or combined across methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from beyondnn.core.trace import TraceResult
from beyondnn.schema import (
    CAUSAL_RELATIONS,
    ActivationRecord,
    Assessment,
    AttributionRecord,
    AttributionReduction,
    CausalEffect,
    Claim,
    ClaimTestResult,
    ClaimTestSpec,
    EvidenceSelection,
    InputRecord,
    InterventionRecord,
    Outcome,
    OutputRecord,
    TraceLimitation,
)

__all__ = [
    "FAITHFULNESS_PROTOCOLS",
    "AttributionView",
    "AuditView",
    "ClaimView",
    "Coverage",
    "FaithfulnessView",
    "InterventionView",
]

#: Every Phase-5 faithfulness protocol (claim tests and diagnostics), in display order.
FAITHFULNESS_PROTOCOLS = (
    "comprehensiveness",
    "sufficiency",
    "removal_curve",
    "retention_curve",
    "stability",
    "method_agreement",
    "baseline_sensitivity",
    "ig_step_sensitivity",
)

_DECISIVE = frozenset({Outcome.SUPPORTS, Outcome.CONTRADICTS})


@dataclass(frozen=True, slots=True, eq=False)
class AttributionView:
    """One attribution: the ATTRIBUTED record, its explicit reductions, the attributed
    OBSERVED input or MEASURED activation, and the limitations scoped to it."""

    record: AttributionRecord
    reductions: tuple[AttributionReduction, ...]
    attributed: InputRecord | ActivationRecord
    limitations: tuple[TraceLimitation, ...]
    source: TraceResult

    @property
    def value(self) -> torch.Tensor:
        """The raw attribution tensor, as retained (no ranking, no reduction)."""
        return self.source.tensor(self.record.value)


@dataclass(frozen=True, slots=True, eq=False)
class InterventionView:
    """One controlled comparison: the intervention spec, its INTERVENTIONAL effect, the
    paired OBSERVED outputs, the patch source context (if any), and its limitations."""

    intervention: InterventionRecord
    effect: CausalEffect
    baseline_output: OutputRecord
    intervention_output: OutputRecord
    source_activation: ActivationRecord | None
    source_input: InputRecord | None
    limitations: tuple[TraceLimitation, ...]
    source: TraceResult


@dataclass(frozen=True, slots=True, eq=False)
class ClaimView:
    """A declared claim with every recorded test (spec + result, in recorded order) and
    its assessments under the supplied policies. Outcomes are never reduced to booleans."""

    claim: Claim
    tests: tuple[tuple[ClaimTestSpec, ClaimTestResult], ...]
    assessments: tuple[Assessment, ...]

    @property
    def is_causal(self) -> bool:
        return self.claim.relation in CAUSAL_RELATIONS

    @property
    def decisive_tests(self) -> tuple[tuple[ClaimTestSpec, ClaimTestResult], ...]:
        return tuple(t for t in self.tests if t[1].outcome in _DECISIVE)

    @property
    def causal_test_performed(self) -> bool:
        """A causal claim with at least one SUPPORTS/CONTRADICTS result (the schema only
        allows those with INTERVENTIONAL/ESTIMATED_CAUSAL evidence)."""
        return self.is_causal and bool(self.decisive_tests)


@dataclass(frozen=True, slots=True, eq=False)
class FaithfulnessView:
    """One faithfulness claim test: the declared claim and spec, the ClaimTestResult,
    the evidence selection, the primary INTERVENTIONAL effect, its matched controls,
    and the limitations scoped to the result. Nothing is summarised further."""

    claim: Claim
    spec: ClaimTestSpec
    result: ClaimTestResult
    selection: EvidenceSelection | None
    effect: CausalEffect | None
    controls: tuple[CausalEffect, ...]
    limitations: tuple[TraceLimitation, ...]


@dataclass(frozen=True, slots=True)
class Coverage:
    """Which kinds of evidence exist and which evaluations never happened.

    Coverage, not confidence: absence of a method is not "low confidence", and there
    is no score. ``faithfulness_evaluated`` is true only when at least one Phase-5
    protocol result is composed, and ``faithfulness_protocols`` names exactly which;
    it never means "the explanation is faithful". ``concepts_evaluated`` is true only
    when at least one Phase-6 concept validation is composed; ``concepts_validated``
    only when at least one of them derived VALIDATED_CONCEPT (within its own scope).
    """

    NOT_EVALUATED: ClassVar[tuple[str, ...]] = (
        "faithfulness",
        "comprehensiveness",
        "sufficiency",
        "concept validation",
    )

    measured_internal_states: bool
    attribution_available: bool
    intervention_effect_available: bool
    estimated_causal_available: bool
    claims_declared: bool
    claims_tested: bool
    causal_claim_tested: bool
    faithfulness_evaluated: bool = False
    concepts_validated: bool = False
    faithfulness_protocols: tuple[str, ...] = ()
    concepts_evaluated: bool = False

    def __post_init__(self) -> None:
        if self.faithfulness_evaluated != bool(self.faithfulness_protocols):
            raise ValueError(
                "faithfulness is evaluated only by recorded faithfulness protocol results; "
                "faithfulness_evaluated must be true exactly when protocols are named"
            )
        unknown = [p for p in self.faithfulness_protocols if p not in FAITHFULNESS_PROTOCOLS]
        if unknown:
            raise ValueError(f"unknown faithfulness protocols {unknown}")
        if self.concepts_validated and not self.concepts_evaluated:
            raise ValueError(
                "concepts can be marked validated only when a concept validation is composed"
            )

    @property
    def not_evaluated(self) -> tuple[str, ...]:
        """What was not evaluated (Phase-4 wording when no faithfulness protocol ran)."""
        concept = () if self.concepts_evaluated else ("concept validation",)
        if not self.faithfulness_protocols:
            return (*self.NOT_EVALUATED[:-1], *concept)
        missing = tuple(p for p in FAITHFULNESS_PROTOCOLS if p not in self.faithfulness_protocols)
        return (*missing, *concept)


@dataclass(frozen=True, slots=True, eq=False)
class ConceptView:
    """A composed concept validation (dataset-scoped context), the feature's value on
    the reference input (``None`` if its site was not recorded), and the
    VALIDATED_CONCEPT activation record if one was composed."""

    validation: Any
    value: float | None
    activation: Any | None

    @property
    def validated(self) -> bool:
        return bool(self.validation.semantic_status.value == "validated_concept")

    @property
    def limitations(self) -> tuple[Any, ...]:
        v = self.validation
        traces = [v.trace, v.encoding.trace, *(u.trace for u in (*v.use, *v.additional))]
        return tuple(
            lim
            for t in traces
            for lim in t.limitations
            if not lim.code.startswith(
                ("FUNCTIONAL_OPS", "PARTIAL_SITE", "SELECTED_SITE", "NON_TENSOR")
            )
        )


@dataclass(frozen=True, slots=True, eq=False)
class AuditView:
    """A composed Phase-7 audit report restricted to the reference input (ADR-047): the
    per-sample claims with their group for this sample, the finite-sample / population
    claims as context, and the concept audits (dataset-scoped context)."""

    report: Any
    sample: str | None
    claims: tuple[tuple[Any, Any], ...]
    context: tuple[Any, ...]
    concepts: tuple[Any, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.report.plan.id,
            "sample": self.sample,
            "claims": [
                {
                    "claim": c.name,
                    "standing": g.standing.value,
                    "findings": sorted({f"{f.kind.value}:{f.code}" for f in g.findings}),
                    "distribution": dict(c.distribution),
                }
                for c, g in self.claims
            ],
            "context": [
                {"claim": c.name, "scope": c.claim.scope.value, "standing": c.standing.value}
                for c in self.context
            ],
            "concepts": [
                {"concept": k.concept.concept, "standing": k.standing.value} for k in self.concepts
            ],
        }
