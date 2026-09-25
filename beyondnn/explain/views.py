"""Immutable views over composed evidence (Phase 4; ADR-031).

Views hold references to the source traces' own record objects: nothing is copied
into a weaker duplicate, nothing is ranked, scored, or combined across methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

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
    it never means "the explanation is faithful". ``concepts_validated`` is always
    ``False`` (no concept validation exists).
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
    concepts_validated: Literal[False] = False
    faithfulness_protocols: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.faithfulness_evaluated != bool(self.faithfulness_protocols):
            raise ValueError(
                "faithfulness is evaluated only by recorded faithfulness protocol results; "
                "faithfulness_evaluated must be true exactly when protocols are named"
            )
        unknown = [p for p in self.faithfulness_protocols if p not in FAITHFULNESS_PROTOCOLS]
        if unknown:
            raise ValueError(f"unknown faithfulness protocols {unknown}")
        if self.concepts_validated is not False:
            raise ValueError("no concept validation exists; concepts cannot be marked validated")

    @property
    def not_evaluated(self) -> tuple[str, ...]:
        """What was not evaluated (Phase-4 wording when no faithfulness protocol ran)."""
        if not self.faithfulness_protocols:
            return self.NOT_EVALUATED
        missing = tuple(p for p in FAITHFULNESS_PROTOCOLS if p not in self.faithfulness_protocols)
        return (*missing, "concept validation")
