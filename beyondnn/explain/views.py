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
    InputRecord,
    InterventionRecord,
    Outcome,
    OutputRecord,
    TraceLimitation,
)

__all__ = ["AttributionView", "ClaimView", "Coverage", "InterventionView"]

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


@dataclass(frozen=True, slots=True)
class Coverage:
    """Which kinds of evidence exist and which evaluations never happened.

    Coverage, not confidence: absence of a method is not "low confidence", and there
    is no score. ``faithfulness_evaluated`` and ``concepts_validated`` are always
    ``False`` in Phase 4 (no faithfulness protocol and no concept validation exist).
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
    faithfulness_evaluated: Literal[False] = False
    concepts_validated: Literal[False] = False

    def __post_init__(self) -> None:
        if self.faithfulness_evaluated is not False or self.concepts_validated is not False:
            raise ValueError(
                "faithfulness and concept validation are not evaluated by any Phase-4 "
                "protocol; they cannot be marked as evaluated"
            )
