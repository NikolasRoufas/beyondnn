"""Epistemic status and the other closed vocabularies of the schema.

``EvidenceStatus`` is deliberately a plain :class:`enum.Enum`, not a ``str`` or
``int`` enum: statuses describe *different kinds* of evidence, not levels of a
hierarchy, so they cannot be ordered (``<`` raises ``TypeError``). What may be
derived from what is stated explicitly in :data:`ALLOWED_PARENT_STATUSES`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import Enum
from types import MappingProxyType
from typing import Protocol

from .errors import EvidenceRuleError

__all__ = [
    "ALLOWED_PARENT_STATUSES",
    "CAUSAL_EVIDENCE_STATUSES",
    "CAUSAL_RELATIONS",
    "EstimandScope",
    "EvidenceStatus",
    "Outcome",
    "Relation",
    "Verdict",
    "check_derivation",
]


class EvidenceStatus(Enum):
    """How a piece of evidence was obtained.

    OBSERVED:
        A value that crossed the model boundary unchanged (inputs, outputs).
    MEASURED:
        Directly observed internal model state (e.g. a module output read by a
        hook). Stays MEASURED when read during an intervened execution: the
        intervention is execution context and belongs in provenance (ADR-017).
    ATTRIBUTED:
        A method-relative score (gradient, integrated gradients, ...).
    INTERVENTIONAL:
        A directly measured *effect* of a specified intervention on specified
        inputs, i.e. a comparison of intervened against baseline behaviour: one
        instance or an exact summary of a finite sample (ADR-013). Raw state read
        while an intervention is active is MEASURED, not INTERVENTIONAL.
    ESTIMATED_CAUSAL:
        An approximation of an interventional quantity, or any causal quantity
        about a population beyond the measured inputs (ADR-013).
    VALIDATED_CONCEPT:
        An activation of a concept that passed a declared validation protocol.
    GENERATED:
        Produced by a model, LLM, or template. Never scientific evidence.
    """

    OBSERVED = "observed"
    MEASURED = "measured"
    ATTRIBUTED = "attributed"
    INTERVENTIONAL = "interventional"
    ESTIMATED_CAUSAL = "estimated_causal"
    VALIDATED_CONCEPT = "validated_concept"
    GENERATED = "generated"


class EstimandScope(Enum):
    """What population a (causal) quantity is about (ADR-013).

    INSTANCE:
        Exactly one identified input.
    FINITE_SAMPLE:
        Exactly the ``n`` identified inputs that were measured; nothing beyond.
    POPULATION:
        A distribution, dataset, or future inputs beyond the measured examples.
    """

    INSTANCE = "instance"
    FINITE_SAMPLE = "finite_sample"
    POPULATION = "population"


class Relation(Enum):
    """What a claim asserts about its subject and target (ADR-012)."""

    NECESSARY_FOR = "necessary_for"
    SUFFICIENT_FOR = "sufficient_for"
    INCREASES = "increases"
    DECREASES = "decreases"
    ATTRIBUTED_TO = "attributed_to"
    ENCODES = "encodes"


class Outcome(Enum):
    """The outcome of running one declared test on one claim."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    INCONCLUSIVE = "inconclusive"
    NOT_APPLICABLE = "not_applicable"
    ERRORED = "errored"


class Verdict(Enum):
    """A claim's standing under a policy, derived from test results. Not a probability."""

    UNTESTED = "untested"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    MIXED = "mixed"
    INCONCLUSIVE = "inconclusive"


#: Relations that assert a causal dependence. Deciding them requires causal evidence.
CAUSAL_RELATIONS: frozenset[Relation] = frozenset(
    {Relation.NECESSARY_FOR, Relation.SUFFICIENT_FOR, Relation.INCREASES, Relation.DECREASES}
)

#: Evidence statuses that can decide a causal relation.
CAUSAL_EVIDENCE_STATUSES: frozenset[EvidenceStatus] = frozenset(
    {EvidenceStatus.INTERVENTIONAL, EvidenceStatus.ESTIMATED_CAUSAL}
)

_S = EvidenceStatus

#: For a record of status ``key``, the statuses its ``derived_from`` parents may have.
#: ``derived_from`` means "values computed from"; the context that *selected* what
#: to compute (e.g. an attribution used to choose which unit to ablate) belongs in
#: provenance parameters, not in lineage. Nothing but GENERATED derives from GENERATED.
ALLOWED_PARENT_STATUSES: Mapping[EvidenceStatus, frozenset[EvidenceStatus]] = MappingProxyType(
    {
        _S.OBSERVED: frozenset({_S.OBSERVED}),
        _S.MEASURED: frozenset({_S.OBSERVED, _S.MEASURED}),
        _S.ATTRIBUTED: frozenset({_S.OBSERVED, _S.MEASURED, _S.ATTRIBUTED}),
        _S.INTERVENTIONAL: frozenset({_S.OBSERVED, _S.MEASURED, _S.INTERVENTIONAL}),
        _S.ESTIMATED_CAUSAL: frozenset(
            {_S.OBSERVED, _S.MEASURED, _S.ATTRIBUTED, _S.INTERVENTIONAL, _S.ESTIMATED_CAUSAL}
        ),
        _S.VALIDATED_CONCEPT: frozenset(
            {
                _S.OBSERVED,
                _S.MEASURED,
                _S.INTERVENTIONAL,
                _S.ESTIMATED_CAUSAL,
                _S.VALIDATED_CONCEPT,
            }
        ),
        _S.GENERATED: frozenset(_S),
    }
)


class _HasStatus(Protocol):
    @property
    def record_id(self) -> str: ...
    @property
    def status(self) -> EvidenceStatus | None: ...


def check_derivation(child: EvidenceStatus | None, parents: Iterable[_HasStatus]) -> None:
    """Raise :class:`EvidenceRuleError` if ``child`` may not derive from ``parents``.

    Records without an evidence status (claims, specs, results, assessments,
    limitations) may derive from anything. Evidence records may only derive from
    evidence records whose status is allowed by :data:`ALLOWED_PARENT_STATUSES`.
    """
    if child is None:
        return
    allowed = ALLOWED_PARENT_STATUSES[child]
    for parent in parents:
        if parent.status is None:
            raise EvidenceRuleError(
                f"a {child.value} record cannot derive from non-evidence record {parent.record_id}"
            )
        if parent.status not in allowed:
            raise EvidenceRuleError(
                f"a {child.value} record cannot derive from {parent.status.value} record "
                f"{parent.record_id}; allowed parent statuses: "
                f"{sorted(s.value for s in allowed)}"
            )
