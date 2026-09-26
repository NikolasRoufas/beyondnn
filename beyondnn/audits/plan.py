"""Constructors for audit plans (plan §28). Every value is explicit: ``None`` means
"not declared" and the report says so; nothing is defaulted silently."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from beyondnn.schema import (
    AlternativeCriteria,
    AssessmentPolicy,
    AuditAxis,
    AuditedClaim,
    AuditedConcept,
    AuditPlan,
    ConceptPolicy,
    CounterexampleRule,
    EstimandScope,
    EvidenceRequirement,
    Invariance,
    ModelDeclaration,
    Relation,
    SampleTarget,
    SelectionSubject,
    SemanticStatus,
    Site,
    SiteIO,
    Subject,
    TargetSpec,
)

__all__ = [
    "alternative",
    "checkpoint_of",
    "claim",
    "concept",
    "counterexample_rule",
    "invariance",
    "plan",
    "requirement",
    "selection",
]


def checkpoint_of(model: Any) -> str:
    """The model's FULL state digest (the checkpoint an audit plan is about)."""
    from beyondnn.provenance import fingerprint_model

    return fingerprint_model(model).state_digest


def _site(site: str | Site) -> Site:
    if isinstance(site, Site):
        return site
    if site == "input":
        return Site(module="", io=SiteIO.INPUT, output_path="args[0]")
    return Site(module=site)


def _target(target: Any) -> TargetSpec:
    if isinstance(target, TargetSpec):
        return target
    spec = getattr(target, "spec", None)  # an interventions metric
    if spec is not None and hasattr(spec, "target"):
        result = spec.target()
        assert isinstance(result, TargetSpec)
        return result
    raise TypeError("target must be a TargetSpec or an interventions metric")


def selection(site: str | Site, *, method: str, k: int | None) -> SelectionSubject:
    """The units ``method`` (an attribution method name, or ``"declared"``) selects at
    ``site`` on each sample; ``k=None`` means any k (k then becomes an assumption axis)."""
    return SelectionSubject(site=_site(site), method=method, k=k)


def invariance(axis: str, *, min_values: int, values: Sequence[str] = ()) -> Invariance:
    """The claim asserts it holds across ``axis`` (>= ``min_values`` tested values)."""
    return Invariance(axis=AuditAxis(axis), min_values=min_values, values=tuple(values))


def claim(
    name: str,
    *,
    statement: str,
    relation: str | Relation,
    target: Any,
    scope: str | EstimandScope,
    requirement: str,
    sample_targets: Mapping[str, Any] | None = None,
    subject: Subject | None = None,
    selection: SelectionSubject | None = None,
    sample_set: str | None = None,
    population: str | None = None,
    invariant_over: Iterable[Invariance] = (),
) -> AuditedClaim:
    """``target`` is one metric/TargetSpec, or ``None`` with ``sample_targets`` mapping each
    plan sample id to its own declared target (per-sample claims)."""
    return AuditedClaim(
        name=name,
        statement=statement,
        relation=Relation(relation),
        target=None if target is None else _target(target),
        sample_targets=tuple(
            SampleTarget(sample=s, target=_target(t)) for s, t in (sample_targets or {}).items()
        ),
        scope=EstimandScope(scope),
        requirement=requirement,
        subject=subject,
        selection=selection,
        sample_set=sample_set,
        population=population,
        invariant_over=tuple(invariant_over),
    )


def alternative(
    protocol: str, key: str, *, factor: float | None = None, value: float | None = None
) -> AlternativeCriteria:
    """Re-evaluate recorded ``protocol`` results with criterion ``key`` scaled by
    ``factor`` (or set to ``value``); labelled re-evaluations, never verdicts."""
    return AlternativeCriteria(protocol=protocol, key=key, factor=factor, value=value)


def requirement(
    name: str,
    *,
    policy: AssessmentPolicy,
    controls: bool,
    alternatives: Iterable[AlternativeCriteria] = (),
) -> EvidenceRequirement:
    return EvidenceRequirement(
        name=name, policy=policy, controls=controls, alternatives=tuple(alternatives)
    )


def concept(
    concept_id: str,
    *,
    policy: ConceptPolicy,
    asserted: str | SemanticStatus = SemanticStatus.VALIDATED_CONCEPT,
    invariant_over: Iterable[Invariance] = (),
) -> AuditedConcept:
    """A concept asserted to be validated under ``policy`` (the only concept claim)."""
    return AuditedConcept(
        concept=concept_id,
        asserted=SemanticStatus(asserted),
        policy=policy,
        invariant_over=tuple(invariant_over),
    )


def counterexample_rule(
    *,
    max_counterexample_fraction: float | None,
    max_false_positive_rate: float | None,
    max_false_negative_rate: float | None,
) -> CounterexampleRule:
    return CounterexampleRule(
        max_counterexample_fraction=max_counterexample_fraction,
        max_false_positive_rate=max_false_positive_rate,
        max_false_negative_rate=max_false_negative_rate,
    )


def plan(
    *,
    name: str,
    checkpoint: str,
    declared_model: ModelDeclaration | None,
    samples: Iterable[str],
    datasets: Iterable[str],
    claims: Iterable[AuditedClaim],
    requirements: Iterable[EvidenceRequirement],
    concepts: Iterable[AuditedConcept],
    counterexamples: CounterexampleRule,
    naive_auroc: float | None,
) -> AuditPlan:
    """An :class:`~beyondnn.schema.AuditPlan`; every field is required (``None`` or an
    empty sequence where nothing is declared)."""
    return AuditPlan(
        name=name,
        checkpoint=checkpoint,
        declared_model=declared_model,
        samples=tuple(samples),
        datasets=tuple(datasets),
        claims=tuple(claims),
        requirements=tuple(requirements),
        concepts=tuple(concepts),
        counterexamples=counterexamples,
        naive_auroc=naive_auroc,
    )
