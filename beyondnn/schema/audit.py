"""The audit plan (Phase 7; ADR-044, ADR-045).

Pure data. The audit itself lives in :mod:`beyondnn.audits`.

An :class:`AuditPlan` declares, before the audit runs, *what is being claimed* and
*what evidence a claim needs*:

* the scope: the model checkpoint (state digest), the declared model, the exact
  sample identities and concept datasets the evidence must be about;
* the claims (:class:`AuditedClaim`): a formal structure (relation, subject or
  selection, target, estimand scope), a named evidence requirement, and the
  assumption axes the claim asserts to hold across (:class:`Invariance`);
* the evidence requirements (:class:`EvidenceRequirement`): an assessment policy,
  whether controls are required, and declared alternative criteria for threshold
  sensitivity (:class:`AlternativeCriteria`);
* the concepts asserted to be validated (:class:`AuditedConcept`);
* the counterexample caps (:class:`CounterexampleRule`).

Nothing here has a default that changes a result: ``None`` means "not declared" and
the report says so. A plan is a record: its id is a hash of everything it declares.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ._types import Value, require
from .base import BaseRecord, is_record_id, record_kind, register_migration
from .claims import AssessmentPolicy
from .concepts import ConceptPolicy
from .provenance import ModelDeclaration
from .status import CAUSAL_RELATIONS, EstimandScope, Relation, SemanticStatus
from .values import Site, Subject, TargetSpec

__all__ = [
    "AlternativeCriteria",
    "AuditAxis",
    "AuditPlan",
    "AuditedClaim",
    "AuditedConcept",
    "ConfigurationRole",
    "CounterexampleRule",
    "EvidenceRequirement",
    "Invariance",
    "RoleRule",
    "SampleTarget",
    "SelectionSubject",
]

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SELECTION_RELATIONS = frozenset(
    {Relation.NECESSARY_FOR, Relation.DECREASES, Relation.SUFFICIENT_FOR}
)


class AuditAxis(Enum):
    """An assumption a result was obtained under (plan §11-§14)."""

    PROTOCOL = "protocol"
    THRESHOLD = "threshold"
    REPLACEMENT = "replacement"
    K = "k"
    NULL = "null"
    METHOD = "method"
    DATASET = "dataset"


class ConfigurationRole(Enum):
    """The declared role of an assumption value (ADR-048; plan 7.5 §22).

    PRIMARY: the pre-registered analysis; it decides the standing. ALTERNATIVE: a
    reasonable alternative; a reversal is reported, it does not change the standing.
    STRESS_TEST: deliberately extreme or out of distribution; a reversal is reported
    as context."""

    PRIMARY = "primary"
    ALTERNATIVE = "alternative"
    STRESS_TEST = "stress_test"


@dataclass(frozen=True, slots=True, kw_only=True)
class RoleRule(Value):
    """Values of ``axis`` whose key matches the glob ``pattern`` (``fnmatch``) have
    ``role``; ``sample`` restricts the rule to one sample (e.g. the k that p = 10% gives
    on that sample). A configuration's role is the worst role over the declared axes,
    where on one axis the best matching rule counts; a value on a declared axis that no
    rule matches makes the configuration UNDECLARED (reported, never in a standing)."""

    axis: AuditAxis
    pattern: str
    role: ConfigurationRole
    sample: str | None = None

    def _validate(self) -> None:
        require(bool(self.pattern), "RoleRule.pattern must be non-empty")


def _sorted_rules(rules: tuple[RoleRule, ...]) -> tuple[RoleRule, ...]:
    keys = [(r.axis.value, r.sample or "", r.pattern, r.role.value) for r in rules]
    require(len(set(keys)) == len(keys), "duplicate role rules")
    return tuple(r for _, r in sorted(zip(keys, rules, strict=True), key=lambda kr: kr[0]))


@dataclass(frozen=True, slots=True, kw_only=True)
class Invariance(Value):
    """The claim asserts it holds across ``axis``: at least ``min_values`` distinct values
    must be tested, including every value key in ``values`` (if any)."""

    axis: AuditAxis
    min_values: int
    values: tuple[str, ...] = ()

    def _validate(self) -> None:
        require(self.min_values >= 2, "an invariance needs at least two tested values")
        require(len(set(self.values)) == len(self.values), "Invariance.values has duplicates")
        require(len(self.values) <= self.min_values, "min_values must cover the listed values")
        object.__setattr__(self, "values", tuple(sorted(self.values)))


@dataclass(frozen=True, slots=True, kw_only=True)
class SelectionSubject(Value):
    """The units a method selects on each sample: ``method`` is an attribution method
    name, or ``"declared"`` for declared/random unit sets. ``k=None``: any k."""

    site: Site
    method: str
    k: int | None = None

    def _validate(self) -> None:
        require(bool(self.method.strip()), "SelectionSubject.method must be non-empty")
        require(self.k is None or self.k >= 1, "SelectionSubject.k must be >= 1")


@dataclass(frozen=True, slots=True, kw_only=True)
class SampleTarget(Value):
    """The declared target of a per-sample claim on one sample (e.g. a margin between
    that sample's predicted class and its runner-up, fixed from its clean pass)."""

    sample: str
    target: TargetSpec

    def _validate(self) -> None:
        require(bool(self.sample.strip()), "SampleTarget.sample must be non-empty")


@dataclass(frozen=True, slots=True, kw_only=True)
class AuditedClaim(Value):
    """A claim under audit, by formal structure (ADR-016), never by record id.

    ``scope`` INSTANCE: the claim is audited on every plan sample (per-sample; plan
    §21). FINITE_SAMPLE: about exactly the sample set ``sample_set`` (an
    ``Estimand.sample_id``). POPULATION: about ``population``.

    The target is one declared ``target`` or, for per-sample claims whose target depends
    on the sample, ``sample_targets`` (one per plan sample, declared before the audit).
    """

    name: str
    statement: str
    relation: Relation
    target: TargetSpec | None
    scope: EstimandScope
    requirement: str
    sample_targets: tuple[SampleTarget, ...] = ()
    subject: Subject | None = None
    selection: SelectionSubject | None = None
    sample_set: str | None = None
    population: str | None = None
    invariant_over: tuple[Invariance, ...] = ()
    roles: tuple[RoleRule, ...] = ()

    def _validate(self) -> None:
        require(bool(_NAME_RE.match(self.name)), f"invalid claim name {self.name!r}")
        object.__setattr__(self, "roles", _sorted_rules(self.roles))
        require(bool(self.statement.strip()), "AuditedClaim.statement must be non-empty")
        require(bool(_NAME_RE.match(self.requirement)), "invalid requirement name")
        require(
            (self.target is None) != (not self.sample_targets),
            "an audited claim has exactly one of target / sample_targets",
        )
        if self.sample_targets:
            require(
                self.scope is EstimandScope.INSTANCE,
                "sample_targets are declared for per-sample (INSTANCE) claims only",
            )
            samples = [t.sample for t in self.sample_targets]
            require(len(set(samples)) == len(samples), "one sample target per sample")
            object.__setattr__(
                self, "sample_targets", tuple(sorted(self.sample_targets, key=lambda t: t.sample))
            )
        require(
            (self.subject is None) != (self.selection is None),
            "an audited claim has exactly one of subject / selection",
        )
        if self.selection is not None:
            require(
                self.scope is EstimandScope.INSTANCE,
                "a selection claim is audited per sample (INSTANCE scope)",
            )
            require(
                self.relation in _SELECTION_RELATIONS,
                "a selection claim is NECESSARY_FOR, DECREASES or SUFFICIENT_FOR",
            )
        require(
            (self.sample_set is not None) == (self.scope is EstimandScope.FINITE_SAMPLE),
            "sample_set is declared exactly for FINITE_SAMPLE claims",
        )
        require(
            (self.population is not None) == (self.scope is EstimandScope.POPULATION),
            "population is declared exactly for POPULATION claims",
        )
        axes = [i.axis for i in self.invariant_over]
        require(len(set(axes)) == len(axes), "one invariance per axis")
        object.__setattr__(
            self, "invariant_over", tuple(sorted(self.invariant_over, key=lambda i: i.axis.value))
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class AlternativeCriteria(Value):
    """Re-evaluate recorded results of ``protocol`` with criterion ``key`` replaced by
    ``factor`` times its recorded value, or by ``value``. Re-evaluations are labelled
    as such and never enter a verdict (plan §13)."""

    protocol: str
    key: str
    factor: float | None = None
    value: float | None = None

    def _validate(self) -> None:
        require(bool(_KEY_RE.match(self.protocol)), "invalid protocol name")
        require(bool(_KEY_RE.match(self.key)), "invalid criterion key")
        require(
            (self.factor is None) != (self.value is None),
            "an alternative declares exactly one of factor / value",
        )
        require(self.factor is None or self.factor > 0, "factor must be > 0")


@dataclass(frozen=True, slots=True, kw_only=True)
class EvidenceRequirement(Value):
    """A named required-evidence policy: the assessment policy that decides the claim's
    verdict, whether a SUPPORTS result must have declared controls, and alternative
    criteria for threshold sensitivity."""

    name: str
    policy: AssessmentPolicy
    controls: bool
    alternatives: tuple[AlternativeCriteria, ...] = ()

    def _validate(self) -> None:
        require(bool(_NAME_RE.match(self.name)), f"invalid requirement name {self.name!r}")
        keys = [(a.protocol, a.key, a.factor, a.value) for a in self.alternatives]
        require(len(set(keys)) == len(keys), "duplicate alternative criteria")


@dataclass(frozen=True, slots=True, kw_only=True)
class AuditedConcept(Value):
    """A concept (``ConceptRecord`` id) asserted to have status ``asserted`` under
    ``policy``. ``invariant_over`` applies to its tests: NULL over encoding tests,
    REPLACEMENT over use tests, DATASET over validations."""

    concept: str
    asserted: SemanticStatus
    policy: ConceptPolicy
    invariant_over: tuple[Invariance, ...] = ()
    roles: tuple[RoleRule, ...] = ()

    def _validate(self) -> None:
        object.__setattr__(self, "roles", _sorted_rules(self.roles))
        require(is_record_id(self.concept, "concept"), "AuditedConcept.concept is a concept id")
        require(
            self.asserted is SemanticStatus.VALIDATED_CONCEPT,
            "an audited concept asserts VALIDATED_CONCEPT (the only status that is a claim)",
        )
        axes = [i.axis for i in self.invariant_over]
        require(len(set(axes)) == len(axes), "one invariance per axis")
        require(
            all(a in (AuditAxis.NULL, AuditAxis.REPLACEMENT, AuditAxis.DATASET) for a in axes),
            "concept invariances are over null, replacement or dataset",
        )
        object.__setattr__(
            self, "invariant_over", tuple(sorted(self.invariant_over, key=lambda i: i.axis.value))
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CounterexampleRule(Value):
    """Declared caps; ``None`` means no cap was declared (stated in the report).

    ``max_counterexample_fraction``: per-sample claims, CONTRADICTED samples over
    evaluated samples. ``max_false_positive_rate`` / ``max_false_negative_rate``:
    concepts, the recorded held-out counterexample rates."""

    max_counterexample_fraction: float | None
    max_false_positive_rate: float | None
    max_false_negative_rate: float | None

    def _validate(self) -> None:
        for cap in (
            self.max_counterexample_fraction,
            self.max_false_positive_rate,
            self.max_false_negative_rate,
        ):
            require(cap is None or 0.0 <= cap <= 1.0, "counterexample caps are in [0, 1]")


@record_kind("audit_plan", version=2)
@dataclass(frozen=True, slots=True, kw_only=True)
class AuditPlan(BaseRecord):
    """Everything an audit is about and requires, declared before it runs (plan §28).

    ``samples``: sample ids instance evidence must be about. ``datasets``: concept
    dataset ids concept evidence must be about. ``naive_auroc``: if declared, an
    encoding contradicted by its controls although its AUROC reaches this value is
    flagged (``controls_defeat_encoding``).
    """

    name: str
    checkpoint: str
    declared_model: ModelDeclaration | None
    samples: tuple[str, ...]
    datasets: tuple[str, ...]
    claims: tuple[AuditedClaim, ...]
    requirements: tuple[EvidenceRequirement, ...]
    concepts: tuple[AuditedConcept, ...]
    counterexamples: CounterexampleRule
    naive_auroc: float | None = None

    def _validate(self) -> None:
        require(bool(_NAME_RE.match(self.name)), f"invalid plan name {self.name!r}")
        require(bool(_DIGEST_RE.match(self.checkpoint)), "checkpoint is a sha256: state digest")
        require(len(set(self.samples)) == len(self.samples), "duplicate plan samples")
        object.__setattr__(self, "samples", tuple(sorted(self.samples)))
        require(len(set(self.datasets)) == len(self.datasets), "duplicate plan datasets")
        require(
            all(is_record_id(d, "concept_dataset") for d in self.datasets),
            "datasets are concept_dataset ids",
        )
        object.__setattr__(self, "datasets", tuple(sorted(self.datasets)))
        names = [c.name for c in self.claims]
        require(len(set(names)) == len(names), "duplicate claim names")
        object.__setattr__(self, "claims", tuple(sorted(self.claims, key=lambda c: c.name)))
        req = [r.name for r in self.requirements]
        require(len(set(req)) == len(req), "duplicate requirement names")
        object.__setattr__(
            self, "requirements", tuple(sorted(self.requirements, key=lambda r: r.name))
        )
        by_name = {r.name: r for r in self.requirements}
        for claim in self.claims:
            requirement = by_name.get(claim.requirement)
            require(
                requirement is not None,
                f"claim {claim.name} names unknown requirement {claim.requirement!r}",
            )
            assert requirement is not None
            require(
                not claim.sample_targets
                or {t.sample for t in claim.sample_targets} == set(self.samples),
                f"claim {claim.name}: sample_targets must declare exactly the plan samples",
            )
            require(
                claim.relation not in CAUSAL_RELATIONS
                or bool(requirement.policy.protocols_for(claim.relation)),
                f"claim {claim.name}: requirement {requirement.name} names no protocol for "
                f"causal relation {claim.relation.value}",
            )
        concepts = [c.concept for c in self.concepts]
        require(len(set(concepts)) == len(concepts), "duplicate audited concepts")
        object.__setattr__(self, "concepts", tuple(sorted(self.concepts, key=lambda c: c.concept)))
        require(
            self.naive_auroc is None or 0.5 <= self.naive_auroc <= 1.0,
            "naive_auroc is in [0.5, 1]",
        )
        require(
            bool(self.claims or self.concepts), "an audit plan declares at least one claim/concept"
        )


@register_migration("audit_plan", 1)
def _audit_plan_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    """v1 claims and concepts declared no configuration roles (ADR-048): ``roles=[]``
    keeps that meaning (every configuration undeclared; Phase-7 standings)."""
    return data | {
        "claims": [c | {"roles": []} for c in data["claims"]],
        "concepts": [c | {"roles": []} for c in data["concepts"]],
    }
