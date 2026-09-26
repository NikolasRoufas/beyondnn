"""Claims, declared tests, test results, and derived assessments (ADR-012, ADR-013).

Pipeline: Measurement -> Claim -> Test -> Evidence -> Assessment -> presentation.

* A :class:`Claim` is an immutable proposition. It has no status field.
* A :class:`ClaimTestSpec` declares its protocol, parameters, and decision
  criteria *before* it runs; its id (a content hash) changes if any of them change.
* A :class:`ClaimTestResult` records one test's outcome with the evidence it cites.
  Decisive outcomes on causal relations require causal evidence about the claim's
  estimand.
* An :class:`Assessment` is derived from results under an explicit
  :class:`AssessmentPolicy`; a stored verdict that does not follow from its
  results is rejected. There is no confidence number (ADR-007).

All types are container-neutral: they reference other records by id and carry
the attributes needed for local checks. Containers verify references with
:func:`~beyondnn.schema.base.verify_ref`.

No test is executed here: runners and protocols arrive in Phase 2.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, ClassVar

from ._canonical import EMPTY_JSON, JsonMap
from ._types import Value, require
from .base import (
    BaseRecord,
    EvidenceRef,
    RecordRef,
    is_record_id,
    record_kind,
    register_migration,
)
from .errors import EvidenceRuleError
from .status import CAUSAL_EVIDENCE_STATUSES, CAUSAL_RELATIONS, Outcome, Relation, Verdict
from .values import ClaimSource, Estimand, Subject, TargetSpec

__all__ = [
    "Assessment",
    "AssessmentPolicy",
    "Claim",
    "ClaimRef",
    "ClaimTestResult",
    "ClaimTestSpec",
    "PolicyRequirement",
    "ResultRef",
    "SpecRef",
    "derive_verdict",
]

_PROTOCOL_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_DECISIVE = frozenset({Outcome.SUPPORTS, Outcome.CONTRADICTS})


def _sorted_relations(relations: Iterable[Relation]) -> tuple[Relation, ...]:
    return tuple(sorted(relations, key=lambda r: r.value))


# --------------------------------------------------------------------------- claim


@record_kind("claim", version=3)
@dataclass(frozen=True, slots=True, kw_only=True)
class Claim(BaseRecord):
    """A testable proposition about the model's computation.

    ``statement`` is descriptive text for humans and is never evidence. The claim's
    standing is not stored here: it is an :class:`Assessment` derived from results.

    Record identity is not semantic equivalence (ADR-016): the id covers every
    field, including the wording of ``statement`` and ``provenance_id``, so two
    formally equivalent claims can have different ids. Never use ``id`` to decide
    whether two claims mean the same thing; that must be based on the formal
    structure (subject, relation, target, estimand).
    """

    statement: str
    relation: Relation
    subject: Subject
    target: TargetSpec
    estimand: Estimand
    source: ClaimSource

    def _validate(self) -> None:
        require(bool(self.statement.strip()), "Claim.statement must be non-empty")


@dataclass(frozen=True, slots=True, kw_only=True)
class ClaimRef(Value):
    """Reference to a claim with the attributes needed to check results locally."""

    claim_id: str
    relation: Relation
    estimand: Estimand

    def _validate(self) -> None:
        require(is_record_id(self.claim_id, "claim"), f"malformed claim id {self.claim_id!r}")

    @classmethod
    def to(cls, claim: BaseRecord) -> ClaimRef:
        if not isinstance(claim, Claim):
            raise TypeError("ClaimRef.to() requires a Claim")
        return cls(claim_id=claim.id, relation=claim.relation, estimand=claim.estimand)


# ---------------------------------------------------------------------- test spec


@record_kind("claim_test_spec")
@dataclass(frozen=True, slots=True, kw_only=True)
class ClaimTestSpec(BaseRecord):
    """A test protocol with its parameters and decision criteria, declared before running.

    ``criteria`` must be non-empty and may not share keys with ``params``. The
    spec's id is a hash of all fields; :attr:`criteria_digest` fingerprints the
    criteria alone. No thresholds are defined by BeyondNN in schema 0.1.
    """

    protocol: str
    protocol_version: int
    applicable_relations: tuple[Relation, ...]
    criteria: JsonMap
    params: JsonMap = EMPTY_JSON

    def _validate(self) -> None:
        require(bool(_PROTOCOL_RE.match(self.protocol)), f"invalid protocol {self.protocol!r}")
        require(self.protocol_version >= 1, "ClaimTestSpec.protocol_version must be >= 1")
        require(len(self.applicable_relations) > 0, "applicable_relations must be non-empty")
        require(
            len(set(self.applicable_relations)) == len(self.applicable_relations),
            "applicable_relations has duplicates",
        )
        object.__setattr__(
            self, "applicable_relations", _sorted_relations(self.applicable_relations)
        )
        require(len(self.criteria) > 0, "ClaimTestSpec.criteria must declare decision criteria")
        overlap = sorted(set(self.criteria) & set(self.params))
        require(not overlap, f"keys appear in both params and criteria: {overlap}")

    @property
    def criteria_digest(self) -> str:
        """``"sha256:<hex>"`` of the canonical JSON of ``criteria``."""
        return "sha256:" + self.criteria.digest()


@dataclass(frozen=True, slots=True, kw_only=True)
class SpecRef(Value):
    spec_id: str
    protocol: str
    applicable_relations: tuple[Relation, ...]

    def _validate(self) -> None:
        require(is_record_id(self.spec_id, "claim_test_spec"), "malformed spec id")
        object.__setattr__(
            self, "applicable_relations", _sorted_relations(self.applicable_relations)
        )

    @classmethod
    def to(cls, spec: BaseRecord) -> SpecRef:
        if not isinstance(spec, ClaimTestSpec):
            raise TypeError("SpecRef.to() requires a ClaimTestSpec")
        return cls(
            spec_id=spec.id,
            protocol=spec.protocol,
            applicable_relations=spec.applicable_relations,
        )


# -------------------------------------------------------------------- test result


@record_kind("claim_test_result")
@dataclass(frozen=True, slots=True, kw_only=True)
class ClaimTestResult(BaseRecord):
    """The outcome of running one spec against one claim.

    Invariants:

    * ``ERRORED`` if and only if ``error`` is set.
    * If the spec does not apply to the claim's relation, the outcome can only be
      ``NOT_APPLICABLE`` or ``ERRORED``.
    * ``SUPPORTS``/``CONTRADICTS`` require at least one evidence reference.
    * For causal relations, ``SUPPORTS``/``CONTRADICTS`` require causal evidence
      (INTERVENTIONAL or ESTIMATED_CAUSAL) whose estimand is the claim's estimand:
      same instance, same finite sample, or same population. Finite-sample
      evidence can therefore never decide a population claim.
    """

    REQUIRES_PROVENANCE: ClassVar[bool] = True

    claim: ClaimRef
    spec: SpecRef
    outcome: Outcome
    evidence: tuple[EvidenceRef, ...] = ()
    statistics: JsonMap = EMPTY_JSON
    error: str | None = None

    def _validate(self) -> None:
        ids = [e.record_id for e in self.evidence]
        require(len(set(ids)) == len(ids), "ClaimTestResult.evidence has duplicate ids")
        object.__setattr__(
            self, "evidence", tuple(sorted(self.evidence, key=lambda e: e.record_id))
        )
        if self.outcome is Outcome.ERRORED:
            require(bool(self.error), "an ERRORED result must describe the error")
        else:
            require(self.error is None, "only ERRORED results carry an error")

        relation = self.claim.relation
        if relation not in self.spec.applicable_relations and self.outcome not in (
            Outcome.NOT_APPLICABLE,
            Outcome.ERRORED,
        ):
            raise EvidenceRuleError(
                f"protocol {self.spec.protocol!r} does not apply to {relation.value}; "
                f"outcome must be NOT_APPLICABLE, not {self.outcome.value}"
            )
        if self.outcome not in _DECISIVE:
            return
        if not self.evidence:
            raise EvidenceRuleError(f"a {self.outcome.value} result must cite evidence")
        if relation in CAUSAL_RELATIONS:
            target = self.claim.estimand
            if not any(
                e.status in CAUSAL_EVIDENCE_STATUSES
                and e.estimand is not None
                and target.covers(e.estimand)
                for e in self.evidence
            ):
                raise EvidenceRuleError(
                    f"a {self.outcome.value} result for causal relation {relation.value} "
                    "requires INTERVENTIONAL or ESTIMATED_CAUSAL evidence about the claim's "
                    f"estimand ({target.scope.value}); none of the cited evidence qualifies"
                )

    @classmethod
    def for_claim(
        cls,
        claim: Claim,
        spec: ClaimTestSpec,
        *,
        outcome: Outcome,
        provenance_id: str,
        evidence: Iterable[BaseRecord | EvidenceRef] = (),
        statistics: JsonMap | None = None,
        error: str | None = None,
        derived_from: tuple[RecordRef, ...] = (),
    ) -> ClaimTestResult:
        """Build a result with refs taken from the actual claim, spec, and evidence."""
        refs = tuple(e if isinstance(e, EvidenceRef) else EvidenceRef.to(e) for e in evidence)
        return cls(
            claim=ClaimRef.to(claim),
            spec=SpecRef.to(spec),
            outcome=outcome,
            evidence=refs,
            statistics=statistics if statistics is not None else EMPTY_JSON,
            error=error,
            provenance_id=provenance_id,
            derived_from=derived_from,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ResultRef(Value):
    result_id: str
    protocol: str
    outcome: Outcome

    def _validate(self) -> None:
        require(is_record_id(self.result_id, "claim_test_result"), "malformed result id")

    @classmethod
    def to(cls, result: BaseRecord) -> ResultRef:
        if not isinstance(result, ClaimTestResult):
            raise TypeError("ResultRef.to() requires a ClaimTestResult")
        return cls(result_id=result.id, protocol=result.spec.protocol, outcome=result.outcome)


# --------------------------------------------------------------------- assessment


@dataclass(frozen=True, slots=True, kw_only=True)
class PolicyRequirement(Value):
    """Protocols that must each have a SUPPORTS result before ``relation`` is SUPPORTED."""

    relation: Relation
    protocols: tuple[str, ...]

    def _validate(self) -> None:
        require(len(self.protocols) > 0, "PolicyRequirement.protocols must be non-empty")
        require(all(_PROTOCOL_RE.match(p) for p in self.protocols), "invalid protocol name")
        require(len(set(self.protocols)) == len(self.protocols), "duplicate protocols")
        object.__setattr__(self, "protocols", tuple(sorted(self.protocols)))


@dataclass(frozen=True, slots=True, kw_only=True)
class AssessmentPolicy(Value):
    """A named, versioned rule for turning results into a verdict.

    A causal relation can only be assessed under a policy that names at least one
    required protocol for it; there is no implicit default. BeyondNN ships no
    policy in schema 0.1.
    """

    name: str
    version: int
    requirements: tuple[PolicyRequirement, ...] = ()

    def _validate(self) -> None:
        require(bool(_PROTOCOL_RE.match(self.name)), f"invalid policy name {self.name!r}")
        require(self.version >= 1, "AssessmentPolicy.version must be >= 1")
        relations = [r.relation for r in self.requirements]
        require(len(set(relations)) == len(relations), "one requirement per relation")
        object.__setattr__(
            self, "requirements", tuple(sorted(self.requirements, key=lambda r: r.relation.value))
        )

    def protocols_for(self, relation: Relation) -> tuple[str, ...]:
        for requirement in self.requirements:
            if requirement.relation is relation:
                return requirement.protocols
        return ()


def derive_verdict(
    relation: Relation, policy: AssessmentPolicy, results: Iterable[ResultRef]
) -> tuple[Verdict, tuple[str, ...]]:
    """Derive ``(verdict, required_but_missing)`` from result outcomes.

    * No results other than NOT_APPLICABLE: ``UNTESTED``.
    * SUPPORTS and CONTRADICTS both present: ``MIXED``.
    * CONTRADICTS only: ``CONTRADICTED``.
    * Every protocol the policy requires has a SUPPORTS result, at least one
      SUPPORTS exists, and nothing contradicts: ``SUPPORTED``.
    * Otherwise (including ERRORED and INCONCLUSIVE outcomes): ``INCONCLUSIVE``.

    Raises :class:`EvidenceRuleError` for a causal relation the policy does not cover.
    """
    results = tuple(results)
    required = policy.protocols_for(relation)
    if relation in CAUSAL_RELATIONS and not required:
        raise EvidenceRuleError(
            f"policy {policy.name}/v{policy.version} names no protocol for causal relation "
            f"{relation.value}; a causal claim cannot be assessed without one"
        )
    supported = {r.protocol for r in results if r.outcome is Outcome.SUPPORTS}
    missing = tuple(p for p in required if p not in supported)
    considered = [r for r in results if r.outcome is not Outcome.NOT_APPLICABLE]
    contradicted = any(r.outcome is Outcome.CONTRADICTS for r in results)
    if not considered:
        verdict = Verdict.UNTESTED
    elif supported and contradicted:
        verdict = Verdict.MIXED
    elif contradicted:
        verdict = Verdict.CONTRADICTED
    elif supported and not missing:
        verdict = Verdict.SUPPORTED
    else:
        verdict = Verdict.INCONCLUSIVE
    return verdict, missing


@record_kind("assessment")
@dataclass(frozen=True, slots=True, kw_only=True)
class Assessment(BaseRecord):
    """A claim's derived standing under a policy.

    Construct with :meth:`derive`. Direct construction is allowed (decoding needs
    it) but the stored ``verdict`` and ``required_but_missing`` must equal what
    :func:`derive_verdict` computes from ``results``, or construction fails.
    """

    claim: ClaimRef
    policy: AssessmentPolicy
    results: tuple[ResultRef, ...]
    verdict: Verdict
    required_but_missing: tuple[str, ...] = ()

    def _validate(self) -> None:
        ids = [r.result_id for r in self.results]
        require(len(set(ids)) == len(ids), "Assessment.results has duplicate ids")
        object.__setattr__(self, "results", tuple(sorted(self.results, key=lambda r: r.result_id)))
        verdict, missing = derive_verdict(self.claim.relation, self.policy, self.results)
        if (verdict, missing) != (self.verdict, self.required_but_missing):
            raise EvidenceRuleError(
                f"verdict {self.verdict.value} (missing {list(self.required_but_missing)}) "
                f"does not follow from the results under {self.policy.name}/v"
                f"{self.policy.version}; derived {verdict.value} (missing {list(missing)})"
            )

    @classmethod
    def derive(
        cls,
        claim: Claim,
        results: Iterable[ClaimTestResult],
        policy: AssessmentPolicy,
        *,
        provenance_id: str | None = None,
    ) -> Assessment:
        """Assess ``claim`` from ``results`` (which must all be about ``claim``)."""
        claim_ref = ClaimRef.to(claim)
        results = tuple(results)
        for result in results:
            if result.claim != claim_ref:
                raise EvidenceRuleError(f"result {result.id} is not about claim {claim.id}")
        refs = tuple(ResultRef.to(r) for r in results)
        verdict, missing = derive_verdict(claim.relation, policy, refs)
        return cls(
            claim=claim_ref,
            policy=policy,
            results=refs,
            verdict=verdict,
            required_but_missing=missing,
            provenance_id=provenance_id,
            derived_from=tuple(RecordRef.to(r) for r in results),
        )


@register_migration("claim", 2)
def _claim_v2_to_v3(data: dict[str, Any]) -> dict[str, Any]:
    """v2 subjects were sites/units only: ``feature=None`` keeps that meaning (ADR-040)."""
    subject = data.get("subject")
    if not isinstance(subject, dict) or "feature" in subject:
        raise ValueError("a claim v2 payload has a subject without feature")
    return data | {"subject": subject | {"feature": None}}


@register_migration("claim", 1)
def _claim_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    """v1 subjects had last-axis units: ``unit_axes=None`` keeps that meaning (ADR-034)."""
    subject = data.get("subject")
    if not isinstance(subject, dict) or "unit_axes" in subject:
        raise ValueError("a claim v1 payload has a subject without unit_axes")
    return data | {"subject": subject | {"unit_axes": None}}
