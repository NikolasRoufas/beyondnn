"""Intervention-backed claim tests (Phase 2; ADR-018 protocol registry, ADR-028).

One protocol, ``intervention_threshold`` v1: a claim about site S, metric M, and
estimand E is decided by the ``CausalEffect`` of an intervention at S measured on M
over exactly E, against a threshold declared *before* running (``min_effect``):

* NECESSARY_FOR / DECREASES: SUPPORTS iff ``effect <= -min_effect``, else CONTRADICTS;
* INCREASES: SUPPORTS iff ``effect >= min_effect``, else CONTRADICTS;
* any mismatch of site/leaf, metric, estimand, or operation: NOT_APPLICABLE.

This supports only precisely scoped statements such as "under zero ablation of S on
input X, M decreased by at least T". It says nothing about other inputs, other
interventions, or redundant paths. SUFFICIENT_FOR is not justified by this protocol
and no protocol justifies it yet, so sufficiency claims cannot be assessed.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from beyondnn.schema import (
    AssessmentPolicy,
    CausalEffect,
    Claim,
    ClaimTestResult,
    ClaimTestSpec,
    EvidenceRuleError,
    InterventionOperation,
    InterventionRecord,
    JsonMap,
    Outcome,
    PolicyRequirement,
    Relation,
)

__all__ = [
    "INTERVENTION_POLICY",
    "INTERVENTION_THRESHOLD",
    "PROTOCOLS",
    "check_policy",
    "evaluate_claim",
    "threshold_spec",
]

INTERVENTION_THRESHOLD = "intervention_threshold"
_DECREASING = frozenset({Relation.NECESSARY_FOR, Relation.DECREASES})

#: Which relations each registered protocol can justify. SUFFICIENT_FOR: none.
PROTOCOLS: Mapping[str, frozenset[Relation]] = MappingProxyType(
    {
        INTERVENTION_THRESHOLD: frozenset(
            {Relation.NECESSARY_FOR, Relation.DECREASES, Relation.INCREASES}
        )
    }
)

#: The Phase-2 assessment policy: the causal relations above need a SUPPORTS result
#: from ``intervention_threshold``. SUFFICIENT_FOR is deliberately absent.
INTERVENTION_POLICY = AssessmentPolicy(
    name="intervention_threshold_policy",
    version=1,
    requirements=tuple(
        PolicyRequirement(relation=r, protocols=(INTERVENTION_THRESHOLD,))
        for r in (Relation.NECESSARY_FOR, Relation.DECREASES, Relation.INCREASES)
    ),
)


def check_policy(policy: AssessmentPolicy) -> None:
    """Raise unless every protocol a policy requires is registered and justifies its relation."""
    for requirement in policy.requirements:
        for protocol in requirement.protocols:
            justified = PROTOCOLS.get(protocol)
            if justified is None or requirement.relation not in justified:
                raise EvidenceRuleError(
                    f"policy {policy.name}/v{policy.version}: protocol {protocol!r} does not "
                    f"justify {requirement.relation.value}"
                )


def threshold_spec(
    *,
    operation: InterventionOperation,
    min_effect: float,
    relations: tuple[Relation, ...] = (
        Relation.NECESSARY_FOR,
        Relation.DECREASES,
        Relation.INCREASES,
    ),
) -> ClaimTestSpec:
    """Declare an ``intervention_threshold`` test before running it."""
    if not min_effect > 0:
        raise ValueError("min_effect must be > 0")
    if any(r not in PROTOCOLS[INTERVENTION_THRESHOLD] for r in relations):
        raise EvidenceRuleError("intervention_threshold does not justify all of these relations")
    return ClaimTestSpec(
        protocol=INTERVENTION_THRESHOLD,
        protocol_version=1,
        applicable_relations=relations,
        criteria=JsonMap({"min_effect": float(min_effect)}),
        params=JsonMap({"operation": operation.value}),
    )


def evaluate_claim(
    claim: Claim,
    spec: ClaimTestSpec,
    effect: CausalEffect,
    intervention: InterventionRecord,
    *,
    provenance_id: str,
) -> ClaimTestResult:
    """Decide ``claim`` from ``effect`` under ``spec`` (see module docstring)."""
    if spec.protocol != INTERVENTION_THRESHOLD or spec.protocol_version != 1:
        raise ValueError("evaluate_claim implements intervention_threshold v1 only")
    threshold = spec.criteria.get("min_effect")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not threshold > 0:
        raise ValueError("spec criteria must declare min_effect > 0")
    if intervention.id not in {r.record_id for r in effect.interventions}:
        raise EvidenceRuleError("the effect does not belong to this intervention")

    applicable = (
        claim.relation in spec.applicable_relations
        and claim.relation in PROTOCOLS[INTERVENTION_THRESHOLD]
        and spec.params.get("operation") == intervention.operation.value
        and claim.subject.site == intervention.site
        and claim.subject.units is None
        and claim.target == effect.metric.target()
        and claim.estimand == effect.estimand
    )
    if not applicable:
        return ClaimTestResult.for_claim(
            claim, spec, outcome=Outcome.NOT_APPLICABLE, provenance_id=provenance_id
        )
    t = float(threshold)
    holds = effect.effect <= -t if claim.relation in _DECREASING else effect.effect >= t
    return ClaimTestResult.for_claim(
        claim,
        spec,
        outcome=Outcome.SUPPORTS if holds else Outcome.CONTRADICTS,
        evidence=(effect,),
        statistics=JsonMap({"effect": effect.effect, "min_effect": t}),
        provenance_id=provenance_id,
    )
