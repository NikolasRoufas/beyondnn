"""Claim-deciding faithfulness protocols: ``comprehensiveness/v1``, ``sufficiency/v1``.

Pure functions of existing records, so the same code decides a result when a test
runs and re-derives it when evidence is composed (ADR-031/033).

* comprehensiveness: ``drop = F(x) - F(remove(S))`` (the INTERVENTIONAL effect's
  negation). SUPPORTS iff ``drop >= min_drop`` and, if declared,
  ``fraction_below >= min_fraction_below`` (the share of matched random removals with
  a smaller drop). Relations: NECESSARY_FOR, DECREASES.
* sufficiency: ``drop = F(x) - F(retain(S))``. SUPPORTS iff ``drop <= max_drop`` and,
  if declared, ``fraction_above >= min_fraction_above`` (the share of matched random
  retained sets with a larger drop). Relation: SUFFICIENT_FOR, in the declared,
  site-relative sense only.
* A perturbation that changed nothing (replacement equal to the original at every
  replaced position) gives INCONCLUSIVE: the zero effect is real but uninformative.
* Any mismatch of relation, site, units, mode, call, target, sample, controls, or
  replacement gives NOT_APPLICABLE.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from beyondnn.protocols import COMPREHENSIVENESS, PROTOCOLS, SUFFICIENCY
from beyondnn.schema import (
    AssessmentPolicy,
    CausalEffect,
    Claim,
    ClaimTestResult,
    ClaimTestSpec,
    InterventionRecord,
    JsonMap,
    Outcome,
    PolicyRequirement,
    Relation,
)

from .stats import control_fractions, mc_p_value

__all__ = ["COMPREHENSIVENESS_POLICY", "DECIDABLE", "SUFFICIENCY_POLICY", "evaluate"]

#: Independent of the registry (defence in depth, as for intervention_threshold).
DECIDABLE: Mapping[str, frozenset[Relation]] = {
    COMPREHENSIVENESS: frozenset({Relation.NECESSARY_FOR, Relation.DECREASES}),
    SUFFICIENCY: frozenset({Relation.SUFFICIENT_FOR}),
}


#: NECESSARY_FOR / DECREASES need a SUPPORTS result from comprehensiveness/v1.
COMPREHENSIVENESS_POLICY = AssessmentPolicy(
    name="comprehensiveness_policy",
    version=1,
    requirements=tuple(
        PolicyRequirement(relation=r, protocols=(COMPREHENSIVENESS,))
        for r in (Relation.NECESSARY_FOR, Relation.DECREASES)
    ),
)

#: SUFFICIENT_FOR needs a SUPPORTS result from sufficiency/v1 (site-relative sense).
SUFFICIENCY_POLICY = AssessmentPolicy(
    name="sufficiency_policy",
    version=1,
    requirements=(PolicyRequirement(relation=Relation.SUFFICIENT_FOR, protocols=(SUFFICIENCY,)),),
)


def _drop(effect: CausalEffect) -> float:
    return effect.baseline_value - effect.intervention_value


def evaluate(
    claim: Claim,
    spec: ClaimTestSpec,
    primary: CausalEffect,
    controls: Sequence[CausalEffect],
    interventions: Mapping[str, InterventionRecord],
    *,
    no_op: bool,
    provenance_id: str,
) -> ClaimTestResult:
    """Decide ``claim`` from the primary perturbation's effect and its controls."""
    protocol = spec.protocol
    if protocol not in DECIDABLE or spec.protocol_version != 1:
        raise ValueError(f"not a faithfulness claim protocol: {protocol}/v{spec.protocol_version}")
    retain = protocol == SUFFICIENCY
    params = spec.params
    main = interventions[primary.interventions[0].record_id]
    k = params.get("k")
    control_records = [interventions[c.interventions[0].record_id] for c in controls]
    declared_controls = params.get("controls")
    applicable = (
        claim.relation in spec.applicable_relations
        and claim.relation in PROTOCOLS[protocol]
        and claim.relation in DECIDABLE[protocol]
        and claim.subject.site == main.site
        and claim.subject.units == main.units
        and main.retain is retain
        and main.call_index == params.get("call_index")
        and main.units is not None
        and len(main.units) == k
        and claim.target == primary.metric.target()
        and claim.estimand == primary.estimand
        and all(c.metric == primary.metric and c.estimand == primary.estimand for c in controls)
        and all(
            r.site == main.site
            and r.call_index == main.call_index
            and r.retain is retain
            and r.units is not None
            and len(r.units) == k
            and (r.operation, r.constant, r.value) == (main.operation, main.constant, main.value)
            for r in control_records
        )
        and (
            (declared_controls is None and not controls)
            or (
                isinstance(declared_controls, Mapping)
                and declared_controls.get("n") == len(controls)
            )
        )
    )
    if not applicable:
        return ClaimTestResult.for_claim(
            claim, spec, outcome=Outcome.NOT_APPLICABLE, provenance_id=provenance_id
        )
    drop = _drop(primary)
    stats: dict[str, Any] = {
        "drop": drop,
        "baseline_value": primary.baseline_value,
        "perturbed_value": primary.intervention_value,
        "no_op": no_op,
        "selected_effect": primary.id,
        "control_effects": [c.id for c in controls],
    }
    criteria = spec.criteria
    if controls:
        control_drops = [_drop(c) for c in controls]
        fractions = control_fractions(drop, control_drops)
        stats |= {"control_drops": control_drops, "n_controls": len(controls)} | fractions
        # one-sided: the probability that a matched random set does at least as well
        stats["mc_p_value"] = (
            mc_p_value(drop, control_drops)
            if not retain
            else mc_p_value(-drop, [-d for d in control_drops])
        )
    if no_op:
        outcome = Outcome.INCONCLUSIVE
    elif not retain:
        holds = drop >= float(criteria["min_drop"])  # type: ignore[arg-type]
        if "min_fraction_below" in criteria:
            holds = holds and stats["fraction_below"] >= float(criteria["min_fraction_below"])  # type: ignore[arg-type]
        outcome = Outcome.SUPPORTS if holds else Outcome.CONTRADICTS
    else:
        holds = drop <= float(criteria["max_drop"])  # type: ignore[arg-type]
        if "min_fraction_above" in criteria:
            holds = holds and stats["fraction_above"] >= float(criteria["min_fraction_above"])  # type: ignore[arg-type]
        outcome = Outcome.SUPPORTS if holds else Outcome.CONTRADICTS
    return ClaimTestResult.for_claim(
        claim,
        spec,
        outcome=outcome,
        evidence=(primary, *controls),
        statistics=JsonMap(stats),
        provenance_id=provenance_id,
    )
