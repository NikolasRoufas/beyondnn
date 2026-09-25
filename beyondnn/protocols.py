"""The claim-test protocol registry (ADR-018, ADR-028, ADR-030).

Each registered protocol states which claim relations it can justify. Policies are
validated against this registry (:func:`check_policy`), so no policy can decide a
relation with a protocol that does not justify it.

* ``intervention_threshold`` v1 (Phase 2): NECESSARY_FOR, DECREASES, INCREASES,
  decided by INTERVENTIONAL ``CausalEffect`` records.
* ``attribution_threshold`` v1 (Phase 3): ATTRIBUTED_TO only, decided by ATTRIBUTED
  ``AttributionRecord`` records. It is a statement about an attribution method's
  output, never about causation.

* ``comprehensiveness`` v1 (Phase 5): NECESSARY_FOR, DECREASES, decided by the
  INTERVENTIONAL effect of removing a declared/selected unit set (with optional
  matched random controls).
* ``sufficiency`` v1 (Phase 5): SUFFICIENT_FOR, decided by the INTERVENTIONAL effect
  of retaining only the unit set within its site. This is the only protocol that can
  decide SUFFICIENT_FOR, and only in that declared, site-relative sense.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from beyondnn.schema import AssessmentPolicy, EvidenceRuleError, Relation

__all__ = [
    "ATTRIBUTION_THRESHOLD",
    "COMPREHENSIVENESS",
    "DIAGNOSTIC_PROTOCOLS",
    "INTERVENTION_THRESHOLD",
    "PROTOCOLS",
    "SUFFICIENCY",
    "check_policy",
]

INTERVENTION_THRESHOLD = "intervention_threshold"
ATTRIBUTION_THRESHOLD = "attribution_threshold"
COMPREHENSIVENESS = "comprehensiveness"
SUFFICIENCY = "sufficiency"

#: Which relations each registered protocol can justify. SUFFICIENT_FOR: none.
PROTOCOLS: Mapping[str, frozenset[Relation]] = MappingProxyType(
    {
        INTERVENTION_THRESHOLD: frozenset(
            {Relation.NECESSARY_FOR, Relation.DECREASES, Relation.INCREASES}
        ),
        ATTRIBUTION_THRESHOLD: frozenset({Relation.ATTRIBUTED_TO}),
        COMPREHENSIVENESS: frozenset({Relation.NECESSARY_FOR, Relation.DECREASES}),
        SUFFICIENCY: frozenset({Relation.SUFFICIENT_FOR}),
    }
)

#: Phase-5 diagnostic protocols (ADR-033): they produce ``ProtocolResult`` records and
#: never decide a claim, so they are deliberately not in ``PROTOCOLS``.
DIAGNOSTIC_PROTOCOLS: frozenset[str] = frozenset(
    {
        "removal_curve",
        "retention_curve",
        "stability",
        "counterexample",
        "paired_control",
        "method_agreement",
        "baseline_sensitivity",
        "ig_step_sensitivity",
    }
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
