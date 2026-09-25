"""The claim-test protocol registry (ADR-018, ADR-028, ADR-030).

Each registered protocol states which claim relations it can justify. Policies are
validated against this registry (:func:`check_policy`), so no policy can decide a
relation with a protocol that does not justify it.

* ``intervention_threshold`` v1 (Phase 2): NECESSARY_FOR, DECREASES, INCREASES,
  decided by INTERVENTIONAL ``CausalEffect`` records.
* ``attribution_threshold`` v1 (Phase 3): ATTRIBUTED_TO only, decided by ATTRIBUTED
  ``AttributionRecord`` records. It is a statement about an attribution method's
  output, never about causation.

SUFFICIENT_FOR is justified by no protocol and cannot be assessed.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from beyondnn.schema import AssessmentPolicy, EvidenceRuleError, Relation

__all__ = ["ATTRIBUTION_THRESHOLD", "INTERVENTION_THRESHOLD", "PROTOCOLS", "check_policy"]

INTERVENTION_THRESHOLD = "intervention_threshold"
ATTRIBUTION_THRESHOLD = "attribution_threshold"

#: Which relations each registered protocol can justify. SUFFICIENT_FOR: none.
PROTOCOLS: Mapping[str, frozenset[Relation]] = MappingProxyType(
    {
        INTERVENTION_THRESHOLD: frozenset(
            {Relation.NECESSARY_FOR, Relation.DECREASES, Relation.INCREASES}
        ),
        ATTRIBUTION_THRESHOLD: frozenset({Relation.ATTRIBUTED_TO}),
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
