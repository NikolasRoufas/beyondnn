"""Instance vs finite-sample vs population causal quantities (ADR-013)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from beyondnn.schema import (
    ClaimTestResult,
    Estimand,
    EstimandScope,
    EvidenceRef,
    EvidenceRuleError,
    EvidenceStatus,
    Outcome,
    Relation,
    SchemaError,
)

POP = Estimand.population_of("mnist_test_distribution", "mean")
FINITE = Estimand.finite_sample("eval-512", 512, "mean")


def _decide(
    mk: SimpleNamespace, claim_estimand: Estimand, *evidence: EvidenceRef
) -> ClaimTestResult:
    return ClaimTestResult.for_claim(
        mk.claim(estimand=claim_estimand),
        mk.spec(),
        outcome=Outcome.SUPPORTS,
        evidence=evidence,
        provenance_id=mk.PROV,
    )


# --------------------------------------------------------------- estimand validation


def test_constructors_set_the_scope() -> None:
    assert Estimand.instance("x0").scope is EstimandScope.INSTANCE
    assert FINITE.scope is EstimandScope.FINITE_SAMPLE
    assert POP.scope is EstimandScope.POPULATION


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scope": EstimandScope.INSTANCE, "n": 1},
        {"scope": EstimandScope.INSTANCE, "sample_id": "x", "n": 2},
        {"scope": EstimandScope.INSTANCE, "sample_id": "x", "n": 1, "aggregation": "mean"},
        {"scope": EstimandScope.FINITE_SAMPLE, "sample_id": "s", "n": 0, "aggregation": "mean"},
        {"scope": EstimandScope.FINITE_SAMPLE, "sample_id": "s", "n": 3},
        {"scope": EstimandScope.FINITE_SAMPLE, "n": 3, "aggregation": "mean"},
        {
            "scope": EstimandScope.FINITE_SAMPLE,
            "sample_id": "s",
            "n": 3,
            "aggregation": "mean",
            "population": "p",
        },
        {"scope": EstimandScope.POPULATION, "aggregation": "mean"},
        {"scope": EstimandScope.POPULATION, "population": "p"},
        {"scope": EstimandScope.POPULATION, "population": "p", "aggregation": "mean", "n": 0},
        {"scope": EstimandScope.INSTANCE, "sample_id": "has space", "n": 1},
    ],
)
def test_invalid_estimands(kwargs: dict[str, Any]) -> None:
    with pytest.raises(SchemaError):
        Estimand(**kwargs)


def test_large_samples_are_never_promoted_to_population() -> None:
    huge = Estimand.finite_sample("everything-we-had", 10**9, "mean")
    assert huge.scope is EstimandScope.FINITE_SAMPLE
    assert not POP.covers(huge)


def test_finite_sample_aggregate_of_exact_effects_is_interventional() -> None:
    ref = EvidenceRef(
        record_id="causal_effect:" + "c" * 32,
        kind="causal_effect",
        status=EvidenceStatus.INTERVENTIONAL,
        estimand=FINITE,
    )
    assert ref.status is EvidenceStatus.INTERVENTIONAL


def test_population_quantities_cannot_be_interventional() -> None:
    with pytest.raises(EvidenceRuleError, match="POPULATION scope"):
        EvidenceRef(
            record_id="causal_effect:" + "c" * 32,
            kind="causal_effect",
            status=EvidenceStatus.INTERVENTIONAL,
            estimand=POP,
        )


def test_status_follows_estimand_scope_for_effect_records(
    mk: SimpleNamespace, fake_effect_kind: Any
) -> None:
    finite = fake_effect_kind(estimand=FINITE, effect=-0.3, provenance_id=mk.PROV)
    pop = fake_effect_kind(estimand=POP, effect=-0.3, provenance_id=mk.PROV)
    assert finite.status is EvidenceStatus.INTERVENTIONAL
    assert pop.status is EvidenceStatus.ESTIMATED_CAUSAL
    assert EvidenceRef.to(finite).estimand == FINITE
    assert EvidenceRef.to(pop).status is EvidenceStatus.ESTIMATED_CAUSAL


# --------------------------------------------------------------- matching claims and evidence


def test_finite_sample_evidence_cannot_decide_a_population_claim(mk: SimpleNamespace) -> None:
    finite_ref = mk.causal_ref(estimand=FINITE)
    with pytest.raises(EvidenceRuleError, match="population"):
        _decide(mk, POP, finite_ref)


def test_population_evidence_decides_a_population_claim(mk: SimpleNamespace) -> None:
    pop_ref = mk.causal_ref(estimand=POP, status=EvidenceStatus.ESTIMATED_CAUSAL)
    assert _decide(mk, POP, pop_ref).outcome is Outcome.SUPPORTS


def test_population_evidence_must_be_about_the_same_population(mk: SimpleNamespace) -> None:
    other = Estimand.population_of("cifar_test_distribution", "mean")
    ref = mk.causal_ref(estimand=other, status=EvidenceStatus.ESTIMATED_CAUSAL)
    with pytest.raises(EvidenceRuleError):
        _decide(mk, POP, ref)
    median = Estimand.population_of("mnist_test_distribution", "median")
    ref = mk.causal_ref(estimand=median, status=EvidenceStatus.ESTIMATED_CAUSAL)
    with pytest.raises(EvidenceRuleError):
        _decide(mk, POP, ref)


def test_finite_sample_claim_needs_evidence_on_the_same_sample(mk: SimpleNamespace) -> None:
    assert _decide(mk, FINITE, mk.causal_ref(estimand=FINITE)).outcome is Outcome.SUPPORTS
    for other in (
        Estimand.finite_sample("eval-512b", 512, "mean"),
        Estimand.finite_sample("eval-512", 511, "mean"),
        Estimand.finite_sample("eval-512", 512, "median"),
        Estimand.instance("eval-512"),
    ):
        with pytest.raises(EvidenceRuleError):
            _decide(mk, FINITE, mk.causal_ref(estimand=other))


def test_population_evidence_cannot_decide_a_finite_sample_claim(mk: SimpleNamespace) -> None:
    pop_ref = mk.causal_ref(estimand=POP, status=EvidenceStatus.ESTIMATED_CAUSAL)
    with pytest.raises(EvidenceRuleError):
        _decide(mk, FINITE, pop_ref)


def test_instance_claim_needs_evidence_on_the_same_instance(mk: SimpleNamespace) -> None:
    x0 = Estimand.instance("x0")
    assert _decide(mk, x0, mk.causal_ref(estimand=x0)).outcome is Outcome.SUPPORTS
    with pytest.raises(EvidenceRuleError):
        _decide(mk, x0, mk.causal_ref(estimand=Estimand.instance("x1")))


def test_approximate_instance_estimates_are_allowed_as_estimated_causal(
    mk: SimpleNamespace,
) -> None:
    x0 = Estimand.instance("x0")
    approx = mk.causal_ref(estimand=x0, status=EvidenceStatus.ESTIMATED_CAUSAL)
    assert _decide(mk, x0, approx).evidence[0].status is EvidenceStatus.ESTIMATED_CAUSAL


def test_non_causal_claims_do_not_check_estimands(mk: SimpleNamespace) -> None:
    claim = mk.claim(relation=Relation.ENCODES, estimand=POP)
    spec = mk.spec(protocol="detection", relations=(Relation.ENCODES,))
    result = ClaimTestResult.for_claim(
        claim,
        spec,
        outcome=Outcome.SUPPORTS,
        evidence=(EvidenceRef.to(mk.activation()),),
        provenance_id=mk.PROV,
    )
    assert result.outcome is Outcome.SUPPORTS
