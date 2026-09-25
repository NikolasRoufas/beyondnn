"""Claim / test / result / assessment invariants (ADR-012)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from operator import attrgetter
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from beyondnn.schema import (
    CAUSAL_RELATIONS,
    Assessment,
    AssessmentPolicy,
    Claim,
    ClaimRef,
    ClaimTestResult,
    EvidenceRef,
    EvidenceRuleError,
    EvidenceStatus,
    JsonMap,
    Outcome,
    PolicyRequirement,
    Relation,
    ResultRef,
    SchemaError,
    SpecRef,
    Verdict,
    derive_verdict,
    verify_ref,
)
from beyondnn.schema.base import BaseRecord

NEC = Relation.NECESSARY_FOR
POLICY = AssessmentPolicy(
    name="test_policy",
    version=1,
    requirements=(
        PolicyRequirement(relation=NEC, protocols=("ablation_necessity", "random_baseline")),
        PolicyRequirement(relation=Relation.SUFFICIENT_FOR, protocols=("sufficiency_patch",)),
    ),
)


def _result(
    mk: SimpleNamespace,
    outcome: Outcome,
    *,
    claim: Claim | None = None,
    spec: Any = None,
    evidence: tuple[Any, ...] | None = None,
    error: str | None = None,
) -> ClaimTestResult:
    claim = claim or mk.claim()
    spec = spec or mk.spec()
    if evidence is None:
        evidence = (mk.causal_ref(),) if outcome in (Outcome.SUPPORTS, Outcome.CONTRADICTS) else ()
    return ClaimTestResult.for_claim(
        claim, spec, outcome=outcome, evidence=evidence, error=error, provenance_id=mk.PROV
    )


# --------------------------------------------------------------- 5: no truth status on claims


def test_claim_has_no_status_or_verdict_field(mk: SimpleNamespace) -> None:
    names = {f.name for f in dataclasses.fields(Claim)}
    assert names.isdisjoint({"status", "verdict", "supported", "truth", "confidence"})
    claim = mk.claim()
    assert claim.status is None
    # Frozen + slots: assignment fails (TypeError on some CPython versions).
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
        claim.verdict = Verdict.SUPPORTED
    with pytest.raises(TypeError):
        Claim(verdict=Verdict.SUPPORTED)  # type: ignore[call-arg]


def test_claim_statement_is_required(mk: SimpleNamespace) -> None:
    with pytest.raises(SchemaError):
        mk.claim(statement="   ")


def test_rewording_a_claim_changes_its_identity(mk: SimpleNamespace) -> None:
    assert mk.claim(statement="a").id != mk.claim(statement="b").id


# --------------------------------------------------------------- 3 + 4: explicit criteria


def test_specs_must_declare_criteria(mk: SimpleNamespace) -> None:
    with pytest.raises(SchemaError, match="decision criteria"):
        mk.spec(criteria={})


def test_params_and_criteria_may_not_overlap(mk: SimpleNamespace) -> None:
    with pytest.raises(SchemaError, match="both params and criteria"):
        mk.spec(criteria={"alpha": 0.05}, params={"alpha": 0.01})


def test_criteria_digest_is_deterministic_and_order_independent(mk: SimpleNamespace) -> None:
    a = mk.spec(criteria={"direction": "decrease", "alpha": 0.05})
    b = mk.spec(criteria={"alpha": 0.05, "direction": "decrease"})
    assert a.criteria_digest == b.criteria_digest
    assert a.criteria_digest.startswith("sha256:")
    assert len(a.criteria_digest) == len("sha256:") + 64
    assert (
        a.criteria_digest == "sha256:" + JsonMap({"alpha": 0.05, "direction": "decrease"}).digest()
    )


def test_changing_criteria_changes_spec_identity(mk: SimpleNamespace) -> None:
    a = mk.spec(criteria={"alpha": 0.05})
    b = mk.spec(criteria={"alpha": 0.01})
    assert a.criteria_digest != b.criteria_digest
    assert a.id != b.id


def test_results_bind_to_the_exact_spec(mk: SimpleNamespace) -> None:
    original = mk.spec(criteria={"alpha": 0.05})
    retuned = mk.spec(criteria={"alpha": 0.2})
    result = _result(mk, Outcome.SUPPORTS, spec=original)
    assert result.spec.spec_id == original.id
    verify_ref(result.spec, original)
    with pytest.raises(EvidenceRuleError):
        verify_ref(result.spec, retuned)


def test_spec_validation(mk: SimpleNamespace) -> None:
    for bad in (
        {"protocol": "Bad-Name"},
        {"relations": ()},
        {"relations": (NEC, NEC)},
    ):
        with pytest.raises(SchemaError):
            mk.spec(**bad)
    spec = mk.spec(relations=(Relation.SUFFICIENT_FOR, Relation.ENCODES, NEC))
    assert spec.applicable_relations == tuple(
        sorted(spec.applicable_relations, key=lambda r: r.value)
    )


# ------------------------------------------ 1: causal support needs causal evidence


@pytest.mark.parametrize("relation", sorted(CAUSAL_RELATIONS, key=attrgetter("value")))
@pytest.mark.parametrize("outcome", [Outcome.SUPPORTS, Outcome.CONTRADICTS])
def test_decisive_causal_results_require_causal_evidence(
    mk: SimpleNamespace, relation: Relation, outcome: Outcome
) -> None:
    claim = mk.claim(relation=relation)
    spec = mk.spec(relations=(relation,))
    measured = EvidenceRef.to(mk.activation())
    with pytest.raises(EvidenceRuleError, match="requires INTERVENTIONAL or ESTIMATED_CAUSAL"):
        _result(mk, outcome, claim=claim, spec=spec, evidence=(measured,))
    ok = _result(mk, outcome, claim=claim, spec=spec, evidence=(measured, mk.causal_ref()))
    assert ok.outcome is outcome


def test_attribution_cannot_support_a_causal_claim(mk: SimpleNamespace) -> None:
    attributed = EvidenceRef(
        record_id="attribution:" + "0" * 32, kind="attribution", status=EvidenceStatus.ATTRIBUTED
    )
    with pytest.raises(EvidenceRuleError):
        _result(mk, Outcome.SUPPORTS, evidence=(attributed,))


def test_decisive_results_must_cite_evidence(mk: SimpleNamespace) -> None:
    claim = mk.claim(relation=Relation.ENCODES)
    spec = mk.spec(protocol="detection", relations=(Relation.ENCODES,))
    with pytest.raises(EvidenceRuleError, match="must cite evidence"):
        _result(mk, Outcome.SUPPORTS, claim=claim, spec=spec, evidence=())
    ok = _result(
        mk, Outcome.SUPPORTS, claim=claim, spec=spec, evidence=(EvidenceRef.to(mk.activation()),)
    )
    assert ok.outcome is Outcome.SUPPORTS


def test_non_decisive_results_need_no_evidence(mk: SimpleNamespace) -> None:
    assert _result(mk, Outcome.INCONCLUSIVE).evidence == ()
    assert _result(mk, Outcome.NOT_APPLICABLE).evidence == ()


def test_inapplicable_protocols_can_only_be_not_applicable(mk: SimpleNamespace) -> None:
    spec = mk.spec(protocol="detection", relations=(Relation.ENCODES,))
    with pytest.raises(EvidenceRuleError, match="does not apply"):
        _result(mk, Outcome.SUPPORTS, spec=spec)
    with pytest.raises(EvidenceRuleError, match="does not apply"):
        _result(mk, Outcome.INCONCLUSIVE, spec=spec)
    assert _result(mk, Outcome.NOT_APPLICABLE, spec=spec).outcome is Outcome.NOT_APPLICABLE


def test_errored_iff_error_message(mk: SimpleNamespace) -> None:
    with pytest.raises(SchemaError):
        _result(mk, Outcome.ERRORED)
    with pytest.raises(SchemaError):
        _result(mk, Outcome.INCONCLUSIVE, error="boom")
    assert _result(mk, Outcome.ERRORED, error="boom").error == "boom"


def test_results_require_provenance(mk: SimpleNamespace) -> None:
    claim, spec = mk.claim(), mk.spec()
    with pytest.raises(SchemaError, match="requires provenance_id"):
        ClaimTestResult(
            claim=ClaimRef.to(claim), spec=SpecRef.to(spec), outcome=Outcome.INCONCLUSIVE
        )


def test_duplicate_evidence_is_rejected(mk: SimpleNamespace) -> None:
    ref = mk.causal_ref()
    with pytest.raises(SchemaError, match="duplicate"):
        _result(mk, Outcome.SUPPORTS, evidence=(ref, ref))


# ------------------------------------------ 2: generated cannot satisfy evidence


def test_generated_records_cannot_satisfy_evidence(
    mk: SimpleNamespace, scratch_registry: Any
) -> None:
    @scratch_registry("summary")
    @dataclass(frozen=True, slots=True, kw_only=True)
    class Summary(BaseRecord):
        STATUS: ClassVar[EvidenceStatus | None] = EvidenceStatus.GENERATED
        text: str

    summary = Summary(text="unit 4 clearly drives class 2", provenance_id=mk.PROV)
    with pytest.raises(EvidenceRuleError, match="GENERATED"):
        _result(mk, Outcome.SUPPORTS, evidence=(summary, mk.causal_ref()))


# --------------------------------------------------------------- 6: assessments are derived


def test_assessment_derive_follows_results(mk: SimpleNamespace) -> None:
    claim = mk.claim()
    results = [
        _result(mk, Outcome.SUPPORTS, claim=claim, spec=mk.spec("ablation_necessity")),
        _result(mk, Outcome.SUPPORTS, claim=claim, spec=mk.spec("random_baseline")),
    ]
    assessment = Assessment.derive(claim, results, POLICY)
    assert assessment.verdict is Verdict.SUPPORTED
    assert assessment.required_but_missing == ()
    assert assessment.claim == ClaimRef.to(claim)
    assert {r.record_id for r in assessment.derived_from} == {r.id for r in results}


def test_manually_asserted_verdicts_are_rejected(mk: SimpleNamespace) -> None:
    claim = mk.claim()
    result = _result(mk, Outcome.INCONCLUSIVE, claim=claim)
    with pytest.raises(EvidenceRuleError, match="does not follow"):
        Assessment(
            claim=ClaimRef.to(claim),
            policy=POLICY,
            results=(ResultRef.to(result),),
            verdict=Verdict.SUPPORTED,
        )
    with pytest.raises(EvidenceRuleError, match="does not follow"):
        Assessment(claim=ClaimRef.to(claim), policy=POLICY, results=(), verdict=Verdict.SUPPORTED)


def test_assessments_have_no_confidence_number() -> None:
    names = {f.name for f in dataclasses.fields(Assessment)}
    assert not any("confidence" in n or "score" in n or "probability" in n for n in names)


def test_one_required_protocol_missing_is_not_supported(mk: SimpleNamespace) -> None:
    claim = mk.claim()
    only_ablation = [_result(mk, Outcome.SUPPORTS, claim=claim, spec=mk.spec("ablation_necessity"))]
    assessment = Assessment.derive(claim, only_ablation, POLICY)
    assert assessment.verdict is Verdict.INCONCLUSIVE
    assert assessment.required_but_missing == ("random_baseline",)


def test_results_must_be_about_the_claim(mk: SimpleNamespace) -> None:
    claim, other = mk.claim(), mk.claim(statement="another claim")
    result = _result(mk, Outcome.SUPPORTS, claim=other)
    with pytest.raises(EvidenceRuleError, match="not about claim"):
        Assessment.derive(claim, [result], POLICY)


def test_causal_claims_need_a_policy_that_names_protocols(mk: SimpleNamespace) -> None:
    empty = AssessmentPolicy(name="empty", version=1)
    with pytest.raises(EvidenceRuleError, match="names no protocol"):
        Assessment.derive(mk.claim(), [], empty)
    for relation in (Relation.INCREASES, Relation.DECREASES):
        with pytest.raises(EvidenceRuleError):
            derive_verdict(relation, POLICY, [])


def test_non_causal_claims_can_be_assessed_without_protocol_requirements(
    mk: SimpleNamespace,
) -> None:
    claim = mk.claim(relation=Relation.ENCODES)
    spec = mk.spec(protocol="detection", relations=(Relation.ENCODES,))
    result = _result(
        mk, Outcome.SUPPORTS, claim=claim, spec=spec, evidence=(EvidenceRef.to(mk.activation()),)
    )
    empty = AssessmentPolicy(name="empty", version=1)
    assert Assessment.derive(claim, [result], empty).verdict is Verdict.SUPPORTED


def _refs(*pairs: tuple[str, Outcome]) -> list[ResultRef]:
    return [
        ResultRef(result_id=f"claim_test_result:{i:032x}", protocol=p, outcome=o)
        for i, (p, o) in enumerate(pairs)
    ]


S, C, INC, NA, E = (
    Outcome.SUPPORTS,
    Outcome.CONTRADICTS,
    Outcome.INCONCLUSIVE,
    Outcome.NOT_APPLICABLE,
    Outcome.ERRORED,
)


@pytest.mark.parametrize(
    ("pairs", "verdict", "missing"),
    [
        ((), Verdict.UNTESTED, ("ablation_necessity", "random_baseline")),
        (
            (("ablation_necessity", NA),),
            Verdict.UNTESTED,
            ("ablation_necessity", "random_baseline"),
        ),
        (
            (("ablation_necessity", E),),
            Verdict.INCONCLUSIVE,
            ("ablation_necessity", "random_baseline"),
        ),
        (
            (("ablation_necessity", INC),),
            Verdict.INCONCLUSIVE,
            ("ablation_necessity", "random_baseline"),
        ),
        ((("ablation_necessity", S), ("random_baseline", S)), Verdict.SUPPORTED, ()),
        (
            (("ablation_necessity", S), ("random_baseline", S), ("counterexample", C)),
            Verdict.MIXED,
            (),
        ),
        (
            (("ablation_necessity", C),),
            Verdict.CONTRADICTED,
            ("ablation_necessity", "random_baseline"),
        ),
        (
            (("ablation_necessity", S), ("random_baseline", INC)),
            Verdict.INCONCLUSIVE,
            ("random_baseline",),
        ),
        ((("other_protocol", S),), Verdict.INCONCLUSIVE, ("ablation_necessity", "random_baseline")),
    ],
)
def test_verdict_derivation_table(
    pairs: tuple[tuple[str, Outcome], ...], verdict: Verdict, missing: tuple[str, ...]
) -> None:
    assert derive_verdict(NEC, POLICY, _refs(*pairs)) == (verdict, missing)


def test_assessment_result_order_is_canonical(mk: SimpleNamespace) -> None:
    claim = mk.claim()
    results = [
        _result(mk, Outcome.SUPPORTS, claim=claim, spec=mk.spec("ablation_necessity")),
        _result(mk, Outcome.SUPPORTS, claim=claim, spec=mk.spec("random_baseline")),
    ]
    a = Assessment.derive(claim, results, POLICY)
    b = Assessment.derive(claim, list(reversed(results)), POLICY)
    assert a.results == b.results


def test_policy_validation() -> None:
    with pytest.raises(SchemaError):
        PolicyRequirement(relation=NEC, protocols=())
    with pytest.raises(SchemaError):
        PolicyRequirement(relation=NEC, protocols=("a", "a"))
    with pytest.raises(SchemaError, match="one requirement per relation"):
        AssessmentPolicy(
            name="p",
            version=1,
            requirements=(
                PolicyRequirement(relation=NEC, protocols=("a",)),
                PolicyRequirement(relation=NEC, protocols=("b",)),
            ),
        )
    with pytest.raises(SchemaError):
        AssessmentPolicy(name="p", version=0)


# --------------------------------------------------------------- references


def test_verify_ref_detects_mismatched_references(mk: SimpleNamespace) -> None:
    claim = mk.claim()
    verify_ref(ClaimRef.to(claim), claim)
    forged = ClaimRef(claim_id=claim.id, relation=Relation.ENCODES, estimand=claim.estimand)
    with pytest.raises(EvidenceRuleError):
        verify_ref(forged, claim)
    rec = mk.activation()
    verify_ref(EvidenceRef.to(rec), rec)
    with pytest.raises(TypeError):
        verify_ref(rec.site, rec)


def test_policy_must_cover_the_specific_causal_relation(mk: SimpleNamespace) -> None:
    # A policy covering NECESSARY_FOR says nothing about SUFFICIENT_FOR.
    nec_only = AssessmentPolicy(
        name="nec_only",
        version=1,
        requirements=(PolicyRequirement(relation=NEC, protocols=("ablation_necessity",)),),
    )
    Assessment.derive(mk.claim(), [], nec_only)
    with pytest.raises(EvidenceRuleError, match="names no protocol"):
        Assessment.derive(mk.claim(relation=Relation.SUFFICIENT_FOR), [], nec_only)


def test_assessment_payload_without_policy_requirements_is_rejected(mk: SimpleNamespace) -> None:
    from beyondnn.schema import DecodeError, from_dict, to_dict

    env = to_dict(Assessment.derive(mk.claim(), [], POLICY))
    env["data"]["policy"]["requirements"] = []
    mk.reseal(env)
    with pytest.raises(DecodeError) as info:
        from_dict(env)
    assert isinstance(info.value.__cause__, EvidenceRuleError)


def test_evidence_order_does_not_change_result_identity(mk: SimpleNamespace) -> None:
    measured = EvidenceRef.to(mk.activation())
    a = _result(mk, Outcome.SUPPORTS, evidence=(measured, mk.causal_ref()))
    b = _result(mk, Outcome.SUPPORTS, evidence=(mk.causal_ref(), measured))
    assert a.id == b.id
