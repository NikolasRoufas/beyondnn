"""Phase 2 schema: InterventionRecord, MetricSpec, CausalEffect (ADR-028)."""

from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace
from typing import Any

import pytest

from beyondnn.schema import (
    CausalEffect,
    ClaimTestResult,
    Estimand,
    EstimandScope,
    EvidenceRef,
    EvidenceRuleError,
    EvidenceStatus,
    InterventionOperation,
    InterventionRecord,
    JsonMap,
    MetricDeclaration,
    MetricSpec,
    Outcome,
    OutputRecord,
    RecordRef,
    SchemaError,
    Site,
    SiteIO,
    TensorRef,
    from_json,
    to_json,
)

ZERO = InterventionOperation.ZERO
DIGEST = "sha256:" + "a" * 64
RETAINED = TensorRef(
    shape=(1,), dtype="float32", device="cpu", storage_key=DIGEST, content_digest=DIGEST
)
METRIC = MetricSpec(name="select", builtin=True, params=JsonMap({"index": [0]}))
PROV = "prov:x"


def _zero(**kw: Any) -> InterventionRecord:
    return InterventionRecord(site=Site(module="a"), operation=ZERO, **kw)


def _outputs() -> tuple[RecordRef, RecordRef]:
    a = OutputRecord(pass_index=0, provenance_id=PROV)
    b = OutputRecord(pass_index=1, provenance_id=PROV)
    return RecordRef.to(a), RecordRef.to(b)


def _effect(**kw: Any) -> CausalEffect:
    base: dict[str, Any] = {
        "interventions": (RecordRef.to(_zero()),),
        "metric": METRIC,
        "estimand": Estimand.instance("sha256:x"),
        "baseline_value": 2.0,
        "intervention_value": 0.5,
        "effect": 0.5 - 2.0,
        "provenance_id": PROV,
        "derived_from": _outputs(),
    }
    return CausalEffect(**(base | kw))


# ----------------------------------------------------------------- InterventionRecord


def test_operations_have_exact_fields() -> None:
    _zero()
    with pytest.raises(SchemaError):
        _zero(constant=0.0)
    InterventionRecord(
        site=Site(module="a"), operation=InterventionOperation.CONSTANT, constant=1.5
    )
    InterventionRecord(
        site=Site(module="a"), operation=InterventionOperation.CONSTANT, value=RETAINED
    )
    for bad in ({}, {"constant": 1.0, "value": RETAINED}, {"constant": math.inf}):
        with pytest.raises(SchemaError):
            InterventionRecord(
                site=Site(module="a"), operation=InterventionOperation.CONSTANT, **bad
            )


def test_patch_needs_a_retained_value_and_a_measured_activation_source(mk: SimpleNamespace) -> None:
    source = RecordRef.to(mk.activation())
    InterventionRecord(
        site=Site(module="a"), operation=InterventionOperation.PATCH, value=RETAINED, source=source
    )
    for bad in (
        {"value": RETAINED},
        {"source": source},
        {"value": dataclasses.replace(RETAINED, content_digest=None), "source": source},
        {"value": RETAINED, "source": RecordRef.to(mk.input())},
    ):
        with pytest.raises(SchemaError):
            InterventionRecord(site=Site(module="a"), operation=InterventionOperation.PATCH, **bad)


def test_only_non_root_module_outputs_can_be_intervened() -> None:
    with pytest.raises(SchemaError, match="OUTPUT"):
        InterventionRecord(site=Site(module="a", io=SiteIO.INPUT), operation=ZERO)
    with pytest.raises(SchemaError, match="root"):
        InterventionRecord(site=Site(module=""), operation=ZERO)
    with pytest.raises(SchemaError):
        _zero(call_index=-1)


def test_intervention_identity_is_deterministic_and_a_spec_not_evidence() -> None:
    assert _zero().id == _zero().id
    assert _zero(call_index=1).id != _zero().id
    assert _zero().status is None
    with pytest.raises(EvidenceRuleError, match="not an evidence record"):
        EvidenceRef.to(_zero())


def test_metric_spec_names() -> None:
    MetricSpec(
        name="custom:my_metric",
        builtin=False,
        declaration=MetricDeclaration(implementation_revision="v1"),
    )
    with pytest.raises(SchemaError, match="MetricDeclaration"):
        MetricSpec(name="custom:my_metric", builtin=False)
    with pytest.raises(SchemaError, match="automatically"):
        MetricSpec(
            name="select", builtin=True, declaration=MetricDeclaration(implementation_revision="v1")
        )
    with pytest.raises(SchemaError, match=r"declaration\.config"):
        MetricSpec(
            name="custom:m",
            builtin=False,
            params=JsonMap({"k": 1}),
            declaration=MetricDeclaration(implementation_revision="v1"),
        )
    with pytest.raises(SchemaError):
        MetricSpec(name="my_metric", builtin=False)
    with pytest.raises(SchemaError):
        MetricSpec(name="custom:x", builtin=True)


# ----------------------------------------------------------------- CausalEffect


def test_instance_and_finite_sample_effects_are_interventional() -> None:
    instance = _effect()
    assert instance.status is EvidenceStatus.INTERVENTIONAL
    parts = tuple(
        RecordRef.to(_effect(estimand=Estimand.instance(f"sha256:{i}"))) for i in range(3)
    )
    finite = _effect(
        estimand=Estimand.finite_sample("sample:abc", 3, "mean"),
        baseline_value=2.0,
        intervention_value=0.5,
        effect=-1.5,
        derived_from=parts,
    )
    assert finite.status is EvidenceStatus.INTERVENTIONAL


def test_population_effects_need_an_estimator_and_are_estimated_causal() -> None:
    pop = Estimand.population_of("distribution", "mean")
    with pytest.raises(EvidenceRuleError, match="estimator"):
        _effect(estimand=pop, derived_from=())
    estimated = _effect(estimand=pop, estimator="bootstrap_mean", derived_from=())
    assert estimated.status is EvidenceStatus.ESTIMATED_CAUSAL
    with pytest.raises(SchemaError):
        _effect(estimator="bootstrap_mean")  # instance effects are exact


def test_effect_sign_convention_is_enforced() -> None:
    with pytest.raises(SchemaError, match="intervention_value - baseline_value"):
        _effect(effect=2.0 - 0.5)  # reversed sign


def test_instance_effects_derive_from_the_paired_outputs() -> None:
    with pytest.raises(SchemaError, match="paired"):
        _effect(derived_from=())
    with pytest.raises(SchemaError):
        _effect(interventions=(RecordRef.to(_zero()), RecordRef.to(_zero(call_index=1))))


def test_finite_sample_effect_needs_exactly_n_parts() -> None:
    parts = (RecordRef.to(_effect()),)
    with pytest.raises(SchemaError, match="exactly n"):
        _effect(estimand=Estimand.finite_sample("sample:abc", 2, "mean"), derived_from=parts)


def test_non_finite_values_are_rejected() -> None:
    with pytest.raises(SchemaError):
        _effect(baseline_value=math.nan)


def test_effect_status_cannot_be_supplied() -> None:
    with pytest.raises(TypeError):
        _effect(status=EvidenceStatus.MEASURED)


def test_effects_round_trip_and_back_causal_claim_results(mk: SimpleNamespace) -> None:
    effect = _effect()
    assert from_json(to_json(effect)).id == effect.id
    ref = EvidenceRef.to(effect)
    assert ref.status is EvidenceStatus.INTERVENTIONAL
    assert ref.estimand is not None
    assert ref.estimand.scope is EstimandScope.INSTANCE
    claim = mk.claim(estimand=effect.estimand)
    result = ClaimTestResult.for_claim(
        claim, mk.spec(), outcome=Outcome.SUPPORTS, evidence=(effect,), provenance_id=PROV
    )
    assert result.outcome is Outcome.SUPPORTS


def _as_v1(effect: CausalEffect) -> dict[str, Any]:
    from beyondnn.schema.codec import _expected_id, to_dict

    payload = to_dict(effect)
    del payload["data"]["metric"]["declaration"]
    payload["record_version"] = 1
    payload["id"] = _expected_id("causal_effect", 1, payload["data"])
    return payload


def test_v1_builtin_effects_migrate_and_v1_caller_metric_effects_are_refused() -> None:
    from beyondnn.schema import DecodeError, from_dict

    migrated = from_dict(_as_v1(_effect()))
    assert isinstance(migrated, CausalEffect)
    assert migrated.metric == METRIC
    custom = MetricSpec(
        name="custom:m", builtin=False, declaration=MetricDeclaration(implementation_revision="v1")
    )
    with pytest.raises(DecodeError, match="MetricDeclaration"):
        from_dict(_as_v1(_effect(metric=custom)))
