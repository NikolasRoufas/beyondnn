"""Envelope round-trips and strict, versioned decoding (ADR-014)."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from beyondnn.schema import (
    SCHEMA_VERSION,
    ActivationRecord,
    AspectOutcome,
    Assessment,
    AssessmentPolicy,
    AssessmentSummary,
    AttributionBaseline,
    AttributionMethodSpec,
    AttributionRecord,
    AttributionReduction,
    BaselineKind,
    BaseRecord,
    CausalEffect,
    CheckOutcome,
    ClaimTestResult,
    ConceptActivation,
    ConceptDataset,
    ConceptPolicy,
    ConceptRecord,
    ConceptValidation,
    DecodeError,
    EnvironmentIdentity,
    Estimand,
    EvidenceRuleError,
    EvidenceSelection,
    ExecutionContext,
    ExecutionMode,
    ExecutionOccurrence,
    FeatureBasis,
    FeatureRecord,
    FeatureSource,
    FingerprintMethod,
    GeneratedLabel,
    IntegrityError,
    InterventionOperation,
    InterventionRecord,
    JsonMap,
    LabelSource,
    MethodIdentity,
    MetricSpec,
    ModelIdentity,
    Outcome,
    OutputRecord,
    PolicyRequirement,
    ProtocolResult,
    ProvenanceRecord,
    Randomness,
    RecordRef,
    Relation,
    SAEIdentity,
    SelectionSource,
    SemanticStatus,
    Site,
    SiteIO,
    TensorRef,
    TensorStats,
    TraceLimitation,
    UnknownLimitationCodeError,
    UnknownRecordKindError,
    UnsupportedVersionError,
    Verdict,
    from_dict,
    from_json,
    to_dict,
    to_json,
)
from beyondnn.schema.base import register_migration, registered_kinds
from beyondnn.schema.codec import _expected_id


def _samples(mk: SimpleNamespace) -> list[BaseRecord]:
    inp = mk.input(display="What is the capital of France?")
    act = mk.activation(parents=(RecordRef.to(inp),))
    claim = mk.claim(estimand=Estimand.finite_sample("eval-8", 8, "mean"))
    spec = mk.spec(criteria={"direction": "decrease", "nested": {"a": [1, 2.5, None, True]}})
    result = ClaimTestResult.for_claim(
        claim,
        spec,
        outcome=Outcome.SUPPORTS,
        evidence=(mk.causal_ref(estimand=claim.estimand),),
        statistics=JsonMap({"effect": -0.31, "n": 8}),
        provenance_id=mk.PROV,
    )
    policy = AssessmentPolicy(
        name="p",
        version=1,
        requirements=(
            PolicyRequirement(relation=Relation.NECESSARY_FOR, protocols=("ablation_necessity",)),
        ),
    )
    assessment = Assessment.derive(claim, [result], policy, provenance_id=mk.PROV)
    limitation = TraceLimitation(code="NO_CAUSAL_EVIDENCE", applies_to=(act.id,))
    provenance = ProvenanceRecord(
        model=ModelIdentity(
            model_class="tests.Net",
            method=FingerprintMethod.FULL,
            algorithm_version=1,
            structure_digest="sha256:" + "1" * 64,
            state_digest="sha256:" + "2" * 64,
            parameter_tensors=2,
            parameter_elements=10,
            buffer_tensors=0,
            buffer_elements=0,
        ),
        environment=EnvironmentIdentity(
            python_implementation="CPython",
            python_version="3.12.0",
            torch_version="2.12.0",
            beyondnn_version="0.0.0.dev0",
            platform_system="Linux",
            platform_machine="x86_64",
        ),
        execution=ExecutionContext(
            mode=ExecutionMode.INTERVENTION,
            intervention_id="intervention:x",
            device="cpu",
            training=False,
            grad_enabled=False,
            randomness=Randomness(declared_seed=0),
        ),
        method=MethodIdentity(name="forward_hook", version="1", params=JsonMap({"k": [1, 2]})),
    )
    occurrence = ExecutionOccurrence(
        provenance_id=provenance.id, started_at="2026-09-25T03:00:00.123456Z"
    )
    intervention = InterventionRecord(
        site=Site(module="layers.0", output_path="[0]"), operation=InterventionOperation.ZERO
    )
    out_a = OutputRecord(pass_index=0, provenance_id=mk.PROV)
    out_b = OutputRecord(pass_index=1, provenance_id=mk.PROV)
    effect = CausalEffect(
        interventions=(RecordRef.to(intervention),),
        metric=MetricSpec(name="select", builtin=True, params=JsonMap({"index": [0]})),
        estimand=Estimand.instance("sha256:x"),
        baseline_value=1.5,
        intervention_value=0.25,
        effect=0.25 - 1.5,
        provenance_id=mk.PROV,
        derived_from=(RecordRef.to(out_a), RecordRef.to(out_b)),
    )
    digest = "sha256:" + "b" * 64
    retained = TensorRef(
        shape=(1, 2), dtype="float32", device="cpu", storage_key=digest, content_digest=digest
    )
    attribution = AttributionRecord(
        method=AttributionMethodSpec(
            name="integrated_gradients",
            implementation="beyondnn",
            implementation_version="1",
            params=JsonMap({"n_steps": 8, "rule": "riemann_middle"}),
        ),
        target=MetricSpec(name="select", builtin=True, params=JsonMap({"index": [0, 0]})),
        site=Site(module="", io=SiteIO.INPUT, output_path="args[0]"),
        pass_index=0,
        sample_id="sha256:x",
        baseline=AttributionBaseline(kind=BaselineKind.ZERO),
        value=retained,
        target_value=21.0,
        diagnostics=JsonMap(
            {"baseline_target_value": 0.0, "attribution_sum": 21.0, "completeness_delta": 0.0}
        ),
        provenance_id=mk.PROV,
        derived_from=(RecordRef.to(inp), RecordRef.to(mk.output())),
    )
    scalar = TensorRef(
        shape=(), dtype="float64", device="cpu", storage_key=digest, content_digest=digest
    )
    reduction = AttributionReduction(
        reduction="abs_sum",
        dims=(0, 1),
        value=scalar,
        scalar=21.0,
        provenance_id=mk.PROV,
        derived_from=(RecordRef.to(attribution),),
    )
    selection = EvidenceSelection(
        site=Site(module="", io=SiteIO.INPUT, output_path="args[0]"),
        sample_id="sha256:x",
        source=SelectionSource.DECLARED,
        rule="declared",
        order=(1, 0),
        n_units=2,
        k=2,
        provenance_id=mk.PROV,
    )
    protocol_result = ProtocolResult(
        protocol="removal_curve",
        protocol_version=1,
        params=JsonMap({"points": [0, 1]}),
        criteria=JsonMap({"max_counterexamples": 0}),
        measurements=JsonMap({"drops": [0.0, 1.5]}),
        outcomes=(AspectOutcome(aspect="max_counterexamples", outcome=CheckOutcome.PASS),),
        samples=("sha256:x",),
        provenance_id=mk.PROV,
    )
    feature_vec = TensorRef(
        shape=(4,), dtype="float32", device="cpu", storage_key=digest, content_digest=digest
    )
    neuron = FeatureRecord(
        basis=FeatureBasis.NEURON,
        site=Site(module="layers.0"),
        axis=1,
        pooling="none",
        index=2,
        source=FeatureSource(kind="declared"),
    )
    sae = FeatureRecord(
        basis=FeatureBasis.SAE,
        site=Site(module="layers.0"),
        axis=1,
        pooling="mean",
        direction=feature_vec,
        source=FeatureSource(kind="sae"),
        model_state_digest=digest,
        sae=SAEIdentity(
            checkpoint="sae:toy@1",
            latent=3,
            d_sae=8,
            activation="relu",
            encoder=feature_vec,
            b_dec=feature_vec,
            b_enc=-0.5,
            reconstruction=JsonMap({"mse": 0.01}),
        ),
    )
    label = GeneratedLabel(
        text="responds to x0",
        feature=sae.id,
        generator="toy-llm",
        revision="v1",
        provenance_id=mk.PROV,
    )
    concept = ConceptRecord(
        label=label.text,
        definition="x0 > 0",
        feature=sae.id,
        label_source=LabelSource.GENERATED,
        generated_label=label.id,
    )
    cdata = ConceptDataset(
        name="toy",
        label_source="x0 > 0",
        samples=("s1", "s2", "s3"),
        labels=(0, 1, 1),
        splits=("train", "val", "test"),
    )
    summary = AssessmentSummary(
        assessment_id=assessment.id,
        claim_id=claim.id,
        relation=Relation.ENCODES,
        protocol="concept_encoding",
        verdict=Verdict.SUPPORTED,
        controls_declared=True,
    )
    validation = ConceptValidation(
        concept=RecordRef.to(concept),
        feature=RecordRef.to(sae),
        dataset=RecordRef.to(cdata),
        policy=ConceptPolicy(name="concept_validation_v1", version=1),
        encoding=summary,
        use=(),
        counterexamples=protocol_result.id,
        false_positive_rate=0.1,
        false_negative_rate=0.0,
        model_state_digest=digest,
        scope="toy",
        semantic_status=SemanticStatus.PROPOSED_CONCEPT,
        unmet=("0 use claim(s); the policy requires 1",),
    )
    concept_act = ConceptActivation(
        validation=validation.id,
        concept=concept.id,
        feature=sae.id,
        sample_id="s1",
        value=0.25,
        provenance_id=mk.PROV,
        derived_from=(RecordRef.to(act),),
    )
    return [
        neuron,
        sae,
        label,
        concept,
        cdata,
        validation,
        concept_act,
        selection,
        protocol_result,
        attribution,
        reduction,
        inp,
        mk.output(),
        act,
        claim,
        spec,
        result,
        assessment,
        limitation,
        provenance,
        occurrence,
        intervention,
        effect,
    ]


def test_samples_cover_every_registered_kind(mk: SimpleNamespace) -> None:
    assert {r.KIND for r in _samples(mk)} == set(registered_kinds())


def test_round_trip_preserves_every_record(mk: SimpleNamespace) -> None:
    for rec in _samples(mk):
        back = from_json(to_json(rec))
        assert back == rec
        assert back.id == rec.id
        assert type(back) is type(rec)
        assert back.status is rec.status
        assert to_json(back) == to_json(rec)


def test_envelope_carries_version_information(mk: SimpleNamespace) -> None:
    env = to_dict(mk.activation())
    assert set(env) == {"schema_version", "kind", "record_version", "id", "status", "data"}
    assert env["schema_version"] == SCHEMA_VERSION == "0.1"
    assert env["record_version"] == 1
    assert env["status"] == "measured"
    back = to_dict(from_dict(json.loads(json.dumps(env))))
    assert back["schema_version"] == "0.1"
    assert back["record_version"] == 1


def test_to_json_is_canonical_and_deterministic(mk: SimpleNamespace) -> None:
    text = to_json(mk.activation())
    assert text == to_json(mk.activation())
    assert text == json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"))


def test_non_finite_stats_round_trip(mk: SimpleNamespace) -> None:
    stats = TensorStats(
        numel=2, mean=math.nan, std=math.nan, min=-math.inf, max=math.inf, l2_norm=math.inf
    )
    ref = TensorRef(shape=(2,), dtype="float32", device="cpu", stats=stats)
    rec = mk.activation()
    import dataclasses

    rec = dataclasses.replace(rec, value=ref)
    text = to_json(rec)
    assert '"NaN"' in text
    assert '"-Infinity"' in text
    back = from_json(text)
    assert isinstance(back, ActivationRecord)
    assert back.value.stats is not None
    assert math.isnan(back.value.stats.mean)
    assert back.id == rec.id


# --------------------------------------------------------------- unknown kinds / versions


def test_unknown_kind_fails_clearly(mk: SimpleNamespace) -> None:
    env = to_dict(mk.activation())
    env["kind"] = "future_kind"
    with pytest.raises(UnknownRecordKindError, match="newer BeyondNN"):
        from_dict(env)


def test_newer_record_version_is_rejected(mk: SimpleNamespace) -> None:
    env = to_dict(mk.activation())
    env["record_version"] = 2
    with pytest.raises(UnsupportedVersionError, match="newer than supported"):
        from_dict(env)


@pytest.mark.parametrize("version", ["0.2", "1.0", "0.1.0", "abc", 0.1, None])
def test_incompatible_schema_versions_are_rejected(mk: SimpleNamespace, version: Any) -> None:
    env = to_dict(mk.activation())
    env["schema_version"] = version
    with pytest.raises(DecodeError):
        from_dict(env)


@pytest.mark.parametrize("version", [0, -1, True, "1", 1.0])
def test_invalid_record_versions_are_rejected(mk: SimpleNamespace, version: Any) -> None:
    env = to_dict(mk.activation())
    env["record_version"] = version
    with pytest.raises(DecodeError):
        from_dict(env)


def test_older_versions_need_a_migration(
    scratch_registry: Callable[..., Any], mk: SimpleNamespace
) -> None:
    @scratch_registry("widget", version=2)
    @dataclass(frozen=True, slots=True, kw_only=True)
    class Widget(BaseRecord):
        size: int
        unit: str

    current = Widget(size=3, unit="cm")
    old_env = {
        "schema_version": SCHEMA_VERSION,
        "kind": "widget",
        "record_version": 1,
        "id": _expected_id("widget", 1, {"provenance_id": None, "derived_from": [], "size": 3}),
        "status": None,
        "data": {"provenance_id": None, "derived_from": [], "size": 3},
    }
    with pytest.raises(UnsupportedVersionError, match="no migration"):
        from_dict(copy.deepcopy(old_env))

    @register_migration("widget", 1)
    def _add_unit(data: dict[str, Any]) -> dict[str, Any]:
        return data | {"unit": "cm"}

    migrated = from_dict(copy.deepcopy(old_env))
    assert migrated == current
    assert migrated.id == current.id


# --------------------------------------------------------------- malformed payloads


def _env(mk: SimpleNamespace) -> dict[str, Any]:
    return to_dict(mk.activation())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda e: e.pop("status"),
        lambda e: e.update(extra=1),
        lambda e: e["data"].pop("call_index"),
        lambda e: e["data"].update(colour="red"),
        lambda e: e["data"].update(call_index="0"),
        lambda e: e["data"].update(call_index=True),
        lambda e: e["data"]["value"].update(shape="2x3"),
        lambda e: e["data"]["value"]["stats"].update(mean="1.5"),
        lambda e: e["data"]["site"].update(io="sideways"),
        lambda e: e["data"].update(site=["layers.0"]),
        lambda e: e["data"].update(derived_from={}),
    ],
)
def test_malformed_payloads_are_rejected(
    mk: SimpleNamespace, mutate: Callable[[dict[str, Any]], Any]
) -> None:
    env = _env(mk)
    mutate(env)
    with pytest.raises(DecodeError):
        from_dict(env)


@pytest.mark.parametrize("payload", [None, [], "activation", 3])
def test_non_object_payloads_are_rejected(payload: Any) -> None:
    with pytest.raises(DecodeError):
        from_dict(payload)


def test_tampered_id_is_detected(mk: SimpleNamespace) -> None:
    env = _env(mk)
    env["data"]["call_index"] = 7
    with pytest.raises(IntegrityError, match="does not match content"):
        from_dict(env)


def test_tampered_status_is_detected(mk: SimpleNamespace) -> None:
    env = _env(mk)
    env["status"] = "interventional"
    with pytest.raises(IntegrityError, match="stored status"):
        from_dict(env)


def test_payload_violating_an_invariant_is_rejected_with_cause(mk: SimpleNamespace) -> None:
    rec = next(r for r in _samples(mk) if r.KIND == "claim_test_result")
    env = to_dict(rec)
    env["data"]["evidence"] = []
    mk.reseal(env)
    with pytest.raises(DecodeError) as info:
        from_dict(env)
    assert isinstance(info.value.__cause__, EvidenceRuleError)


def test_forged_assessment_verdict_is_rejected(mk: SimpleNamespace) -> None:
    rec = next(r for r in _samples(mk) if r.KIND == "assessment")
    env = to_dict(rec)
    env["data"]["verdict"] = "contradicted"
    mk.reseal(env)
    with pytest.raises(DecodeError) as info:
        from_dict(env)
    assert isinstance(info.value.__cause__, EvidenceRuleError)


def test_unknown_limitation_code_in_payload(mk: SimpleNamespace) -> None:
    env = to_dict(TraceLimitation(code="NO_ATTRIBUTION"))
    env["data"]["code"] = "FUTURE_CODE"
    mk.reseal(env)
    with pytest.raises(DecodeError) as info:
        from_dict(env)
    assert isinstance(info.value.__cause__, UnknownLimitationCodeError)


@pytest.mark.parametrize(
    "text",
    [
        "{not json",
        '{"a": NaN}',
        '{"a": Infinity}',
    ],
)
def test_invalid_json_text(text: str) -> None:
    with pytest.raises(DecodeError):
        from_json(text)


def test_invalid_float_strings(mk: SimpleNamespace) -> None:
    env = _env(mk)
    env["data"]["value"]["stats"]["mean"] = "nan"
    mk.reseal(env)
    with pytest.raises(DecodeError, match="invalid float string"):
        from_dict(env)


def test_generated_evidence_in_payload_is_rejected(mk: SimpleNamespace) -> None:
    rec = next(r for r in _samples(mk) if r.KIND == "claim_test_result")
    env = to_dict(rec)
    env["data"]["evidence"][0]["status"] = "generated"
    mk.reseal(env)
    with pytest.raises(DecodeError) as info:
        from_dict(env)
    assert isinstance(info.value.__cause__, EvidenceRuleError)


def test_site_defaults_are_serialised_explicitly(mk: SimpleNamespace) -> None:
    env = to_dict(mk.activation())
    assert env["data"]["site"] == {"module": "layers.0", "io": "output", "output_path": ""}
    assert Site(module="layers.0") == from_dict(env).site  # type: ignore[attr-defined]


def test_replaced_id_with_unchanged_content_is_detected(mk: SimpleNamespace) -> None:
    env = _env(mk)
    env["id"] = mk.activation("layers.9").id  # a valid id, but of different content
    with pytest.raises(IntegrityError, match="does not match content"):
        from_dict(env)


def test_id_with_wrong_kind_prefix_is_detected(mk: SimpleNamespace) -> None:
    env = _env(mk)
    env["id"] = "input:" + env["id"].split(":", 1)[1]
    with pytest.raises(IntegrityError):
        from_dict(env)
