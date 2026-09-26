"""Phase 6: concepts and concept validation (ADR-039..043).

Ground-truth expectations are pre-registered in docs/PHASE_6_PLAN.md §21 (the full-N
run is experiments/phase6/ground_truth.py); these tests use smaller control counts."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
import torch

import beyondnn as bnn
from beyondnn._testing.concept_models import ConceptToy, concept_inputs
from beyondnn.concepts import ConceptError
from beyondnn.concepts.verify import ConceptVerificationError, verify_feature, verify_validation
from beyondnn.explain import EvidenceIntegrityError, ModelMismatchError, SampleMismatchError
from beyondnn.schema import (
    ConceptValidation,
    EvidenceRef,
    EvidenceRuleError,
    EvidenceStatus,
    FeatureBasis,
    InputRecord,
    InterventionOperation,
    Outcome,
    Relation,
    SchemaError,
    SemanticStatus,
    Verdict,
)
from beyondnn.schema.codec import _expected_id

C, iv = bnn.concepts, bnn.interventions
MODEL = ConceptToy().eval()
X = concept_inputs(300, seed=0)
SPLITS = ["train"] * 150 + ["val"] * 50 + ["test"] * 100
ENC = C.encoding_criteria(min_fraction_below=0.95)
USE = C.use_criteria(min_change=0.25, min_fraction_beyond_controls=0.9)


def data(col: int, name: str | None = None, x: torch.Tensor = X) -> C.ConceptData:
    return C.dataset(
        rows(x),
        (x[:, col] > 0).long().tolist(),
        SPLITS,
        name=name or f"toy-x{col}",
        label_source=f"x{col} > 0",
    )


def unit(*idx: int) -> torch.Tensor:
    v = torch.zeros(12)
    v[list(idx)] = 1.0
    out: torch.Tensor = v / v.norm()
    return out


def rows(x: torch.Tensor) -> list[torch.Tensor]:
    return [x[i : i + 1] for i in range(x.shape[0])]


def enc_controls() -> list[C.Control]:
    return [
        C.random_directions(60, seed=1, distribution="isotropic"),
        C.label_permutation(60, seed=2),
    ]


def use_controls(feature: C.Feature, n: int = 15) -> list[C.Control]:
    if feature.record.basis is FeatureBasis.NEURON:
        return [C.random_neurons(n, seed=3)]
    return [C.random_directions(n, seed=3, distribution="isotropic")]


def full(
    feature: C.Feature,
    d: C.ConceptData,
    target: int,
    *,
    label: str = "concept",
    reference: C.Reference | None = None,
    model: torch.nn.Module = MODEL,
) -> tuple[C.Concept, C.EncodingResult, C.UseResult, C.ConceptValidationResult]:
    concept = C.propose(feature, label=label, definition=label)
    enc = C.encoding_test(model, concept, d, controls=enc_controls(), criteria=ENC)
    use = C.use_test(
        model,
        concept,
        d,
        target=iv.metrics.select([0, target]),
        relation="decreases",
        intervention=C.remove(reference or C.zero()),
        controls=use_controls(feature),
        criteria=USE,
    )
    return concept, enc, use, C.validate(concept, encoding=enc, use=[use])


# ------------------------------------------------------------------ ground-truth cases


@pytest.mark.parametrize(
    "make", [lambda: C.neuron("hidden", 0), lambda: C.direction("hidden", unit(0))]
)
def test_case_a_encoded_and_used_is_validated(make: Any) -> None:
    feature = make()
    assert feature.semantic_status is SemanticStatus.UNLABELED_FEATURE
    concept, enc, use, val = full(feature, data(0), 0, label="x0 > 0")
    assert concept.semantic_status is SemanticStatus.PROPOSED_CONCEPT
    assert enc.outcome is Outcome.SUPPORTS
    assert enc.auroc == 1.0
    assert use.outcome is Outcome.SUPPORTS
    positives = X[150 + 50 :][X[200:, 0] > 0, 0]
    assert use.mean_effect == pytest.approx(-3 * float(positives.mean()), rel=1e-5)
    assert val.semantic_status is SemanticStatus.VALIDATED_CONCEPT
    assert val.unmet == ()
    verify_validation(val)


@pytest.mark.parametrize(
    "make", [lambda: C.neuron("hidden", 1), lambda: C.direction("hidden", unit(1))]
)
def test_case_b_decodable_but_unused_stays_proposed(make: Any) -> None:
    _, enc, use, val = full(make(), data(1), 0, label="x1 > 0")
    assert enc.outcome is Outcome.SUPPORTS
    assert enc.verdict is Verdict.SUPPORTED
    assert use.mean_effect == 0.0
    assert use.outcome is Outcome.CONTRADICTS
    assert use.verdict is Verdict.CONTRADICTED
    assert val.semantic_status is SemanticStatus.PROPOSED_CONCEPT
    assert any("contradicted" in u for u in val.unmet)
    text = val.describe()
    assert "ENCODING CLAIM: SUPPORTED" in text
    assert "USE CLAIM (decreases, remove with zero reference): CONTRADICTED" in text
    assert "VALIDATED_CONCEPT" not in text
    codes = {lim.code for lim in enc.trace.limitations}
    assert "DECODABILITY_NOT_USE" in codes
    verify_validation(val)


def test_case_d_redundancy_is_not_read_as_unused() -> None:
    d = data(4)
    _, _, neuron_use, neuron_val = full(C.neuron("hidden", 4), d, 2)
    assert neuron_use.outcome is Outcome.CONTRADICTS
    assert neuron_use.mean_effect == 0.0
    assert "SINGLE_FEATURE_TEST_MISSES_REDUNDANCY" in {
        lim.code for lim in neuron_use.trace.limitations
    }
    _, _, dir_use, dir_val = full(C.direction("hidden", unit(4, 5)), d, 2)
    assert dir_use.outcome is Outcome.SUPPORTS
    assert dir_val.semantic_status is SemanticStatus.VALIDATED_CONCEPT
    # the neuron alone is sufficient (site-relative), though not necessary
    concept = neuron_val.concept
    keep = C.use_test(
        MODEL,
        concept,
        d,
        target=iv.metrics.select([0, 2]),
        relation="sufficient_for",
        intervention=C.retain(C.zero()),
        controls=[C.random_neurons(15, seed=3)],
        criteria=C.use_criteria(max_change=0.05, min_fraction_beyond_controls=0.9),
    )
    assert keep.outcome is Outcome.SUPPORTS
    assert keep.claim.relation is Relation.SUFFICIENT_FOR
    val = C.validate(
        concept, encoding=neuron_val.encoding, use=list(neuron_val.use), additional=[keep]
    )
    assert val.semantic_status is SemanticStatus.PROPOSED_CONCEPT
    assert val.record.additional[0].verdict is Verdict.SUPPORTED
    verify_validation(val)


def test_case_g_permuted_labels_fail_their_controls() -> None:
    perm = torch.randperm(300, generator=torch.Generator().manual_seed(30)).tolist()
    labels = (X[:, 0] > 0).long().tolist()
    d = C.dataset(rows(X), [labels[p] for p in perm], SPLITS, name="perm", label_source="permuted")
    concept = C.propose(C.neuron("hidden", 0), label="x0 > 0", definition="x0 > 0")
    enc = C.encoding_test(MODEL, concept, d, controls=enc_controls(), criteria=ENC)
    assert enc.outcome is Outcome.CONTRADICTS


def test_case_h_counterexamples_are_listed_and_threshold_is_from_val() -> None:
    concept = C.propose(C.neuron("hidden", 7), label="x7 > 0", definition="x7 > 0")
    d = data(7)
    enc = C.encoding_test(MODEL, concept, d, controls=enc_controls(), criteria=ENC)
    m = enc.counterexamples.measurements
    fp = m["false_positives"]
    assert isinstance(fp, tuple)
    assert m["false_positive_rate"] == len(fp) / m["n_non_concept"]  # type: ignore[operator]
    assert m["false_positive_rate"] >= 0.2  # type: ignore[operator]
    test_ids = {d.record.samples[i] for i in d.record.indices("test")}
    assert all(str(row[0]) in test_ids for row in fp)  # type: ignore[index]


def test_random_directions_rarely_pass_but_naive_criteria_do() -> None:
    splits = ["train"] * 150 + ["val"] * 50 + ["test"] * 16
    x = concept_inputs(216, seed=20)
    d = C.dataset(rows(x), (x[:, 0] > 0).long().tolist(), splits, name="small", label_source="x0")
    naive = supported = 0
    for s in range(20):
        v = torch.randn(12, generator=torch.Generator().manual_seed(1000 + s))
        concept = C.propose(C.direction("hidden", v), label="x0 > 0", definition="random")
        enc = C.encoding_test(MODEL, concept, d, controls=enc_controls(), criteria=ENC)
        naive += enc.auroc >= 0.7
        supported += enc.outcome is Outcome.SUPPORTS
    assert naive >= 2
    assert supported <= 2


# ------------------------------------------------------------------ mandatory declarations


def test_controls_and_interventions_are_mandatory() -> None:
    d = data(0)
    concept = C.propose(C.neuron("hidden", 0), label="x0", definition="x0 > 0")
    with pytest.raises(ConceptError, match="control"):
        C.encoding_test(MODEL, concept, d, controls=[], criteria=ENC)
    with pytest.raises(ConceptError, match="mandatory"):
        C.encoding_criteria(min_fraction_below=0.2)
    with pytest.raises(ConceptError, match="distribution"):
        C.random_directions(10, seed=0, distribution="gaussian")
    common: dict[str, Any] = {"target": iv.metrics.select([0, 0]), "relation": "decreases"}
    with pytest.raises(ConceptError, match="declare the intervention"):
        C.use_test(
            MODEL,
            concept,
            d,
            intervention=None,  # type: ignore[arg-type]
            controls=use_controls(concept.feature),
            criteria=USE,
            **common,
        )
    with pytest.raises(ConceptError, match="explicit reference"):
        C.remove(None)  # type: ignore[arg-type]
    with pytest.raises(ConceptError, match="control"):
        C.use_test(
            MODEL, concept, d, intervention=C.remove(C.zero()), controls=[], criteria=USE, **common
        )
    with pytest.raises(ConceptError, match="random_directions or random_neurons"):
        C.use_test(
            MODEL,
            concept,
            d,
            intervention=C.remove(C.zero()),
            controls=[C.label_permutation(5, seed=0)],
            criteria=USE,
            **common,
        )
    with pytest.raises(ConceptError, match="neurons use random_neurons"):
        C.use_test(
            MODEL,
            concept,
            d,
            intervention=C.remove(C.zero()),
            controls=[C.random_directions(5, seed=0, distribution="isotropic")],
            criteria=USE,
            **common,
        )
    with pytest.raises(ConceptError, match="retain"):
        C.use_test(
            MODEL,
            concept,
            d,
            intervention=C.retain(C.zero()),
            controls=use_controls(concept.feature),
            criteria=USE,
            **common,
        )
    with pytest.raises(ConceptError, match="min_change"):
        C.use_criteria(min_fraction_beyond_controls=0.9)


# ------------------------------------------------------------------ discovery and scope


def test_fitting_uses_train_only_and_rederives() -> None:
    d = data(0)
    feature = C.fit_direction(MODEL, d, site="hidden")
    assert feature.record.source.split == "train"
    assert feature.record.source.dataset == d.id
    verify_feature(feature, d.record)
    v = feature.vector()
    assert int(v.abs().argmax()) == 0  # the fitted direction points at h0
    other = data(0, name="another-dataset")
    concept = C.propose(feature, label="x0", definition="x0 > 0")
    other = C.dataset(
        rows(X),
        (X[:, 0] > 0).long().tolist(),
        ["test"] * 100 + ["train"] * 150 + ["val"] * 50,
        name="shuffled",
        label_source="x0",
    )
    with pytest.raises(ConceptError, match="dataset it was derived on"):
        C.encoding_test(MODEL, concept, other, controls=enc_controls(), criteria=ENC)
    with pytest.raises(ConceptVerificationError, match="another concept dataset"):
        verify_feature(feature, other.record)


def test_search_records_candidates_and_rederives() -> None:
    d = data(4)
    feature = C.search_neurons(MODEL, d, site="hidden")
    assert feature.record.index in (4, 5)
    params = feature.record.source.params
    assert params["candidate_count"] == 12
    assert params["sign"] == 1
    verify_feature(feature, d.record)


def test_a_concept_is_bound_to_its_checkpoint() -> None:
    other = ConceptToy(scale=2.0).eval()
    d = data(0)
    feature = C.fit_direction(MODEL, d, site="hidden")
    concept = C.propose(feature, label="x0", definition="x0 > 0")
    with pytest.raises(ConceptError, match="another model checkpoint"):
        C.encoding_test(other, concept, d, controls=enc_controls(), criteria=ENC)
    _, _, _, val = full(C.neuron("hidden", 0), d, 0)
    x = X[:1]
    with pytest.raises(ModelMismatchError, match="another model checkpoint"):
        bnn.compose(bnn.trace(other, x, sites=["hidden"], retention="cpu"), concepts=[val])
    with pytest.raises(ConceptError, match="another model checkpoint"):
        C.activation(other, x, validation=val)


# ------------------------------------------------------------------ labels and statuses


def test_generated_labels_stay_generated_and_proposed() -> None:
    wrong = C.neuron("hidden", 1)
    gen = C.generated_label(MODEL, wrong, "x0 > 0", generator="toy-labeler", revision="v1")
    assert gen.record.status is EvidenceStatus.GENERATED
    with pytest.raises(EvidenceRuleError, match="GENERATED"):
        EvidenceRef.to(gen.record)
    assert "GENERATED_LABEL_UNVERIFIED" in {lim.code for lim in gen.store.limitations}
    concept = C.propose(wrong, definition="x0 > 0", generated=gen)
    assert concept.semantic_status is SemanticStatus.PROPOSED_CONCEPT
    enc = C.encoding_test(MODEL, concept, data(0), controls=enc_controls(), criteria=ENC)
    assert enc.claim.source.kind.value == "generated"
    assert enc.outcome is Outcome.CONTRADICTS
    with pytest.raises(ConceptError, match="another feature"):
        C.propose(C.neuron("hidden", 0), definition="x", generated=gen)


def test_a_validation_status_cannot_be_forged() -> None:
    _, _, _, val = full(C.neuron("hidden", 1), data(1), 0)
    record = val.record
    with pytest.raises(SchemaError, match="does not follow"):
        dataclasses.replace(record, semantic_status=SemanticStatus.VALIDATED_CONCEPT, unmet=())
    with pytest.raises(SchemaError, match="requires >= 1"):
        C.POLICY_V1.__class__(name="lax", version=1, min_use_claims=0)


def test_evidence_about_another_concept_is_refused() -> None:
    concept_a, enc_a, use_a, val_a = full(C.neuron("hidden", 0), data(0), 0)
    _, enc_b, _, _ = full(C.neuron("hidden", 1), data(1), 0)
    with pytest.raises(ConceptError, match="another concept"):
        C.validate(concept_a, encoding=enc_b, use=[use_a])
    swapped = C.ConceptValidationResult(val_a.trace, concept_a, enc_b, (use_a,))
    with pytest.raises(ConceptVerificationError):
        verify_validation(swapped)
    no_use = C.ConceptValidationResult(val_a.trace, concept_a, enc_a, ())
    with pytest.raises(ConceptVerificationError, match="use summaries"):
        verify_validation(no_use)


def test_concept_activations_exist_only_for_validated_concepts() -> None:
    _, _, _, val_a = full(C.neuron("hidden", 0), data(0), 0)
    _, _, _, val_b = full(C.neuron("hidden", 1), data(1), 0)
    x = torch.tensor([[1.5, 2.0] + [0.0] * 10])
    act_a = C.activation(MODEL, x, validation=val_a)
    act_b = C.activation(MODEL, x, validation=val_b)
    assert act_a.record is not None
    assert act_a.record.status is EvidenceStatus.VALIDATED_CONCEPT
    assert act_a.value == 1.5
    assert act_b.record is None
    assert act_b.value == 2.0
    assert act_b.status is SemanticStatus.PROPOSED_CONCEPT


# ------------------------------------------------------------------ structured WHY


def test_why_shows_encoding_and_use_separately_and_refuses_mismatches() -> None:
    _, _, _, val_a = full(C.neuron("hidden", 0), data(0), 0, label="x0 > 0")
    _, _, _, val_b = full(C.neuron("hidden", 1), data(1), 0, label="x1 > 0")
    x = torch.tensor([[1.5, 2.0] + [0.0] * 10])
    act = C.activation(MODEL, x, validation=val_a)
    ref = bnn.trace(MODEL, x, sites=["hidden"], retention="cpu")
    response = bnn.compose(ref, concepts=[val_a, val_b, act])
    text = response.render()
    assert "CONCEPTS  [dataset-scoped" in text
    assert "ENCODING CLAIM: SUPPORTED" in text
    assert "CONTRADICTED" in text
    assert "value on this input: 1.5 [VALIDATED_CONCEPT activation]" in text
    assert "value on this input: 2 [MEASURED feature activation]" in text
    cov = response.why.coverage
    assert cov.concepts_evaluated
    assert cov.concepts_validated
    assert "concept validation" not in cov.not_evaluated
    rows = response.to_dict()["concepts"]
    assert [r["semantic_status"] for r in rows] == ["validated_concept", "proposed_concept"]
    # a VALIDATED activation without its validation, and one about another input
    with pytest.raises(EvidenceIntegrityError, match="validation composed"):
        bnn.compose(ref, concepts=[act])
    other = bnn.trace(MODEL, X[:1], sites=["hidden"], retention="cpu")
    with pytest.raises(SampleMismatchError):
        bnn.compose(other, concepts=[val_a, act])
    # no concepts: Phase-4/5 coverage is unchanged
    cov0 = bnn.compose(ref).why.coverage
    assert (cov0.concepts_evaluated, cov0.concepts_validated) == (False, False)


# ------------------------------------------------------------------ SAE adapter


def _toy_sae() -> dict[str, Any]:
    g = torch.Generator().manual_seed(5)
    enc = torch.randn(12, 16, generator=g) * 0.1
    enc[0, 3] = 1.0
    dec = torch.randn(16, 12, generator=g) * 0.1
    dec[3, 0] = 1.0
    return {"encoder": enc, "decoder": dec, "b_enc": torch.zeros(16), "b_dec": torch.zeros(12)}


def test_sae_features_are_validated_like_any_feature() -> None:
    sae = _toy_sae()
    feature = C.sae_feature(MODEL, "hidden", latent=3, checkpoint="toy-sae@1", **sae)
    x = X[:1]
    h = MODEL.hidden(x)[0].double()
    expected = float(torch.relu(h @ sae["encoder"][:, 3].double()))
    concept = C.propose(feature, label="x0 > 0", definition="x0 > 0")
    act = C.encoding_test(MODEL, concept, data(0), controls=enc_controls(), criteria=ENC)
    codes = {lim.code for lim in act.trace.limitations}
    assert {"SAE_FEATURE_SPLITTING", "SAE_FEATURE_ABSORPTION", "SAE_RECONSTRUCTION_ERROR"} <= codes
    from beyondnn.concepts._core import feature_values

    assert feature_values(MODEL.hidden(x), feature.record, feature.store) == pytest.approx(expected)
    other = C.sae_feature(MODEL, "hidden", latent=3, checkpoint="toy-sae@2", **sae)
    assert other.id != feature.id  # a different SAE checkpoint is a different feature
    use = C.use_test(
        MODEL,
        concept,
        data(0),
        target=iv.metrics.select([0, 0]),
        relation="decreases",
        intervention=C.remove(C.zero()),
        controls=use_controls(feature),
        criteria=USE,
    )
    assert "SAE_INTERVENTION_IS_PROJECTION" in {lim.code for lim in use.trace.limitations}
    verify_validation(C.validate(concept, encoding=act, use=[use]))


# ------------------------------------------------------------------ interventions and schema


def test_direction_interventions_are_exact_and_recorded() -> None:
    x = torch.tensor([[1.0, 2.0] + [0.0] * 10])
    ref = torch.full((1, 12), 0.5)
    v = unit(0, 1)
    r = iv.intervene(
        MODEL,
        x,
        intervention=iv.direction("hidden", v, axis=1, reference=ref),
        metric=iv.metrics.select([0, 0]),
    )
    # h0 -> h0 - <h - r, v>v_0 = 1 - ((0.5 + 1.5)/√2)(1/√2) = 0 -> y0 drops by 3
    assert r.value == pytest.approx(-3.0, abs=1e-5)
    rec = r.intervention
    assert rec.operation is InterventionOperation.DIRECTION
    assert rec.direction is not None
    assert rec.direction_axis == 1
    assert rec.value is not None
    assert "DIRECTION_REPLACEMENT_MAY_BE_OOD" in {lim.code for lim in r.trace.limitations}
    kept = iv.intervene(
        MODEL,
        x,
        intervention=iv.direction("hidden", v, axis=1, retain=True),
        metric=iv.metrics.select([0, 0]),
    )
    assert kept.value == pytest.approx(3 * (1.5 - 1.0), abs=1e-5)  # h0 -> 1.5
    with pytest.raises(ValueError, match="non-zero"):
        iv.direction("hidden", torch.zeros(12), axis=1)


def _forge(tmp_path: Path, trace: Any, kind: str, mutate: Any, version: int) -> Any:
    target = tmp_path / f"t-{kind}"
    trace.save(target)
    document = json.loads((target / "trace.json").read_text())
    renamed: dict[str, str] = {}
    for env in document["records"]:
        text = json.dumps(env["data"])
        for new, old in renamed.items():
            text = text.replace(new, old)
        env["data"] = json.loads(text)
        if env["kind"] == kind:
            mutate(env["data"])
            env["record_version"] = version
        new_id = env["id"]
        env["id"] = _expected_id(env["kind"], env["record_version"], env["data"])
        renamed[new_id] = env["id"]
    (target / "trace.json").write_text(json.dumps(document))
    return bnn.load_trace(target)


def test_old_interventions_and_claims_migrate_without_invented_fields(tmp_path: Path) -> None:
    r = iv.intervene(
        MODEL,
        X[:1],
        intervention=iv.zero("hidden", units=(0,)),
        metric=iv.metrics.select([0, 0]),
        claims=[],
    )

    def v3(data: dict[str, Any]) -> None:
        data.pop("direction")
        data.pop("direction_axis")

    loaded = _forge(tmp_path, r.trace, "intervention", v3, 3)
    (rec,) = [x for x in loaded.records if x.KIND == "intervention"]
    assert rec.direction is None
    assert rec.direction_axis is None
    _, enc, _, _ = full(C.neuron("hidden", 0), data(0), 0)

    def v2(data: dict[str, Any]) -> None:
        data["subject"].pop("feature")

    claims = [x for x in _forge(tmp_path, enc.trace, "claim", v2, 2).records if x.KIND == "claim"]
    assert [c.subject.feature for c in claims] == [None]


def test_concept_traces_persist_with_their_feature_tensors(tmp_path: Path) -> None:
    feature = C.direction("hidden", unit(0))
    _, enc, _, _ = full(feature, data(0), 0)
    enc.trace.save(tmp_path / "enc")
    loaded = bnn.load_trace(tmp_path / "enc")
    assert loaded.get(feature.id) == feature.record
    assert torch.equal(loaded.tensor(feature.record.direction), feature.vector())  # type: ignore[arg-type]


def test_trace_indexes_agree_with_a_scan() -> None:
    _, _, use, _ = full(C.neuron("hidden", 0), data(0), 0)
    t = use.trace
    scan = tuple(r for r in t.records if isinstance(r, InputRecord))
    assert t.inputs == scan
    for inp in scan:
        assert inp.pass_index is not None
        assert t.input_of_pass(inp.pass_index) is inp
    assert ConceptValidation.RECORD_VERSION == 1
