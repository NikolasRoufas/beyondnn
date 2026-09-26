"""Re-derivation of concept records from raw records (used by composition; ADR-042).

Nothing is trusted because BeyondNN produced it: a composed concept validation is
re-derived from the MEASURED activations and INTERVENTIONAL effects it rests on. The
statistics, outcome, result id, counterexamples and assessment of every test, the
regenerated controls, the intervention records (the feature and reference that were
actually used), the fitted or searched feature, and finally the derived semantic
status. Any difference raises :class:`ConceptVerificationError`. Nothing here runs a
model.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import torch

from beyondnn.attribution.spec import tensor_digest
from beyondnn.core.trace import TraceResult
from beyondnn.schema import (
    Assessment,
    CausalEffect,
    Claim,
    ClaimTestResult,
    ClaimTestSpec,
    ConceptDataset,
    ConceptRecord,
    ConceptValidation,
    Estimand,
    EstimandScope,
    FeatureBasis,
    FeatureRecord,
    GeneratedLabel,
    InterventionOperation,
    InterventionRecord,
    JsonMap,
    LabelSource,
    ProtocolResult,
    Relation,
)

from ._core import (
    ConceptError,
    Control,
    auroc,
    feature_values,
    pooled_vectors,
    random_directions_for,
    random_indices_for,
    sample_set_id,
)
from .encoding import (
    COUNTEREXAMPLE_PROTOCOL,
    ENCODING_POLICY,
    ENCODING_PROTOCOL,
    EncodingCriteria,
    EncodingResult,
    counterexamples,
    evaluate_encoding,
)
from .features import Feature, site_tensors
from .use import USE_POLICY, USE_PROTOCOL, UseCriteria, UseResult, evaluate_use

__all__ = [
    "ConceptVerificationError",
    "verify_encoding",
    "verify_encoding_trace",
    "verify_feature",
    "verify_feature_record",
    "verify_use",
    "verify_use_trace",
    "verify_validation",
    "verify_validation_trace",
]

R = TypeVar("R")
_DIRECTION_TOLERANCE = 1e-4


class ConceptVerificationError(ValueError):
    """A concept record does not follow from the records it rests on."""


def _one(trace: TraceResult, kind: type[R], where: str) -> R:
    found = [r for r in trace.records if isinstance(r, kind)]
    if len(found) != 1:
        raise ConceptVerificationError(f"{where}: expected one {kind.__name__}, found {len(found)}")
    return found[0]


def _record(trace: TraceResult, record_id: object, kind: type[R], where: str) -> R:
    if not isinstance(record_id, str):
        raise ConceptVerificationError(f"{where}: {record_id!r} is not a record id")
    try:
        record = trace.get(record_id)
    except KeyError:
        raise ConceptVerificationError(f"{where}: {record_id} is not in the trace") from None
    if not isinstance(record, kind):
        raise ConceptVerificationError(f"{where}: {record_id} is not a {kind.__name__}")
    return record


def _controls(params: JsonMap) -> list[Control]:
    out = []
    raw = params.get("controls")
    if not isinstance(raw, tuple) or not raw:
        raise ConceptVerificationError("the spec declares no controls")
    for c in raw:
        if not isinstance(c, JsonMap):
            raise ConceptVerificationError("malformed control declaration")
        kind, n, seed = c.get("kind"), c.get("n"), c.get("seed")
        dist = c.get("distribution")
        if not (isinstance(kind, str) and isinstance(n, int) and isinstance(seed, int)):
            raise ConceptVerificationError("malformed control declaration")
        out.append(Control(kind, n, seed, dist if isinstance(dist, str) else None))
    return out


def _definitions(
    trace: TraceResult, params: JsonMap, where: str
) -> tuple[FeatureRecord, ConceptRecord, ConceptDataset]:
    feature = _record(trace, params.get("feature"), FeatureRecord, where)
    concept = _record(trace, params.get("concept"), ConceptRecord, where)
    data = _record(trace, params.get("dataset"), ConceptDataset, where)
    if concept.feature != feature.id:
        raise ConceptVerificationError(f"{where}: the concept is about another feature")
    if concept.label_source is LabelSource.GENERATED:
        label = _record(trace, concept.generated_label, GeneratedLabel, where)
        if label.feature != feature.id or label.text != concept.label:
            raise ConceptVerificationError(f"{where}: the generated label does not match")
    return feature, concept, data


def _model_digest(trace: TraceResult, feature: FeatureRecord, where: str) -> str:
    digests = {trace.origin(i).model.state_digest for i in trace.inputs}
    if len(digests) != 1:
        raise ConceptVerificationError(f"{where}: the trace spans several model states")
    (digest,) = digests
    if feature.model_state_digest is not None and feature.model_state_digest != digest:
        raise ConceptVerificationError(f"{where}: the feature was derived on another checkpoint")
    return digest


def _floats(values: Sequence[float]) -> list[float]:
    return [float(v) for v in values]


# ------------------------------------------------------------------ features


def verify_feature(feature: Feature, data: ConceptDataset | None = None) -> None:
    """Re-derive a fitted (mean difference) or searched (train AUROC) feature from its
    fitting recording; declared and SAE features have nothing to re-derive."""
    verify_feature_record(feature.record, feature.fit_trace, data)


def needs_derivation(record: FeatureRecord) -> bool:
    """Whether ``record`` was fitted or searched (and so can be re-derived)."""
    return record.source.kind in ("fit", "search") and record.basis is not FeatureBasis.SAE


def verify_feature_record(
    record: FeatureRecord, recording: TraceResult | None, data: ConceptDataset | None
) -> None:
    """Re-derive a fitted or searched feature from any recording that holds the site's
    activations for every train sample of ``data`` (the fitting recording, or an encoding
    test's trace, which records the train split too)."""
    source = record.source
    if not needs_derivation(record):
        return
    if recording is None or data is None:
        raise ConceptVerificationError(f"{record.id}: the fitting recording/dataset is missing")
    if data.id != source.dataset:
        raise ConceptVerificationError(f"{record.id}: derived on another concept dataset")
    train = data.indices("train")
    if source.params.get("train_set") != sample_set_id([data.samples[i] for i in train]):
        raise ConceptVerificationError(f"{record.id}: fitted on another sample set than train")
    probe = FeatureRecord(
        basis=FeatureBasis.NEURON,
        site=record.site,
        call_index=record.call_index,
        axis=record.axis,
        pooling=record.pooling,
        index=0,
        source=_declared(),
    )
    rows = torch.stack(
        [
            pooled_vectors(t, probe)
            for _, t in site_tensors(recording, data, train, record.site, record.call_index)
        ]
    )
    labels = [data.labels[i] for i in train]
    if record.basis is FeatureBasis.DIRECTION:
        y = torch.tensor(labels, dtype=torch.bool)
        v = (rows[y].mean(dim=0) - rows[~y].mean(dim=0)).float().contiguous()
        assert record.direction is not None
        if tensor_digest(v) != record.direction.content_digest:
            raise ConceptVerificationError(f"{record.id}: the direction does not re-derive")
        return
    best: tuple[float, int, int] | None = None
    for j in range(rows.shape[1]):
        a = auroc(rows[:, j].tolist(), labels)
        score, sign = (a, 1) if a >= 1 - a else (1 - a, -1)
        if best is None or score > best[0]:
            best = (score, j, sign)
    assert best is not None
    if (best[1], best[2], rows.shape[1]) != (
        record.index,
        source.params.get("sign"),
        source.params.get("candidate_count"),
    ):
        raise ConceptVerificationError(f"{record.id}: the searched neuron does not re-derive")


def _declared() -> Any:
    from beyondnn.schema import FeatureSource

    return FeatureSource(kind="declared")


# ------------------------------------------------------------------ encoding


def verify_encoding(result: EncodingResult) -> None:
    _, concept, data = verify_encoding_trace(result.trace)
    if (concept.id, data.id) != (result.concept.id, result.data.id):
        raise ConceptVerificationError("encoding: the result is about another concept/dataset")


def verify_encoding_trace(
    trace: TraceResult,
) -> tuple[FeatureRecord, ConceptRecord, ConceptDataset]:
    """Re-derive an encoding test from its trace alone (records and retained tensors; no
    live objects), e.g. after ``load_trace``. Returns the feature, concept and dataset
    records the test is about."""
    where = "encoding"
    claim = _one(trace, Claim, where)
    spec = _one(trace, ClaimTestSpec, where)
    stored = _one(trace, ClaimTestResult, where)
    assessment = _one(trace, Assessment, where)
    if spec.protocol != ENCODING_PROTOCOL or claim.relation is not Relation.ENCODES:
        raise ConceptVerificationError(f"{where}: not a concept_encoding test of an ENCODES claim")
    feature, concept, data = _definitions(trace, spec.params, where)
    _model_digest(trace, feature, where)
    train, val, test = data.indices("train"), data.indices("val"), data.indices("test")
    test_ids = [data.samples[i] for i in test]
    expected_target = {"concept": concept.id, "dataset": data.id, "split": "test"}
    if (
        claim.subject.feature != feature.id
        or claim.subject.site != feature.site
        or claim.target.metric != "concept_label"
        or dict(claim.target.params.to_plain()) != expected_target
        or claim.estimand != Estimand.finite_sample(sample_set_id(test_ids), len(test), "auroc")
    ):
        raise ConceptVerificationError(
            f"{where}: the claim does not match its feature/concept/split"
        )
    criteria = spec.criteria
    min_below = criteria.get("min_fraction_below")
    min_auroc = criteria.get("min_auroc")
    if not isinstance(min_below, float):
        raise ConceptVerificationError(f"{where}: no control criterion was declared")
    crit = EncodingCriteria(min_below, min_auroc if isinstance(min_auroc, float) else None)
    sign = spec.params.get("expected_sign")
    if sign not in (1, -1):
        raise ConceptVerificationError(f"{where}: invalid expected sign")
    assert isinstance(sign, int)
    order = [*train, *val, *test]
    rows = dict(
        zip(order, site_tensors(trace, data, order, feature.site, feature.call_index), strict=True)
    )
    values = {i: feature_values(t, feature, trace) for i, (_, t) in rows.items()}
    pooled = {i: pooled_vectors(t, feature) for i, (_, t) in rows.items()}
    labels = [data.labels[i] for i in test]
    try:
        outcome, statistics = evaluate_encoding(
            values=[values[i] for i in test],
            labels=labels,
            sign=sign,
            basis=feature.basis,
            index=feature.index,
            pooled_test=torch.stack([pooled[i] for i in test]),
            pooled_train=torch.stack([pooled[i] for i in train]) if len(train) >= 2 else None,
            controls=_controls(spec.params),
            criteria=crit,
        )
    except ConceptError as exc:
        raise ConceptVerificationError(f"{where}: {exc}") from None
    rebuilt = ClaimTestResult.for_claim(
        claim,
        spec,
        outcome=outcome,
        evidence=[rows[i][0] for i in test],
        statistics=JsonMap(statistics),
        provenance_id=stored.provenance_id or "",
    )
    if rebuilt.id != stored.id:
        raise ConceptVerificationError(
            f"{stored.id} ({stored.outcome.value}) does not follow from its activations "
            f"(re-derived {outcome.value})"
        )
    ce = [p for p in trace.records if isinstance(p, ProtocolResult)]
    if len(ce) != 1 or ce[0].protocol != COUNTEREXAMPLE_PROTOCOL:
        raise ConceptVerificationError(f"{where}: exactly one counterexample record is required")
    expected_ce = counterexamples(
        [values[i] for i in val],
        [data.labels[i] for i in val],
        [values[i] for i in test],
        labels,
        test_ids,
        sign,
    )
    if JsonMap(expected_ce) != ce[0].measurements:
        raise ConceptVerificationError(f"{where}: the counterexamples do not re-derive (dropped?)")
    if (
        Assessment.derive(
            claim, [stored], ENCODING_POLICY, provenance_id=assessment.provenance_id
        ).id
        != assessment.id
    ):
        raise ConceptVerificationError(f"{where}: the assessment does not follow from the result")
    return feature, concept, data


# ------------------------------------------------------------------ use


def _reference_matches(record: InterventionRecord, ref: Any, basis: FeatureBasis) -> bool:
    if not isinstance(ref, JsonMap):
        return False
    digest = None if record.value is None else record.value.content_digest
    if ref.get("kind") == "zero":
        if basis is FeatureBasis.NEURON:
            return record.operation is InterventionOperation.ZERO
        return record.operation is InterventionOperation.DIRECTION and digest is None
    op = (
        InterventionOperation.CONSTANT
        if basis is FeatureBasis.NEURON
        else InterventionOperation.DIRECTION
    )
    return record.operation is op and digest == ref.get("digest")


def verify_use(result: UseResult) -> None:
    _, concept, data = verify_use_trace(result.trace)
    if (concept.id, data.id) != (result.concept.id, result.data.id):
        raise ConceptVerificationError("use: the result is about another concept/dataset")


def verify_use_trace(trace: TraceResult) -> tuple[FeatureRecord, ConceptRecord, ConceptDataset]:
    """Re-derive a use test from its trace alone (see :func:`verify_encoding_trace`)."""
    where = "use"
    claim = _one(trace, Claim, where)
    spec = _one(trace, ClaimTestSpec, where)
    stored = _one(trace, ClaimTestResult, where)
    assessment = _one(trace, Assessment, where)
    if spec.protocol != USE_PROTOCOL:
        raise ConceptVerificationError(f"{where}: not a concept_intervention test")
    feature, concept, data = _definitions(trace, spec.params, where)
    _model_digest(trace, feature, where)
    subset = spec.params.get("subset")
    label = {"positive": 1, "negative": 0, "all": None}
    if subset not in label:
        raise ConceptVerificationError(f"{where}: invalid subset")
    evaluated = data.indices("test", label[subset])  # type: ignore[index]
    eval_ids = [data.samples[i] for i in evaluated]
    estimand = Estimand.finite_sample(sample_set_id(eval_ids), len(eval_ids), "mean")
    if claim.subject.feature != feature.id or claim.estimand != estimand:
        raise ConceptVerificationError(f"{where}: the claim does not match its feature/subset")
    stats = stored.statistics
    primary = _record(trace, stats.get("primary_effect"), CausalEffect, where)
    control_ids = stats.get("control_effects")
    if not isinstance(control_ids, tuple):
        raise ConceptVerificationError(f"{where}: missing control effects")
    controls_fx = [_record(trace, c, CausalEffect, where) for c in control_ids]
    order = {s: k for k, s in enumerate(eval_ids)}
    for effect in (primary, *controls_fx):
        parts = [_record(trace, r.record_id, CausalEffect, where) for r in effect.derived_from]
        samples = [p.estimand.sample_id for p in parts]
        if (
            effect.estimand != estimand
            or any(p.estimand.scope is not EstimandScope.INSTANCE for p in parts)
            or sorted(samples, key=lambda s: order.get(s or "", -1)) != eval_ids
            or any(p.interventions != effect.interventions for p in parts)
            or effect.metric != primary.metric
        ):
            raise ConceptVerificationError(f"{where}: {effect.id} is not the mean over the subset")
        ordered = sorted(parts, key=lambda p: order[p.estimand.sample_id or ""])
        mean = sum(p.effect for p in ordered) / len(ordered)
        if not math.isclose(mean, effect.effect, rel_tol=1e-12, abs_tol=1e-12):
            raise ConceptVerificationError(f"{where}: {effect.id} is not the mean of its parts")
    if claim.target != primary.metric.target():
        raise ConceptVerificationError(f"{where}: the claim target is not the measured metric")
    intervention = spec.params.get("intervention")
    if not isinstance(intervention, JsonMap):
        raise ConceptVerificationError(f"{where}: the intervention is not declared")
    keep = intervention.get("mode") == "retain"
    ref = intervention.get("reference")

    def check(
        record: InterventionRecord, *, index: int | None, vector: torch.Tensor | None
    ) -> None:
        ok = record.site == feature.site and record.call_index == feature.call_index
        ok = ok and record.retain is keep and _reference_matches(record, ref, feature.basis)
        if feature.basis is FeatureBasis.NEURON:
            ok = ok and record.units == (index,) and record.unit_axes == (feature.axis,)
        else:
            assert record.direction is not None
            ok = ok and record.direction_axis == feature.axis
            if vector is None:
                assert feature.direction is not None
                ok = ok and record.direction.content_digest == feature.direction.content_digest
            else:
                got = trace.tensor(record.direction).double()
                ok = ok and float((got - vector.double()).abs().max()) <= _DIRECTION_TOLERANCE
        if not ok:
            raise ConceptVerificationError(
                f"{where}: intervention {record.id} is not the declared intervention on the "
                "declared feature (or control)"
            )

    def intervention_of(effect: CausalEffect) -> InterventionRecord:
        (ref_,) = effect.interventions
        return _record(trace, ref_.record_id, InterventionRecord, where)

    check(intervention_of(primary), index=feature.index, vector=None)
    controls = _controls(spec.params)
    sample_record = next(a for a in trace.activations if a.site == feature.site)
    d_size = sample_record.value.shape[feature.axis]
    train = data.indices("train")
    pooled_train = None
    if any(c.distribution == "covariance" for c in controls):
        pooled_train = torch.stack(
            [
                pooled_vectors(t, feature)
                for _, t in site_tensors(trace, data, train, feature.site, feature.call_index)
            ]
        )
    k = 0
    for control in controls:
        if control.kind == "random_neurons":
            assert feature.index is not None
            for j in random_indices_for(control, d_size, feature.index):
                check(intervention_of(controls_fx[k]), index=j, vector=None)
                k += 1
        else:
            for v in random_directions_for(control, d_size, pooled_train):
                check(intervention_of(controls_fx[k]), index=None, vector=v)
                k += 1
    if k != len(controls_fx):
        raise ConceptVerificationError(f"{where}: the controls do not match their declaration")
    criteria = spec.criteria
    try:
        crit = UseCriteria(
            criteria["min_fraction_beyond_controls"],  # type: ignore[arg-type]
            criteria.get("min_change"),  # type: ignore[arg-type]
            criteria.get("max_change"),  # type: ignore[arg-type]
        )
    except KeyError:
        raise ConceptVerificationError(f"{where}: no control criterion was declared") from None
    outcome, sup = evaluate_use(
        claim.relation, primary.effect, [c.effect for c in controls_fx], crit
    )
    rebuilt_stats = JsonMap(
        sup
        | {
            "mean_effect": primary.effect,
            "n": len(evaluated),
            "primary_effect": primary.id,
            "control_effects": [c.id for c in controls_fx],
            "control_mean_effects": [c.effect for c in controls_fx],
            "control_features": list(stats.get("control_features") or ()),  # type: ignore[arg-type]
        }
    )
    rebuilt = ClaimTestResult.for_claim(
        claim,
        spec,
        outcome=outcome,
        evidence=[primary, *controls_fx],
        statistics=rebuilt_stats,
        provenance_id=stored.provenance_id or "",
    )
    if rebuilt.id != stored.id:
        raise ConceptVerificationError(
            f"{stored.id} ({stored.outcome.value}) does not follow from its effects "
            f"(re-derived {outcome.value})"
        )
    if (
        Assessment.derive(claim, [stored], USE_POLICY, provenance_id=assessment.provenance_id).id
        != assessment.id
    ):
        raise ConceptVerificationError(f"{where}: the assessment does not follow from the result")
    return feature, concept, data


# ------------------------------------------------------------------ validation


def verify_validation(validation: Any) -> None:
    """Re-derive a :class:`ConceptValidationResult` entirely (see module docstring)."""
    from .validate import summary

    record = validation.record
    concept = validation.concept
    if (record.concept.record_id, record.feature.record_id) != (concept.id, concept.feature.id):
        raise ConceptVerificationError("the validation is about another concept or feature")
    data = _record(validation.trace, record.dataset.record_id, ConceptDataset, "validation")
    verify_feature(concept.feature, data)
    verify_encoding(validation.encoding)
    for u in (*validation.use, *validation.additional):
        verify_use(u)
    results = [validation.encoding, *validation.use, *validation.additional]
    for r in results:
        feat = _record(r.trace, record.feature.record_id, FeatureRecord, "validation")
        if feat != concept.feature.record or r.data.id != data.id:
            raise ConceptVerificationError("a test is about another feature or dataset")
        if _model_digest(r.trace, feat, "validation") != record.model_state_digest:
            raise ConceptVerificationError("a test ran on another model checkpoint")
    if summary(validation.encoding) != record.encoding:
        raise ConceptVerificationError("the encoding summary does not follow from its assessment")

    def key(u: Any) -> str:
        return str(u.assessment_id)

    if sorted((summary(u) for u in validation.use), key=key) != sorted(
        record.use, key=key
    ) or sorted((summary(a) for a in validation.additional), key=key) != sorted(
        record.additional, key=key
    ):
        raise ConceptVerificationError("the use summaries do not follow from their assessments")
    ce = validation.encoding.counterexamples
    if (
        ce.id != record.counterexamples
        or ce.measurements["false_positive_rate"] != record.false_positive_rate
        or ce.measurements["false_negative_rate"] != record.false_negative_rate
    ):
        raise ConceptVerificationError("the counterexample rates do not follow from the record")


def verify_validation_trace(
    trace: TraceResult, locate: Callable[[str], TraceResult]
) -> tuple[TraceResult, tuple[TraceResult, ...]]:
    """Re-derive a validation record from traces alone (e.g. after ``load_trace``).

    ``locate(assessment_id)`` returns the trace holding that test's assessment (raising
    ``KeyError`` if none was supplied). Every test is re-derived from its trace, and the
    summaries, counterexample rates, model checkpoint and derived status must follow.
    The feature's own derivation is checked separately (:func:`verify_feature_record`).
    Returns the encoding trace and the use/additional traces.
    """
    from .validate import record_summary

    record = _one(trace, ConceptValidation, "validation")
    ids = (record.concept.record_id, record.feature.record_id, record.dataset.record_id)

    def test_trace(assessment_id: str) -> TraceResult:
        try:
            return locate(assessment_id)
        except KeyError:
            raise ConceptVerificationError(
                f"validation: the test with assessment {assessment_id} was not supplied"
            ) from None

    encoding = test_trace(record.encoding.assessment_id)
    uses = tuple(test_trace(s.assessment_id) for s in (*record.use, *record.additional))
    checked = [(encoding, verify_encoding_trace(encoding))]
    checked += [(t, verify_use_trace(t)) for t in uses]
    for t, (feature, concept, data) in checked:
        if (concept.id, feature.id, data.id) != ids:
            raise ConceptVerificationError("a test is about another concept, feature or dataset")
        if _model_digest(t, feature, "validation") != record.model_state_digest:
            raise ConceptVerificationError("a test ran on another model checkpoint")
    if record_summary(encoding) != record.encoding:
        raise ConceptVerificationError("the encoding summary does not follow from its assessment")
    for t, s in zip(uses, (*record.use, *record.additional), strict=True):
        if record_summary(t) != s:
            raise ConceptVerificationError("a use summary does not follow from its assessment")
    ce = _one(encoding, ProtocolResult, "validation")
    if (
        ce.id != record.counterexamples
        or ce.measurements["false_positive_rate"] != record.false_positive_rate
        or ce.measurements["false_negative_rate"] != record.false_negative_rate
    ):
        raise ConceptVerificationError("the counterexample rates do not follow from the record")
    return encoding, uses
