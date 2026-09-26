"""Concept validation: derive a semantic status from recorded assessments (plan §6-§7).

:func:`validate` runs nothing. It checks that the encoding and use results are about
the same concept, feature, dataset and model, summarises their assessments, and
builds a ``concept_validation`` record whose status is *derived* under the declared
policy (VALIDATED_CONCEPT only if every requirement holds; otherwise
PROPOSED_CONCEPT with the unmet requirements listed). There is no concept score.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from beyondnn.core.samples import sample_id
from beyondnn.core.trace import TraceResult, recording
from beyondnn.schema import (
    Assessment,
    AssessmentSummary,
    Claim,
    ClaimTestSpec,
    ConceptActivation,
    ConceptPolicy,
    ConceptValidation,
    MethodIdentity,
    ModelDeclaration,
    ProvenanceRecord,
    RecordRef,
    Relation,
    SemanticStatus,
    TraceLimitation,
    derive_semantic_status,
)

from ._core import ConceptError, copy_record, feature_values, new_store
from .data import Concept
from .encoding import EncodingResult
from .use import UseResult

__all__ = [
    "POLICY_V1",
    "ConceptActivationResult",
    "ConceptValidationResult",
    "activation",
    "load_validation",
    "validate",
]

#: The v1 concept policy: SUPPORTED encoding (with controls) + >= 1 SUPPORTED use claim
#: (with controls) + recorded counterexamples. No rate caps (rates are shown).
POLICY_V1 = ConceptPolicy(name="concept_validation_v1", version=1)

_CONTROL_KEYS = {
    "concept_encoding": "min_fraction_below",
    "concept_intervention": "min_fraction_beyond_controls",
}


def summary(result: EncodingResult | UseResult) -> AssessmentSummary:
    """What a validation relies on from one test (re-derived by composition)."""
    return record_summary(result.trace)


def record_summary(trace: TraceResult) -> AssessmentSummary:
    """:func:`summary` from a test's trace alone (one claim, spec and assessment)."""
    (spec,) = [r for r in trace.records if isinstance(r, ClaimTestSpec)]
    (claim,) = [r for r in trace.records if isinstance(r, Claim)]
    (assessment,) = [r for r in trace.records if isinstance(r, Assessment)]
    controls = spec.params.get("controls")
    key = _CONTROL_KEYS.get(spec.protocol)
    declared = bool(controls) and key is not None and key in spec.criteria
    return AssessmentSummary(
        assessment_id=assessment.id,
        claim_id=claim.id,
        relation=claim.relation,
        protocol=spec.protocol,
        verdict=assessment.verdict,
        controls_declared=declared,
    )


def _model_digest(trace: TraceResult) -> str:
    digests = {trace.origin(i).model.state_digest for i in trace.inputs}
    if len(digests) != 1:
        raise ConceptError("a concept test trace spans several model states")
    (digest,) = digests
    return digest


@dataclass(frozen=True, slots=True, eq=False)
class ConceptValidationResult:
    """The validation record (in its own trace) with the results it summarises."""

    trace: TraceResult
    concept: Concept
    encoding: EncodingResult
    use: tuple[UseResult, ...]
    additional: tuple[UseResult, ...] = ()

    @property
    def record(self) -> ConceptValidation:
        (record,) = [r for r in self.trace.records if isinstance(r, ConceptValidation)]
        return record

    @property
    def semantic_status(self) -> SemanticStatus:
        return self.record.semantic_status

    @property
    def unmet(self) -> tuple[str, ...]:
        return self.record.unmet

    @property
    def scope(self) -> str:
        return self.record.scope

    def describe(self) -> str:
        """Scoped wording (request §39); never 'the model uses C'."""
        r = self.record
        lines = [
            f"concept {self.concept.record.label!r} ({self.concept.record.label_source.value} "
            f"label) on feature {r.feature.record_id}: {r.semantic_status.value.upper()} "
            f"under {r.policy.name}/v{r.policy.version}",
            f"  scope: {r.scope}",
            f"  ENCODING CLAIM: {r.encoding.verdict.value.upper()} "
            f"(held-out AUROC {self.encoding.auroc:.3f} against declared controls)",
        ]
        for u, res in zip(r.use, self._ordered_use(), strict=True):
            lines.append(
                f"  USE CLAIM ({u.relation.value}, {res.intervention.mode} with "
                f"{res.intervention.reference.name} reference): {u.verdict.value.upper()} "
                f"(mean effect {res.mean_effect:+.4g})"
            )
        for a, res in zip(r.additional, self._ordered_additional(), strict=True):
            lines.append(
                f"  additional ({a.relation.value}, {res.intervention.mode} with "
                f"{res.intervention.reference.name} reference): {a.verdict.value.upper()}"
            )
        lines.append(
            f"  counterexamples: false-positive rate {r.false_positive_rate:.3f}, "
            f"false-negative rate {r.false_negative_rate:.3f}"
        )
        for reason in r.unmet:
            lines.append(f"  unmet: {reason}")
        return "\n".join(lines)

    def _ordered_use(self) -> list[UseResult]:
        by = {u.assessment.id: u for u in self.use}
        return [by[s.assessment_id] for s in self.record.use]

    def _ordered_additional(self) -> list[UseResult]:
        by = {u.assessment.id: u for u in self.additional}
        return [by[s.assessment_id] for s in self.record.additional]


def validate(
    concept: Concept,
    *,
    encoding: EncodingResult,
    use: Sequence[UseResult],
    additional: Sequence[UseResult] = (),
    policy: ConceptPolicy = POLICY_V1,
) -> ConceptValidationResult:
    """Derive the concept's semantic status from its tests under ``policy`` (see module
    docstring). Refuses results about another concept, feature, dataset or model."""
    use, additional = tuple(use), tuple(additional)
    results: list[EncodingResult | UseResult] = [encoding, *use, *additional]
    for r in results:
        params = r.spec.params
        if (params.get("concept"), params.get("feature"), params.get("dataset")) != (
            concept.id,
            concept.feature.id,
            encoding.data.id,
        ):
            raise ConceptError("a test result is about another concept, feature or dataset")
        if r.concept.id != concept.id:
            raise ConceptError("a test result was run for another concept object")
    digests = {_model_digest(r.trace) for r in results}
    if len(digests) != 1:
        raise ConceptError("the encoding and use tests ran on different model checkpoints")
    (digest,) = digests
    for u in use:
        if u.claim.relation not in (Relation.DECREASES, Relation.INCREASES):
            raise ConceptError("use claims are DECREASES/INCREASES; pass others as additional")
    ce = encoding.counterexamples
    fp = ce.measurements["false_positive_rate"]
    fn = ce.measurements["false_negative_rate"]
    assert isinstance(fp, float)
    assert isinstance(fn, float)
    enc_summary = summary(encoding)
    use_summaries = tuple(summary(u) for u in use)
    status, unmet = derive_semantic_status(policy, enc_summary, use_summaries, fp, fn)
    feature = concept.feature.record
    ds = encoding.data.record
    site_name = feature.site.module + (
        f"/{feature.site.output_path}" if feature.site.output_path else ""
    )
    scope = (
        f"dataset {ds.name} (test split, n={len(ds.indices('test'))}); model {digest[:23]}; "
        f"site {site_name}"
        f" axis {feature.axis}; "
        + "; ".join(
            f"{u.claim.relation.value} {u.claim.target.metric} under "
            f"{u.intervention.mode}/{u.intervention.reference.name}"
            for u in (*use, *additional)
        )
    )
    store = new_store()
    copy_record(store, concept.feature.store, feature)
    if concept.generated is not None:
        copy_record(store, concept.generated.store, concept.generated.record)
    copy_record(store, concept.store, concept.record)
    copy_record(store, encoding.data.store, ds)
    record = ConceptValidation(
        concept=RecordRef.to(concept.record),
        feature=RecordRef.to(feature),
        dataset=RecordRef.to(ds),
        policy=policy,
        encoding=enc_summary,
        use=use_summaries,
        additional=tuple(summary(a) for a in additional),
        counterexamples=ce.id,
        false_positive_rate=fp,
        false_negative_rate=fn,
        model_state_digest=digest,
        scope=scope,
        semantic_status=status,
        unmet=unmet,
    )
    store._add(record)
    store._add(
        TraceLimitation(code="CONCEPT_SCOPE_LIMITED", detail=scope[:200], applies_to=(record.id,))
    )
    store._seal()
    return ConceptValidationResult(store, concept, encoding, use, additional)


# ------------------------------------------------------------------ concept activations


@dataclass(frozen=True, slots=True, eq=False)
class ConceptActivationResult:
    """A feature's value on one input. ``record`` is a VALIDATED_CONCEPT
    ``ConceptActivation`` only if the validation is VALIDATED; otherwise ``None`` and
    ``value`` is a MEASURED-derived feature activation of a PROPOSED concept."""

    trace: TraceResult
    value: float
    status: SemanticStatus
    record: ConceptActivation | None


def activation(
    model: nn.Module,
    *inputs: Any,
    validation: ConceptValidationResult,
    model_kwargs: dict[str, Any] | None = None,
    declared_model: ModelDeclaration | None = None,
) -> ConceptActivationResult:
    """The concept's feature activation on one input (one clean traced pass). A
    VALIDATED_CONCEPT record is created only for a VALIDATED validation on this exact
    model checkpoint (ADR-009)."""
    feature = validation.concept.feature
    kwargs = dict(model_kwargs or {})
    rec = recording(
        model, sites=[feature.record.site.module], retention="cpu", declared_model=declared_model
    )
    out: dict[str, Any] = {}

    def finalize(trace: TraceResult) -> None:
        (inp,) = trace.inputs
        origin = trace.origin(inp)
        if origin.model.state_digest != validation.record.model_state_digest:
            raise ConceptError("the validation was derived on another model checkpoint")
        act = trace.activation(
            feature.record.site.module,
            output_path=feature.record.site.output_path,
            pass_index=inp.pass_index,
            call_index=feature.record.call_index,
        )
        value = feature_values(trace.tensor(act), feature.record, feature.store)
        out["value"] = value
        if validation.semantic_status is not SemanticStatus.VALIDATED_CONCEPT:
            out["record"] = None
            return
        provenance = trace._add(
            ProvenanceRecord(
                model=origin.model,
                environment=origin.environment,
                execution=origin.execution,
                method=MethodIdentity(
                    name="concepts:concept_activation",
                    version="1",
                    params=_json({"validation": validation.record.id}),
                ),
                declared_model=origin.declared_model,
            )
        )
        copy_record(trace, feature.store, feature.record)
        record = ConceptActivation(
            validation=validation.record.id,
            concept=validation.concept.id,
            feature=feature.id,
            sample_id=sample_id(*inputs, model_kwargs=kwargs),
            value=value,
            provenance_id=provenance.id,
            derived_from=(RecordRef.to(act),),
        )
        out["record"] = trace._add(record)
        trace._add(
            TraceLimitation(
                code="CONCEPT_SCOPE_LIMITED",
                detail=validation.record.scope[:200],
                applies_to=(record.id,),
            )
        )

    with torch.no_grad(), rec:
        model(*inputs, **kwargs)
        rec._finalizers.append(finalize)
    status = validation.semantic_status
    return ConceptActivationResult(rec.result, out["value"], status, out.get("record"))


def _json(data: dict[str, Any]) -> Any:
    from beyondnn.schema import JsonMap

    return JsonMap(data)


def load_validation(traces: Sequence[Any]) -> ConceptValidationResult:
    """Rebuild a :class:`ConceptValidationResult` from saved traces alone (ADR-050).

    ``traces`` are ``TraceResult`` objects or saved trace paths: the validation's trace and
    the traces of its encoding and use tests (anything else is ignored). Everything is
    re-derived first (``verify_validation_trace``; the fitted feature from the encoding
    trace's train-split activations). The result can be composed into a WHY
    (``bnn.compose(trace, concepts=[...])``) and audited; its dataset has no inputs and its
    references have no tensors, so it can never be used to run a new test.
    """
    import os
    from pathlib import Path

    from beyondnn.core.persistence import load_trace
    from beyondnn.schema import (
        ConceptDataset,
        ConceptRecord,
        FeatureRecord,
        GeneratedLabel,
        JsonMap,
    )

    from .data import ConceptData, GeneratedLabelResult
    from .encoding import EncodingResult
    from .features import Feature
    from .use import FeatureIntervention, Reference, UseResult
    from .verify import (
        ConceptVerificationError,
        needs_derivation,
        verify_feature_record,
        verify_validation_trace,
    )

    loaded = [load_trace(Path(t)) if isinstance(t, (str, os.PathLike)) else t for t in traces]
    if not all(isinstance(t, TraceResult) for t in loaded):
        raise TypeError("load_validation takes TraceResult objects or saved trace paths")
    holders = [t for t in loaded if any(isinstance(r, ConceptValidation) for r in t.records)]
    if len(holders) != 1:
        raise ConceptError(f"expected exactly one validation trace, got {len(holders)}")
    (vtrace,) = holders
    by_assessment = {r.id: t for t in loaded for r in t.records if isinstance(r, Assessment)}

    def locate(assessment_id: str) -> TraceResult:
        return by_assessment[assessment_id]

    encoding_trace, use_traces = verify_validation_trace(vtrace, locate)
    (record,) = [r for r in vtrace.records if isinstance(r, ConceptValidation)]

    def one(kind: type, record_id: str) -> Any:
        found = vtrace.get(record_id)
        if not isinstance(found, kind):
            raise ConceptError(f"{record_id} is not a {kind.__name__} in the validation trace")
        return found

    feature_record = one(FeatureRecord, record.feature.record_id)
    concept_record = one(ConceptRecord, record.concept.record_id)
    dataset_record = one(ConceptDataset, record.dataset.record_id)
    fit_trace = None
    if needs_derivation(feature_record):
        try:
            verify_feature_record(feature_record, encoding_trace, dataset_record)
        except ConceptError as exc:  # the recording lacks train samples: not re-derivable
            raise ConceptVerificationError(
                f"{feature_record.id}: the fitted feature cannot be re-derived from the "
                f"supplied traces ({exc})"
            ) from None
        fit_trace = encoding_trace
    feature = Feature(feature_record, vtrace, fit_trace)
    generated = None
    if concept_record.generated_label is not None:
        generated = GeneratedLabelResult(
            one(GeneratedLabel, concept_record.generated_label), vtrace
        )
    concept = Concept(concept_record, feature, vtrace, generated)
    data = ConceptData(dataset_record, (), (), vtrace)

    def intervention_of(trace: TraceResult) -> FeatureIntervention:
        (spec,) = [r for r in trace.records if isinstance(r, ClaimTestSpec)]
        declared = spec.params.get("intervention")
        assert isinstance(declared, JsonMap)
        ref = declared.get("reference")
        assert isinstance(ref, JsonMap)
        return FeatureIntervention(
            str(declared.get("mode")),
            Reference(str(ref.get("kind")), str(ref.get("name")), None),
        )

    uses = [UseResult(t, concept, data, intervention_of(t)) for t in use_traces]
    n_use = len(record.use)
    return ConceptValidationResult(
        vtrace,
        concept,
        EncodingResult(encoding_trace, concept, data),
        tuple(uses[:n_use]),
        tuple(uses[n_use:]),
    )
