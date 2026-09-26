"""The ENCODES claim test, ``concept_encoding`` v1 (plan §9, §11, §13; ADR-042).

"Feature F encodes concept C" means only: on the declared concept dataset's held-out
**test** split, F's fixed 1-D readout separates concept from non-concept samples
(AUROC, declared sign) better than the declared controls, by the declared criteria.
It never means the model uses C (``DECODABILITY_NOT_USE``).

The evidence is the MEASURED site activations of the test samples (one clean pass
per sample, all splits in one recording). Nothing is trained at evaluation time; a
fitted direction was fitted on the train split only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import torch
from torch import nn

from beyondnn.core.trace import Recording, TraceResult
from beyondnn.schema import (
    Assessment,
    AssessmentPolicy,
    Claim,
    ClaimSource,
    ClaimSourceKind,
    ClaimTestResult,
    ClaimTestSpec,
    Estimand,
    FeatureBasis,
    JsonMap,
    LabelSource,
    MethodIdentity,
    ModelDeclaration,
    Outcome,
    PolicyRequirement,
    ProtocolResult,
    ProvenanceRecord,
    RecordRef,
    Relation,
    Subject,
    TargetSpec,
    TraceLimitation,
    Verdict,
)

from ._core import (
    ConceptError,
    Control,
    auroc,
    copy_record,
    feature_values,
    permutations_for,
    pooled_vectors,
    quantiles,
    random_directions_for,
    random_indices_for,
    sample_set_id,
    superiority,
)
from .data import Concept, ConceptData
from .features import site_tensors

__all__ = [
    "ENCODING_POLICY",
    "ENCODING_PROTOCOL",
    "EncodingCriteria",
    "EncodingResult",
    "encoding_criteria",
    "encoding_test",
    "label_permutation",
    "random_directions",
    "random_neurons",
]

ENCODING_PROTOCOL = "concept_encoding"
COUNTEREXAMPLE_PROTOCOL = "concept_counterexamples"
ENCODING_POLICY = AssessmentPolicy(
    name="concept_encoding_v1",
    version=1,
    requirements=(PolicyRequirement(relation=Relation.ENCODES, protocols=(ENCODING_PROTOCOL,)),),
)


# ------------------------------------------------------------------ declarations


def random_directions(n: int, *, seed: int, distribution: str) -> Control:
    """``n`` random unit directions at the feature's site and axis. ``distribution``
    must be declared: ``isotropic`` (N(0, I)) or ``covariance`` (N(0, Σ_train), the
    stronger null inside the activation subspace)."""
    if distribution not in ("isotropic", "covariance"):
        raise ConceptError("distribution must be declared: 'isotropic' or 'covariance'")
    return Control("random_directions", _count(n), _seed(seed), distribution)


def random_neurons(n: int, *, seed: int) -> Control:
    """``n`` random other indices of the feature's axis (neuron features)."""
    return Control("random_neurons", _count(n), _seed(seed))


def label_permutation(n: int, *, seed: int) -> Control:
    """``n`` permutations of the held-out labels scored with the observed feature
    (encoding tests only)."""
    return Control("label_permutation", _count(n), _seed(seed))


def _count(n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ConceptError("a control needs n >= 1 draws")
    return n


def _seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConceptError("control seeds are ints >= 0 (declared explicitly)")
    return seed


@dataclass(frozen=True, slots=True)
class EncodingCriteria:
    """Declared before running. ``min_fraction_below`` applies to every control: the
    observed AUROC's superiority over it must reach this value (mandatory).
    ``min_auroc`` is an optional absolute floor."""

    min_fraction_below: float
    min_auroc: float | None = None

    def as_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"min_fraction_below": self.min_fraction_below}
        if self.min_auroc is not None:
            out["min_auroc"] = self.min_auroc
        return out


def encoding_criteria(
    *, min_fraction_below: float, min_auroc: float | None = None
) -> EncodingCriteria:
    if not 0.5 <= min_fraction_below <= 1.0:
        raise ConceptError("min_fraction_below is in [0.5, 1] (a control criterion is mandatory)")
    if min_auroc is not None and not 0.5 <= min_auroc <= 1.0:
        raise ConceptError("min_auroc is in [0.5, 1]")
    return EncodingCriteria(min_fraction_below, min_auroc)


# ------------------------------------------------------------------ evaluation (shared)


def evaluate_encoding(
    *,
    values: Sequence[float],
    labels: Sequence[int],
    sign: int,
    basis: FeatureBasis,
    index: int | None,
    pooled_test: torch.Tensor,
    pooled_train: torch.Tensor | None,
    controls: Sequence[Control],
    criteria: EncodingCriteria,
) -> tuple[Outcome, dict[str, Any]]:
    """The protocol's decision from raw values (used by the runner and verification)."""
    signed = [sign * v for v in values]
    observed = auroc(signed, labels)
    rows = []
    holds = criteria.min_auroc is None or observed >= criteria.min_auroc
    for control in controls:
        if control.kind == "label_permutation":
            scores = [
                auroc(signed, [labels[j] for j in perm])
                for perm in permutations_for(control, len(labels))
            ]
        elif control.kind == "random_directions":
            dirs = random_directions_for(control, pooled_test.shape[1], pooled_train)
            scores = [_free(auroc((pooled_test @ v).tolist(), labels)) for v in dirs]
        elif control.kind == "random_neurons":
            if basis is not FeatureBasis.NEURON or index is None:
                raise ConceptError("random-neuron controls apply to neuron features")
            idx = random_indices_for(control, pooled_test.shape[1], index)
            scores = [_free(auroc(pooled_test[:, j].tolist(), labels)) for j in idx]
        else:
            raise ConceptError(f"unknown control {control.kind}")
        stats = superiority(observed, scores)
        holds = holds and stats["superiority"] >= criteria.min_fraction_below
        rows.append(control.identity() | stats | quantiles(scores) | {"values": scores})
    statistics = {
        "auroc": observed,
        "expected_sign": sign,
        "n_test": len(labels),
        "n_concept": sum(labels),
        "controls": rows,
    }
    return (Outcome.SUPPORTS if holds else Outcome.CONTRADICTS), statistics


def _free(a: float) -> float:
    """Sign-free AUROC for random features (they have no preferred sign)."""
    return max(a, 1.0 - a)


def counterexamples(
    val_values: Sequence[float],
    val_labels: Sequence[int],
    test_values: Sequence[float],
    test_labels: Sequence[int],
    test_samples: Sequence[str],
    sign: int,
) -> dict[str, Any]:
    """Threshold on the **val** split (max balanced accuracy, ties to the lower threshold),
    then every test false positive / false negative (plan §13)."""
    signed_val = [sign * v for v in val_values]
    if len(set(val_labels)) < 2:
        raise ConceptError("the val split needs both concept and non-concept samples")
    points = sorted(set(signed_val))
    candidates = [points[0] - 1.0] + [(a + b) / 2 for a, b in pairwise(points)] + [points[-1] + 1.0]
    best: tuple[float, float] | None = None
    for t in candidates:
        tpr = _rate([s >= t for s, y in zip(signed_val, val_labels, strict=True) if y == 1])
        tnr = _rate([s < t for s, y in zip(signed_val, val_labels, strict=True) if y == 0])
        score = (tpr + tnr) / 2
        if best is None or score > best[0]:
            best = (score, t)
    assert best is not None
    val_bacc, threshold = best
    fp: list[tuple[str, float]] = []
    fn: list[tuple[str, float]] = []
    for sample, v, y in zip(test_samples, test_values, test_labels, strict=True):
        s = sign * v
        if y == 0 and s >= threshold:
            fp.append((sample, v))
        elif y == 1 and s < threshold:
            fn.append((sample, v))
    n_pos = sum(test_labels)
    n_neg = len(test_labels) - n_pos
    fp.sort(key=lambda r: (-sign * r[1], r[0]))
    fn.sort(key=lambda r: (sign * r[1], r[0]))
    return {
        "threshold": threshold,
        "val_balanced_accuracy": val_bacc,
        "false_positives": [list(r) for r in fp],
        "false_negatives": [list(r) for r in fn],
        "false_positive_rate": len(fp) / n_neg,
        "false_negative_rate": len(fn) / n_pos,
        "n_concept": n_pos,
        "n_non_concept": n_neg,
    }


def _rate(flags: list[bool]) -> float:
    return sum(flags) / len(flags) if flags else 0.0


def claim_source(concept: Concept) -> ClaimSource:
    if concept.record.label_source is LabelSource.GENERATED:
        return ClaimSource(kind=ClaimSourceKind.GENERATED, detail=concept.record.generated_label)
    return ClaimSource(kind=ClaimSourceKind.USER, detail=concept.record.label_source.value)


# ------------------------------------------------------------------ runner


@dataclass(frozen=True, slots=True, eq=False)
class EncodingResult:
    """An ENCODES claim test: its trace (activations, copies of the feature/concept/
    dataset records, claim, spec, result, counterexamples, assessment)."""

    trace: TraceResult
    concept: Concept
    data: ConceptData

    def _one(self, kind: type) -> Any:
        (record,) = [r for r in self.trace.records if isinstance(r, kind)]
        return record

    @property
    def claim(self) -> Claim:
        record: Claim = self._one(Claim)
        return record

    @property
    def spec(self) -> ClaimTestSpec:
        record: ClaimTestSpec = self._one(ClaimTestSpec)
        return record

    @property
    def result(self) -> ClaimTestResult:
        record: ClaimTestResult = self._one(ClaimTestResult)
        return record

    @property
    def assessment(self) -> Assessment:
        record: Assessment = self._one(Assessment)
        return record

    @property
    def counterexamples(self) -> ProtocolResult:
        record: ProtocolResult = self._one(ProtocolResult)
        return record

    @property
    def outcome(self) -> Outcome:
        return self.result.outcome

    @property
    def verdict(self) -> Verdict:
        return self.assessment.verdict

    @property
    def auroc(self) -> float:
        value = self.result.statistics["auroc"]
        assert isinstance(value, float)
        return value


def _check_scope(model: nn.Module, concept: Concept, data: ConceptData) -> str:
    from beyondnn.provenance import fingerprint_model

    feature = concept.feature.record
    digest = fingerprint_model(model).state_digest
    if feature.model_state_digest is not None and feature.model_state_digest != digest:
        raise ConceptError(
            "the feature was derived on another model checkpoint; cross-checkpoint "
            "evaluation is a separate test (plan §2, request §37)"
        )
    if feature.source.dataset is not None and feature.source.dataset != data.id:
        raise ConceptError(
            "a fitted/searched feature is evaluated only on the concept dataset it was "
            "derived on (its test split is guaranteed disjoint from its train split)"
        )
    if len(data.inputs) != len(data.record.samples):
        raise ConceptError(
            "this concept dataset was reconstructed from records (load_validation) and has no "
            "inputs; tests cannot be re-run with it"
        )
    for i in data.record.indices("test"):
        from beyondnn.core.samples import sample_id

        if sample_id(*data.inputs[i], model_kwargs=data.kwargs[i]) != data.record.samples[i]:
            raise ConceptError("a dataset input no longer matches its recorded identity")
    return digest


def derived_provenance(trace: TraceResult, name: str, params: dict[str, Any]) -> ProvenanceRecord:
    """Provenance of a derived record: the conditions of the recording's first clean
    pass, with a ``concepts:<name>`` method (as Phase 5 does)."""
    first = min(trace.inputs, key=lambda i: i.pass_index or 0)
    ref = trace.origin(first)
    record = ProvenanceRecord(
        model=ref.model,
        environment=ref.environment,
        execution=ref.execution,
        method=MethodIdentity(name=f"concepts:{name}", version="1", params=JsonMap(params)),
        declared_model=ref.declared_model,
    )
    stored = trace._add(record)
    assert isinstance(stored, ProvenanceRecord)
    return stored


def copy_definitions(trace: TraceResult, concept: Concept, data: ConceptData) -> None:
    copy_record(trace, concept.feature.store, concept.feature.record)
    if concept.generated is not None:
        copy_record(trace, concept.generated.store, concept.generated.record)
    copy_record(trace, concept.store, concept.record)
    copy_record(trace, data.store, data.record)


def encoding_test(
    model: nn.Module,
    concept: Concept,
    data: ConceptData,
    *,
    controls: Sequence[Control],
    criteria: EncodingCriteria,
    statement: str | None = None,
    declared_model: ModelDeclaration | None = None,
) -> EncodingResult:
    """Run ``concept_encoding`` v1 (see module docstring). ``controls`` and ``criteria``
    are mandatory; the feature's declared sign is used for the observed AUROC."""
    if not controls:
        raise ConceptError("an encoding test needs at least one control (request §16)")
    if not isinstance(criteria, EncodingCriteria):
        raise ConceptError("criteria must come from encoding_criteria()")
    controls = tuple(controls)
    for c in controls:
        if not isinstance(c, Control):
            raise ConceptError(
                "controls come from random_directions/random_neurons/label_permutation"
            )
    _check_scope(model, concept, data)
    feature = concept.feature.record
    ds = data.record
    train, val, test = ds.indices("train"), ds.indices("val"), ds.indices("test")
    if not val or not test:
        raise ConceptError("an encoding test needs val (threshold) and test (evaluation) splits")
    order = [*train, *val, *test]
    rec = Recording(
        model, sites=[feature.site.module], retention="cpu", declared_model=declared_model
    )
    sign = concept.feature.expected_sign
    holder: list[TraceResult] = []

    def finalize(trace: TraceResult) -> None:
        store = concept.feature.store
        rows = {
            i: r
            for i, r in zip(
                order,
                site_tensors(trace, data, order, feature.site, feature.call_index),
                strict=True,
            )
        }
        values = {i: feature_values(t, feature, store) for i, (_, t) in rows.items()}
        pooled = {i: pooled_vectors(t, feature) for i, (_, t) in rows.items()}
        labels = [ds.labels[i] for i in test]
        pooled_train = torch.stack([pooled[i] for i in train]) if len(train) >= 2 else None
        outcome, statistics = evaluate_encoding(
            values=[values[i] for i in test],
            labels=labels,
            sign=sign,
            basis=feature.basis,
            index=feature.index,
            pooled_test=torch.stack([pooled[i] for i in test]),
            pooled_train=pooled_train,
            controls=controls,
            criteria=criteria,
        )
        copy_definitions(trace, concept, data)
        test_ids = [ds.samples[i] for i in test]
        params = {
            "concept": concept.id,
            "feature": feature.id,
            "dataset": ds.id,
            "split": "test",
            "readout": "feature_activation (fixed 1-D; nothing trained at evaluation)",
            "metric": "auroc",
            "expected_sign": sign,
            "controls": [c.identity() for c in controls],
        }
        provenance = derived_provenance(trace, ENCODING_PROTOCOL, {"feature": feature.id})
        claim = Claim(
            statement=statement
            or f"feature {feature.id} encodes {concept.record.label!r} on {ds.name} (test split)",
            relation=Relation.ENCODES,
            subject=Subject(site=feature.site, feature=feature.id),
            target=TargetSpec(
                metric="concept_label",
                params=JsonMap({"concept": concept.id, "dataset": ds.id, "split": "test"}),
            ),
            estimand=Estimand.finite_sample(sample_set_id(test_ids), len(test_ids), "auroc"),
            source=claim_source(concept),
        )
        spec = ClaimTestSpec(
            protocol=ENCODING_PROTOCOL,
            protocol_version=1,
            applicable_relations=(Relation.ENCODES,),
            criteria=JsonMap(criteria.as_json()),
            params=JsonMap(params),
        )
        trace._add(claim)
        trace._add(spec)
        result = ClaimTestResult.for_claim(
            claim,
            spec,
            outcome=outcome,
            evidence=[rows[i][0] for i in test],
            statistics=JsonMap(statistics),
            provenance_id=provenance.id,
        )
        trace._add(result)
        ce = counterexamples(
            [values[i] for i in val],
            [ds.labels[i] for i in val],
            [values[i] for i in test],
            labels,
            test_ids,
            sign,
        )
        trace._add(
            ProtocolResult(
                protocol=COUNTEREXAMPLE_PROTOCOL,
                protocol_version=1,
                params=JsonMap(
                    {
                        "threshold_rule": "val split: max balanced accuracy, ties to the lower",
                        "concept": concept.id,
                        "feature": feature.id,
                        "dataset": ds.id,
                        "expected_sign": sign,
                    }
                ),
                measurements=JsonMap(ce),
                samples=tuple(test_ids),
                provenance_id=provenance.id,
                derived_from=tuple(RecordRef.to(rows[i][0]) for i in (*val, *test)),
            )
        )
        assessment = Assessment.derive(claim, [result], ENCODING_POLICY)
        trace._add(assessment)
        codes = ["DECODABILITY_NOT_USE", "FEATURE_MAY_CARRY_OTHER_INFORMATION"]
        if feature.basis is FeatureBasis.SAE:
            codes += [
                "SAE_FEATURE_SPLITTING",
                "SAE_FEATURE_ABSORPTION",
                "SAE_POLYSEMANTICITY_NOT_EXCLUDED",
                "SAE_RECONSTRUCTION_ERROR",
                "SAE_MISSING_FEATURES",
            ]
        for code in codes:
            trace._add(TraceLimitation(code=code, applies_to=(result.id,)))
        holder.append(trace)

    with torch.no_grad(), rec:
        for i in order:
            model(*data.inputs[i], **data.kwargs[i])
        rec._finalizers.append(finalize)
    return EncodingResult(rec.result, concept, data)
