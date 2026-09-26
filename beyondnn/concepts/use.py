"""The causal-use claim test, ``concept_intervention`` v1 (plan §10-§12; ADR-042).

"Intervening on feature F by the declared intervention changes target Y on the
declared evaluation subset, beyond the same intervention on matched random features":

* ``remove`` (DECREASES / INCREASES): the feature's value (neuron) or coordinate
  (direction, SAE decoder direction) is replaced by the declared reference's;
* ``retain`` (SUFFICIENT_FOR): only the feature is kept, everything else at the
  site comes from the reference (site-relative sufficiency).

The intervention and its reference are **required**; nothing defaults to zero. The
evidence is INTERVENTIONAL: one Phase-2 comparison family (paired baseline and
intervention passes per sample), and a FINITE_SAMPLE mean effect for the feature and
for every control. Causal language is scoped to this intervention, subset and target.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

import beyondnn.interventions as iv
from beyondnn.core.trace import TraceResult
from beyondnn.interventions.runner import ComparisonFamily
from beyondnn.provenance import fingerprint_model
from beyondnn.schema import (
    Assessment,
    AssessmentPolicy,
    CausalEffect,
    Claim,
    ClaimTestResult,
    ClaimTestSpec,
    Estimand,
    FeatureBasis,
    FeatureRecord,
    JsonMap,
    ModelDeclaration,
    Outcome,
    PolicyRequirement,
    RecordRef,
    Relation,
    Subject,
    TraceLimitation,
    Verdict,
)

from ._core import (
    ConceptError,
    Control,
    pooled_vectors,
    random_directions_for,
    random_indices_for,
    sample_set_id,
    superiority,
)
from .data import Concept, ConceptData
from .encoding import _check_scope, claim_source, copy_definitions, derived_provenance
from .features import site_tensors

__all__ = [
    "USE_POLICY",
    "USE_PROTOCOL",
    "FeatureIntervention",
    "Reference",
    "UseCriteria",
    "UseResult",
    "reference",
    "remove",
    "retain",
    "use_criteria",
    "use_test",
    "zero",
]

USE_PROTOCOL = "concept_intervention"
_RELATIONS = {
    "decreases": Relation.DECREASES,
    "increases": Relation.INCREASES,
    "sufficient_for": Relation.SUFFICIENT_FOR,
}
USE_POLICY = AssessmentPolicy(
    name="concept_use_v1",
    version=1,
    requirements=tuple(
        PolicyRequirement(relation=r, protocols=(USE_PROTOCOL,)) for r in _RELATIONS.values()
    ),
)


# ------------------------------------------------------------------ declarations


@dataclass(frozen=True, slots=True, eq=False)
class Reference:
    """What replaces the feature (removal) or everything else (retention): zeros, or a
    tensor of the site leaf's exact shape (e.g. a train-mean activation), named."""

    kind: str
    name: str
    tensor: torch.Tensor | None = None

    def identity(self) -> dict[str, Any]:
        if self.kind == "tensor" and self.tensor is None:
            raise ConceptError(
                f"reference {self.name!r} was reconstructed from records (load_validation) and "
                "has no tensor; it cannot be used to run a test"
            )
        if self.tensor is None:
            return {"kind": "zero", "name": self.name}
        from beyondnn.attribution.spec import tensor_digest

        return {"kind": "tensor", "name": self.name, "digest": tensor_digest(self.tensor)}


def zero() -> Reference:
    """The zero reference (declared explicitly; zero is not neutral: OOD limitations)."""
    return Reference("zero", "zero")


def reference(tensor: torch.Tensor, *, name: str) -> Reference:
    """A reference tensor of the site leaf's exact shape (e.g. ``"train_mean"``)."""
    if not isinstance(tensor, torch.Tensor) or (
        tensor.is_floating_point() and not bool(torch.isfinite(tensor).all())
    ):
        raise ConceptError("a reference is a finite tensor")
    if not name or not name.replace("_", "").isalnum():
        raise ConceptError("a reference has a name token (letters, digits, _)")
    return Reference(
        "tensor", name, tensor.detach().to("cpu").clone(memory_format=torch.contiguous_format)
    )


@dataclass(frozen=True, slots=True, eq=False)
class FeatureIntervention:
    """``mode``: ``remove`` (replace the feature by the reference's) or ``retain``
    (keep only the feature; everything else from the reference)."""

    mode: str
    reference: Reference

    def identity(self) -> dict[str, Any]:
        return {"mode": self.mode, "reference": self.reference.identity()}


def remove(reference: Reference) -> FeatureIntervention:
    if not isinstance(reference, Reference):
        raise ConceptError("remove() needs an explicit reference: zero() or reference(...)")
    return FeatureIntervention("remove", reference)


def retain(reference: Reference) -> FeatureIntervention:
    if not isinstance(reference, Reference):
        raise ConceptError("retain() needs an explicit reference: zero() or reference(...)")
    return FeatureIntervention("retain", reference)


@dataclass(frozen=True, slots=True)
class UseCriteria:
    """Declared before running. DECREASES/INCREASES: ``min_change`` (> 0, in target units)
    and ``min_fraction_beyond_controls``; SUFFICIENT_FOR: ``max_change`` (>= 0) and
    ``min_fraction_beyond_controls`` (controls change the target more)."""

    min_fraction_beyond_controls: float
    min_change: float | None = None
    max_change: float | None = None

    def as_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"min_fraction_beyond_controls": self.min_fraction_beyond_controls}
        if self.min_change is not None:
            out["min_change"] = self.min_change
        if self.max_change is not None:
            out["max_change"] = self.max_change
        return out


def use_criteria(
    *,
    min_fraction_beyond_controls: float,
    min_change: float | None = None,
    max_change: float | None = None,
) -> UseCriteria:
    if not 0.5 <= min_fraction_beyond_controls <= 1.0:
        raise ConceptError(
            "min_fraction_beyond_controls is in [0.5, 1] (mandatory control criterion)"
        )
    if (min_change is None) == (max_change is None):
        raise ConceptError(
            "declare min_change (decreases/increases) or max_change (sufficient_for)"
        )
    if min_change is not None and not min_change > 0:
        raise ConceptError("min_change must be > 0")
    if max_change is not None and not max_change >= 0:
        raise ConceptError("max_change must be >= 0")
    return UseCriteria(min_fraction_beyond_controls, min_change, max_change)


# ------------------------------------------------------------------ interventions


def feature_intervention(
    feature: FeatureRecord,
    intervention: FeatureIntervention,
    *,
    index: int | None = None,
    vector: torch.Tensor | None = None,
) -> iv.Intervention:
    """The Phase-2 intervention on a feature (or on a control feature of the same kind:
    another ``index`` or another ``vector``)."""
    site, keep = feature.site, intervention.mode == "retain"
    ref = intervention.reference.tensor
    if feature.basis is FeatureBasis.NEURON:
        unit = feature.index if index is None else index
        assert unit is not None
        if ref is None:
            return iv.zero(
                site.module,
                output_path=site.output_path,
                call_index=feature.call_index,
                units=(unit,),
                retain=keep,
                unit_axes=(feature.axis,),
            )
        return iv.constant(
            site.module,
            ref,
            output_path=site.output_path,
            call_index=feature.call_index,
            units=(unit,),
            retain=keep,
            unit_axes=(feature.axis,),
        )
    assert vector is not None
    return iv.direction(
        site.module,
        vector,
        axis=feature.axis,
        reference=ref,
        retain=keep,
        output_path=site.output_path,
        call_index=feature.call_index,
    )


# ------------------------------------------------------------------ evaluation (shared)


def evaluate_use(
    relation: Relation, mean_effect: float, control_effects: Sequence[float], criteria: UseCriteria
) -> tuple[Outcome, dict[str, Any]]:
    if relation is Relation.DECREASES:
        observed, controls = -mean_effect, [-c for c in control_effects]
        holds = criteria.min_change is not None and mean_effect <= -criteria.min_change
    elif relation is Relation.INCREASES:
        observed, controls = mean_effect, list(control_effects)
        holds = criteria.min_change is not None and mean_effect >= criteria.min_change
    else:
        observed, controls = -abs(mean_effect), [-abs(c) for c in control_effects]
        holds = criteria.max_change is not None and abs(mean_effect) <= criteria.max_change
    stats = superiority(observed, controls)
    holds = holds and stats["superiority"] >= criteria.min_fraction_beyond_controls
    return (Outcome.SUPPORTS if holds else Outcome.CONTRADICTS), stats


# ------------------------------------------------------------------ runner


@dataclass(frozen=True, slots=True, eq=False)
class UseResult:
    trace: TraceResult
    concept: Concept
    data: ConceptData
    intervention: FeatureIntervention

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
    def outcome(self) -> Outcome:
        return self.result.outcome

    @property
    def verdict(self) -> Verdict:
        return self.assessment.verdict

    @property
    def mean_effect(self) -> float:
        value = self.result.statistics["mean_effect"]
        assert isinstance(value, float)
        return value


def _subset(data: ConceptData, subset: str) -> tuple[int, ...]:
    if subset == "positive":
        idx = data.record.indices("test", 1)
    elif subset == "negative":
        idx = data.record.indices("test", 0)
    elif subset == "all":
        idx = data.record.indices("test")
    else:
        raise ConceptError("subset is 'positive', 'negative' or 'all' (of the test split)")
    if not idx:
        raise ConceptError(f"the {subset!r} test subset is empty")
    return idx


def use_test(
    model: nn.Module,
    concept: Concept,
    data: ConceptData,
    *,
    target: iv.metrics.Metric,
    relation: str,
    intervention: FeatureIntervention,
    controls: Sequence[Control],
    criteria: UseCriteria,
    subset: str = "positive",
    statement: str | None = None,
    declared_model: ModelDeclaration | None = None,
) -> UseResult:
    """Run ``concept_intervention`` v1 (see module docstring). ``intervention``,
    ``controls`` and ``criteria`` are mandatory; ``relation`` is ``decreases`` /
    ``increases`` (with ``remove``) or ``sufficient_for`` (with ``retain``)."""
    if relation not in _RELATIONS:
        raise ConceptError(f"relation is one of {sorted(_RELATIONS)}")
    rel = _RELATIONS[relation]
    if not isinstance(intervention, FeatureIntervention):
        raise ConceptError("declare the intervention: remove(reference) or retain(reference)")
    if (intervention.mode == "retain") != (rel is Relation.SUFFICIENT_FOR):
        raise ConceptError("remove() tests decreases/increases; retain() tests sufficient_for")
    if (criteria.max_change is not None) != (rel is Relation.SUFFICIENT_FOR):
        raise ConceptError("sufficient_for uses max_change; decreases/increases use min_change")
    if not controls:
        raise ConceptError("a use test needs at least one control (request §16)")
    controls = tuple(controls)
    feature = concept.feature.record
    for c in controls:
        if not isinstance(c, Control) or c.kind == "label_permutation":
            raise ConceptError("use controls are random_directions or random_neurons")
        if (c.kind == "random_neurons") != (feature.basis is FeatureBasis.NEURON):
            raise ConceptError("neurons use random_neurons; directions/SAE use random_directions")
    _check_scope(model, concept, data)
    ds = data.record
    evaluated = _subset(data, subset)
    train = ds.indices("train") if any(c.distribution == "covariance" for c in controls) else ()
    d_size = _axis_size(model, data, evaluated[0], feature)
    pooled_train = None
    if train:
        trace0 = _record_train(model, data, train, feature, declared_model)
        pooled_train = torch.stack(
            [
                pooled_vectors(t, feature)
                for _, t in site_tensors(trace0, data, train, feature.site, feature.call_index)
            ]
        )
    specs, control_features = _plan(concept, feature, intervention, controls, d_size, pooled_train)
    holder: list[TraceResult] = []
    # Train samples (covariance controls only) are baseline-only groups of the same family,
    # so the use trace itself holds the activations the controls are re-derived from.
    offset = len(train)

    def extend(trace: TraceResult, family: ComparisonFamily) -> None:
        eval_ids = [ds.samples[i] for i in evaluated]
        estimand = Estimand.finite_sample(sample_set_id(eval_ids), len(eval_ids), "mean")
        provenance = derived_provenance(trace, "concept_intervention_mean", {"feature": feature.id})
        aggregates = []
        for j in range(len(specs)):
            parts = [family.effects[offset + g][j] for g in range(len(evaluated))]
            record = family.records[offset][j]
            n = len(parts)
            base = sum(p.baseline_value for p in parts) / n
            inter = sum(p.intervention_value for p in parts) / n
            aggregate = CausalEffect(
                interventions=(RecordRef.to(record),),
                metric=target.spec,
                estimand=estimand,
                baseline_value=base,
                intervention_value=inter,
                effect=sum(p.effect for p in parts) / n,
                provenance_id=provenance.id,
                derived_from=tuple(RecordRef.to(p) for p in parts),
            )
            stored = trace._add(aggregate)
            assert isinstance(stored, CausalEffect)
            aggregates.append(stored)
        primary, rest = aggregates[0], aggregates[1:]
        outcome, stats = evaluate_use(rel, primary.effect, [a.effect for a in rest], criteria)
        copy_definitions(trace, concept, data)
        claim = Claim(
            statement=statement
            or (
                f"{intervention.mode} feature {feature.id} ({intervention.reference.name} "
                f"reference) {relation.replace('_', ' ')} {target.spec.name} on the {subset} "
                f"test subset of {ds.name}"
            ),
            relation=rel,
            subject=Subject(site=feature.site, feature=feature.id),
            target=target.spec.target(),
            estimand=estimand,
            source=claim_source(concept),
        )
        spec = ClaimTestSpec(
            protocol=USE_PROTOCOL,
            protocol_version=1,
            applicable_relations=(rel,),
            criteria=JsonMap(criteria.as_json()),
            params=JsonMap(
                {
                    "concept": concept.id,
                    "feature": feature.id,
                    "dataset": ds.id,
                    "subset": subset,
                    "intervention": intervention.identity(),
                    "controls": [c.identity() for c in controls],
                    "statistic": "finite-sample mean effect (intervened - baseline)",
                }
            ),
        )
        trace._add(claim)
        trace._add(spec)
        result = ClaimTestResult.for_claim(
            claim,
            spec,
            outcome=outcome,
            evidence=aggregates,
            statistics=JsonMap(
                stats
                | {
                    "mean_effect": primary.effect,
                    "n": len(evaluated),
                    "primary_effect": primary.id,
                    "control_effects": [a.id for a in rest],
                    "control_mean_effects": [a.effect for a in rest],
                    "control_features": control_features,
                }
            ),
            provenance_id=provenance.id,
        )
        trace._add(result)
        trace._add(Assessment.derive(claim, [result], USE_POLICY))
        codes = ["SINGLE_FEATURE_TEST_MISSES_REDUNDANCY"]
        if feature.basis is not FeatureBasis.NEURON:
            codes.append("DIRECTION_INTERVENTION_MAY_ACTIVATE_DORMANT_PATHWAYS")
        if feature.basis is FeatureBasis.SAE:
            codes.append("SAE_INTERVENTION_IS_PROJECTION")
        if rel is Relation.SUFFICIENT_FOR:
            codes.append("SITE_RELATIVE_SUFFICIENCY")
        for code in codes:
            trace._add(TraceLimitation(code=code, applies_to=(result.id,)))
        holder.append(trace)

    iv.compare_family(
        model,
        [(data.inputs[i], data.kwargs[i], []) for i in train]
        + [(data.inputs[i], data.kwargs[i], specs) for i in evaluated],
        target,
        sites=[feature.site.module],
        retention="cpu" if train else "summary",
        declared_model=declared_model,
        extend=extend,
    )
    return UseResult(holder[0], concept, data, intervention)


def _axis_size(model: nn.Module, data: ConceptData, i: int, feature: FeatureRecord) -> int:
    trace = _record_train(model, data, (i,), feature, None)
    ((_, tensor),) = site_tensors(trace, data, (i,), feature.site, feature.call_index)
    return int(tensor.shape[feature.axis])


def _record_train(
    model: nn.Module,
    data: ConceptData,
    indices: Sequence[int],
    feature: FeatureRecord,
    declared_model: ModelDeclaration | None,
) -> TraceResult:
    from .features import record_site

    return record_site(model, data, indices, feature.site, declared_model=declared_model)


def _plan(
    concept: Concept,
    feature: FeatureRecord,
    intervention: FeatureIntervention,
    controls: Sequence[Control],
    d_size: int,
    pooled_train: torch.Tensor | None,
) -> tuple[list[iv.Intervention], list[Any]]:
    if feature.basis is FeatureBasis.NEURON:
        specs = [feature_intervention(feature, intervention)]
    else:
        specs = [feature_intervention(feature, intervention, vector=concept.feature.vector())]
    described: list[Any] = []
    for control in controls:
        if control.kind == "random_neurons":
            assert feature.index is not None
            for j in random_indices_for(control, d_size, feature.index):
                specs.append(feature_intervention(feature, intervention, index=j))
                described.append(j)
        else:
            for v in random_directions_for(control, d_size, pooled_train):
                vec = v.float()
                specs.append(feature_intervention(feature, intervention, vector=vec))
                from beyondnn.attribution.spec import tensor_digest

                described.append(tensor_digest(vec.contiguous()))
    return specs, described


def model_digest(model: nn.Module) -> str:
    return fingerprint_model(model).state_digest
