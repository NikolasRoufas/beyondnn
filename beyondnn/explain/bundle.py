"""``EvidenceBundle``: the validated composition behind a Phase-4 WHY (ADR-031).

An evidence bundle is **not evidence**. It organises already-computed evidence for
ONE explanation context:

* a reference :class:`~beyondnn.core.trace.TraceResult` (one root pass: the
  OBSERVED input/output and the MEASURED activations);
* :class:`~beyondnn.attribution.AttributionResult`\\ s (ATTRIBUTED);
* :class:`~beyondnn.interventions.InterventionResult`\\ s (INTERVENTIONAL);
* declared claims, the test results already recorded with the evidence, and
  assessments derived under caller-supplied, registry-checked policies.

Composition runs nothing: no model, autograd, Captum, hooks, RNG, or tensor
writes. It revalidates everything it is given and **refuses** (never merges)
evidence that cannot honestly belong to one context:

* record ids must match content; references must resolve in their own trace; one
  id may not carry two contents;
* one automatic ``ModelIdentity`` and one ``ModelDeclaration`` across every
  provenance record;
* one exact input sample (``InputRecord.sample_id``) for all target-specific
  evidence and claims (a patch source is shown as source context);
* INSTANCE scope only (finite-sample and population evidence are refused here);
* one scalar target (``MetricSpec``) across attributions, effects and claims;
* every decisive claim-test result is re-derived from its cited evidence with its
  registered protocol and must be identical.

The source traces remain the scientific source of truth; the bundle holds
references to their record objects and never copies or modifies them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Any

from beyondnn.attribution import AttributionResult
from beyondnn.attribution import evaluate_claim as evaluate_attribution_claim
from beyondnn.core.trace import _REF_TARGET, TraceResult, _refs
from beyondnn.interventions import InterventionResult
from beyondnn.interventions import evaluate_claim as evaluate_intervention_claim
from beyondnn.protocols import (
    ATTRIBUTION_THRESHOLD,
    INTERVENTION_THRESHOLD,
    PROTOCOLS,
    check_policy,
)
from beyondnn.schema import (
    Assessment,
    AssessmentPolicy,
    AttributionRecord,
    BaseRecord,
    CausalEffect,
    Claim,
    ClaimTestResult,
    ClaimTestSpec,
    EstimandScope,
    InterventionRecord,
    MetricSpec,
    Outcome,
    ProvenanceRecord,
    TargetSpec,
    TraceLimitation,
    to_json,
    verify_ref,
)
from beyondnn.schema.base import _compute_id

__all__ = [
    "CompositionError",
    "EvidenceBundle",
    "EvidenceIntegrityError",
    "ModelMismatchError",
    "SampleMismatchError",
    "TargetMismatchError",
    "UnsupportedScopeError",
]


class CompositionError(ValueError):
    """Evidence cannot be composed into one explanation context."""


class EvidenceIntegrityError(CompositionError):
    """A record, reference, or claim-test result does not verify."""


class ModelMismatchError(CompositionError):
    """Evidence comes from a different model identity or model declaration."""


class SampleMismatchError(CompositionError):
    """Target-specific evidence is not about the reference input (exactly)."""


class TargetMismatchError(CompositionError):
    """Target-specific evidence does not share one scalar target."""


class UnsupportedScopeError(CompositionError):
    """Non-instance evidence or claims in an instance-level explanation."""


_BUILD = object()


def _check_source(trace: TraceResult) -> None:
    """Every record's id matches its content and every reference resolves in ``trace``."""
    records = {r.id: r for r in trace.records}
    for record in trace.records:
        if _compute_id(record) != record.id:
            raise EvidenceIntegrityError(f"record {record.id} does not match its content")
        for ref in _refs(record):
            target = records.get(getattr(ref, _REF_TARGET[type(ref)]))
            if target is None:
                raise EvidenceIntegrityError(f"{record.id}: dangling reference {ref}")
            try:
                verify_ref(ref, target)
            except Exception as exc:
                raise EvidenceIntegrityError(f"{record.id}: {exc}") from exc
        if record.provenance_id is not None and not isinstance(
            records.get(record.provenance_id), ProvenanceRecord
        ):
            raise EvidenceIntegrityError(f"{record.id}: provenance does not resolve")


class EvidenceBundle:
    """A finalised, validated composition for one explanation context. Build with
    :meth:`compose`; it has no mutating methods."""

    __slots__ = (
        "_index",
        "assessments",
        "attributions",
        "claim_target",
        "claims",
        "interventions",
        "policies",
        "reference",
        "results",
        "sample_id",
        "sources",
        "target",
    )

    reference: TraceResult
    attributions: tuple[AttributionResult, ...]
    interventions: tuple[InterventionResult, ...]
    claims: tuple[Claim, ...]
    results: tuple[ClaimTestResult, ...]
    policies: tuple[AssessmentPolicy, ...]
    assessments: tuple[Assessment, ...]
    target: MetricSpec | None
    claim_target: TargetSpec | None
    sample_id: str | None
    sources: tuple[TraceResult, ...]
    _index: Mapping[str, tuple[BaseRecord, TraceResult]]

    def __init__(self, token: object, **fields: Any) -> None:
        if token is not _BUILD:
            raise TypeError("build an EvidenceBundle with EvidenceBundle.compose(...)")
        for name, value in fields.items():
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("an EvidenceBundle is immutable")

    def __repr__(self) -> str:
        return (
            f"EvidenceBundle(attributions={len(self.attributions)}, "
            f"interventions={len(self.interventions)}, claims={len(self.claims)})"
        )

    # ------------------------------------------------------------------ queries

    def get(self, record_id: str) -> BaseRecord:
        """A record of any source trace, by id (the source's own object)."""
        return self._index[record_id][0]

    def source_of(self, record: BaseRecord | str) -> TraceResult:
        """The trace that holds ``record`` (the first source, in composition order)."""
        record_id = record if isinstance(record, str) else record.id
        if record_id not in self._index:
            raise KeyError(f"{record_id} is not part of this bundle's source traces")
        return self._index[record_id][1]

    def origin(self, record: BaseRecord | str) -> ProvenanceRecord:
        """The provenance record of ``record``, from its own source trace."""
        return self.source_of(record).origin(record)

    def results_for(self, claim: Claim) -> tuple[ClaimTestResult, ...]:
        return tuple(r for r in self.results if r.claim.claim_id == claim.id)

    @property
    def has_target_evidence(self) -> bool:
        return bool(self.attribution_records or self.effects)

    @property
    def attribution_records(self) -> tuple[AttributionRecord, ...]:
        found = [a.record for a in self.attributions]
        found += [r for r in self.reference.records if isinstance(r, AttributionRecord)]
        return tuple(_unique(found))

    @property
    def effects(self) -> tuple[CausalEffect, ...]:
        return tuple(_unique(i.effect for i in self.interventions))

    @property
    def limitations(self) -> tuple[TraceLimitation, ...]:
        """Every limitation of every source trace, deduplicated by record identity, in
        source order (reference first) and each trace's own order."""
        return tuple(_unique(lim for src in self.sources for lim in src.limitations))

    @property
    def provenance(self) -> tuple[ProvenanceRecord, ...]:
        return tuple(_unique(p for src in self.sources for p in src.provenance))

    # ------------------------------------------------------------------ composition

    @classmethod
    def compose(
        cls,
        trace: TraceResult,
        *,
        attributions: Sequence[AttributionResult] = (),
        interventions: Sequence[InterventionResult] = (),
        claims: Sequence[Claim] = (),
        policies: Sequence[AssessmentPolicy] = (),
    ) -> EvidenceBundle:
        """Validate and compose (see module docstring). Runs no model and no method."""
        if not isinstance(trace, TraceResult):
            raise TypeError("the reference must be a TraceResult")
        if trace.passes != 1:
            raise ValueError(
                f"an explanation covers exactly one root invocation; this trace has "
                f"{trace.passes} passes"
            )
        for name, items, kind in (
            ("attributions", attributions, AttributionResult),
            ("interventions", interventions, InterventionResult),
            ("claims", claims, Claim),
            ("policies", policies, AssessmentPolicy),
        ):
            if isinstance(items, (str, bytes)) or not all(isinstance(i, kind) for i in items):
                raise TypeError(f"{name} must be a sequence of {kind.__name__}")
        attributions, interventions = tuple(attributions), tuple(interventions)

        sources = tuple(
            _unique_objects(
                [trace, *(a.trace for a in attributions), *(i.trace for i in interventions)]
            )
        )
        index: dict[str, tuple[BaseRecord, TraceResult]] = {}
        canonical: dict[str, str] = {}
        for src in sources:
            _check_source(src)
            for record in src.records:
                text = to_json(record)
                if record.id in canonical:
                    if canonical[record.id] != text:
                        raise EvidenceIntegrityError(
                            f"record id {record.id} carries different content in two traces"
                        )
                    continue
                canonical[record.id] = text
                index[record.id] = (record, src)
        for a in attributions:
            if a.trace.get(a.record.id) is not a.record:
                raise EvidenceIntegrityError("an attribution record is not in its own trace")
        for i in interventions:
            if i.trace.get(i.effect.id) is not i.effect:
                raise EvidenceIntegrityError("an intervention effect is not in its own trace")

        reference_input = trace.input
        reference = trace.origin(reference_input)
        for src in sources:
            for p in src.provenance:
                if p.model != reference.model:
                    raise ModelMismatchError(
                        "evidence from a different model identity (fingerprint "
                        f"{p.model.state_digest[:19]}... vs {reference.model.state_digest[:19]}...)"
                    )
                if p.declared_model != reference.declared_model:
                    raise ModelMismatchError(
                        "evidence under a different ModelDeclaration (declared config/revision); "
                        "equal weights do not make two declared models the same"
                    )

        sample = reference_input.sample_id
        bundle_claims = list(_unique(claims))
        for src in sources:
            bundle_claims += [c for c in src.records if isinstance(c, Claim)]
        bundle_claims = list(_unique(bundle_claims))
        attribution_records = _unique(
            [a.record for a in attributions]
            + [r for r in trace.records if isinstance(r, AttributionRecord)]
        )
        effects = list(_unique(i.effect for i in interventions))
        target_specific = bool(attribution_records or effects or bundle_claims)
        if target_specific and sample is None:
            raise SampleMismatchError(
                "the reference input has no exact sample identity (a trace recorded before "
                "ADR-031, or an input without a deterministic identity); target-specific "
                "evidence cannot be matched to it"
            )

        def same_sample(value: str | None, what: str) -> None:
            if value != sample:
                raise SampleMismatchError(
                    f"{what} is about sample {value}, not the reference input {sample}; "
                    "evidence about another input is never merged into this explanation"
                )

        for record in attribution_records:
            same_sample(record.sample_id, f"attribution {record.id}")
            source = index[record.id][1]
            (attributed_input,) = [r for r in source.inputs if r.pass_index == record.pass_index]
            same_sample(attributed_input.sample_id, "the attribution's reference pass")
        for i in interventions:
            effect = i.effect
            if effect.estimand.scope is not EstimandScope.INSTANCE:
                raise UnsupportedScopeError(
                    f"{effect.estimand.scope.value} effects are not part of an instance-level "
                    "explanation (the population/finite-sample view is a later feature)"
                )
            same_sample(effect.estimand.sample_id, f"causal effect {effect.id}")
            baseline = i.baseline_output
            (baseline_input,) = [r for r in i.trace.inputs if r.pass_index == baseline.pass_index]
            same_sample(baseline_input.sample_id, "the intervention's baseline pass")
        for claim in bundle_claims:
            if claim.estimand.scope is not EstimandScope.INSTANCE:
                raise UnsupportedScopeError(
                    f"claim {claim.id} has a {claim.estimand.scope.value} estimand; only "
                    "instance claims belong to an instance-level explanation"
                )
            same_sample(claim.estimand.sample_id, f"claim {claim.id}")

        metrics = _unique([r.target for r in attribution_records] + [e.metric for e in effects])
        if len(metrics) > 1:
            raise TargetMismatchError(
                "target-specific evidence uses different targets "
                f"({', '.join(m.name + ' ' + str(m.params.to_plain()) for m in metrics)}); "
                "one explanation has one scalar target"
            )
        target = metrics[0] if metrics else None
        claim_targets = _unique(c.target for c in bundle_claims)
        if target is not None:
            claim_targets = _unique([target.target(), *claim_targets])
        if len(claim_targets) > 1:
            raise TargetMismatchError("claims and evidence refer to different targets")
        claim_target = claim_targets[0] if claim_targets else None

        claim_ids = {c.id for c in bundle_claims}
        results = [r for src in sources for r in src.records if isinstance(r, ClaimTestResult)]
        results = list(_unique(results))
        for result in results:
            _revalidate(result, index, claim_ids)

        policy_list = tuple(_unique(policies))
        for policy in policy_list:
            check_policy(policy)
        assessments: list[Assessment] = []
        for claim in bundle_claims:
            about = [r for r in results if r.claim.claim_id == claim.id]
            for policy in policy_list:
                if policy.protocols_for(claim.relation):
                    assessments.append(Assessment.derive(claim, about, policy))

        return cls(
            _BUILD,
            reference=trace,
            attributions=attributions,
            interventions=interventions,
            claims=tuple(bundle_claims),
            results=tuple(results),
            policies=policy_list,
            assessments=tuple(assessments),
            target=target,
            claim_target=claim_target,
            sample_id=sample,
            sources=sources,
            _index=MappingProxyType(index),
        )


def _unique(items: Iterable[Any]) -> list[Any]:
    """Order-preserving deduplication by value (records: by id and content)."""
    out: list[Any] = []
    seen: set[Any] = set()
    for item in items:
        key = item.id if isinstance(item, BaseRecord) else item
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _unique_objects(items: Iterable[Any]) -> list[Any]:
    out: list[Any] = []
    for item in items:
        if not any(item is o for o in out):
            out.append(item)
    return out


def _revalidate(
    result: ClaimTestResult,
    index: Mapping[str, tuple[BaseRecord, TraceResult]],
    claim_ids: set[str],
) -> None:
    """Re-derive a recorded claim-test result from its cited evidence; refuse forgeries."""
    if result.claim.claim_id not in claim_ids:
        raise EvidenceIntegrityError(f"{result.id} is about a claim that is not present")
    claim = index[result.claim.claim_id][0]
    spec = index[result.spec.spec_id][0]
    assert isinstance(claim, Claim)
    assert isinstance(spec, ClaimTestSpec)
    if spec.protocol not in PROTOCOLS:
        raise EvidenceIntegrityError(
            f"{result.id} uses unregistered protocol {spec.protocol!r}; it cannot be revalidated"
        )
    for ev in result.evidence:
        cited = index.get(ev.record_id)
        if cited is None or cited[0].status is not ev.status:
            raise EvidenceIntegrityError(
                f"{result.id} cites evidence that is not present as stated"
            )
    if result.outcome is Outcome.NOT_APPLICABLE and not result.evidence:
        return  # cites nothing and can support nothing
    if result.outcome in (Outcome.SUPPORTS, Outcome.CONTRADICTS) and (
        claim.relation not in PROTOCOLS[spec.protocol]
    ):
        raise EvidenceIntegrityError(
            f"{result.id}: protocol {spec.protocol!r} does not justify "
            f"{claim.relation.value}; its outcome cannot decide this claim"
        )
    cited_records = [index[ev.record_id] for ev in result.evidence]
    if spec.protocol == INTERVENTION_THRESHOLD:
        effects = [(r, t) for r, t in cited_records if isinstance(r, CausalEffect)]
        if len(effects) != 1:
            raise EvidenceIntegrityError(f"{result.id} must cite exactly one causal effect")
        effect = effects[0][0]
        assert isinstance(effect, CausalEffect)
        intervention = index[effect.interventions[0].record_id][0]
        assert isinstance(intervention, InterventionRecord)
        expected = evaluate_intervention_claim(
            claim, spec, effect, intervention, provenance_id=result.provenance_id or ""
        )
    elif spec.protocol == ATTRIBUTION_THRESHOLD:
        records = [(r, t) for r, t in cited_records if isinstance(r, AttributionRecord)]
        if len(records) != 1:
            raise EvidenceIntegrityError(f"{result.id} must cite exactly one attribution")
        record, source = records[0]
        assert isinstance(record, AttributionRecord)
        expected = evaluate_attribution_claim(claim, spec, record, source.tensor(record.value))
    else:  # a registered protocol without a revalidator
        raise EvidenceIntegrityError(f"no revalidator for protocol {spec.protocol!r}")
    if expected.id != result.id:
        raise EvidenceIntegrityError(
            f"{result.id} ({result.outcome.value}) does not follow from its evidence under "
            f"{spec.protocol} (re-derived {expected.outcome.value})"
        )
