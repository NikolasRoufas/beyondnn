"""INPUT -> [TARGET] -> WHY -> OUTPUT: a presentation view over composed evidence.

Phase 1 (M1.8; ADR-026): with only a trace, WHY answers "what internal evidence
was measured while this output was produced?" and nothing more. That meaning is
unchanged.

Phase 4 (ADR-031; ADR-026 amendment): WHY can also present evidence that was
computed explicitly beforehand (``bnn.attribute``, ``bnn.intervene``) and composed
with :func:`compose` / :meth:`ExplainResponse.from_evidence`. Evidence stays
grouped by epistemic status: OBSERVED, MEASURED, ATTRIBUTED, INTERVENTIONAL and
ESTIMATED_CAUSAL are separate sections; claims appear only if they were declared,
with every recorded test and the assessments of the supplied policies. Nothing is
ranked, scored, combined across methods, or generated. Faithfulness and concepts
are explicitly not evaluated.

The traces are the scientific source of truth: views reference their record
objects. A response is rebuilt, not persisted: reload the traces
(``load_trace``, ``AttributionResult.from_trace``, ``InterventionResult.from_trace``)
and compose again.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from beyondnn.attribution import AttributionResult
from beyondnn.core.trace import TraceResult
from beyondnn.interventions import InterventionResult
from beyondnn.schema import (
    CAUSAL_EVIDENCE_STATUSES,
    ActivationRecord,
    Assessment,
    AssessmentPolicy,
    AttributionRecord,
    AttributionReduction,
    BaseRecord,
    CausalEffect,
    Claim,
    ClaimTestSpec,
    EvidenceStatus,
    InputRecord,
    MetricSpec,
    NamedTensor,
    OutputRecord,
    ProvenanceRecord,
    TargetSpec,
    TensorRef,
    TraceLimitation,
)

from .bundle import EvidenceBundle
from .views import AttributionView, ClaimView, Coverage, InterventionView

__all__ = ["EXPLANATION_LIMITATIONS", "ExplainResponse", "Why", "compose"]

#: Emitted by an explanation whose composed evidence lacks that kind of evidence
#: (always, for ``explain()``, which runs none of these methods).
EXPLANATION_LIMITATIONS = ("NO_ATTRIBUTION", "NO_CAUSAL_EVIDENCE", "NO_CLAIMS_TESTED")


@dataclass(frozen=True, slots=True, eq=False)
class Why:
    """Structured WHY over one explanation context (see module docstring).

    Measured-only (no target-specific evidence): answers ``QUESTION`` and not
    ``NOT_ANSWERED``, exactly as in Phase 1. With attributions or interventions it
    answers ``TARGET_QUESTION`` and still not ``TARGET_NOT_ANSWERED``.
    """

    QUESTION: ClassVar[str] = "What internal evidence was measured while this output was produced?"
    NOT_ANSWERED: ClassVar[str] = "Which internal state caused the output? (no causal evidence yet)"
    TARGET_QUESTION: ClassVar[str] = (
        "For this target and input: what was measured, what did the declared attribution "
        "methods assign, what changed under the declared interventions, and what did the "
        "declared claim tests find?"
    )
    TARGET_NOT_ANSWERED: ClassVar[str] = (
        "The faithfulness, completeness, or sufficiency of this evidence (not evaluated), "
        "what any activation means (no concept validation), and anything beyond the "
        "declared, tested claims"
    )

    trace: TraceResult
    bundle: EvidenceBundle | None = None

    def __post_init__(self) -> None:
        if self.bundle is None:
            object.__setattr__(self, "bundle", EvidenceBundle.compose(self.trace))
        elif self.bundle.reference is not self.trace:
            raise ValueError("the bundle's reference trace is not this trace")

    @property
    def _b(self) -> EvidenceBundle:
        assert self.bundle is not None
        return self.bundle

    # ------------------------------------------------------------ target and question

    @property
    def target(self) -> MetricSpec | None:
        """The one scalar target of the target-specific evidence (``None``: measured-only)."""
        return self._b.target

    @property
    def target_spec(self) -> TargetSpec | None:
        """The claim-level target (the metric's ``target()``, or the claims' target)."""
        return self._b.claim_target

    @property
    def question(self) -> str:
        return self.TARGET_QUESTION if self._target_view else self.QUESTION

    @property
    def not_answered(self) -> str:
        return self.TARGET_NOT_ANSWERED if self._target_view else self.NOT_ANSWERED

    @property
    def _target_view(self) -> bool:
        return self._b.has_target_evidence or bool(self._b.claims)

    # ------------------------------------------------------------ sections by status

    @property
    def observations(self) -> tuple[InputRecord, OutputRecord]:
        """The OBSERVED input and output of the reference pass."""
        return (self.trace.input, self.trace.output)

    @property
    def activations(self) -> tuple[ActivationRecord, ...]:
        """MEASURED activation records of the reference pass, in execution order (the
        trace's own objects)."""
        return self.trace.activations

    @property
    def measurements(self) -> tuple[ActivationRecord, ...]:
        """Alias of :attr:`activations`."""
        return self.activations

    @property
    def attributions(self) -> tuple[AttributionRecord, ...]:
        """ATTRIBUTED records: method-relative scores for the target, not causal effects.
        Only present if an attribution was explicitly run and composed; never ranked."""
        return self._b.attribution_records

    @property
    def attribution_views(self) -> tuple[AttributionView, ...]:
        views = []
        for record in self.attributions:
            src = self._b.source_of(record)
            attributed = next(
                self._b.get(r.record_id)
                for r in record.derived_from
                if r.kind in ("input", "activation")
            )
            assert isinstance(attributed, (InputRecord, ActivationRecord))
            views.append(
                AttributionView(
                    record=record,
                    reductions=tuple(
                        r
                        for r in src.records
                        if isinstance(r, AttributionReduction)
                        and r.derived_from[0].record_id == record.id
                    ),
                    attributed=attributed,
                    limitations=tuple(
                        lim for lim in src.limitations if record.id in lim.applies_to
                    ),
                    source=src,
                )
            )
        return tuple(views)

    @property
    def effects(self) -> tuple[CausalEffect, ...]:
        """INTERVENTIONAL causal effects (instance scope), in composition order."""
        return tuple(e for e in self._b.effects if e.status is EvidenceStatus.INTERVENTIONAL)

    @property
    def estimated_causal(self) -> tuple[CausalEffect, ...]:
        """ESTIMATED_CAUSAL records: always its own section (none can be composed into an
        instance-level explanation in Phase 4)."""
        return tuple(e for e in self._b.effects if e.status is EvidenceStatus.ESTIMATED_CAUSAL)

    @property
    def intervention_views(self) -> tuple[InterventionView, ...]:
        views = []
        for result in self._b.interventions:
            intervention = result.intervention
            source_activation = None
            source_input = None
            if intervention.source is not None:
                act = result.trace.get(intervention.source.record_id)
                assert isinstance(act, ActivationRecord)
                source_activation = act
                source_input = next(
                    r for r in result.trace.inputs if r.pass_index == act.pass_index
                )
            views.append(
                InterventionView(
                    intervention=intervention,
                    effect=result.effect,
                    baseline_output=result.baseline_output,
                    intervention_output=result.intervention_output,
                    source_activation=source_activation,
                    source_input=source_input,
                    limitations=tuple(
                        lim
                        for lim in result.trace.limitations
                        if result.effect.id in lim.applies_to
                    ),
                    source=result.trace,
                )
            )
        return tuple(views)

    @property
    def interventions(self) -> tuple[InterventionView, ...]:
        """Alias of :attr:`intervention_views`."""
        return self.intervention_views

    def by_status(self, status: EvidenceStatus) -> tuple[BaseRecord, ...]:
        """The presented evidence records with ``status``, in presentation order."""
        return tuple(r for r in self._presented if r.status is status)

    @property
    def _presented(self) -> tuple[BaseRecord, ...]:
        reductions = [r for v in self.attribution_views for r in v.reductions]
        return (
            *self.observations,
            *self.activations,
            *self.attributions,
            *reductions,
            *self._b.effects,
        )

    @property
    def evidence_statuses(self) -> frozenset[EvidenceStatus]:
        """Evidence statuses of the presented evidence (measured-only: OBSERVED, MEASURED)."""
        return frozenset(r.status for r in self._presented if r.status is not None)

    # ------------------------------------------------------------ claims

    @property
    def claims(self) -> tuple[ClaimView, ...]:
        """Declared claims (caller order, then those recorded with the evidence), each with
        every recorded test and its assessments. None are generated."""
        views = []
        for claim in self._b.claims:
            tests = []
            for result in self._b.results_for(claim):
                spec = self._b.get(result.spec.spec_id)
                assert isinstance(spec, ClaimTestSpec)
                tests.append((spec, result))
            views.append(
                ClaimView(
                    claim=claim,
                    tests=tuple(tests),
                    assessments=tuple(
                        a for a in self._b.assessments if a.claim.claim_id == claim.id
                    ),
                )
            )
        return tuple(views)

    @property
    def assessments(self) -> tuple[Assessment, ...]:
        return self._b.assessments

    # ------------------------------------------------------------ limitations, provenance

    @property
    def limitations(self) -> tuple[TraceLimitation, ...]:
        """Every limitation of the composed evidence (deduplicated by record identity,
        details preserved), then the explanation-level ones that apply."""
        own = self._b.limitations
        codes = {lim.code for lim in own}
        present = {
            "NO_ATTRIBUTION": bool(self.attributions),
            "NO_CAUSAL_EVIDENCE": bool(self.evidence_statuses & CAUSAL_EVIDENCE_STATUSES),
            "NO_CLAIMS_TESTED": bool(self._b.results),
        }
        extra = tuple(
            TraceLimitation(code=c)
            for c in EXPLANATION_LIMITATIONS
            if c not in codes and not present[c]
        )
        return own + extra

    @property
    def provenance(self) -> tuple[ProvenanceRecord, ...]:
        return self._b.provenance

    def origin(self, record: BaseRecord | str) -> ProvenanceRecord:
        """Where ``record`` came from: its provenance record in its own source trace."""
        return self._b.origin(record)

    # ------------------------------------------------------------ coverage and unknowns

    @property
    def coverage(self) -> Coverage:
        claims = self.claims
        return Coverage(
            measured_internal_states=bool(self.activations),
            attribution_available=bool(self.attributions),
            intervention_effect_available=bool(self.effects),
            estimated_causal_available=bool(self.estimated_causal),
            claims_declared=bool(claims),
            claims_tested=any(c.decisive_tests for c in claims),
            causal_claim_tested=any(c.causal_test_performed for c in claims),
        )

    @property
    def unanswered(self) -> tuple[str, ...]:
        """Deterministic list of questions this evidence does not answer."""
        cov = self.coverage
        out = []
        if not cov.measured_internal_states:
            out.append("What internal states were measured? (no sites were recorded)")
        if not cov.attribution_available:
            out.append("What would an attribution method assign? (no attribution was run)")
        if not cov.intervention_effect_available:
            out.append("What changes under a controlled intervention? (no intervention was run)")
        for view in self.claims:
            if view.is_causal and not view.causal_test_performed:
                out.append(
                    f"Does the claim {view.claim.statement!r} hold? (no decisive causal test)"
                )
        out.append("Is this evidence faithful, comprehensive, or sufficient? (not evaluated)")
        out.append("Does any activation correspond to a concept? (no concept validation)")
        return tuple(out)


# ------------------------------------------------------------------ rendering helpers


def _describe(ref: TensorRef) -> str:
    text = f"{ref.dtype} {tuple(ref.shape)} on {ref.device}"
    if ref.stats is not None:
        s = ref.stats
        text += f"  mean={s.mean:.6g} std={s.std:.6g} min={s.min:.6g} max={s.max:.6g}"
    return text


def _tensors(tensors: tuple[NamedTensor, ...]) -> list[str]:
    return [f"  {t.path}: {_describe(t.ref)}" for t in tensors] or ["  (no tensor leaves)"]


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _metric(spec: MetricSpec) -> str:
    kind = "built-in metric" if spec.builtin else "caller metric (declared, unverified)"
    text = f"{spec.name} {_json(spec.params.to_plain())} [{kind}]"
    if spec.declaration is not None:
        text += f" declared revision {spec.declaration.implementation_revision}"
    return text


def _site(module: str, io: str, path: str) -> str:
    where = module or "<model>"
    return f"{where} {io}" + (f" {path}" if path else "")


_MAX_VALUES = 16


def _codes(lims: Sequence[TraceLimitation]) -> str:
    return ", ".join(lim.code for lim in lims) or "(none scoped)"


def _values(values: list[float]) -> str:
    shown = ", ".join(f"{v:.6g}" for v in values[:_MAX_VALUES])
    more = f", ... ({len(values) - _MAX_VALUES} more)" if len(values) > _MAX_VALUES else ""
    return f"[{shown}{more}]"


def _limitation_lines(lims: Sequence[TraceLimitation], indent: str) -> list[str]:
    lines = []
    for lim in lims:
        detail = f" [{lim.detail}]" if lim.detail else ""
        lines.append(
            f"{indent}- {lim.code} ({lim.severity.value}): {lim.definition.meaning}{detail}"
        )
    return lines


@dataclass(frozen=True, slots=True, eq=False)
class ExplainResponse:
    """INPUT -> WHY -> OUTPUT for one explanation context (one reference root pass)."""

    trace: TraceResult
    bundle: EvidenceBundle | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.trace, TraceResult):
            raise TypeError("ExplainResponse requires a TraceResult")
        if self.trace.passes != 1:
            raise ValueError(
                f"an explanation covers exactly one root invocation; this trace has "
                f"{self.trace.passes} passes"
            )
        if self.bundle is None:
            object.__setattr__(self, "bundle", EvidenceBundle.compose(self.trace))
        elif not isinstance(self.bundle, EvidenceBundle) or self.bundle.reference is not self.trace:
            raise ValueError("the bundle must be composed over this trace")

    @classmethod
    def from_trace(cls, trace: TraceResult) -> ExplainResponse:
        """Rebuild the response from a (e.g. loaded) single-pass trace."""
        return cls(trace)

    @classmethod
    def from_evidence(
        cls,
        trace: TraceResult,
        *,
        attributions: Sequence[AttributionResult] = (),
        interventions: Sequence[InterventionResult] = (),
        claims: Sequence[Claim] = (),
        policies: Sequence[AssessmentPolicy] = (),
    ) -> ExplainResponse:
        """Compose already-computed evidence (see :class:`EvidenceBundle`). Runs no model."""
        bundle = EvidenceBundle.compose(
            trace,
            attributions=attributions,
            interventions=interventions,
            claims=claims,
            policies=policies,
        )
        return cls(trace, bundle)

    @property
    def input(self) -> InputRecord:
        return self.trace.input

    @property
    def output(self) -> OutputRecord:
        return self.trace.output

    @property
    def why(self) -> Why:
        return Why(self.trace, self.bundle)

    # ------------------------------------------------------------------ rendering

    def render(self) -> str:
        """Deterministic plain-text view derived only from the structured records.

        Not generated prose and not reasoning: every line restates a record. The
        measured-only form keeps every Phase-1 line and appends NOT EVALUATED.
        """
        why = self.why
        if not why._target_view:
            return "\n".join([*self._render_measured(why), "", *self._render_not_evaluated(why)])
        lines = ["INPUT  [observed]", *_tensors(self.input.tensors)]
        lines.append(f"  sample: {self.input.sample_id}")
        lines += ["", "TARGET"]
        target = why.target
        if target is not None:
            lines.append(f"  {_metric(target)}")
        else:
            spec = why.target_spec
            assert spec is not None
            lines.append(f"  {spec.metric} {_json(spec.params.to_plain())} [claim target]")
        lines += ["", "WHY  [evidence grouped by epistemic status; statuses are never combined]"]
        lines.append(f"  answers: {why.question}")
        lines.append(f"  does not answer: {why.not_answered}")
        lines += self._render_measurements(why)
        lines += self._render_attributions(why)
        lines += self._render_interventions(why)
        lines += ["", "  ESTIMATED_CAUSAL", "    (none)"]
        lines += self._render_claims(why)
        lines += self._render_coverage(why)
        lines += ["", "  LIMITATIONS", *_limitation_lines(why.limitations, "    ")]
        lines += ["", *self._render_not_evaluated(why)]
        lines += ["", "OUTPUT  [observed]", *_tensors(self.output.tensors)]
        return "\n".join(lines)

    def _render_measured(self, why: Why) -> list[str]:
        sites = sorted({(a.site.module, a.site.io.value) for a in why.activations})
        lines = ["INPUT  [observed]", *_tensors(self.input.tensors), ""]
        lines.append("WHY  [measured internal evidence; not a causal or attributed explanation]")
        lines.append(f"  answers: {Why.QUESTION}")
        lines.append(f"  does not answer: {Why.NOT_ANSWERED}")
        lines.append(
            f"  {len(why.activations)} measured activation records at {len(sites)} site(s)"
        )
        for a in why.activations:
            leaf = f" {a.site.output_path}" if a.site.output_path else ""
            where = f"{a.site.module} {a.site.io.value}{leaf}"
            lines.append(
                f"    {where}  pass {a.pass_index} call {a.call_index}: {_describe(a.value)}"
            )
        statuses = ", ".join(sorted(s.value for s in why.evidence_statuses))
        lines.append(f"  evidence statuses present: {statuses}")
        lines.append("  limitations:")
        lines += _limitation_lines(why.limitations, "    ")
        lines += ["", "OUTPUT  [observed]", *_tensors(self.output.tensors)]
        return lines

    @staticmethod
    def _render_measurements(why: Why) -> list[str]:
        lines = ["", "  MEASURED  [internal states read during the reference pass]"]
        if not why.activations:
            lines.append("    (none recorded)")
        for a in why.activations:
            lines.append(
                f"    {_site(a.site.module, a.site.io.value, a.site.output_path)}  pass "
                f"{a.pass_index} call {a.call_index}: {_describe(a.value)}"
            )
        return lines

    @staticmethod
    def _render_attributions(why: Why) -> list[str]:
        lines = ["", "  ATTRIBUTED  [method-relative scores; not intervention effects; not ranked]"]
        views = why.attribution_views
        if not views:
            lines.append("    (none: no attribution method was run)")
        for n, view in enumerate(views, 1):
            r = view.record
            m = r.method
            lines.append(
                f"    attribution {n}: {m.name} ({m.implementation} {m.implementation_version}) "
                f"{_json(m.params.to_plain())}"
            )
            lines.append(
                f"      site: {_site(r.site.module, r.site.io.value, r.site.output_path)}  "
                f"call {r.call_index}  pass {r.pass_index}"
            )
            lines.append(f"      target: {_metric(r.target)}  value at input {r.target_value:.6g}")
            if r.baseline is None:
                lines.append("      baseline: (not used by this method)")
            else:
                b = r.baseline
                extra = "" if b.value is None else f" {b.value.content_digest}"
                where = "" if b.input_path is None else f" replacing {b.input_path}"
                lines.append(f"      baseline: {b.kind.value}{extra}{where}")
            values = view.value.double().flatten().tolist()
            lines.append(
                f"      attribution {tuple(r.value.shape)} (index order): {_values(values)}"
            )
            for red in view.reductions:
                shown = (
                    f"{red.scalar:.6g}"
                    if red.scalar is not None
                    else _values(view.source.tensor(red.value).flatten().tolist())
                )
                lines.append(f"      reduction {red.reduction} over dims {list(red.dims)}: {shown}")
            if r.diagnostics:
                diag = ", ".join(
                    f"{k}={float(v):.6g}"  # type: ignore[arg-type]
                    for k, v in sorted(r.diagnostics.items())
                )
                lines.append(f"      diagnostics: {diag}")
            lines.append(f"      limitations: {_codes(view.limitations)}")
        return lines

    @staticmethod
    def _render_interventions(why: Why) -> list[str]:
        lines = ["", "  INTERVENTIONAL  [effects of declared interventions on this input]"]
        views = why.intervention_views
        if not views:
            lines.append("    (none: no intervention was run)")
        for n, view in enumerate(views, 1):
            i, e = view.intervention, view.effect
            op = i.operation.value
            if i.constant is not None:
                op += f" {i.constant:.6g}"
            lines.append(
                f"    intervention {n}: {op} at "
                f"{_site(i.site.module, i.site.io.value, i.site.output_path)} call {i.call_index}"
            )
            if view.source_input is not None:
                lines.append(
                    f"      patch source context: sample {view.source_input.sample_id} "
                    f"(pass {view.source_input.pass_index})"
                )
            lines.append(f"      target: {_metric(e.metric)}")
            lines.append(
                f"      under this intervention the target changed by {e.effect:.6g} "
                f"(baseline {e.baseline_value:.6g} -> intervention {e.intervention_value:.6g})"
            )
            lines.append(f"      scope: {e.estimand.scope.value} (exactly this input)")
            lines.append(f"      limitations: {_codes(view.limitations)}")
        return lines

    @staticmethod
    def _render_claims(why: Why) -> list[str]:
        lines = ["", "  CLAIMS  [declared; tested only by registered protocols]"]
        views = why.claims
        if not views:
            lines.append("    (none declared)")
        for n, view in enumerate(views, 1):
            c = view.claim
            s = c.subject.site
            units = "" if c.subject.units is None else f" units {list(c.subject.units)}"
            lines.append(f"    claim {n}: {c.statement!r}")
            lines.append(
                f"      relation {c.relation.value}; subject "
                f"{_site(s.module, s.io.value, s.output_path)}{units}; "
                f"estimand {c.estimand.scope.value}"
            )
            if not view.tests:
                lines.append("      tests: (none recorded)")
            for spec, result in view.tests:
                evidence = ", ".join(
                    f"{ev.record_id} [{ev.status.value}]" for ev in result.evidence
                )
                lines.append(
                    f"      test {spec.protocol} v{spec.protocol_version} "
                    f"{_json(spec.criteria.to_plain())}: {result.outcome.value}"
                    + (f"  evidence: {evidence}" if evidence else "  (no evidence cited)")
                )
            if view.is_causal and not view.causal_test_performed:
                lines.append("      no decisive causal test was performed for this causal claim")
            if not view.assessments:
                lines.append(
                    "      assessment: not assessed (no supplied policy covers this relation)"
                )
            for a in view.assessments:
                missing = (
                    f"; required protocols without a supports result: "
                    f"{', '.join(a.required_but_missing)}"
                    if a.required_but_missing
                    else ""
                )
                lines.append(
                    f"      assessment {a.policy.name} v{a.policy.version}: "
                    f"{a.verdict.value}{missing}"
                )
        return lines

    @staticmethod
    def _render_coverage(why: Why) -> list[str]:
        cov = why.coverage
        yes = {True: "yes", False: "no"}
        return [
            "",
            "  COVERAGE  [which evidence exists; not a confidence]",
            f"    measured internal states: {yes[cov.measured_internal_states]}",
            f"    attribution: {yes[cov.attribution_available]}",
            f"    intervention effects: {yes[cov.intervention_effect_available]}",
            f"    estimated causal effects: {yes[cov.estimated_causal_available]}",
            f"    claims declared: {yes[cov.claims_declared]}; tested: {yes[cov.claims_tested]}; "
            f"causal claim tested: {yes[cov.causal_claim_tested]}",
        ]

    @staticmethod
    def _render_not_evaluated(why: Why) -> list[str]:
        return [
            "NOT EVALUATED",
            *[f"  - {item}" for item in Coverage.NOT_EVALUATED],
            *[f"  unanswered: {q}" for q in why.unanswered],
        ]

    # ------------------------------------------------------------------ structured export

    def to_dict(self) -> dict[str, Any]:
        """A deterministic, JSON-compatible presentation summary (record ids plus the
        values shown by :meth:`render`). Presentation only: the traces remain the
        scientific records and the codec their serialisation."""
        why = self.why
        cov = why.coverage
        return {
            "input": {"record": self.input.id, "sample_id": self.input.sample_id},
            "output": {"record": self.output.id},
            "target": None if why.target is None else why.target.target().metric,
            "target_params": None if why.target is None else why.target.params.to_plain(),
            "measured": [a.id for a in why.activations],
            "attributed": [
                {
                    "record": v.record.id,
                    "method": v.record.method.name,
                    "implementation": v.record.method.implementation,
                    "site": v.record.site.module,
                    "leaf": v.record.site.output_path,
                    "call_index": v.record.call_index,
                    "values": v.value.double().flatten().tolist(),
                    "reductions": [r.id for r in v.reductions],
                    "diagnostics": v.record.diagnostics.to_plain(),
                    "limitations": [lim.code for lim in v.limitations],
                }
                for v in why.attribution_views
            ],
            "interventional": [
                {
                    "record": v.effect.id,
                    "operation": v.intervention.operation.value,
                    "site": v.intervention.site.module,
                    "leaf": v.intervention.site.output_path,
                    "call_index": v.intervention.call_index,
                    "baseline_value": v.effect.baseline_value,
                    "intervention_value": v.effect.intervention_value,
                    "effect": v.effect.effect,
                    "scope": v.effect.estimand.scope.value,
                    "limitations": [lim.code for lim in v.limitations],
                }
                for v in why.intervention_views
            ],
            "estimated_causal": [e.id for e in why.estimated_causal],
            "claims": [
                {
                    "record": v.claim.id,
                    "statement": v.claim.statement,
                    "relation": v.claim.relation.value,
                    "tests": [
                        {"protocol": s.protocol, "result": r.id, "outcome": r.outcome.value}
                        for s, r in v.tests
                    ],
                    "assessments": [
                        {
                            "policy": a.policy.name,
                            "version": a.policy.version,
                            "verdict": a.verdict.value,
                        }
                        for a in v.assessments
                    ],
                }
                for v in why.claims
            ],
            "limitations": [lim.id for lim in why.limitations],
            "coverage": {
                "measured_internal_states": cov.measured_internal_states,
                "attribution_available": cov.attribution_available,
                "intervention_effect_available": cov.intervention_effect_available,
                "estimated_causal_available": cov.estimated_causal_available,
                "claims_declared": cov.claims_declared,
                "claims_tested": cov.claims_tested,
                "causal_claim_tested": cov.causal_claim_tested,
                "faithfulness_evaluated": cov.faithfulness_evaluated,
                "concepts_validated": cov.concepts_validated,
            },
            "not_evaluated": list(Coverage.NOT_EVALUATED),
            "unanswered": list(why.unanswered),
        }


def compose(
    trace: TraceResult,
    *,
    attributions: Sequence[AttributionResult] = (),
    interventions: Sequence[InterventionResult] = (),
    claims: Sequence[Claim] = (),
    policies: Sequence[AssessmentPolicy] = (),
) -> ExplainResponse:
    """Compose already-computed evidence into one INPUT -> TARGET -> WHY -> OUTPUT view.

    ``trace`` is the reference single-pass trace (``handle.trace(x, sites=...)``).
    Attributions and interventions must have been run explicitly beforehand; this
    runs no model and no method. ``claims`` adds declared claims (claims recorded with
    the evidence are included automatically). ``policies`` are the explicit,
    versioned assessment policies to apply (e.g. ``bnn.interventions.INTERVENTION_POLICY``,
    ``bnn.attribution.ATTRIBUTION_POLICY``). Incompatible evidence is refused.
    """
    return ExplainResponse.from_evidence(
        trace,
        attributions=attributions,
        interventions=interventions,
        claims=claims,
        policies=policies,
    )
