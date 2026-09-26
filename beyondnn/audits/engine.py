"""The audit (plan §1, §6-§24): deterministic, model-free, no score.

``audit(evidence, plan=plan)`` classifies every plan claim and concept from the
re-derived, in-scope evidence by the fixed rules of the pre-registered plan
(docs/PHASE_7_PLAN.md §22). Contradictions are listed, never resolved; missing
evidence is NOT_EVALUATED, never positive or negative.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import combinations
from typing import Any

from beyondnn.protocols import (
    CONCEPT_ENCODING,
    CONCEPT_INTERVENTION,
    PROTOCOLS,
    check_policy,
)
from beyondnn.schema import (
    CAUSAL_RELATIONS,
    Assessment,
    AuditedClaim,
    AuditedConcept,
    AuditPlan,
    Claim,
    ConceptRecord,
    EstimandScope,
    EvidenceRequirement,
    Invariance,
    LabelSource,
    Outcome,
    ProtocolResult,
    Relation,
    ResultRef,
    SemanticStatus,
    Verdict,
    derive_verdict,
)

from .evidence import AXES, EvidenceSet, Problem, ResultEntry, ValidationEntry, _key, collect_traces
from .report import (
    AuditReport,
    ClaimAudit,
    ConceptAudit,
    Coverage,
    Diagnostic,
    EvidenceSummary,
    Finding,
    FindingKind,
    FindingSeverity,
    GroupAudit,
    InventoryEntry,
    Standing,
    TestEntry,
    aggregate,
)

__all__ = ["AuditPlanError", "audit"]

K, S = FindingKind, FindingSeverity
_DECISIVE = (Outcome.SUPPORTS, Outcome.CONTRADICTS)
#: Findings that make a claim UNSUPPORTED as stated (plan §10, §15, §22).
_UNSUPPORTING = frozenset(
    {
        "attribution_is_not_intervention",
        "decodability_is_not_use",
        "generated_label_is_not_validation",
        "narrower_estimand",
        "invariance_untested",
        "missing_required_controls",
        "counterexample_cap_exceeded",
        "counterexample_heavy",
        "decodable_not_used",
    }
)


class AuditPlanError(ValueError):
    """The plan cannot be audited (plan §24)."""


# ------------------------------------------------------------------ entry point


def audit(evidence: Iterable[Any], *, plan: AuditPlan, model: Any = None) -> AuditReport:
    """Audit ``evidence`` (traces, saved trace paths, or result objects) under ``plan``.

    Runs no model. ``model`` (optional) is only fingerprinted: it must be the plan's
    checkpoint, or the audit refuses (``ModelMismatchError``)."""
    if not isinstance(plan, AuditPlan):
        raise TypeError("plan must be an AuditPlan (see beyondnn.audits.plan)")
    _check_plan(plan)
    if model is not None:
        from beyondnn.explain.bundle import ModelMismatchError
        from beyondnn.provenance import fingerprint_model

        digest = fingerprint_model(model).state_digest
        if digest != plan.checkpoint:
            raise ModelMismatchError(
                f"the model ({digest[:19]}...) is not the plan's checkpoint "
                f"({plan.checkpoint[:19]}...); an audit never applies to another checkpoint"
            )
    if isinstance(evidence, (str, bytes)):
        raise TypeError("evidence must be a sequence (of traces, paths, or result objects)")
    ev = EvidenceSet.build(collect_traces(evidence))
    return _Auditor(plan, ev).run()


def _check_plan(plan: AuditPlan) -> None:
    for requirement in plan.requirements:
        try:
            check_policy(requirement.policy)
        except Exception as exc:
            raise AuditPlanError(f"requirement {requirement.name}: {exc}") from None
        for alt in requirement.alternatives:
            if alt.protocol not in PROTOCOLS:
                raise AuditPlanError(f"alternative criteria for unregistered {alt.protocol!r}")
            if alt.protocol == CONCEPT_ENCODING:
                raise AuditPlanError(
                    "alternative criteria are not re-evaluable for concept_encoding "
                    "(declare a second recorded encoding test instead)"
                )


# ------------------------------------------------------------------ auditor


class _Auditor:
    def __init__(self, plan: AuditPlan, ev: EvidenceSet) -> None:
        self.plan = plan
        self.ev = ev
        self.requirements = {r.name: r for r in plan.requirements}
        self.samples = set(plan.samples)
        self.datasets = set(plan.datasets)
        self.matched_claims: dict[str, set[str]] = {}
        self.untested: list[tuple[str, str]] = []

    # ------------------------------------------------------------------ run

    def run(self) -> AuditReport:
        for entry in self.ev.results:
            entry.exclusion = self._exclusion(entry)
        for v in self.ev.validations:
            v.exclusion = self._validation_exclusion(v)
        claims = [self._claim(c) for c in self.plan.claims]
        claims = self._cross_claim(claims)
        concepts = tuple(self._concept(c) for c in self.plan.concepts)
        findings = self._report_findings()
        usable = [e for e in self.ev.results if e.usable]
        return AuditReport(
            plan=self.plan,
            evidence=EvidenceSummary(
                digest=self.ev.digest(),
                traces=len(self.ev.traces) + len(self.ev.failed_traces),
                records=self.ev.records,
                results=len(self.ev.results),
                results_included=len(usable),
                results_excluded=len(self.ev.results) - len(usable),
            ),
            inventory=self._inventory(),
            claims=tuple(claims),
            concepts=concepts,
            findings=findings,
            diagnostics=tuple(
                Diagnostic(
                    record=r.id,
                    protocol=r.protocol,
                    samples=len(r.samples),
                    verified=ok,
                    outcomes=tuple((a.aspect, a.outcome.value) for a in r.outcomes),
                )
                for r, ok in sorted(self.ev.diagnostics, key=lambda d: d[0].id)
            ),
            coverage=self._coverage(claims, concepts),
            limitations=_count(lim.code for t in self.ev.traces for lim in t.limitations),
        )

    # ------------------------------------------------------------------ scope

    def _exclusion(self, entry: ResultEntry) -> Problem | None:
        if entry.problem is not None:
            return None  # already unusable (integrity / missing source)
        if entry.checkpoint != self.plan.checkpoint:
            return Problem(
                "provenance_mismatch",
                "other_checkpoint",
                f"{entry.id} was recorded on checkpoint {str(entry.checkpoint)[:19]}..., not the "
                f"plan's {self.plan.checkpoint[:19]}...",
            )
        if entry.declared_model != self.plan.declared_model:
            return Problem(
                "provenance_mismatch",
                "other_model_declaration",
                f"{entry.id} was recorded under another model declaration",
            )
        sample = entry.sample
        if sample is not None and sample not in self.samples:
            return Problem(
                "scope_mismatch",
                "sample_out_of_scope",
                f"{entry.id} is about sample {sample}, which the plan does not declare",
            )
        dataset = entry.dataset
        if entry.spec.protocol in (CONCEPT_ENCODING, CONCEPT_INTERVENTION) and (
            dataset not in self.datasets
        ):
            return Problem(
                "scope_mismatch",
                "dataset_out_of_scope",
                f"{entry.id} is about concept dataset {dataset}, which the plan does not declare",
            )
        return None

    def _validation_exclusion(self, v: ValidationEntry) -> Problem | None:
        if v.problem is not None:
            return None
        if v.record.model_state_digest != self.plan.checkpoint:
            return Problem(
                "provenance_mismatch",
                "other_checkpoint",
                f"{v.record.id} was validated on another checkpoint",
            )
        if v.record.dataset.record_id not in self.datasets:
            return Problem(
                "scope_mismatch",
                "dataset_out_of_scope",
                f"{v.record.id} is about a concept dataset the plan does not declare",
            )
        return None

    # ------------------------------------------------------------------ claims

    def _structure(self, c: AuditedClaim, e: ResultEntry, *, method_known: bool = True) -> bool:
        claim = e.claim
        if claim.relation is not c.relation or claim.target != _target_for(c, e.sample):
            return False
        if c.subject is not None:
            return claim.subject == c.subject
        sel = c.selection
        assert sel is not None
        s = e.selection
        if s is None or s.site != sel.site or (sel.k is not None and s.k != sel.k):
            return False
        if method_known and e.method != sel.method:
            return False
        return s.k is not None and claim.subject.units == s.selected

    def _group_of(self, c: AuditedClaim, e: ResultEntry) -> tuple[str, str | None]:
        """("match", group) | ("narrower", group) | ("other_set", None) | ("none", None)."""
        est = e.claim.estimand
        if c.scope is EstimandScope.INSTANCE:
            return (
                ("match", est.sample_id) if est.scope is EstimandScope.INSTANCE else ("none", None)
            )
        if c.scope is EstimandScope.FINITE_SAMPLE:
            if est.scope is EstimandScope.FINITE_SAMPLE:
                return ("match", None) if est.sample_id == c.sample_set else ("other_set", None)
            return ("narrower", None) if est.scope is EstimandScope.INSTANCE else ("none", None)
        if est.scope is EstimandScope.POPULATION and est.population == c.population:
            return "match", None
        return "narrower", None

    def _related(
        self, c: AuditedClaim, group: str | None
    ) -> tuple[list[str], list[Finding], list[ResultEntry]]:
        """Non-deciding evidence about the same subject (plan §10 O1/O2)."""
        ids: list[str] = []
        findings: list[Finding] = []
        attribution_results: list[ResultEntry] = []
        if c.relation not in CAUSAL_RELATIONS:
            return ids, findings, attribution_results
        if c.subject is not None:
            for e in self.ev.results:
                if not e.usable or e.claim.subject != c.subject:
                    continue
                r = e.claim.relation
                if r is Relation.ATTRIBUTED_TO and e.claim.target == _target_for(c, e.sample):
                    same = c.scope is not EstimandScope.INSTANCE or e.sample == group
                    if same:
                        ids.append(e.id)
                        attribution_results.append(e)
                elif r is Relation.ENCODES and c.subject.feature is not None:
                    ids.append(e.id)
                    if not any(f.code == "decodability_is_not_use" for f in findings):
                        findings.append(
                            _f(
                                K.EVIDENCE_TYPE_MISMATCH,
                                "decodability_is_not_use",
                                S.BLOCKING,
                                c.name,
                                "only decodability (ENCODES) evidence exists for this feature; "
                                "decodability is not use",
                            )
                        )
            if attribution_results:
                findings.append(
                    _f(
                        K.EVIDENCE_TYPE_MISMATCH,
                        "attribution_is_not_intervention",
                        S.BLOCKING,
                        c.name,
                        "only ATTRIBUTED evidence exists for this causal claim; an attribution "
                        "is not an intervention",
                        records=tuple(e.id for e in attribution_results),
                    )
                )
        else:
            sel = c.selection
            assert sel is not None
            for record, trace in self.ev.attributions:
                if (
                    record.sample_id == group
                    and record.site == sel.site
                    and record.method.name == sel.method
                    and record.target.target() == _target_for(c, group)
                    and trace.origin(record).model.state_digest == self.plan.checkpoint
                ):
                    ids.append(record.id)
            if ids:
                findings.append(
                    _f(
                        K.EVIDENCE_TYPE_MISMATCH,
                        "attribution_is_not_intervention",
                        S.BLOCKING,
                        c.name,
                        f"only {sel.method} attributions exist for this sample; the selected units "
                        "were never intervened on",
                        records=tuple(ids),
                    )
                )
        return ids, findings, attribution_results

    def _claim(self, c: AuditedClaim) -> ClaimAudit:
        requirement = self.requirements[c.requirement]
        matched: dict[str | None, list[ResultEntry]] = {}
        narrower: list[ResultEntry] = []
        excluded: dict[str | None, list[ResultEntry]] = {}
        claim_level: list[Finding] = []
        for e in self.ev.results:
            unresolved = e.problem is not None and e.problem.code == "selection_source_not_supplied"
            if not self._structure(c, e, method_known=not unresolved):
                continue
            kind, group = self._group_of(c, e)
            if kind == "none":
                continue
            if not e.usable:
                problem = e.problem or e.exclusion
                assert problem is not None
                if kind == "match" and (
                    c.scope is not EstimandScope.INSTANCE or group in self.samples
                ):
                    excluded.setdefault(group, []).append(e)
                else:
                    claim_level.append(_problem_finding(problem, c.name, e, S.QUALIFYING))
                continue
            if kind == "other_set":
                claim_level.append(
                    _f(
                        K.SCOPE_MISMATCH,
                        "other_sample_set",
                        S.QUALIFYING,
                        c.name,
                        f"{e.id} is about another sample set than the claim declares",
                        records=(e.id,),
                    )
                )
                continue
            self.matched_claims.setdefault(e.claim.id, set()).add(c.name)
            if kind == "narrower":
                narrower.append(e)
            else:
                matched.setdefault(group, []).append(e)
        per_sample = c.scope is EstimandScope.INSTANCE
        groups_keys: list[str | None] = [None]
        if per_sample:
            groups_keys = [*sorted(self.samples)]
        groups = []
        for g in groups_keys:
            groups.append(
                self._group(c, requirement, g, matched.get(g, []), excluded.get(g, []), narrower)
            )
        distribution = _count(g.standing.value for g in groups)
        counterexamples = tuple(
            g.sample for g in groups if g.standing is Standing.CONTRADICTED and g.sample
        )
        findings: list[Finding] = list(claim_level)
        standing: Standing | None
        if per_sample:
            standing = None
            findings += aggregate((f for g in groups for f in g.findings), c.name)
            findings += self._cap(c, groups, counterexamples)
        else:
            standing = groups[0].standing
            findings += groups[0].findings
        limitations = self._limitations([e for es in matched.values() for e in es])
        return ClaimAudit(
            claim=c,
            per_sample=per_sample,
            standing=standing,
            groups=tuple(groups),
            distribution=tuple(sorted(distribution)),
            counterexamples=tuple(sorted(counterexamples)),
            findings=tuple(_dedupe(findings)),
            limitations=limitations,
        )

    def _group(
        self,
        c: AuditedClaim,
        requirement: EvidenceRequirement,
        group: str | None,
        matched: list[ResultEntry],
        excluded: list[ResultEntry],
        narrower: list[ResultEntry],
    ) -> GroupAudit:
        name = c.name
        samples = (group,) if group is not None else ()
        findings: list[Finding] = []
        verdict, missing = derive_verdict(
            c.relation, requirement.policy, [ResultRef.to(e.result) for e in matched]
        )
        tests = [
            TestEntry(e.id, e.spec.protocol, e.outcome.value, tuple(sorted(e.axes.items())))
            for e in sorted(matched, key=lambda e: e.id)
        ]
        decisive: list[tuple[dict[str, str], Outcome, str]] = [
            (e.axes, e.outcome, e.id) for e in matched if e.outcome in _DECISIVE
        ]
        for alt in requirement.alternatives:
            for e in matched:
                outcome = self.ev.reevaluate(e, alt)
                if outcome is None:
                    continue
                how = f"x{alt.factor!r}" if alt.factor is not None else f"={alt.value!r}"
                label = f"{alt.key}{how}"
                axes = dict(e.axes) | {"threshold": f"{e.axes['threshold']}|alt:{label}"}
                tests.append(
                    TestEntry(
                        e.id, e.spec.protocol, outcome.value, tuple(sorted(axes.items())), label
                    )
                )
                decisive.append((axes, outcome, e.id))
        # inconclusive evidence is never ignored
        weak = [e for e in matched if e.outcome not in _DECISIVE]
        if weak:
            findings.append(
                _f(
                    K.INCONCLUSIVE_EVIDENCE,
                    "inconclusive_results",
                    S.QUALIFYING,
                    name,
                    f"{len(weak)} matched result(s) are "
                    + ", ".join(sorted({e.outcome.value for e in weak})),
                    records=tuple(sorted(e.id for e in weak)),
                    samples=samples,
                )
            )
        sup = [d for d in decisive if d[1] is Outcome.SUPPORTS]
        con = [d for d in decisive if d[1] is Outcome.CONTRADICTS]
        unexplained = False
        if sup and con:
            sensitivity, unexplained = _disagreement(sup, con, name, samples)
            findings += sensitivity
        ran = {e.spec.protocol for e in matched}
        missing = tuple(p for p in missing if p not in ran)  # never recorded (not: contradicted)
        if missing and matched:  # with nothing matched, the standing already says so
            findings.append(
                _f(
                    K.MISSING_EVIDENCE,
                    "required_protocol_missing",
                    S.QUALIFYING,
                    name,
                    f"requirement {requirement.name} needs a SUPPORTS result from "
                    f"{', '.join(missing)}; none was recorded",
                    samples=samples,
                )
            )
        if requirement.controls:
            supports = [e for e in matched if e.outcome is Outcome.SUPPORTS]
            if supports and not any(e.controlled for e in supports):
                findings.append(
                    _f(
                        K.MISSING_CONTROL,
                        "missing_required_controls",
                        S.BLOCKING,
                        name,
                        f"requirement {requirement.name} requires controls; no SUPPORTS result "
                        "declared a control criterion",
                        records=tuple(sorted(e.id for e in supports)),
                        samples=samples,
                    )
                )
        values = {a: sorted({e.axes[a] for e in matched}) for a in AXES}
        values["threshold"] = sorted(
            {d[0]["threshold"] for d in decisive} | set(values["threshold"])
        )
        for inv in c.invariant_over:
            findings += self._invariance(name, inv, values, samples, bool(matched))
        related: list[str] = []
        if not matched:
            related, type_findings, _ = self._related(c, group)
            findings += [_with_samples(f, samples) for f in type_findings]
            if narrower and c.scope is not EstimandScope.INSTANCE:
                findings.append(
                    _f(
                        K.SCOPE_MISMATCH,
                        "narrower_estimand",
                        S.BLOCKING,
                        name,
                        "only evidence with a narrower estimand exists "
                        f"({len(narrower)} result(s): "
                        + ", ".join(
                            f"{k} {n}" for k, n in _count(e.outcome.value for e in narrower)
                        )
                        + f"); it cannot decide a {c.scope.value} claim",
                        records=tuple(sorted(e.id for e in narrower)),
                    )
                )
                related += [e.id for e in narrower]
        elif con and c.relation in CAUSAL_RELATIONS:
            _, _, attribution = self._related(c, group)
            agree = [e for e in attribution if e.outcome is Outcome.SUPPORTS]
            if agree:
                findings.append(
                    _f(
                        K.PROTOCOL_DISAGREEMENT,
                        "attribution_intervention_disagree",
                        S.QUALIFYING,
                        name,
                        "an attribution test SUPPORTS importance of this subject while an "
                        "interventional test CONTRADICTS the causal claim",
                        records=tuple(sorted(e.id for e in agree) + sorted(d[2] for d in con)),
                        samples=samples,
                    )
                )
        for e in excluded:
            problem = e.problem or e.exclusion
            assert problem is not None
            severity = (
                S.BLOCKING if (problem.kind == "integrity_failure" or not matched) else S.QUALIFYING
            )
            findings.append(_problem_finding(problem, name, e, severity, samples))
        standing = _standing(verdict, sup, con, unexplained, findings)
        if standing is Standing.NOT_EVALUATED and not related and not excluded:
            findings.append(
                _f(
                    K.NOT_EVALUATED,
                    "no_evidence",
                    S.BLOCKING,
                    name,
                    "no in-scope evidence was supplied for this claim"
                    + (" on this sample" if group is not None else ""),
                    samples=samples,
                )
            )
        return GroupAudit(
            sample=group,
            standing=standing,
            verdict=verdict.value,
            tests=tuple(sorted(tests, key=lambda t: (t.result, t.alternative or ""))),
            findings=tuple(_dedupe(findings)),
            axis_values=tuple((a, tuple(v)) for a, v in sorted(values.items()) if v),
            related=tuple(sorted(related)),
        )

    def _invariance(
        self,
        name: str,
        inv: Invariance,
        values: dict[str, list[str]],
        samples: tuple[str, ...],
        any_evidence: bool,
    ) -> list[Finding]:
        tested = values.get(inv.axis.value, [])
        tested = [v for v in tested if v != "-"]
        absent = [v for v in inv.values if v not in tested]
        if not any_evidence or (len(tested) >= inv.min_values and not absent):
            return []
        self.untested.append((name, inv.axis.value))
        return [
            _f(
                K.MISSING_EVIDENCE,
                "invariance_untested",
                S.BLOCKING,
                name,
                f"the claim asserts invariance over {inv.axis.value} (>= {inv.min_values} values"
                + (f" incl. {list(inv.values)}" if inv.values else "")
                + f"); {len(tested)} value(s) tested"
                + (f", missing {absent}" if absent else ""),
                axis=inv.axis.value,
                values=tuple(("tested", v) for v in tested),
                samples=samples,
            )
        ]

    def _cap(
        self, c: AuditedClaim, groups: Sequence[GroupAudit], counter: Sequence[str]
    ) -> list[Finding]:
        evaluated = [g for g in groups if g.standing is not Standing.NOT_EVALUATED]
        if not counter:
            return []
        cap = self.plan.counterexamples.max_counterexample_fraction
        fraction = len(counter) / len(evaluated)
        detail = (
            f"{len(counter)} of {len(evaluated)} evaluated samples are CONTRADICTED "
            f"(fraction {fraction:.4g}; "
            + ("no cap declared)" if cap is None else f"declared cap {cap:.4g})")
        )
        exceeded = cap is not None and fraction > cap
        return [
            _f(
                K.COUNTEREXAMPLE_FOUND,
                "counterexample_cap_exceeded" if exceeded else "counterexamples_present",
                S.BLOCKING if exceeded else S.QUALIFYING,
                c.name,
                detail,
                samples=tuple(sorted(counter)),
            )
        ]

    def _cross_claim(self, claims: list[ClaimAudit]) -> list[ClaimAudit]:
        """PROTOCOL_DISAGREEMENT between plan claims about the same subject/selection and
        target with different relations (plan §8b)."""
        extra: dict[str, list[Finding]] = {}
        for a, b in combinations(claims, 2):
            ca, cb = a.claim, b.claim
            same_subject = (ca.subject, ca.selection) == (cb.subject, cb.selection)
            same_target = (ca.target, ca.sample_targets) == (cb.target, cb.sample_targets)
            if not same_subject or not same_target or ca.scope is not cb.scope:
                continue
            if ca.relation is cb.relation or ca.sample_set != cb.sample_set:
                continue
            gb = {g.sample: g for g in b.groups}
            for ga in a.groups:
                other = gb.get(ga.sample)
                if other is None:
                    continue
                pair = {ga.standing, other.standing}
                if pair != {Standing.SUPPORTED, Standing.CONTRADICTED}:
                    continue
                supported = ca if ga.standing is Standing.SUPPORTED else cb
                code = "relations_disagree"
                if supported.relation in (Relation.NECESSARY_FOR, Relation.DECREASES) and (
                    Relation.SUFFICIENT_FOR in (ca.relation, cb.relation)
                ):
                    code = "necessary_not_sufficient"
                elif supported.relation is Relation.SUFFICIENT_FOR:
                    code = "sufficient_not_necessary"
                samples = (ga.sample,) if ga.sample else ()
                detail = (
                    f"{ca.name} ({ca.relation.value}) is {ga.standing.value} while {cb.name} "
                    f"({cb.relation.value}) is {other.standing.value}"
                )
                for claim in (ca, cb):
                    extra.setdefault(claim.name, []).append(
                        _f(
                            K.PROTOCOL_DISAGREEMENT,
                            code,
                            S.QUALIFYING,
                            claim.name,
                            detail,
                            samples=samples,
                        )
                    )
        out = []
        for c in claims:
            more = extra.get(c.name, [])
            if not more:
                out.append(c)
                continue
            level = aggregate(more, c.name) if c.per_sample else tuple(more)
            groups = c.groups
            if not c.per_sample:
                g = groups[0]
                groups = (
                    GroupAudit(
                        g.sample,
                        g.standing,
                        g.verdict,
                        g.tests,
                        (*g.findings, *more),
                        g.axis_values,
                        g.related,
                    ),
                )
            else:
                by_sample: dict[str | None, list[Finding]] = {}
                for f in more:
                    by_sample.setdefault(f.samples[0] if f.samples else None, []).append(f)
                groups = tuple(
                    GroupAudit(
                        g.sample,
                        g.standing,
                        g.verdict,
                        g.tests,
                        (*g.findings, *by_sample.get(g.sample, [])),
                        g.axis_values,
                        g.related,
                    )
                    for g in groups
                )
            out.append(
                ClaimAudit(
                    claim=c.claim,
                    per_sample=c.per_sample,
                    standing=c.standing,
                    groups=groups,
                    distribution=c.distribution,
                    counterexamples=c.counterexamples,
                    findings=tuple(_dedupe((*c.findings, *level))),
                    limitations=c.limitations,
                )
            )
        return out

    # ------------------------------------------------------------------ concepts

    def _concept(self, ac: AuditedConcept) -> ConceptAudit:
        cid = ac.concept
        found = self.ev.index.get(cid)
        name = cid
        if found is None or not isinstance(found[0], ConceptRecord):
            return ConceptAudit(
                concept=ac,
                label=None,
                label_source=None,
                standing=Standing.NOT_EVALUATED,
                validations=(),
                tests=(),
                findings=(
                    _f(
                        K.NOT_EVALUATED,
                        "concept_not_supplied",
                        S.BLOCKING,
                        name,
                        "the concept record was not in the supplied evidence",
                    ),
                ),
                false_positives=(),
                false_negatives=(),
                limitations=(),
            )
        concept = found[0]
        assert isinstance(concept, ConceptRecord)
        findings: list[Finding] = []
        validations: list[ValidationEntry] = []
        for v in self.ev.validations:
            if v.record.concept.record_id != cid:
                continue
            problem = v.problem or v.exclusion
            if problem is None and v.record.policy != ac.policy:
                problem = Problem(
                    "scope_mismatch",
                    "policy_differs",
                    f"{v.record.id} was derived under {v.record.policy.name}/v"
                    f"{v.record.policy.version}, not the plan's {ac.policy.name}/v"
                    f"{ac.policy.version}",
                )
            if problem is not None:
                findings.append(
                    Finding(
                        kind=FindingKind(problem.kind),
                        code=problem.code,
                        severity=S.BLOCKING
                        if problem.kind == "integrity_failure"
                        else S.QUALIFYING,
                        subject=name,
                        detail=problem.detail,
                        records=(v.record.id,),
                    )
                )
                continue
            if v.feature.startswith("not_rederived"):
                findings.append(
                    _f(
                        K.MISSING_EVIDENCE,
                        "feature_derivation_not_rederived",
                        S.QUALIFYING,
                        name,
                        f"the fitted/searched feature could not be re-derived: {v.feature}",
                        records=(v.record.feature.record_id,),
                    )
                )
            validations.append(v)
        tests: list[ResultEntry] = []
        for e in self.ev.results:
            if e.concept != cid:
                continue
            if not e.usable:
                problem = e.problem or e.exclusion
                assert problem is not None
                findings.append(_problem_finding(problem, name, e, S.QUALIFYING))
                continue
            tests.append(e)
        encodings = [e for e in tests if e.spec.protocol == CONCEPT_ENCODING]
        uses = [e for e in tests if e.spec.protocol == CONCEPT_INTERVENTION]
        unexplained = False
        disagreement = False
        for group in _by_structure(encodings) + _by_structure(uses):
            sup = [(e.axes, e.outcome, e.id) for e in group if e.outcome is Outcome.SUPPORTS]
            con = [(e.axes, e.outcome, e.id) for e in group if e.outcome is Outcome.CONTRADICTS]
            if sup and con:
                disagreement = True
                more, bad = _disagreement(sup, con, name, (), suffix="_sensitive")
                findings += more
                unexplained = unexplained or bad
        statuses = {v.record.semantic_status for v in validations}
        if len(statuses) > 1:
            disagreement = True
            more, bad = self._validation_disagreement(validations, name)
            findings += more
            unexplained = unexplained or bad
        if (
            any(e.outcome is Outcome.SUPPORTS for e in encodings)
            and uses
            and not any(e.outcome is Outcome.SUPPORTS for e in uses)
        ):
            findings.append(
                _f(
                    K.EVIDENCE_TYPE_MISMATCH,
                    "decodable_not_used",
                    S.BLOCKING,
                    name,
                    "the feature encodes the concept (held-out, above controls) but no use test "
                    "supports use: decodable, not shown to be used",
                    records=tuple(sorted(e.id for e in (*encodings, *uses))),
                )
            )
        validated = [
            v for v in validations if v.record.semantic_status is SemanticStatus.VALIDATED_CONCEPT
        ]
        if concept.label_source is LabelSource.GENERATED:
            if validated:
                findings.append(
                    _f(
                        K.LIMITATION,
                        "generated_label_unverified",
                        S.INFORMATIONAL,
                        name,
                        "the label text is GENERATED; the validation tests the dataset concept, "
                        "not the generated wording",
                        records=(concept.generated_label or "",),
                    )
                )
            else:
                findings.append(
                    _f(
                        K.EVIDENCE_TYPE_MISMATCH,
                        "generated_label_is_not_validation",
                        S.BLOCKING,
                        name,
                        "a GENERATED label is asserted as a validated concept without an in-scope "
                        "VALIDATED validation",
                        records=(concept.generated_label or "",),
                    )
                )
        naive = self.plan.naive_auroc
        if naive is not None:
            for e in encodings:
                auroc = e.result.statistics.get("auroc")
                if (
                    e.outcome is Outcome.CONTRADICTS
                    and isinstance(auroc, float)
                    and max(auroc, 1 - auroc) >= naive
                ):
                    findings.append(
                        _f(
                            K.ASSUMPTION_SENSITIVE,
                            "controls_defeat_encoding",
                            S.QUALIFYING,
                            name,
                            f"AUROC {auroc:.4g} would pass a naive {naive} bar, but the declared "
                            "controls reject the encoding claim",
                            axis="null",
                            records=(e.id,),
                        )
                    )
        other = self._other_concepts(concept.feature, cid)
        if other:
            findings.append(
                _f(
                    K.LIMITATION,
                    "feature_encodes_several_concepts",
                    S.QUALIFYING,
                    name,
                    f"the feature also has SUPPORTED encoding tests for {len(other)} other "
                    "concept(s); it may carry other information (polysemanticity not excluded)",
                    records=tuple(sorted(other)),
                )
            )
        values = {
            "null": sorted({e.axes["null"] for e in encodings}),
            "replacement": sorted({e.axes["replacement"] for e in uses}),
            "dataset": sorted({v.record.dataset.record_id for v in validations}),
        }
        for inv in ac.invariant_over:
            findings += self._invariance(name, inv, values, (), bool(tests or validations))
        fps: set[str] = set()
        fns: set[str] = set()
        rule = self.plan.counterexamples
        for v in validations:
            ce = self._counterexamples(v)
            fps |= set(ce[0])
            fns |= set(ce[1])
            fp, fn = v.record.false_positive_rate, v.record.false_negative_rate
            over = [
                f"false-positive rate {fp:.4g} > {cap:.4g}"
                for cap in [rule.max_false_positive_rate]
                if cap is not None and fp > cap
            ] + [
                f"false-negative rate {fn:.4g} > {cap:.4g}"
                for cap in [rule.max_false_negative_rate]
                if cap is not None and fn > cap
            ]
            caps = f"caps FP {rule.max_false_positive_rate}, FN {rule.max_false_negative_rate}"
            if over:
                findings.append(
                    _f(
                        K.COUNTEREXAMPLE_FOUND,
                        "counterexample_heavy",
                        S.BLOCKING,
                        name,
                        "; ".join(over) + f" ({caps})",
                        records=(v.record.id,),
                        samples=tuple(sorted(set(ce[0]) | set(ce[1]))),
                    )
                )
            elif ce[0] or ce[1]:
                findings.append(
                    _f(
                        K.COUNTEREXAMPLE_FOUND,
                        "counterexamples_present",
                        S.QUALIFYING,
                        name,
                        f"{len(ce[0])} false positive(s), {len(ce[1])} false negative(s) ({caps})",
                        records=(v.record.id,),
                        samples=tuple(sorted(set(ce[0]) | set(ce[1]))),
                    )
                )
        for lim in self._sae_limitations(concept.feature):
            findings.append(
                _f(K.LIMITATION, lim.lower(), S.QUALIFYING, name, f"{lim} applies to this feature")
            )
        if not validations:
            if any(f.code == "generated_label_is_not_validation" for f in findings):
                standing = Standing.UNSUPPORTED
            elif disagreement:
                standing = Standing.MIXED if unexplained else Standing.ASSUMPTION_SENSITIVE
            else:
                standing = Standing.NOT_EVALUATED
                findings.append(
                    _f(
                        K.NOT_EVALUATED,
                        "no_validation",
                        S.BLOCKING,
                        name,
                        "no in-scope concept validation was supplied",
                    )
                )
        elif disagreement:
            standing = Standing.MIXED if unexplained else Standing.ASSUMPTION_SENSITIVE
        elif len(validated) == len(validations) and not any(
            f.code in _UNSUPPORTING for f in findings
        ):
            standing = Standing.SUPPORTED
        else:
            standing = Standing.UNSUPPORTED
        test_entries = tuple(
            TestEntry(e.id, e.spec.protocol, e.outcome.value, tuple(sorted(e.axes.items())))
            for e in sorted(tests, key=lambda e: e.id)
        )
        return ConceptAudit(
            concept=ac,
            label=concept.label,
            label_source=concept.label_source.value,
            standing=standing,
            validations=tuple(
                (v.record.id, v.record.semantic_status.value, v.record.unmet)
                for v in sorted(validations, key=lambda v: v.record.id)
            ),
            tests=test_entries,
            findings=tuple(_dedupe(findings)),
            false_positives=tuple(sorted(fps)),
            false_negatives=tuple(sorted(fns)),
            limitations=self._limitations(tests),
        )

    def _validation_disagreement(
        self, validations: list[ValidationEntry], name: str
    ) -> tuple[list[Finding], bool]:
        rows = []
        for v in validations:
            enc = [e for e in self.ev.results if e.trace is v.encoding]
            use = [e for e in self.ev.results if any(e.trace is t for t in v.uses)]
            axes = {
                "dataset": v.record.dataset.record_id,
                "null": "+".join(sorted({e.axes["null"] for e in enc})),
                "replacement": "+".join(sorted({e.axes["replacement"] for e in use})),
            }
            outcome = (
                Outcome.SUPPORTS
                if v.record.semantic_status is SemanticStatus.VALIDATED_CONCEPT
                else Outcome.CONTRADICTS
            )
            rows.append((axes, outcome, v.record.id))
        sup = [r for r in rows if r[1] is Outcome.SUPPORTS]
        con = [r for r in rows if r[1] is Outcome.CONTRADICTS]
        return _disagreement(
            sup, con, name, (), suffix="_sensitive", sides=("validated", "proposed")
        )

    def _counterexamples(self, v: ValidationEntry) -> tuple[list[str], list[str]]:
        found = self.ev.index.get(v.record.counterexamples)
        if found is None or not isinstance(found[0], ProtocolResult):
            return [], []
        m = found[0].measurements

        def ids(key: str) -> list[str]:
            rows = m.get(key) or ()
            return [str(r[0]) if isinstance(r, tuple) else str(r) for r in rows]  # type: ignore[union-attr]

        return ids("false_positives"), ids("false_negatives")

    def _other_concepts(self, feature: str, concept: str) -> set[str]:
        out = set()
        for e in self.ev.results:
            if (
                e.usable
                and e.claim.relation is Relation.ENCODES
                and e.claim.subject.feature == feature
                and e.concept not in (None, concept)
                and e.outcome is Outcome.SUPPORTS
            ):
                out.add(str(e.concept))
        return out

    def _sae_limitations(self, feature: str) -> list[str]:
        codes = set()
        for t in self.ev.traces:
            for lim in t.limitations:
                if lim.code.startswith("SAE_") and (
                    not lim.applies_to or feature in lim.applies_to
                ):
                    codes.add(lim.code)
        return sorted(codes)

    # ------------------------------------------------------------------ report-level

    def _report_findings(self) -> tuple[Finding, ...]:
        out: list[Finding] = []
        for position, problem in self.ev.failed_traces:
            out.append(
                Finding(
                    kind=FindingKind(problem.kind),
                    code=problem.code,
                    severity=S.BLOCKING,
                    subject="report",
                    detail=f"trace #{position} excluded: {problem.detail}",
                )
            )
        groups: dict[tuple[str, str], list[ResultEntry]] = {}
        for e in self.ev.results:
            found = e.problem or e.exclusion
            if found is not None:
                groups.setdefault((found.kind, found.code), []).append(e)
        for (kind, code), entries in sorted(groups.items()):
            out.append(
                Finding(
                    kind=FindingKind(kind),
                    code=code,
                    severity=S.BLOCKING if kind == "integrity_failure" else S.QUALIFYING,
                    subject="report",
                    detail=f"{len(entries)} result(s) excluded: "
                    + (entries[0].problem or entries[0].exclusion).detail  # type: ignore[union-attr]
                    + (" (first shown)" if len(entries) > 1 else ""),
                    records=tuple(sorted(e.id for e in entries)),
                    samples=tuple(sorted({e.sample for e in entries if e.sample})),
                )
            )
        vgroups: dict[tuple[str, str], list[ValidationEntry]] = {}
        for v in self.ev.validations:
            found = v.problem or v.exclusion
            if found is not None:
                vgroups.setdefault((found.kind, found.code), []).append(v)
        for (kind, code), items in sorted(vgroups.items()):
            out.append(
                Finding(
                    kind=FindingKind(kind),
                    code=code,
                    severity=S.BLOCKING if kind == "integrity_failure" else S.QUALIFYING,
                    subject="report",
                    detail=f"{len(items)} concept validation(s) excluded",
                    records=tuple(sorted(v.record.id for v in items)),
                )
            )
        return tuple(out)

    def _inventory(self) -> tuple[InventoryEntry, ...]:
        by_claim: dict[str, list[ResultEntry]] = {}
        for e in self.ev.results:
            by_claim.setdefault(e.claim.id, []).append(e)
        assessments: dict[str, list[Assessment]] = {}
        claims: dict[str, Claim] = {}
        for record, _ in self.ev.index.values():
            if isinstance(record, Assessment):
                assessments.setdefault(record.claim.claim_id, []).append(record)
            elif isinstance(record, Claim):
                claims[record.id] = record
        out = []
        for cid in sorted(claims):
            claim = claims[cid]
            s = claim.subject
            subject = s.site.module or s.site.output_path or "input"
            if s.units is not None:
                subject += f" units {list(s.units)}"
            if s.unit_axes is not None:
                subject += f" axes {list(s.unit_axes)}"
            if s.feature is not None:
                subject += f" feature {s.feature}"
            out.append(
                InventoryEntry(
                    claim=cid,
                    relation=claim.relation.value,
                    subject=subject,
                    target=f"{claim.target.metric} {_key(claim.target.params)}",
                    scope=claim.estimand.scope.value,
                    sample=claim.estimand.sample_id,
                    results=tuple(
                        sorted(
                            (
                                e.id,
                                e.spec.protocol,
                                e.outcome.value,
                                "included" if e.usable else (e.problem or e.exclusion).code,  # type: ignore[union-attr]
                            )
                            for e in by_claim.get(cid, [])
                        )
                    ),
                    assessments=tuple(
                        sorted(
                            (f"{a.policy.name}/v{a.policy.version}", a.verdict.value)
                            for a in assessments.get(cid, [])
                        )
                    ),
                    matched=tuple(sorted(self.matched_claims.get(cid, ()))),
                )
            )
        return tuple(out)

    def _coverage(self, claims: Sequence[ClaimAudit], concepts: Sequence[ConceptAudit]) -> Coverage:
        usable = [e for e in self.ev.results if e.usable]
        statuses: set[str] = set()
        for t in self.ev.traces:
            for r in t.records:
                if r.status is not None:
                    statuses.add(r.status.value)
        run = sorted({e.spec.protocol for e in usable})
        required = sorted(
            {
                p
                for c in self.plan.claims
                for p in self.requirements[c.requirement].policy.protocols_for(c.relation)
            }
        )
        rule = self.plan.counterexamples
        return Coverage(
            claims=len(claims),
            claims_by_standing=_count(c.standing.value for c in claims if c.standing is not None),
            per_sample_claims=tuple(
                (
                    c.name,
                    len(c.groups),
                    sum(1 for g in c.groups if g.standing is not Standing.NOT_EVALUATED),
                )
                for c in claims
                if c.per_sample
            ),
            concepts=len(concepts),
            concepts_evaluated=sum(1 for k in concepts if k.standing is not Standing.NOT_EVALUATED),
            plan_samples=len(self.samples),
            samples_with_evidence=len({e.sample for e in usable if e.sample in self.samples}),
            evidence_statuses=tuple(sorted(statuses)),
            protocols_run=tuple(run),
            protocols_required_not_run=tuple(p for p in required if p not in run),
            untested_invariances=tuple(sorted(set(self.untested))),
            counterexample_caps=(
                ("max_counterexample_fraction", _cap_text(rule.max_counterexample_fraction)),
                ("max_false_positive_rate", _cap_text(rule.max_false_positive_rate)),
                ("max_false_negative_rate", _cap_text(rule.max_false_negative_rate)),
            ),
        )

    def _limitations(self, entries: Iterable[ResultEntry]) -> tuple[tuple[str, int], ...]:
        codes: list[str] = []
        seen: set[str] = set()
        for e in entries:
            about = {e.id, *(ev.record_id for ev in e.result.evidence)}
            for lim in e.trace.limitations:
                if lim.id in seen:
                    continue
                if not lim.applies_to or about & set(lim.applies_to):
                    seen.add(lim.id)
                    codes.append(lim.code)
        return _count(codes)


# ------------------------------------------------------------------ rules


def _target_for(c: AuditedClaim, sample: str | None) -> Any:
    """The claim's declared target (on ``sample``, for per-sample targets)."""
    if not c.sample_targets:
        return c.target
    for t in c.sample_targets:
        if t.sample == sample:
            return t.target
    return None


def _standing(
    verdict: Verdict,
    sup: list[Any],
    con: list[Any],
    unexplained: bool,
    findings: Sequence[Finding],
) -> Standing:
    """Plan §22 precedence."""
    if sup and con:
        return Standing.MIXED if unexplained else Standing.ASSUMPTION_SENSITIVE
    if verdict is Verdict.CONTRADICTED:
        return Standing.CONTRADICTED
    if any(f.code in _UNSUPPORTING for f in findings):
        return Standing.UNSUPPORTED
    if verdict is Verdict.SUPPORTED:
        return Standing.SUPPORTED
    if verdict is Verdict.INCONCLUSIVE:
        return Standing.INCONCLUSIVE
    return Standing.NOT_EVALUATED


def _disagreement(
    sup: Sequence[tuple[dict[str, str], Outcome, str]],
    con: Sequence[tuple[dict[str, str], Outcome, str]],
    subject: str,
    samples: tuple[str, ...],
    *,
    suffix: str = "",
    sides: tuple[str, str] = ("supports", "contradicts"),
) -> tuple[list[Finding], bool]:
    """Plan §8: explain SUPPORTS/CONTRADICTS pairs by the axes they differ in."""
    single: dict[str, set[tuple[str, str]]] = {}
    single_records: dict[str, set[str]] = {}
    combined: dict[tuple[str, ...], set[str]] = {}
    unexplained: set[str] = set()
    for sa, _, sid in sup:
        for ca, _, cid in con:
            diff = tuple(a for a in sorted(set(sa) | set(ca)) if sa.get(a) != ca.get(a))
            if not diff:
                unexplained |= {sid, cid}
            elif len(diff) == 1:
                (a,) = diff
                single.setdefault(a, set()).update({(sides[0], sa[a]), (sides[1], ca[a])})
                single_records.setdefault(a, set()).update({sid, cid})
            else:
                combined.setdefault(diff, set()).update({sid, cid})
    findings = []
    for axis in sorted(single):
        findings.append(
            _f(
                K.ASSUMPTION_SENSITIVE,
                f"{axis}{suffix}" if suffix else "assumption_sensitive",
                S.QUALIFYING,
                subject,
                f"the outcome changes with {axis}: "
                + "; ".join(f"{side} under {value}" for side, value in sorted(single[axis])),
                axis=axis,
                values=tuple(sorted(single[axis])),
                records=tuple(sorted(single_records[axis])),
                samples=samples,
            )
        )
    if not single and combined:
        diff = min(combined, key=lambda d: (len(d), d))
        axis = "combined:" + "+".join(diff)
        findings.append(
            _f(
                K.ASSUMPTION_SENSITIVE,
                f"combined{suffix}" if suffix else "assumption_sensitive",
                S.QUALIFYING,
                subject,
                f"supporting and contradicting results differ in several assumptions at once "
                f"({', '.join(diff)}); no single assumption explains the disagreement",
                axis=axis,
                records=tuple(sorted(combined[diff])),
                samples=samples,
            )
        )
    if unexplained:
        findings.append(
            _f(
                K.CONTRADICTION,
                "unexplained_contradiction",
                S.BLOCKING,
                subject,
                "results obtained under identical recorded assumptions disagree; the "
                "contradiction is listed, not resolved",
                records=tuple(sorted(unexplained)),
                samples=samples,
            )
        )
    return findings, bool(unexplained)


def _by_structure(entries: Sequence[ResultEntry]) -> list[list[ResultEntry]]:
    groups: dict[tuple[Any, ...], list[ResultEntry]] = {}
    for e in entries:
        c = e.claim
        key = (c.relation, c.subject, c.target, c.estimand)
        groups.setdefault(key, []).append(e)
    return [groups[k] for k in sorted(groups, key=repr)]


def _problem_finding(
    problem: Problem,
    subject: str,
    e: ResultEntry,
    severity: FindingSeverity,
    samples: tuple[str, ...] = (),
) -> Finding:
    return Finding(
        kind=FindingKind(problem.kind),
        code=problem.code,
        severity=severity,
        subject=subject,
        detail=problem.detail,
        records=(e.id,),
        samples=samples or ((e.sample,) if e.sample else ()),
    )


def _f(
    kind: FindingKind,
    code: str,
    severity: FindingSeverity,
    subject: str,
    detail: str,
    *,
    axis: str | None = None,
    values: tuple[tuple[str, str], ...] = (),
    records: tuple[str, ...] = (),
    samples: tuple[str, ...] = (),
) -> Finding:
    return Finding(kind, code, severity, subject, detail, axis, values, records, samples)


def _with_samples(f: Finding, samples: tuple[str, ...]) -> Finding:
    return Finding(
        f.kind, f.code, f.severity, f.subject, f.detail, f.axis, f.values, f.records, samples
    )


def _dedupe(findings: Iterable[Finding]) -> list[Finding]:
    out: list[Finding] = []
    seen: set[Finding] = set()
    for f in findings:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def _count(items: Iterable[str]) -> tuple[tuple[str, int], ...]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return tuple(sorted(counts.items()))


def _cap_text(cap: float | None) -> str:
    return "not declared" if cap is None else repr(cap)
