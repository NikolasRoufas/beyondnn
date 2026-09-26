"""Phase 7 audits: pre-registered scenarios A-N, evidence mutations, persistence,
re-derivation, refusals, and the WHY integration (docs/PHASE_7_PLAN.md)."""

from __future__ import annotations

import json
import subprocess
import sys
from functools import cache
from pathlib import Path
from typing import Any

import pytest
import torch

import beyondnn as bnn
from beyondnn._testing.audit_scenarios import (
    SCENARIOS,
    TARGET,
    Scenario,
    WeightedSum,
    _comp,
    _comp_req,
    _one_unit,
    _plan,
    _unit_claim,
)
from beyondnn.audits import (
    AuditMismatchError,
    AuditPlanError,
    FindingKind,
    FindingSeverity,
    Standing,
)
from beyondnn.audits.engine import _disagreement
from beyondnn.core.samples import sample_id
from beyondnn.core.trace import TraceResult
from beyondnn.schema import (
    Assessment,
    AuditPlan,
    ClaimTestResult,
    Outcome,
    ProtocolResult,
    SchemaError,
    Site,
    Subject,
    from_json,
    to_json,
)

A, F, AU, C, iv = bnn.attribution, bnn.faithfulness, bnn.audits, bnn.concepts, bnn.interventions


@cache
def scenario(name: str) -> Scenario:
    return SCENARIOS[name]()


def outcome_of(s: Scenario, report: Any) -> tuple[str, set[str]]:
    if s.claim is not None:
        c = report.claim(s.claim)
        standing = c.standing.value if c.standing is not None else _only(c.distribution)
        return standing, {f.code for f in c.findings}
    k = report.concept(s.extra["concept"])
    return k.standing.value, {f.code for f in k.findings}


def _only(distribution: tuple[tuple[str, int], ...]) -> str:
    assert len(distribution) == 1, distribution
    return distribution[0][0]


# ------------------------------------------------------------------ scenarios A-N


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario_matches_preregistration(name: str) -> None:
    s = scenario(name)
    report = bnn.audit(s.evidence, plan=s.plan)
    standing, codes = outcome_of(s, report)
    assert standing == s.standing
    assert set(s.codes) <= codes
    if "axis" in s.extra:
        c = report.claim(s.claim)  # type: ignore[arg-type]
        axes = {f.axis for f in c.findings if f.kind is FindingKind.ASSUMPTION_SENSITIVE}
        assert axes == {s.extra["axis"]}


def test_scenario_secondary_expectations() -> None:
    a = scenario("A")
    assert bnn.audit(a.evidence, plan=a.plan).claim("a_site_necessary").distribution == (
        ("supported", 1),
    )
    d = scenario("D")
    report = bnn.audit(d.evidence, plan=d.plan)
    suff = report.claim("d_sufficient")
    assert suff.distribution == (("contradicted", 1),)
    assert "necessary_not_sufficient" in {f.code for f in suff.findings}
    g = scenario("G")
    report = bnn.audit(g.evidence, plan=g.plan)
    assert report.claim("g_use").standing is Standing.CONTRADICTED
    i = scenario("I")
    assert i.extra["status"] == "validated_concept"  # validated by the policy, capped by the plan
    k = bnn.audit(i.evidence, plan=i.plan).concept(i.extra["concept"])
    assert len(k.false_positives) > 0
    heavy = [f for f in k.findings if f.code == "counterexample_heavy"]
    assert heavy
    assert heavy[0].severity is FindingSeverity.BLOCKING
    assert set(heavy[0].samples) >= set(k.false_positives)


def test_missing_evidence_is_not_evaluated_not_contradicted() -> None:
    n = scenario("N")
    c = bnn.audit(n.evidence, plan=n.plan).claim("n_necessary")
    assert c.distribution == (("not_evaluated", 1),)
    assert c.counterexamples == ()
    assert not any(f.kind is FindingKind.COUNTEREXAMPLE_FOUND for f in c.findings)


# ------------------------------------------------------------------ forgery / mutations


def forge(trace: TraceResult, outcome: Outcome) -> TraceResult:
    """A copy of ``trace`` whose one claim-test result claims ``outcome`` (ids consistent:
    the forgery passes integrity checks and can only be caught by re-derivation)."""
    old = next(r for r in trace.records if isinstance(r, ClaimTestResult))
    forged = ClaimTestResult(
        claim=old.claim,
        spec=old.spec,
        outcome=outcome,
        evidence=old.evidence,
        statistics=old.statistics,
        provenance_id=old.provenance_id,
        derived_from=old.derived_from,
    )
    out = TraceResult(trace.config)
    for record in trace.records:
        if record.id == old.id:
            out._add(forged)
        elif not (
            isinstance(record, Assessment)
            or old.id in getattr(record, "applies_to", ())
            or any(d.record_id == old.id for d in record.derived_from)
        ):
            out._add(record)
    for key, tensor in trace._tensors.items():
        out._add_tensor(key, tensor)
    out._seal()
    return out


def test_forged_outcome_is_an_integrity_failure_and_excluded() -> None:
    a = scenario("A")
    forged = forge(a.evidence[0].trace, Outcome.CONTRADICTS)
    report = bnn.audit([forged, *a.evidence[1:]], plan=a.plan)
    c = report.claim("a_unit_necessary")
    kinds = {(f.kind, f.severity) for f in c.findings}
    assert (FindingKind.INTEGRITY_FAILURE, FindingSeverity.BLOCKING) in kinds
    assert c.distribution == (("unsupported", 1),)  # 2 replacements left < 3 declared
    assert report.evidence.results_excluded == 1
    assert any(f.code == "rederivation_failed" for f in report.findings)


def test_inconclusive_relabelled_as_supports_is_caught() -> None:
    model, x = _one_unit()
    same = torch.ones(1, 32)
    same[0, 0] = 2.0  # replacement equal to the value: a no-op, INCONCLUSIVE
    res = _comp(model, x, F.replacement(same))
    assert res.outcome is Outcome.INCONCLUSIVE
    plan = _plan(
        model,
        [sample_id(x)],
        [_unit_claim("c", "necessary_for", "comprehensiveness")],
        [_comp_req(False)],
    )
    honest = bnn.audit([res], plan=plan).claim("c")
    assert honest.distribution == (("inconclusive", 1),)
    assert "inconclusive_results" in {f.code for f in honest.findings}
    forged = bnn.audit([forge(res.trace, Outcome.SUPPORTS)], plan=plan).claim("c")
    assert forged.distribution == (("not_evaluated", 1),)
    assert FindingKind.INTEGRITY_FAILURE in {f.kind for f in forged.findings}


def test_dropping_the_contradicting_result_cannot_produce_supported() -> None:
    e = scenario("E")
    inv = _unit_claim(
        "e_inv",
        "necessary_for",
        "comprehensiveness",
        invariant_over=[AU.invariance("replacement", min_values=2)],
    )
    model, x = _one_unit()
    plan = _plan(model, [sample_id(x)], [inv], [_comp_req(False)])
    full = bnn.audit(e.evidence, plan=plan).claim("e_inv")
    assert full.distribution == (("assumption_sensitive", 1),)
    favourable = bnn.audit(e.evidence[:1], plan=plan).claim("e_inv")
    assert favourable.distribution == (("unsupported", 1),)
    assert "invariance_untested" in {f.code for f in favourable.findings}


def test_changed_target_or_subject_does_not_match_evidence() -> None:
    a = scenario("A")
    model, x = _one_unit()
    other_target = AU.claim(
        "t",
        statement="unit 0 necessary for another output",
        relation="necessary_for",
        target=iv.metrics.select([0, 1]),
        scope="instance",
        requirement="comprehensiveness",
        subject=Subject(site=Site(module="hidden"), units=(0,)),
    )
    other_unit = _unit_claim("u", "necessary_for", "comprehensiveness")
    other_unit = AU.claim(
        "u",
        statement=other_unit.statement,
        relation="necessary_for",
        target=TARGET,
        scope="instance",
        requirement="comprehensiveness",
        subject=Subject(site=Site(module="hidden"), units=(1,)),
    )
    plan = _plan(model, [sample_id(x)], [other_target, other_unit], [_comp_req(True)])
    report = bnn.audit(a.evidence, plan=plan)
    assert report.claim("t").distribution == (("not_evaluated", 1),)
    assert report.claim("u").distribution == (("not_evaluated", 1),)


def test_dataset_scope_mutation_excludes_concept_evidence() -> None:
    i = scenario("I")
    plan = i.plan
    narrowed = AuditPlan(
        name=plan.name,
        checkpoint=plan.checkpoint,
        declared_model=None,
        samples=(),
        datasets=(),
        claims=(),
        requirements=(),
        concepts=plan.concepts,
        counterexamples=plan.counterexamples,
    )
    k = bnn.audit(i.evidence, plan=narrowed).concept(i.extra["concept"])
    assert k.standing is Standing.NOT_EVALUATED
    assert "dataset_out_of_scope" in {f.code for f in k.findings}


def test_removed_counterexamples_are_caught() -> None:
    i = scenario("I")
    validation = i.evidence[0]
    enc = validation.encoding.trace
    ce = next(r for r in enc.records if isinstance(r, ProtocolResult))
    edited = ProtocolResult(
        protocol=ce.protocol,
        protocol_version=ce.protocol_version,
        params=ce.params,
        criteria=ce.criteria,
        measurements=type(ce.measurements)(
            ce.measurements.to_plain() | {"false_positives": [], "false_positive_rate": 0.0}
        ),
        outcomes=ce.outcomes,
        samples=ce.samples,
        target=ce.target,
        provenance_id=ce.provenance_id,
        derived_from=ce.derived_from,
    )
    out = TraceResult(enc.config)
    for record in enc.records:
        out._add(edited if record.id == ce.id else record)
    for key, tensor in enc._tensors.items():
        out._add_tensor(key, tensor)
    out._seal()
    evidence = [validation.trace, out, *(u.trace for u in validation.use), validation.concept]
    report = bnn.audit(evidence, plan=i.plan)
    k = report.concept(i.extra["concept"])
    assert FindingKind.INTEGRITY_FAILURE in {f.kind for f in k.findings}
    assert k.standing is not Standing.SUPPORTED


def test_threshold_sensitivity_by_declared_alternative_criteria() -> None:
    model, x = _one_unit()
    res = _comp(model, x, F.zero(), min_drop=5.0)  # drop 10
    req = AU.requirement(
        "comprehensiveness",
        policy=F.COMPREHENSIVENESS_POLICY,
        controls=False,
        alternatives=[AU.alternative("comprehensiveness", "min_drop", factor=3.0)],
    )
    plan = _plan(
        model, [sample_id(x)], [_unit_claim("c", "necessary_for", "comprehensiveness")], [req]
    )
    c = bnn.audit([res], plan=plan).claim("c")
    assert c.distribution == (("assumption_sensitive", 1),)
    (finding,) = [f for f in c.findings if f.kind is FindingKind.ASSUMPTION_SENSITIVE]
    assert finding.axis == "threshold"
    tests = c.groups[0].tests
    assert {t.alternative for t in tests} == {None, "min_dropx3.0"}
    # the verdict itself rests only on the recorded result
    assert c.groups[0].verdict == "supported"


def test_concept_alternatives_for_encoding_are_refused() -> None:
    model, x = _one_unit()
    req = AU.requirement(
        "enc",
        policy=C.ENCODING_POLICY,
        controls=True,
        alternatives=[AU.alternative("concept_encoding", "min_fraction_below", value=0.9)],
    )
    plan = _plan(
        model,
        [sample_id(x)],
        [_unit_claim("c", "necessary_for", "comprehensiveness")],
        [_comp_req(False), req],
    )
    with pytest.raises(AuditPlanError, match="concept_encoding"):
        bnn.audit([], plan=plan)


# ------------------------------------------------------------------ selection claims


@cache
def _selection_evidence() -> tuple[Any, list[Any], Any]:
    model, x = _one_unit()
    ig = A.integrated_gradients(baseline=A.zero_baseline(), n_steps=8)
    attr = A.attribute(model, x, target=TARGET, method=ig, at=A.layer("hidden"))
    out = []
    for k in (1, 2, 4):
        test = F.comprehensiveness(
            target=TARGET,
            min_drop=5.0,
            statement=f"ig top-{k}",
            controls=F.controls(20, seed=1),
            replacement=F.zero(),
            min_fraction_below=0.9,
        )
        out.append(F.run(model, x, test=test, selection=F.top_k(attr, k=k), attributions=[attr]))
    return attr, out, x


def _selection_plan(k: int | None, invariance: bool) -> AuditPlan:
    model, x = _one_unit()
    claim = AU.claim(
        "ig_necessary",
        statement="the IG top-k units are necessary",
        relation="necessary_for",
        target=TARGET,
        scope="instance",
        requirement="comprehensiveness",
        selection=AU.selection("hidden", method="integrated_gradients", k=k),
        invariant_over=[AU.invariance("k", min_values=3)] if invariance else [],
    )
    plan: AuditPlan = _plan(model, [sample_id(x)], [claim], [_comp_req(True)])
    return plan


def test_selection_claim_k_sensitivity_and_favourable_k() -> None:
    _, results, _ = _selection_evidence()
    full = bnn.audit(results, plan=_selection_plan(None, True)).claim("ig_necessary")
    assert full.distribution == (("assumption_sensitive", 1),)
    assert {f.axis for f in full.findings if f.kind is FindingKind.ASSUMPTION_SENSITIVE} == {"k"}
    favourable = bnn.audit(results[:1], plan=_selection_plan(None, True)).claim("ig_necessary")
    assert favourable.distribution == (("unsupported", 1),)
    fixed = bnn.audit(results, plan=_selection_plan(1, False)).claim("ig_necessary")
    assert fixed.distribution == (("supported", 1),)


def test_selection_method_must_be_verifiable() -> None:
    _, results, _ = _selection_evidence()
    report = bnn.audit([r.trace for r in results], plan=_selection_plan(None, False))
    c = report.claim("ig_necessary")
    assert c.distribution == (("not_evaluated", 1),)
    assert "selection_source_not_supplied" in {f.code for f in c.findings}
    assert "no_evidence" not in {f.code for f in c.findings}


def test_attribution_only_selection_claim_is_unsupported() -> None:
    attr, _, _ = _selection_evidence()
    c = bnn.audit([attr], plan=_selection_plan(None, False)).claim("ig_necessary")
    assert c.distribution == (("unsupported", 1),)
    assert "attribution_is_not_intervention" in {f.code for f in c.findings}


# ------------------------------------------------------------------ aggregation, scope


def _per_sample_evidence() -> tuple[Any, list[Any], list[str]]:
    model = WeightedSum([5.0] + [0.01] * 31).eval()
    xs: list[torch.Tensor] = []
    for value in (2.0, 2.0, 0.5, 0.2):  # drops 10, 10, 2.5, 1 against min_drop 5
        x = torch.ones(1, 32)
        x[0, 0] = value
        x[0, 1] = len(xs)  # distinct samples
        xs.append(x)
    return model, [_comp(model, x, F.zero()) for x in xs], [sample_id(x) for x in xs]


@pytest.mark.parametrize("cap", [None, 0.25, 0.75])
def test_per_sample_distribution_and_counterexample_cap(cap: float | None) -> None:
    model, results, samples = _per_sample_evidence()
    extra = sample_id(torch.zeros(1, 32))
    plan = _plan(
        model,
        [*samples, extra],
        [_unit_claim("c", "necessary_for", "comprehensiveness")],
        [_comp_req(False)],
        counterexamples=AU.counterexample_rule(
            max_counterexample_fraction=cap,
            max_false_positive_rate=None,
            max_false_negative_rate=None,
        ),
    )
    c = bnn.audit(results, plan=plan).claim("c")
    assert c.standing is None  # no claim-level truth value from a distribution
    assert dict(c.distribution) == {"supported": 2, "contradicted": 2, "not_evaluated": 1}
    assert set(c.counterexamples) == set(samples[2:])
    (ce,) = [f for f in c.findings if f.kind is FindingKind.COUNTEREXAMPLE_FOUND]
    if cap is not None and cap < 0.5:
        assert (ce.code, ce.severity) == ("counterexample_cap_exceeded", FindingSeverity.BLOCKING)
    else:
        assert (ce.code, ce.severity) == ("counterexamples_present", FindingSeverity.QUALIFYING)
    assert "2 of 4 evaluated samples" in ce.detail


def test_finite_sample_and_population_claims_from_instance_evidence() -> None:
    model, results, samples = _per_sample_evidence()
    from beyondnn.concepts._core import sample_set_id

    common: dict[str, Any] = {
        "statement": "unit 0 is necessary on these samples",
        "relation": "necessary_for",
        "target": TARGET,
        "requirement": "comprehensiveness",
        "subject": Subject(site=Site(module="hidden"), units=(0,)),
    }
    finite = AU.claim("finite", scope="finite_sample", sample_set=sample_set_id(samples), **common)
    population = AU.claim("population", scope="population", population="all inputs", **common)
    plan = _plan(model, samples, [finite, population], [_comp_req(False)])
    report = bnn.audit(results, plan=plan)
    for name in ("finite", "population"):
        c = report.claim(name)
        assert c.standing is Standing.UNSUPPORTED
        (f,) = [f for f in c.findings if f.code == "narrower_estimand"]
        assert f.severity is FindingSeverity.BLOCKING
        assert "supports 2" in f.detail


def test_unexplained_disagreement_is_mixed_not_resolved() -> None:
    axes = {"protocol": "p", "threshold": "t", "replacement": "zero"}
    findings, unexplained = _disagreement(
        [(axes, Outcome.SUPPORTS, "r1")], [(dict(axes), Outcome.CONTRADICTS, "r2")], "c", ()
    )
    assert unexplained
    assert findings[-1].kind is FindingKind.CONTRADICTION
    assert set(findings[-1].records) == {"r1", "r2"}
    other = dict(axes) | {"replacement": "mean", "k": "3"}
    findings, unexplained = _disagreement(
        [(axes | {"k": "1"}, Outcome.SUPPORTS, "r1")], [(other, Outcome.CONTRADICTS, "r2")], "c", ()
    )
    assert not unexplained
    assert findings[0].axis == "combined:k+replacement"


# ------------------------------------------------------------------ plan


def test_plan_is_a_deterministic_serialisable_record() -> None:
    a = scenario("A")
    text = to_json(a.plan)
    back = from_json(text)
    assert back == a.plan
    assert back.id == a.plan.id
    assert to_json(back) == text
    assert a.plan.id.startswith("audit_plan:")


def test_plan_validation() -> None:
    model, x = _one_unit()
    claim = _unit_claim("c", "necessary_for", "comprehensiveness")
    with pytest.raises(SchemaError, match="unknown requirement"):
        _plan(model, [sample_id(x)], [claim], [])
    with pytest.raises(SchemaError, match="duplicate claim"):
        _plan(model, [sample_id(x)], [claim, claim], [_comp_req(False)])
    with pytest.raises(SchemaError, match="names no protocol"):
        _plan(
            model,
            [sample_id(x)],
            [claim],
            [AU.requirement("comprehensiveness", policy=F.SUFFICIENCY_POLICY, controls=False)],
        )
    with pytest.raises(SchemaError, match="at least two"):
        AU.invariance("k", min_values=1)
    with pytest.raises(SchemaError, match="exactly one of subject"):
        AU.claim(
            "c",
            statement="s",
            relation="necessary_for",
            target=TARGET,
            scope="instance",
            requirement="r",
        )
    with pytest.raises(SchemaError, match="sample_set"):
        AU.claim(
            "c",
            statement="s",
            relation="necessary_for",
            target=TARGET,
            scope="finite_sample",
            requirement="r",
            subject=Subject(site=Site(module="hidden")),
        )


def test_refusals() -> None:
    a = scenario("A")
    with pytest.raises(bnn.explain.bundle.ModelMismatchError):
        bnn.audit(a.evidence, plan=a.plan, model=WeightedSum([1.0] * 32))
    model, _ = _one_unit()
    assert bnn.audit(a.evidence, plan=a.plan, model=model).claim("a_unit_necessary")
    with pytest.raises(TypeError):
        bnn.audit(a.evidence, plan="plan")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        bnn.audit("path", plan=a.plan)
    with pytest.raises(TypeError, match="cannot audit"):
        bnn.audit([object()], plan=a.plan)
    bad = AU.requirement(
        "comprehensiveness",
        policy=bnn.schema.AssessmentPolicy(
            name="bad",
            version=1,
            requirements=(
                bnn.schema.PolicyRequirement(
                    relation=bnn.Relation.NECESSARY_FOR, protocols=("attribution_threshold",)
                ),
            ),
        ),
        controls=False,
    )
    plan = _plan(
        model, a.plan.samples, [_unit_claim("c", "necessary_for", "comprehensiveness")], [bad]
    )
    with pytest.raises(AuditPlanError):
        bnn.audit([], plan=plan)


# ------------------------------------------------------------------ persistence


def test_save_load_audit_equals_in_memory(tmp_path: Path) -> None:
    e, l_ = scenario("E"), scenario("L")
    live = bnn.audit(e.evidence, plan=e.plan).to_dict()
    paths = []
    for i, r in enumerate(e.evidence):
        path = tmp_path / f"e{i}"
        r.trace.save(path)
        paths.append(path)
    assert bnn.audit(paths, plan=e.plan).to_dict() == live
    concept_live = bnn.audit(l_.evidence, plan=l_.plan).to_dict()
    from beyondnn.audits.evidence import collect_traces

    concept_paths = []
    for i, t in enumerate(collect_traces(l_.evidence)):
        path = tmp_path / f"l{i}"
        t.save(path)
        concept_paths.append(path)
    assert bnn.audit(concept_paths, plan=l_.plan).to_dict() == concept_live


def test_restart_load_audit_in_a_fresh_process(tmp_path: Path) -> None:
    f = scenario("F")
    live = bnn.audit(f.evidence, plan=f.plan)
    for i, r in enumerate(f.evidence):
        r.trace.save(tmp_path / f"t{i}")
    (tmp_path / "plan.json").write_text(to_json(f.plan))
    live.save(tmp_path / "report.json")
    code = (
        "import sys, json, pathlib, beyondnn as bnn\n"
        "from beyondnn.schema import from_json\n"
        "d = pathlib.Path(sys.argv[1])\n"
        "plan = from_json((d / 'plan.json').read_text())\n"
        "paths = sorted(p for p in d.iterdir() if p.is_dir())\n"
        "doc = bnn.audits.load_report(d / 'report.json')\n"
        "bnn.audits.verify_report(doc, paths, plan)\n"
        "print(bnn.audit(paths, plan=plan).to_json())\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, check=True
    )
    assert json.loads(out.stdout) == json.loads(live.to_json())


def test_report_save_load_verify_and_edits(tmp_path: Path) -> None:
    c = scenario("C")
    report = bnn.audit(c.evidence, plan=c.plan)
    path = tmp_path / "r.json"
    report.save(path)
    with pytest.raises(FileExistsError):
        report.save(path)
    doc = AU.load_report(path)
    assert AU.verify_report(doc, c.evidence, c.plan).to_dict() == report.to_dict()
    edited = json.loads(json.dumps(doc))
    edited["claims"][0]["groups"][0]["standing"] = "supported"
    with pytest.raises(AuditMismatchError, match="standing"):
        AU.verify_report(edited, c.evidence, c.plan)
    dropped = json.loads(json.dumps(doc))
    dropped["claims"][0]["findings"] = []
    with pytest.raises(AuditMismatchError):
        AU.verify_report(dropped, c.evidence, c.plan)
    with pytest.raises(AuditMismatchError):
        AU.verify_report(doc, c.evidence[:1], c.plan)  # the evidence changed


def test_report_is_deterministic_and_never_says_trustworthy() -> None:
    for name in ("A", "C", "G", "I"):
        s = scenario(name)
        one, two = bnn.audit(s.evidence, plan=s.plan), bnn.audit(s.evidence, plan=s.plan)
        assert one.to_json() == two.to_json()
        text = one.render().lower()
        for word in ("trustworth", "reliable", "is correct", "faithful explanation", "score"):
            if word == "score":
                assert "not a score" in text
                continue
            assert word not in text


def test_report_sections() -> None:
    c = scenario("C")
    report = bnn.audit(c.evidence, plan=c.plan)
    assert report.protocol_disagreements
    assert report.supported == ()
    assert report.per_sample
    b = scenario("B")
    report = bnn.audit(b.evidence, plan=b.plan)
    assert [f.code for f in report.overclaims] == ["attribution_is_not_intervention"]
    e = scenario("E")
    assert bnn.audit(e.evidence, plan=e.plan).sensitivity("replacement")
    g = scenario("G")
    report = bnn.audit(g.evidence, plan=g.plan)
    assert report.contradicted
    assert report.concept_findings
    assert report.coverage.concepts_evaluated == 1


# ------------------------------------------------------------------ WHY


def test_why_audit_view_and_refusals() -> None:
    a = scenario("A")
    report = bnn.audit(a.evidence, plan=a.plan)
    model, x = _one_unit()
    response = bnn.compose(bnn.trace(model, x, sites=["hidden"]), audit=report)
    view = response.why.audit
    assert view is not None
    assert view.sample == sample_id(x)
    assert {c.name: g.standing for c, g in view.claims} == {
        "a_site_necessary": Standing.SUPPORTED,
        "a_unit_necessary": Standing.SUPPORTED,
    }
    text = response.render()
    assert "AUDIT  [plan scenario_a" in text
    assert "a_unit_necessary" in text
    assert response.to_dict()["audit"]["claims"][0]["standing"] == "supported"
    assert "audit" not in bnn.compose(bnn.trace(model, x)).to_dict()
    other = x.clone()
    other[0, 3] = 7.0
    with pytest.raises(bnn.explain.bundle.SampleMismatchError):
        bnn.compose(bnn.trace(model, other), audit=report)
    with pytest.raises(bnn.explain.bundle.ModelMismatchError):
        bnn.compose(bnn.trace(WeightedSum([1.0] * 32), x), audit=report)
    with pytest.raises(TypeError):
        bnn.compose(bnn.trace(model, x), audit=report.to_dict())


# ------------------------------------------------------------------ per-sample targets


class _TwoOut(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from beyondnn._testing.audit_scenarios import _Pass

        self.hidden = _Pass()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hidden(x)
        return torch.stack([5.0 * h[:, 0], 5.0 * h[:, 1]], dim=1)


def test_per_sample_targets_match_each_samples_own_target() -> None:
    model = _TwoOut().eval()
    xs = [torch.full((1, 8), 2.0), torch.full((1, 8), 3.0)]
    targets = [iv.metrics.select([0, 0]), iv.metrics.select([0, 1])]  # e.g. per-sample margins
    results = []
    for x, t in zip(xs, targets, strict=True):
        test = F.comprehensiveness(
            target=t, min_drop=5.0, statement="unit 0 necessary", replacement=F.zero()
        )
        results.append(
            F.run(model, x, test=test, selection=F.units(A.layer("hidden"), (0,), n_units=8))
        )
    samples = [sample_id(x) for x in xs]
    subject = Subject(site=Site(module="hidden"), units=(0,))
    common: dict[str, Any] = {
        "statement": "unit 0 is necessary for each sample's own target",
        "relation": "necessary_for",
        "scope": "instance",
        "requirement": "comprehensiveness",
        "subject": subject,
    }
    per_sample = AU.claim(
        "own", target=None, sample_targets=dict(zip(samples, targets, strict=True)), **common
    )
    single = AU.claim("single", target=targets[0], **common)
    plan = _plan(model, samples, [per_sample, single], [_comp_req(False)])
    report = bnn.audit(results, plan=plan)
    own = report.claim("own")
    assert own.group(samples[0]).standing is Standing.SUPPORTED
    assert own.group(samples[1]).standing is Standing.CONTRADICTED  # unit 0 does not feed y1
    one = report.claim("single")
    assert one.group(samples[0]).standing is Standing.SUPPORTED
    assert one.group(samples[1]).standing is Standing.NOT_EVALUATED  # another target
    with pytest.raises(SchemaError, match="exactly the plan samples"):
        _plan(model, samples[:1], [per_sample], [_comp_req(False)])
    with pytest.raises(SchemaError, match="exactly one of target"):
        AU.claim("x", target=targets[0], sample_targets={samples[0]: targets[0]}, **common)


def test_flagship_example() -> None:
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "examples" / "phase7_audit.py"
    spec = importlib.util.spec_from_file_location("phase7_audit", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report, attribution_only, why = module.build()
    assert report.claim("gradient_top1_necessary").distribution == (("contradicted", 1),)
    ig = report.claim("integrated_gradients_top1_necessary")
    assert ig.distribution == (("assumption_sensitive", 1),)
    assert {f.axis for f in ig.findings if f.kind is FindingKind.ASSUMPTION_SENSITIVE} == {
        "replacement"
    }
    assert {c.distribution for c in attribution_only.claims} == {(("unsupported", 1),)}
    assert "AUDIT  [plan saturated_top1" in why


# ------------------------------------------------------------------ mutation-driven tests


def test_disagreement_precedes_contradiction() -> None:
    """A recorded CONTRADICTS that a declared alternative threshold would reverse is
    ASSUMPTION_SENSITIVE, not CONTRADICTED (plan §22 precedence)."""
    model, x = _one_unit()
    res = _comp(model, x, F.zero(), min_drop=15.0)  # drop 10 < 15
    req = AU.requirement(
        "comprehensiveness",
        policy=F.COMPREHENSIVENESS_POLICY,
        controls=False,
        alternatives=[AU.alternative("comprehensiveness", "min_drop", factor=0.5)],
    )
    plan = _plan(
        model, [sample_id(x)], [_unit_claim("c", "necessary_for", "comprehensiveness")], [req]
    )
    group = bnn.audit([res], plan=plan).claim("c").groups[0]
    assert group.verdict == "contradicted"
    assert group.standing is Standing.ASSUMPTION_SENSITIVE


def test_unexplained_disagreement_sets_mixed() -> None:
    from beyondnn.audits.engine import _standing
    from beyondnn.schema import Verdict

    one = [({}, Outcome.SUPPORTS, "a")]
    two = [({}, Outcome.CONTRADICTS, "b")]
    assert _standing(Verdict.MIXED, one, two, True, ()) is Standing.MIXED
    assert _standing(Verdict.MIXED, one, two, False, ()) is Standing.ASSUMPTION_SENSITIVE


def test_concept_test_outside_the_declared_datasets_is_excluded() -> None:
    s = scenario("L")
    plan = s.plan
    narrowed = AuditPlan(
        name=plan.name,
        checkpoint=plan.checkpoint,
        declared_model=None,
        samples=(),
        datasets=(),
        claims=plan.claims,
        requirements=plan.requirements,
        concepts=(),
        counterexamples=plan.counterexamples,
    )
    c = bnn.audit(s.evidence, plan=narrowed).claim("l_encodes")
    assert c.standing is Standing.NOT_EVALUATED
    assert "dataset_out_of_scope" in {f.code for f in c.findings}


def test_forged_concept_validation_is_caught() -> None:
    """A validation record whose use summary and status were changed consistently (so the
    record itself constructs) is caught by re-deriving the summaries from the tests."""
    import dataclasses

    from beyondnn.schema import (
        ConceptValidation,
        SemanticStatus,
        TraceLimitation,
        Verdict,
        derive_semantic_status,
    )

    g = scenario("G")
    validation = g.evidence[0]
    record = validation.record
    use = tuple(dataclasses.replace(u, verdict=Verdict.SUPPORTED) for u in record.use)
    status, unmet = derive_semantic_status(
        record.policy, record.encoding, use, record.false_positive_rate, record.false_negative_rate
    )
    assert status is SemanticStatus.VALIDATED_CONCEPT
    forged = ConceptValidation(
        concept=record.concept,
        feature=record.feature,
        dataset=record.dataset,
        policy=record.policy,
        encoding=record.encoding,
        use=use,
        counterexamples=record.counterexamples,
        false_positive_rate=record.false_positive_rate,
        false_negative_rate=record.false_negative_rate,
        model_state_digest=record.model_state_digest,
        scope=record.scope,
        semantic_status=status,
        unmet=unmet,
    )
    trace = TraceResult(validation.trace.config)
    for r in validation.trace.records:
        if r.id == record.id:
            trace._add(forged)
        elif not isinstance(r, TraceLimitation):
            trace._add(r)
    trace._seal()
    evidence = [
        trace,
        validation.encoding,
        *validation.use,
        validation.concept,
    ]
    k = bnn.audit(evidence, plan=g.plan).concept(g.extra["concept"])
    assert k.standing is not Standing.SUPPORTED
    assert FindingKind.INTEGRITY_FAILURE in {f.kind for f in k.findings}


def test_concept_null_sensitivity_across_encoding_tests() -> None:
    s = scenario("L")
    k = bnn.audit(s.evidence, plan=s.plan).concept(s.plan.concepts[0].concept)
    assert k.standing is Standing.ASSUMPTION_SENSITIVE
    assert "null_sensitive" in {f.code for f in k.findings}
