"""Phase 7.5 audit refinements: configuration roles, sensitivity profiles,
configuration-level disagreement, uncertainty statements, plan migration and
portability, protocol versions, and the save -> restart -> load -> audit -> WHY loop
(ADR-048..050)."""

from __future__ import annotations

import json
import math
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
    Scenario,
    _comp,
    _comp_req,
    _one_unit,
    _plan,
    _unit_claim,
)
from beyondnn.audits import FindingKind, FindingSeverity, Standing
from beyondnn.audits.evidence import collect_traces
from beyondnn.core.samples import sample_id
from beyondnn.core.trace import TraceResult
from beyondnn.schema import (
    AuditPlan,
    ClaimTestResult,
    ClaimTestSpec,
    Outcome,
    SchemaError,
    from_json,
    to_json,
)

A, F, AU, C, iv = bnn.attribution, bnn.faithfulness, bnn.audits, bnn.concepts, bnn.interventions
DATA = Path(__file__).resolve().parent / "data"


@cache
def scenario(name: str) -> Scenario:
    return SCENARIOS[name]()


def _e_plan(roles: list[Any]) -> AuditPlan:
    model, x = _one_unit()
    claim = _unit_claim("e", "necessary_for", "comprehensiveness", roles=roles)
    plan: AuditPlan = _plan(model, [sample_id(x)], [claim], [_comp_req(False)])
    return plan


# ------------------------------------------------------------------ roles


def test_primary_configuration_decides_and_alternative_reversal_is_reported() -> None:
    e = scenario("E")  # zero replacement SUPPORTS; a near-value tensor CONTRADICTS
    plan = _e_plan(
        [
            AU.role("replacement", "zero", "primary"),
            AU.role("replacement", "tensor*", "alternative"),
        ]
    )
    group = bnn.audit(e.evidence, plan=plan).claim("e").groups[0]
    assert group.standing is Standing.SUPPORTED
    (finding,) = [f for f in group.findings if f.code == "alternative_reverses"]
    assert finding.severity is FindingSeverity.QUALIFYING
    assert finding.axis == "replacement"
    assert {t.role for t in group.tests} == {"primary", "alternative"}


def test_stress_test_reversal_is_context_only() -> None:
    e = scenario("E")
    plan = _e_plan(
        [
            AU.role("replacement", "zero", "primary"),
            AU.role("replacement", "tensor*", "stress_test"),
        ]
    )
    group = bnn.audit(e.evidence, plan=plan).claim("e").groups[0]
    assert group.standing is Standing.SUPPORTED
    (finding,) = [f for f in group.findings if f.code == "stress_test_reverses"]
    assert finding.severity is FindingSeverity.INFORMATIONAL


def test_swapping_the_primary_role_changes_the_standing() -> None:
    e = scenario("E")
    plan = _e_plan(
        [
            AU.role("replacement", "tensor*", "primary"),
            AU.role("replacement", "zero", "alternative"),
        ]
    )
    group = bnn.audit(e.evidence, plan=plan).claim("e").groups[0]
    assert group.standing is Standing.CONTRADICTED
    assert "alternative_reverses" in {f.code for f in group.findings}


def test_undeclared_and_missing_primary_configurations() -> None:
    e = scenario("E")
    only_zero = _e_plan([AU.role("replacement", "zero", "primary")])
    group = bnn.audit(e.evidence, plan=only_zero).claim("e").groups[0]
    assert group.standing is Standing.SUPPORTED  # the undeclared result takes no part
    assert "undeclared_configuration" in {f.code for f in group.findings}
    no_primary = _e_plan([AU.role("replacement", "*", "alternative")])
    group = bnn.audit(e.evidence, plan=no_primary).claim("e").groups[0]
    assert group.standing is Standing.NOT_EVALUATED
    assert "primary_untested" in {f.code for f in group.findings}


def test_no_roles_keeps_phase7_behaviour() -> None:
    e = scenario("E")
    group = bnn.audit(e.evidence, plan=_e_plan([])).claim("e").groups[0]
    assert group.standing is Standing.ASSUMPTION_SENSITIVE
    assert {t.role for t in group.tests} == {"undeclared"}


def test_role_rules_are_validated_and_ordered() -> None:
    with pytest.raises(SchemaError, match="duplicate role rules"):
        _e_plan([AU.role("k", "1", "primary"), AU.role("k", "1", "primary")])
    with pytest.raises(ValueError, match="robust"):
        AU.role("k", "1", "robust")
    a = _e_plan([AU.role("k", "1", "primary"), AU.role("replacement", "zero", "primary")])
    b = _e_plan([AU.role("replacement", "zero", "primary"), AU.role("k", "1", "primary")])
    assert a.id == b.id


# ------------------------------------------------------------------ profile


def test_sensitivity_profile_exposes_raw_structure() -> None:
    e = scenario("E")
    group = bnn.audit(e.evidence, plan=_e_plan([])).claim("e").groups[0]
    profile = group.profile
    assert profile is not None
    assert profile.tested == 2
    assert profile.count(outcome="supports") == 1
    assert profile.count(outcome="contradicts") == 1
    assert profile.sensitive_axes == ("replacement",)
    ((sup, con, diff),) = profile.reversals
    assert sup in profile.supporting
    assert con in profile.contradicting
    assert diff == ("replacement",)
    assert profile.describe().startswith("1 of 2 tested configurations SUPPORT")
    text = json.dumps(bnn.audit(e.evidence, plan=_e_plan([])).to_dict())
    for forbidden in ("score", "confidence", "trustworth", "robustness"):
        assert forbidden not in text.lower()


def test_profile_counts_every_configuration_including_alternatives() -> None:
    model, x = _one_unit()
    res = _comp(model, x, F.zero(), min_drop=5.0)
    req = AU.requirement(
        "comprehensiveness",
        policy=F.COMPREHENSIVENESS_POLICY,
        controls=False,
        alternatives=[
            AU.alternative("comprehensiveness", "min_drop", factor=0.5),
            AU.alternative("comprehensiveness", "min_drop", factor=3.0),
        ],
    )
    claim = _unit_claim(
        "c",
        "necessary_for",
        "comprehensiveness",
        roles=[
            AU.role("threshold", "{*}", "primary"),
            AU.role("threshold", "*|alt:min_dropx0.5", "alternative"),
            AU.role("threshold", "*|alt:min_dropx3.0", "stress_test"),
        ],
    )
    plan = _plan(model, [sample_id(x)], [claim], [req])
    group = bnn.audit([res], plan=plan).claim("c").groups[0]
    assert group.standing is Standing.SUPPORTED
    assert group.profile is not None
    assert group.profile.tested == 3
    assert group.profile.count(role="stress_test", outcome="contradicts") == 1
    assert {f.code for f in group.findings} >= {"stress_test_reverses"}


# ------------------------------------------------------------------ configuration level


def test_configuration_level_disagreement_survives_a_summary_standing() -> None:
    model, x = _one_unit()
    near = torch.ones(1, 32)
    near[0, 0] = 1.8
    comp_zero = _comp(model, x, F.zero())  # SUPPORTS
    comp_near = _comp(model, x, F.replacement(near))  # CONTRADICTS
    sel = F.units(A.layer("hidden"), (0,), n_units=32)
    suff = F.run(
        model,
        x,
        test=F.sufficiency(
            target=iv.metrics.select([0, 0]), max_drop=0.1, statement="s", replacement=F.zero()
        ),
        selection=sel,
    )  # CONTRADICTS (drop 0.31)
    plan = _plan(
        model,
        [sample_id(x)],
        [
            _unit_claim("nec", "necessary_for", "comprehensiveness"),
            _unit_claim("suf", "sufficient_for", "sufficiency"),
        ],
        [
            _comp_req(False),
            AU.requirement("sufficiency", policy=F.SUFFICIENCY_POLICY, controls=False),
        ],
    )
    report = bnn.audit([comp_zero, comp_near, suff], plan=plan)
    nec = report.claim("nec")
    assert nec.groups[0].standing is Standing.ASSUMPTION_SENSITIVE
    (finding,) = [f for f in nec.findings if f.code == "configuration_level_disagreement"]
    assert finding.kind is FindingKind.PROTOCOL_DISAGREEMENT
    assert len(finding.values) == 1
    assert "configuration_level_disagreement" in {f.code for f in report.claim("suf").findings}


def test_configuration_level_disagreement_in_scenario_d() -> None:
    d = scenario("D")
    report = bnn.audit(d.evidence, plan=d.plan)
    codes = {f.code for f in report.claim("d_necessary").findings}
    assert {"necessary_not_sufficient", "configuration_level_disagreement"} <= codes


# ------------------------------------------------------------------ uncertainty


def test_wilson_interval_matches_reference_values() -> None:
    iv5 = AU.wilson(5, 10, quantity="q", unit="samples")
    assert math.isclose(iv5.low, 0.2366, abs_tol=1e-4)
    assert math.isclose(iv5.high, 0.7634, abs_tol=1e-4)
    zero = AU.wilson(0, 40, quantity="q", unit="samples")
    assert zero.low == 0.0
    assert math.isclose(zero.high, 0.0876, abs_tol=1e-4)
    assert AU.wilson(40, 40, quantity="q", unit="samples").high == 1.0
    with pytest.raises(ValueError, match="0 <= k <= n"):
        AU.wilson(3, 2, quantity="q", unit="samples")


def test_bootstrap_is_seeded_recorded_and_paired() -> None:
    values = [float(v) for v in range(20)]
    one = AU.bootstrap(values, quantity="mean effect", unit="samples", seed=7)
    two = AU.bootstrap(values, quantity="mean effect", unit="samples", seed=7)
    other = AU.bootstrap(values, quantity="mean effect", unit="samples", seed=8)
    assert one == two
    assert one != other
    assert (one.seed, one.draws, one.unit, one.n) == (7, 10_000, "samples", 20)
    assert one.low < 9.5 < one.high
    same = AU.paired_bootstrap(values, values, quantity="d", unit="samples", seed=1)
    assert (same.estimate, same.low, same.high) == (0.0, 0.0, 0.0)
    shifted = AU.paired_bootstrap(
        [v + 1 for v in values], values, quantity="d", unit="samples", seed=1
    )
    assert (shifted.low, shifted.high) == (1.0, 1.0)  # pairing kept: every difference is 1
    with pytest.raises(TypeError):
        AU.bootstrap(values, quantity="q", unit="samples", seed=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="1000"):
        AU.bootstrap(values, quantity="q", unit="samples", seed=1, draws=10)


def test_per_sample_claims_carry_wilson_intervals() -> None:
    n = scenario("N")
    claim = bnn.audit(n.evidence, plan=n.plan).claim("n_necessary")
    (interval,) = claim.intervals
    assert (interval.k, interval.n, interval.method) == (1, 1, "wilson")
    assert "NOT_EVALUATED" in interval.quantity


# ------------------------------------------------------------------ plan migration, portability


def test_v1_plan_migrates_to_v2() -> None:
    text = (DATA / "audit_plan_v1.json").read_text()
    plan = from_json(text)
    assert isinstance(plan, AuditPlan)
    assert all(c.roles == () for c in plan.claims)
    assert plan.id != json.loads(text)["id"]  # a new version is a new record
    assert from_json(to_json(plan)) == plan


def test_plan_portability_across_processes(tmp_path: Path) -> None:
    a = scenario("A")
    (tmp_path / "plan.json").write_text(to_json(a.plan))
    for i, t in enumerate(collect_traces(a.evidence)):
        t.save(tmp_path / f"t{i:02d}")
    live = bnn.audit(a.evidence, plan=a.plan).to_json()
    code = (
        "import sys, pathlib, beyondnn as bnn\n"
        "from beyondnn.schema import from_json\n"
        "d = pathlib.Path(sys.argv[1])\n"
        "plan = from_json((d / 'plan.json').read_text())\n"
        "print(bnn.audit(sorted(p for p in d.iterdir() if p.is_dir()), plan=plan).to_json())\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, check=True
    )
    assert json.loads(out.stdout) == json.loads(live)


def test_unsupported_protocol_version_is_excluded() -> None:
    c = scenario("C")
    effect = c.evidence[-1]
    old = next(r for r in effect.trace.records if isinstance(r, ClaimTestResult))
    spec = effect.trace.get(old.spec.spec_id)
    assert isinstance(spec, ClaimTestSpec)
    spec2 = ClaimTestSpec(
        protocol=spec.protocol,
        protocol_version=2,
        applicable_relations=spec.applicable_relations,
        criteria=spec.criteria,
        params=spec.params,
    )
    claim = effect.trace.get(old.claim.claim_id)
    trace = TraceResult(effect.trace.config)
    for record in effect.trace.records:
        trace._add(record)
    trace._add(spec2)
    trace._add(
        ClaimTestResult.for_claim(
            claim,
            spec2,
            outcome=Outcome.NOT_APPLICABLE,
            provenance_id=old.provenance_id or "",
        )
    )
    for key, tensor in effect.trace._tensors.items():
        trace._add_tensor(key, tensor)
    trace._seal()
    report = bnn.audit([trace], plan=c.plan)
    assert "unsupported_protocol_version" in {f.code for f in report.findings}


# ------------------------------------------------------------------ save -> restart -> load


def test_save_restart_load_audit_and_why(tmp_path: Path) -> None:
    """ADR-050: no live Phase-6 result object survives into the fresh process."""
    i = scenario("I")
    for n, t in enumerate(collect_traces(i.evidence)):
        t.save(tmp_path / f"t{n:03d}")
    (tmp_path / "plan.json").write_text(to_json(i.plan))
    live = bnn.audit(i.evidence, plan=i.plan).concept(i.extra["concept"]).standing.value
    code = (
        "import sys, pathlib, beyondnn as bnn\n"
        "from beyondnn.schema import from_json\n"
        "from beyondnn._testing.concept_models import ConceptToy, concept_inputs\n"
        "d = pathlib.Path(sys.argv[1])\n"
        "plan = from_json((d / 'plan.json').read_text())\n"
        "paths = sorted(p for p in d.iterdir() if p.is_dir())\n"
        "report = bnn.audit(paths, plan=plan)\n"
        "validation = bnn.concepts.load_validation(paths)\n"
        "model = ConceptToy().eval()\n"
        "trace = bnn.trace(model, concept_inputs(1, seed=5), sites=['hidden'], retention='cpu')\n"
        "text = bnn.compose(trace, concepts=[validation]).render()\n"
        "print(report.concepts[0].standing.value, validation.semantic_status.value,\n"
        "      'CONCEPTS' in text)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, check=True
    )
    assert out.stdout.split() == [live, "validated_concept", "True"]


def test_reconstructed_validation_cannot_run_new_tests(tmp_path: Path) -> None:
    g = scenario("G")
    paths = []
    for n, t in enumerate(collect_traces(g.evidence)):
        t.save(tmp_path / f"t{n}")
        paths.append(tmp_path / f"t{n}")
    v = C.load_validation(paths)
    from beyondnn._testing.concept_models import ConceptToy

    with pytest.raises(C.ConceptError, match="reconstructed"):
        C.encoding_test(
            ConceptToy().eval(),
            v.concept,
            v.encoding.data,
            controls=[C.random_neurons(5, seed=1)],
            criteria=C.encoding_criteria(min_fraction_below=0.9),
        )
    from beyondnn.concepts.verify import ConceptVerificationError

    with pytest.raises(ConceptVerificationError, match="not supplied"):
        C.load_validation(paths[:1])


def test_public_evidence_helpers_round_trip(tmp_path: Path) -> None:
    """API review F-1/F-2: public sample ids and saving exactly the audited evidence."""
    i = scenario("I")
    _, x = _one_unit()
    assert AU.sample_id(x) == sample_id(x)
    traces = AU.traces_of(i.evidence)
    assert traces == collect_traces(i.evidence)
    paths = AU.save_evidence(i.evidence, tmp_path / "ev")
    assert AU.load_evidence(tmp_path / "ev") == paths
    live = bnn.audit(i.evidence, plan=i.plan)
    AU.verify_report(live.to_dict(), paths, i.plan)  # identical evidence after reload
    with pytest.raises(FileExistsError):
        AU.save_evidence(i.evidence, tmp_path / "ev")
    with pytest.raises(FileNotFoundError):
        AU.load_evidence(tmp_path)
