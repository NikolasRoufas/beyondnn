"""Phase 4: structured WHY / evidence synthesis (ADR-031).

Scenarios A-H and their expected outputs are pre-registered in docs/PHASE_4_PLAN.md.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import beyondnn as bnn
import beyondnn.attribution as A
import beyondnn.interventions as iv
from beyondnn._testing.attribution_models import Linear
from beyondnn._testing.causal_models import Additive, Redundant
from beyondnn._testing.models import TinyMLP
from beyondnn.explain import (
    CompositionError,
    Coverage,
    EvidenceBundle,
    EvidenceIntegrityError,
    ExplainResponse,
    ModelMismatchError,
    SampleMismatchError,
    TargetMismatchError,
    UnsupportedScopeError,
)
from beyondnn.schema import (
    EvidenceStatus,
    InterventionOperation,
    JsonMap,
    ModelDeclaration,
    Outcome,
    Relation,
    Verdict,
)
from beyondnn.schema.codec import _expected_id

SEL = iv.metrics.select([0, 0])
POLICIES = (A.ATTRIBUTION_POLICY, iv.INTERVENTION_POLICY)


def x(a: float, b: float) -> torch.Tensor:
    return torch.tensor([[a, b]], dtype=torch.float32)


def ig(n: int = 8) -> A.Method:
    return A.integrated_gradients(baseline=A.zero_baseline(), n_steps=n)


def codes(response: ExplainResponse) -> list[str]:
    return [lim.code for lim in response.why.limitations]


# ------------------------------------------------------------------ scenario builders


def scenario_e() -> dict[str, Any]:
    """Redundant path: attribution to p is substantial, zeroing p leaves the output."""
    h = bnn.instrument(Redundant().eval())
    inputs = x(3, 5)
    trace = h.trace(inputs, sites=["p", "q"])
    method = ig()
    attributed = A.make_claim(A.layer("p"), SEL, inputs, statement="p is attributed credit for y")
    attr = h.attribute(
        inputs,
        target=SEL,
        method=method,
        at=A.layer("p"),
        claims=[(attributed, A.threshold_spec(method, at=A.layer("p"), min_abs_attribution=2.0))],
    )
    necessary = iv.make_claim(
        iv.zero("p"), SEL, Relation.NECESSARY_FOR, inputs, statement="p is necessary for y"
    )
    effect = h.intervene(
        inputs,
        intervention=iv.zero("p"),
        metric=SEL,
        claims=[
            (necessary, iv.threshold_spec(operation=InterventionOperation.ZERO, min_effect=6.0))
        ],
    )
    response = bnn.compose(trace, attributions=[attr], interventions=[effect], policies=POLICIES)
    return {
        "trace": trace,
        "attr": attr,
        "effect": effect,
        "response": response,
        "attributed": attributed,
        "necessary": necessary,
    }


# ------------------------------------------------------------------ A: measured only


def test_scenario_a_measured_only_keeps_phase_1_meaning() -> None:
    response = bnn.instrument(TinyMLP()).explain(torch.ones(1, 4), sites=["**"])
    why = response.why
    assert why.evidence_statuses == {EvidenceStatus.OBSERVED, EvidenceStatus.MEASURED}
    assert why.target is None
    assert why.target_spec is None
    assert why.attributions == ()
    assert why.effects == ()
    assert why.claims == ()
    assert {"NO_ATTRIBUTION", "NO_CAUSAL_EVIDENCE", "NO_CLAIMS_TESTED"} <= set(codes(response))
    cov = why.coverage
    assert cov.measured_internal_states
    assert not cov.attribution_available
    assert not cov.intervention_effect_available
    assert cov.faithfulness_evaluated is False
    assert cov.concepts_validated is False
    assert why.question == Why_QUESTION
    text = response.render()
    head, tail = text.split("\n\nNOT EVALUATED\n")
    assert head.startswith("INPUT  [observed]")
    assert "WHY  [measured internal evidence; not a causal or attributed explanation]" in head
    assert head.rstrip().splitlines()[-2] == "OUTPUT  [observed]"
    assert "faithfulness" in tail


Why_QUESTION = "What internal evidence was measured while this output was produced?"


def test_scenario_a_via_compose_equals_handle_explain() -> None:
    model = TinyMLP()
    trace = bnn.trace(model, torch.ones(1, 4), sites=["**"])
    assert bnn.compose(trace).render() == ExplainResponse(trace).render()
    assert bnn.compose(trace).to_dict() == ExplainResponse.from_trace(trace).to_dict()


# ------------------------------------------------------------------ B: attribution only


def test_scenario_b_attribution_only_cannot_support_necessity() -> None:
    h = bnn.instrument(Linear().eval())
    inputs = x(3, 5)
    method = ig(16)
    spec = A.threshold_spec(method, min_abs_attribution=5.0)
    attributed = A.make_claim(
        A.input(), SEL, inputs, statement="x0 is attributed credit", units=(0,)
    )
    necessary = bnn.schema.Claim(
        statement="x is necessary for y",
        relation=Relation.NECESSARY_FOR,
        subject=attributed.subject,
        target=attributed.target,
        estimand=attributed.estimand,
        source=attributed.source,
    )
    attr = h.attribute(
        inputs, target=SEL, method=method, claims=[(attributed, spec), (necessary, spec)]
    )
    response = bnn.compose(h.trace(inputs), attributions=[attr], policies=POLICIES)
    why = response.why
    assert torch.allclose(why.attribution_views[0].value, torch.tensor([[6.0, 15.0]]))
    assert why.effects == ()
    by_claim = {v.claim.id: v for v in why.claims}
    a_view, n_view = by_claim[attributed.id], by_claim[necessary.id]
    assert [r.outcome for _, r in a_view.tests] == [Outcome.SUPPORTS]
    assert [a.verdict for a in a_view.assessments] == [Verdict.SUPPORTED]
    assert [r.outcome for _, r in n_view.tests] == [Outcome.NOT_APPLICABLE]
    (assessment,) = n_view.assessments
    assert assessment.verdict is Verdict.UNTESTED
    assert assessment.required_but_missing == ("intervention_threshold",)
    assert not n_view.causal_test_performed
    assert not why.coverage.causal_claim_tested
    assert "NO_CAUSAL_EVIDENCE" in codes(response)
    assert "NO_ATTRIBUTION" not in codes(response)
    text = response.render()
    assert "no decisive causal test was performed for this causal claim" in text
    assert "assessment intervention_threshold_policy v1: untested" in text


# ------------------------------------------------------------------ C: intervention only


def test_scenario_c_intervention_only() -> None:
    h = bnn.instrument(Additive().eval())
    inputs = x(3, 5)
    effect = h.intervene(inputs, intervention=iv.zero("a"), metric=SEL)
    response = bnn.compose(h.trace(inputs), interventions=[effect])
    why = response.why
    (e,) = why.effects
    assert (e.baseline_value, e.intervention_value, e.effect) == (8.0, 5.0, -3.0)
    assert e.status is EvidenceStatus.INTERVENTIONAL
    assert why.attributions == ()
    assert "ZERO_ABLATION_MAY_BE_OOD" in codes(response)
    assert "NO_ATTRIBUTION" in codes(response)
    assert "NO_CAUSAL_EVIDENCE" not in codes(response)
    assert "under this intervention the target changed by -3" in response.render()


# ------------------------------------------------------------------ D: both, no combination


def test_scenario_d_attribution_and_intervention_are_shown_separately() -> None:
    h = bnn.instrument(Additive().eval())
    inputs = x(3, 5)
    attr = h.attribute(inputs, target=SEL, method=ig(), at=A.layer("a"))
    effect = h.intervene(inputs, intervention=iv.zero("a"), metric=SEL)
    response = bnn.compose(
        h.trace(inputs, sites=["a"]), attributions=[attr], interventions=[effect]
    )
    why = response.why
    assert torch.allclose(why.attribution_views[0].value, torch.tensor([[3.0]]))
    assert why.effects[0].effect == -3.0
    assert why.claims == ()  # nothing generated
    assert why.by_status(EvidenceStatus.ATTRIBUTED) == (attr.record,)
    assert why.by_status(EvidenceStatus.INTERVENTIONAL) == (effect.effect,)
    data = response.to_dict()
    assert set(data) >= {"attributed", "interventional"}
    for key in ("score", "confidence", "combined", "agreement", "trust", "importance"):
        assert not any(key in k for k in _all_keys(data))


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in _all_keys(v)}
    if isinstance(value, list):
        return {k for v in value for k in _all_keys(v)}
    return set()


# ------------------------------------------------------------------ E: flagship


def test_scenario_e_redundant_path_shows_both_answers_side_by_side() -> None:
    s = scenario_e()
    why = s["response"].why
    assert torch.allclose(why.attribution_views[0].value, torch.tensor([[3.0]]))
    (effect,) = why.effects
    assert (effect.effect, effect.intervention_value) == (-3.0, 3.0)
    verdicts = {v.claim.id: [a.verdict for a in v.assessments] for v in why.claims}
    assert verdicts[s["attributed"].id] == [Verdict.SUPPORTED]
    assert verdicts[s["necessary"].id] == [Verdict.CONTRADICTED]
    assert {
        "LAYER_ATTRIBUTION_PARTIAL_COVERAGE",
        "ATTRIBUTION_BASELINE_ASSUMPTION",
        "ATTRIBUTION_NUMERICAL_APPROXIMATION",
        "ZERO_ABLATION_MAY_BE_OOD",
    } <= set(codes(s["response"]))
    assert not any("CONFLICT" in c for c in codes(s["response"]))
    assert not any("CONFLICT" in c for c in bnn.schema.LIMITATIONS)
    text = s["response"].render()
    assert "assessment attribution_threshold_policy v1: supported" in text
    assert "assessment intervention_threshold_policy v1: contradicted" in text
    assert text.index("ATTRIBUTED  [") < text.index("INTERVENTIONAL  [") < text.index("CLAIMS")


# ------------------------------------------------------------------ F, G: refusals


def test_scenario_f_mismatched_sample_is_refused() -> None:
    h = bnn.instrument(Additive().eval())
    attr = h.attribute(x(3, 5), target=SEL, method=ig())
    effect = h.intervene(x(1, 1), intervention=iv.zero("a"), metric=SEL)
    with pytest.raises(SampleMismatchError, match="another input"):
        bnn.compose(h.trace(x(3, 5)), attributions=[attr], interventions=[effect])
    with pytest.raises(SampleMismatchError):
        bnn.compose(h.trace(x(5, 3)), attributions=[attr])  # same summary stats, other input


def test_scenario_g_mismatched_target_is_refused() -> None:
    model = TinyMLP().eval()
    inputs = torch.linspace(-1, 1, 4).reshape(1, 4)
    attr = bnn.attribute(model, inputs, target=iv.metrics.select([0, 0]), method=A.gradient())
    effect = bnn.intervene(
        model, inputs, intervention=iv.zero("shared"), metric=iv.metrics.select([0, 1])
    )
    with pytest.raises(TargetMismatchError, match="one scalar target"):
        bnn.compose(bnn.trace(model, inputs), attributions=[attr], interventions=[effect])
    other_claim = iv.make_claim(
        iv.zero("shared"),
        iv.metrics.select([0, 1]),
        Relation.DECREASES,
        inputs,
        statement="other target",
    )
    with pytest.raises(TargetMismatchError):
        bnn.compose(bnn.trace(model, inputs), attributions=[attr], claims=[other_claim])


# ------------------------------------------------------------------ H: mixed


def test_scenario_h_mixed_results_stay_mixed_and_both_are_shown() -> None:
    h = bnn.instrument(Additive().eval())
    inputs = x(3, 5)
    claim = iv.make_claim(iv.zero("a"), SEL, Relation.DECREASES, inputs, statement="a decreases y")
    zero = h.intervene(
        inputs,
        intervention=iv.zero("a"),
        metric=SEL,
        claims=[(claim, iv.threshold_spec(operation=InterventionOperation.ZERO, min_effect=1.0))],
    )
    const = h.intervene(
        inputs,
        intervention=iv.constant("a", 10.0),
        metric=SEL,
        claims=[
            (claim, iv.threshold_spec(operation=InterventionOperation.CONSTANT, min_effect=1.0))
        ],
    )
    response = bnn.compose(
        h.trace(inputs), interventions=[zero, const], policies=[iv.INTERVENTION_POLICY]
    )
    (view,) = response.why.claims
    assert [r.outcome for _, r in view.tests] == [Outcome.SUPPORTS, Outcome.CONTRADICTS]
    assert [a.verdict for a in view.assessments] == [Verdict.MIXED]
    text = response.render()
    assert ": supports" in text
    assert ": contradicts" in text
    assert "assessment intervention_threshold_policy v1: mixed" in text


# ------------------------------------------------------------------ further refusals


def test_different_model_weights_are_refused() -> None:
    inputs = x(3, 5)
    other = Redundant().eval()
    with torch.no_grad():
        other.q.weight.mul_(2)
    attr = bnn.attribute(other, inputs, target=SEL, method=A.gradient())
    with pytest.raises(ModelMismatchError, match="model identity"):
        bnn.compose(bnn.trace(Redundant().eval(), inputs), attributions=[attr])
    effect = bnn.intervene(other, inputs, intervention=iv.zero("p"), metric=SEL)
    with pytest.raises(ModelMismatchError):
        bnn.compose(bnn.trace(Redundant().eval(), inputs), interventions=[effect])


def test_different_model_declarations_are_refused() -> None:
    model = Redundant().eval()
    inputs = x(3, 5)
    a = ModelDeclaration(implementation_revision="git:aaa")
    b = ModelDeclaration(implementation_revision="git:bbb")
    attr = bnn.attribute(model, inputs, target=SEL, method=A.gradient(), declared_model=b)
    with pytest.raises(ModelMismatchError, match="ModelDeclaration"):
        bnn.compose(bnn.trace(model, inputs, declared_model=a), attributions=[attr])
    with pytest.raises(ModelMismatchError, match="ModelDeclaration"):
        bnn.compose(bnn.trace(model, inputs), attributions=[attr])  # declared vs undeclared
    ok = bnn.compose(bnn.trace(model, inputs, declared_model=b), attributions=[attr])
    assert ok.why.attributions == (attr.record,)


def test_finite_sample_and_other_input_claims_are_refused() -> None:
    model = Additive().eval()
    sample = iv.intervene_sample(model, [x(1, 2), x(3, 5)], intervention=iv.zero("a"), metric=SEL)
    with pytest.raises(UnsupportedScopeError, match="finite_sample"):
        bnn.compose(bnn.trace(model, x(3, 5)), interventions=[sample])
    other = iv.make_claim(iv.zero("a"), SEL, Relation.DECREASES, x(1, 1), statement="other x")
    with pytest.raises(SampleMismatchError, match="claim"):
        bnn.compose(bnn.trace(model, x(3, 5)), claims=[other])


def test_a_forged_claim_result_is_refused(tmp_path: Path) -> None:
    s = scenario_e()
    s["effect"].trace.save(tmp_path / "t")
    document = json.loads((tmp_path / "t" / "trace.json").read_text())
    for env in document["records"]:
        if env["kind"] == "claim_test_result":
            assert env["data"]["outcome"] == "contradicts"
            env["data"]["outcome"] = "supports"  # forged, with a consistent id
            env["id"] = _expected_id(env["kind"], env["record_version"], env["data"])
    (tmp_path / "t" / "trace.json").write_text(json.dumps(document))
    forged = iv.InterventionResult.from_trace(bnn.load_trace(tmp_path / "t"))
    with pytest.raises(EvidenceIntegrityError, match="does not follow from its evidence"):
        bnn.compose(s["trace"], interventions=[forged], policies=POLICIES)


def test_a_record_mutated_after_creation_is_refused() -> None:
    s = scenario_e()
    object.__setattr__(s["effect"].effect, "baseline_value", 7.0)
    with pytest.raises(EvidenceIntegrityError, match="does not match its content"):
        bnn.compose(s["trace"], interventions=[s["effect"]])


def test_a_trace_without_input_identity_cannot_anchor_target_evidence(tmp_path: Path) -> None:
    model = Additive().eval()
    inputs = x(3, 5)
    bnn.trace(model, inputs).save(tmp_path / "t")
    document = json.loads((tmp_path / "t" / "trace.json").read_text())
    renamed: dict[str, str] = {}
    for env in document["records"]:
        text = json.dumps(env["data"])
        for new, old in renamed.items():
            text = text.replace(new, old)
        env["data"] = json.loads(text)
        if env["kind"] == "input":  # as written before ADR-031
            env["record_version"] = 2
            env["data"].pop("sample_id")
        new_id = env["id"]
        env["id"] = _expected_id(env["kind"], env["record_version"], env["data"])
        renamed[new_id] = env["id"]
    (tmp_path / "t" / "trace.json").write_text(json.dumps(document))
    loaded = bnn.load_trace(tmp_path / "t")
    assert loaded.input.sample_id is None
    ExplainResponse(loaded)  # measured-only still works
    attr = bnn.attribute(model, inputs, target=SEL, method=A.gradient())
    with pytest.raises(SampleMismatchError, match="no exact sample identity"):
        bnn.compose(loaded, attributions=[attr])


def test_composition_errors_are_value_errors_and_types_are_checked() -> None:
    assert issubclass(CompositionError, ValueError)
    trace = bnn.trace(Additive().eval(), x(1, 2))
    with pytest.raises(TypeError):
        bnn.compose(trace, attributions=[trace])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="compose"):
        EvidenceBundle(object())
    model = Additive().eval()
    with bnn.recording(model) as ctx:
        model(x(1, 2))
        model(x(1, 2))
    with pytest.raises(ValueError, match="exactly one root invocation"):
        bnn.compose(ctx.result)


# ------------------------------------------------------------------ views, immutability


def test_views_reference_the_original_records() -> None:
    s = scenario_e()
    why = s["response"].why
    assert why.attributions[0] is s["attr"].record
    assert why.effects[0] is s["effect"].effect
    assert why.intervention_views[0].intervention is s["effect"].intervention
    assert why.observations[0] is s["trace"].input
    assert all(a is b for a, b in zip(why.activations, s["trace"].activations, strict=True))
    claims = {v.claim.id: v.claim for v in why.claims}
    assert claims[s["necessary"].id] is s["effect"].trace.get(s["necessary"].id)
    result = why.claims[1].tests[0][1]
    assert result is s["effect"].trace.get(result.id)
    assert why.origin(s["attr"].record) is s["attr"].trace.origin(s["attr"].record)
    assert why.origin(s["effect"].effect) is s["effect"].trace.origin(s["effect"].effect)


def test_composition_mutates_nothing_and_is_itself_immutable() -> None:
    s = scenario_e()
    before = {
        id(t): (t.records, dict(t._tensors))
        for t in (s["trace"], s["attr"].trace, s["effect"].trace)
    }
    bnn.compose(
        s["trace"], attributions=[s["attr"]], interventions=[s["effect"]], policies=POLICIES
    ).render()
    for t in (s["trace"], s["attr"].trace, s["effect"].trace):
        records, tensors = before[id(t)]
        assert t.records == records
        assert all(a is b for a, b in zip(t.records, records, strict=True))
        assert set(t._tensors) == set(tensors)
        assert all(t._tensors[k] is v for k, v in tensors.items())
    bundle = s["response"].bundle
    with pytest.raises(AttributeError):
        bundle.target = None
    with pytest.raises(dataclasses.FrozenInstanceError):
        s["response"].why.bundle = None
    with pytest.raises(ValueError, match="evaluated"):
        Coverage(
            measured_internal_states=True,
            attribution_available=True,
            intervention_effect_available=True,
            estimated_causal_available=False,
            claims_declared=True,
            claims_tested=True,
            causal_claim_tested=True,
            faithfulness_evaluated=True,
        )


# ------------------------------------------------------------------ no computation


class _Tripwire(Redundant):
    armed = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.armed:
            raise AssertionError("the model was executed during composition")
        return super().forward(x)


def test_composition_runs_no_model_autograd_hooks_or_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _Tripwire().eval()
    inputs = x(3, 5)
    trace = bnn.trace(model, inputs, sites=["p"])
    attr = bnn.attribute(model, inputs, target=SEL, method=ig(), at=A.layer("p"))
    effect = bnn.intervene(model, inputs, intervention=iv.zero("p"), metric=SEL)
    model.armed = True

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("autograd was used during composition")

    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(nn.Module, "register_forward_hook", forbidden)
    rng = torch.get_rng_state()
    response = bnn.compose(trace, attributions=[attr], interventions=[effect], policies=POLICIES)
    response.render()
    response.to_dict()
    assert torch.equal(rng, torch.get_rng_state())
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


# ------------------------------------------------------------------ determinism, persistence


def test_rendering_and_structure_are_deterministic() -> None:
    a, b = scenario_e()["response"], scenario_e()["response"]
    assert a.render() == b.render()
    assert a.to_dict() == b.to_dict()
    assert json.dumps(a.to_dict(), sort_keys=True) == json.dumps(b.to_dict(), sort_keys=True)


def test_an_explanation_is_reconstructed_from_persisted_evidence(tmp_path: Path) -> None:
    s = scenario_e()
    for name in ("trace",):
        s[name].save(tmp_path / name)
    s["attr"].trace.save(tmp_path / "attr")
    s["effect"].trace.save(tmp_path / "effect")
    rebuilt = bnn.compose(
        bnn.load_trace(tmp_path / "trace"),
        attributions=[A.AttributionResult.from_trace(bnn.load_trace(tmp_path / "attr"))],
        interventions=[iv.InterventionResult.from_trace(bnn.load_trace(tmp_path / "effect"))],
        policies=POLICIES,
    )
    assert rebuilt.to_dict() == s["response"].to_dict()
    assert rebuilt.render() == s["response"].render()
    assert not list(tmp_path.rglob("*.pkl"))


# ------------------------------------------------------------------ language


FORBIDDEN = re.compile(
    r"\b(caused|causes|because|reason|important|the model (thought|used|decided|focused|"
    r"understood)|is faithful|explanation is)\b|confidence\s*[:=]|\d\s*%",
    re.IGNORECASE,
)


def test_rendered_text_makes_no_unlicensed_statements() -> None:
    for response in (
        scenario_e()["response"],
        bnn.compose(
            bnn.trace(Linear().eval(), x(3, 5)),
            attributions=[bnn.attribute(Linear().eval(), x(3, 5), target=SEL, method=ig())],
        ),
        bnn.instrument(TinyMLP()).explain(torch.ones(1, 4), sites=["**"]),
    ):
        text = response.render()
        # Lines that name questions the evidence does NOT answer are exempt.
        stated = "\n".join(
            line
            for line in text.splitlines()
            if not line.strip().startswith(("does not answer:", "unanswered:"))
        )
        found = FORBIDDEN.search(stated)
        assert found is None, found
        assert "does not answer:" in text


def test_the_phase_4_example_script_runs() -> None:
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "examples" / "phase4_redundant_path.py"
    spec = importlib.util.spec_from_file_location("phase4_example", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    response = module.build()
    verdicts = [a.verdict for v in response.why.claims for a in v.assessments]
    assert verdicts == [Verdict.SUPPORTED, Verdict.CONTRADICTED]


# ------------------------------------------------------------------ independent guard tests


def test_the_intervention_protocol_never_decides_attributed_to() -> None:
    from beyondnn.protocols import INTERVENTION_THRESHOLD, PROTOCOLS, check_policy
    from beyondnn.schema import (
        AssessmentPolicy,
        ClaimTestSpec,
        EvidenceRuleError,
        JsonMap,
        PolicyRequirement,
    )

    assert PROTOCOLS[INTERVENTION_THRESHOLD] == frozenset(
        {Relation.NECESSARY_FOR, Relation.DECREASES, Relation.INCREASES}
    )
    with pytest.raises(EvidenceRuleError):
        iv.threshold_spec(
            operation=InterventionOperation.ZERO,
            min_effect=1.0,
            relations=(Relation.ATTRIBUTED_TO,),
        )
    with pytest.raises(EvidenceRuleError, match="does not justify"):
        check_policy(
            AssessmentPolicy(
                name="bad",
                version=1,
                requirements=(
                    PolicyRequirement(
                        relation=Relation.ATTRIBUTED_TO, protocols=(INTERVENTION_THRESHOLD,)
                    ),
                ),
            )
        )
    h = bnn.instrument(Additive().eval())
    inputs = x(3, 5)
    claim = A.make_claim(A.layer("a"), SEL, inputs, statement="a attributed")
    crafted = ClaimTestSpec(
        protocol=INTERVENTION_THRESHOLD,
        protocol_version=1,
        applicable_relations=(Relation.ATTRIBUTED_TO,),
        criteria=JsonMap({"min_effect": 1.0}),
        params=JsonMap({"operation": "zero"}),
    )
    effect = h.intervene(inputs, intervention=iv.zero("a"), metric=SEL, claims=[(claim, crafted)])
    assert [r.outcome for r in effect.claim_results] == [Outcome.NOT_APPLICABLE]
    response = bnn.compose(h.trace(inputs), interventions=[effect], policies=POLICIES)
    (view,) = response.why.claims
    assert [a.verdict for a in view.assessments] == [Verdict.UNTESTED]


def test_statuses_stay_separate_in_every_section() -> None:
    why = scenario_e()["response"].why
    attributed = why.by_status(EvidenceStatus.ATTRIBUTED)
    interventional = why.by_status(EvidenceStatus.INTERVENTIONAL)
    assert {r.status for r in attributed} == {EvidenceStatus.ATTRIBUTED}
    assert {r.status for r in interventional} == {EvidenceStatus.INTERVENTIONAL}
    assert not set(map(id, attributed)) & set(map(id, interventional))
    assert all(e.status is EvidenceStatus.INTERVENTIONAL for e in why.effects)
    assert all(a.status is EvidenceStatus.ATTRIBUTED for a in why.attributions)
    assert why.estimated_causal == ()


def test_intervention_from_another_model_is_refused() -> None:
    inputs = x(3, 5)
    other = Additive().eval()
    with torch.no_grad():
        other.b.weight.mul_(3)
    effect = bnn.intervene(other, inputs, intervention=iv.zero("a"), metric=SEL)
    with pytest.raises(ModelMismatchError, match="model identity"):
        bnn.compose(bnn.trace(Additive().eval(), inputs), interventions=[effect])


def test_intervention_under_another_declaration_is_refused() -> None:
    model = Additive().eval()
    inputs = x(3, 5)
    effect = bnn.intervene(
        model,
        inputs,
        intervention=iv.zero("a"),
        metric=SEL,
        declared_model=ModelDeclaration(config=JsonMap({"variant": "b"})),
    )
    with pytest.raises(ModelMismatchError, match="ModelDeclaration"):
        bnn.compose(
            bnn.trace(
                model, inputs, declared_model=ModelDeclaration(config=JsonMap({"variant": "a"}))
            ),
            interventions=[effect],
        )


def test_two_attributions_with_different_targets_are_refused() -> None:
    model = TinyMLP().eval()
    inputs = torch.ones(1, 4)
    a0 = bnn.attribute(model, inputs, target=iv.metrics.select([0, 0]), method=A.gradient())
    a1 = bnn.attribute(model, inputs, target=iv.metrics.select([0, 2]), method=A.gradient())
    with pytest.raises(TargetMismatchError):
        bnn.compose(bnn.trace(model, inputs), attributions=[a0, a1])


def test_attribution_and_intervention_records_are_the_originals() -> None:
    s = scenario_e()
    why = s["response"].why
    assert all(v.record is s["attr"].record for v in why.attribution_views)
    assert why.by_status(EvidenceStatus.ATTRIBUTED)[0] is s["attr"].record
    assert why.by_status(EvidenceStatus.INTERVENTIONAL)[0] is s["effect"].effect


def test_an_untested_causal_claim_is_shown_as_untested() -> None:
    model = Additive().eval()
    inputs = x(3, 5)
    claim = iv.make_claim(
        iv.zero("a"), SEL, Relation.NECESSARY_FOR, inputs, statement="a is necessary for y"
    )
    response = bnn.compose(
        bnn.trace(model, inputs), claims=[claim], policies=[iv.INTERVENTION_POLICY]
    )
    (view,) = response.why.claims
    assert view.tests == ()
    assert not view.causal_test_performed
    assert [a.verdict for a in view.assessments] == [Verdict.UNTESTED]
    assert not response.why.coverage.causal_claim_tested
    text = response.render()
    assert "no decisive causal test was performed" in text
    assert "tests: (none recorded)" in text


def test_the_attribution_section_never_describes_causes() -> None:
    text = scenario_e()["response"].render()
    section = text[text.index("  ATTRIBUTED  [") : text.index("  INTERVENTIONAL  [")]
    assert "caus" not in section.lower()
    assert "not intervention effects" in section


def test_a_forged_attribution_claim_result_is_refused(tmp_path: Path) -> None:
    s = scenario_e()
    s["attr"].trace.save(tmp_path / "t")
    document = json.loads((tmp_path / "t" / "trace.json").read_text())
    for env in document["records"]:
        if env["kind"] == "claim_test_result":
            assert env["data"]["outcome"] == "supports"
            env["data"]["outcome"] = "contradicts"
            env["id"] = _expected_id(env["kind"], env["record_version"], env["data"])
    (tmp_path / "t" / "trace.json").write_text(json.dumps(document))
    forged = A.AttributionResult.from_trace(bnn.load_trace(tmp_path / "t"))
    with pytest.raises(EvidenceIntegrityError, match="does not follow"):
        bnn.compose(s["trace"], attributions=[forged])


def test_a_mutated_attribution_record_is_refused() -> None:
    s = scenario_e()
    object.__setattr__(s["attr"].record, "target_value", 99.0)
    with pytest.raises(EvidenceIntegrityError, match="does not match its content"):
        bnn.compose(s["trace"], attributions=[s["attr"]])


def test_intervention_only_composition_uses_no_autograd_or_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = bnn.instrument(Additive().eval())
    inputs = x(3, 5)
    trace = h.trace(inputs)
    effect = h.intervene(inputs, intervention=iv.zero("a"), metric=SEL)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("autograd was used during composition")

    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    rng = torch.get_rng_state()
    bnn.compose(trace, interventions=[effect]).render()
    assert torch.equal(rng, torch.get_rng_state())


def test_the_intervention_evaluator_rejects_attributed_to_whatever_the_registry_says(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import beyondnn.interventions.claims as ic
    from beyondnn.schema import ClaimTestSpec

    widened = dict(ic.PROTOCOLS)
    widened[ic.INTERVENTION_THRESHOLD] = widened[ic.INTERVENTION_THRESHOLD] | {
        Relation.ATTRIBUTED_TO
    }
    monkeypatch.setattr(ic, "PROTOCOLS", widened)
    h = bnn.instrument(Additive().eval())
    inputs = x(3, 5)
    claim = A.make_claim(A.layer("a"), SEL, inputs, statement="a attributed")
    crafted = ClaimTestSpec(
        protocol=ic.INTERVENTION_THRESHOLD,
        protocol_version=1,
        applicable_relations=(Relation.ATTRIBUTED_TO,),
        criteria=JsonMap({"min_effect": 1.0}),
        params=JsonMap({"operation": "zero"}),
    )
    effect = h.intervene(inputs, intervention=iv.zero("a"), metric=SEL, claims=[(claim, crafted)])
    assert [r.outcome for r in effect.claim_results] == [Outcome.NOT_APPLICABLE]
