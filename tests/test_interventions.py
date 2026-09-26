"""Phase 2: paired interventions, CausalEffect semantics, ground truth, safety, claims."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import beyondnn as bnn
import beyondnn.interventions as iv
from beyondnn._testing.causal_models import Additive, Gated, Interaction, Redundant
from beyondnn._testing.models import TinyMLP, TinyTransformer
from beyondnn.core.hooks import AliasSiteAmbiguityError
from beyondnn.core.trace import (
    ExternalForwardHooksError,
    TraceConfig,
    TraceIntegrityError,
    TraceResult,
)
from beyondnn.provenance.fingerprint import tensor_bytes
from beyondnn.schema import (
    ActivationRecord,
    Assessment,
    AssessmentPolicy,
    CausalEffect,
    EstimandScope,
    EvidenceRuleError,
    EvidenceStatus,
    ExecutionMode,
    ExecutionOccurrence,
    InterventionOperation,
    Outcome,
    PolicyRequirement,
    Relation,
    Verdict,
    verify_ref,
)

SEL = iv.metrics.select([0, 0])


def x(a: float, b: float) -> torch.Tensor:
    return torch.tensor([[a, b]], dtype=torch.float32)


def hook_count(model: nn.Module) -> int:
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


def run(model: nn.Module, *inputs: Any, **kw: Any) -> iv.InterventionResult:
    kw.setdefault("metric", SEL)
    return iv.intervene(model.eval(), *inputs, **kw)


# ------------------------------------------ ground truth (docs/PHASE_2_PLAN.md)


@pytest.mark.parametrize(("x0", "x1"), [(3.0, 5.0), (-2.0, 0.5), (0.0, 1.0)])
def test_additive_zero_ablation_effect_is_minus_x0(x0: float, x1: float) -> None:
    r = run(Additive(), x(x0, x1), intervention=iv.zero("a"))
    assert r.baseline_value == x0 + x1
    assert r.intervention_value == x1
    assert r.value == -x0


def test_additive_constant_replacement_effect_is_c_minus_x0() -> None:
    r = run(Additive(), x(3.0, 5.0), intervention=iv.constant("a", 10.0))
    assert r.value == 10.0 - 3.0


def test_additive_patch_effect_is_source_minus_target() -> None:
    r = run(Additive(), x(3.0, 5.0), intervention=iv.patch("a", x(7.0, 1.0)))
    assert r.value == 7.0 - 3.0
    assert r.trace.passes == 3  # source, baseline, intervention


def test_patching_from_an_identical_source_has_zero_effect() -> None:
    r = run(Additive(), x(3.0, 5.0), intervention=iv.patch("a", x(3.0, 5.0)))
    assert r.value == 0.0
    assert "PATCH_SOURCE_CONTEXT_DIFFERS" not in [lim.code for lim in r.limitations]


def test_replacing_an_activation_with_its_own_value_has_zero_effect() -> None:
    r = run(Additive(), x(3.0, 5.0), intervention=iv.constant("a", torch.tensor([[3.0]])))
    assert r.value == 0.0


def test_gated_zero_gate_removes_the_output() -> None:
    r = run(Gated(), x(3.0, 5.0), intervention=iv.zero("gate"))
    assert (r.baseline_value, r.intervention_value, r.value) == (15.0, 0.0, -15.0)


def test_redundant_path_ablation_does_not_remove_the_output() -> None:
    r = run(Redundant(), x(3.0, 5.0), intervention=iv.zero("p"))
    assert (r.baseline_value, r.intervention_value, r.value) == (6.0, 3.0, -3.0)
    assert r.intervention_value != 0.0  # ablating one path is not "the output needs p"


def test_interaction_effect_depends_on_context() -> None:
    assert run(Interaction(), x(3.0, 0.0), intervention=iv.zero("a")).value == 0.0
    assert run(Interaction(), x(3.0, 5.0), intervention=iv.zero("a")).value == -15.0


def test_same_intervention_repeated_is_deterministic() -> None:
    a = run(Additive(), x(3.0, 5.0), intervention=iv.zero("a"))
    b = run(Additive(), x(3.0, 5.0), intervention=iv.zero("a"))
    assert a.effect.id == b.effect.id
    ids = [
        [r.id for r in t.records if not isinstance(r, ExecutionOccurrence)]
        for t in (a.trace, b.trace)
    ]
    assert ids[0] == ids[1]


# ----------------------------------------------------------------- statuses and provenance


def test_baseline_clean_intervention_marked_and_effect_interventional() -> None:
    r = run(Additive(), x(3.0, 5.0), intervention=iv.zero("a"), sites=["**"], retention="cpu")
    trace = r.trace
    base_prov = trace.origin(r.baseline_output)
    int_prov = trace.origin(r.intervention_output)
    assert base_prov.execution.mode is ExecutionMode.CLEAN
    assert base_prov.execution.intervention_id is None
    assert int_prov.execution.mode is ExecutionMode.INTERVENTION
    assert int_prov.execution.intervention_id == r.intervention.id
    assert int_prov.model == base_prov.model
    assert r.effect.status is EvidenceStatus.INTERVENTIONAL
    assert r.effect.estimand.scope is EstimandScope.INSTANCE
    assert r.effect.estimand.sample_id == iv.sample_id(x(3.0, 5.0))
    assert trace.origin(r.effect).method.name == "intervention_effect"
    # activations stay MEASURED, including the replaced one, which holds zeros
    assert all(a.status is EvidenceStatus.MEASURED for a in trace.activations)
    replaced = trace.activation("a", pass_index=r.intervention_output.pass_index)
    original = trace.activation("a", pass_index=r.baseline_output.pass_index)
    assert torch.equal(trace.tensor(replaced), torch.zeros(1, 1))
    assert torch.equal(trace.tensor(original), torch.tensor([[3.0]]))
    statuses = {rec.status for rec in trace.records}
    assert EvidenceStatus.ESTIMATED_CAUSAL not in statuses
    assert statuses <= {
        None,
        EvidenceStatus.OBSERVED,
        EvidenceStatus.MEASURED,
        EvidenceStatus.INTERVENTIONAL,
    }


def test_patch_references_its_provenance_bearing_source() -> None:
    r = run(Additive(), x(3.0, 5.0), intervention=iv.patch("a", x(7.0, 1.0)))
    record = r.intervention
    assert record.operation is InterventionOperation.PATCH
    assert record.source is not None
    source = r.trace.get(record.source.record_id)
    assert isinstance(source, ActivationRecord)
    verify_ref(record.source, source)
    assert source.pass_index == 0
    assert r.trace.origin(source).execution.mode is ExecutionMode.CLEAN
    assert record.value is not None
    patched = r.trace.tensor(record.value)
    assert torch.equal(patched, torch.tensor([[7.0]]))


def test_intervention_specs_have_deterministic_identity() -> None:
    a = run(Additive(), x(3.0, 5.0), intervention=iv.zero("a")).intervention
    b = run(Additive(), x(9.0, 9.0), intervention=iv.zero("a")).intervention
    assert a.id == b.id  # same site/leaf/call/operation
    assert run(Additive(), x(3.0, 5.0), intervention=iv.zero("b")).intervention.id != a.id


def test_finite_sample_effect_is_an_exact_mean_and_stays_interventional() -> None:
    samples = [x(1.0, 0.0), x(2.0, 0.0), x(6.0, 0.0)]
    r = iv.intervene_sample(Additive().eval(), samples, intervention=iv.zero("a"), metric=SEL)
    assert r.effect.estimand.scope is EstimandScope.FINITE_SAMPLE
    assert r.effect.estimand.n == 3
    assert r.value == pytest.approx(-3.0)
    assert r.effect.status is EvidenceStatus.INTERVENTIONAL
    assert len(r.instance_effects) == 3
    assert sorted(e.effect for e in r.instance_effects) == [-6.0, -2.0, -1.0]
    assert {ref.record_id for ref in r.effect.derived_from} == {e.id for e in r.instance_effects}


# ----------------------------------------------------------------- limitations


def codes(r: iv.InterventionResult) -> set[str]:
    return {lim.code for lim in r.limitations}


def test_limitations_follow_the_operation_and_metric() -> None:
    assert "ZERO_ABLATION_MAY_BE_OOD" in codes(run(Additive(), x(1, 2), intervention=iv.zero("a")))
    const = run(Additive(), x(1, 2), intervention=iv.constant("a", 4.0))
    assert "CONSTANT_REPLACEMENT_MAY_BE_OOD" in codes(const)
    assert "ZERO_ABLATION_MAY_BE_OOD" not in codes(const)
    assert "PATCH_SOURCE_CONTEXT_DIFFERS" in codes(
        run(Additive(), x(1, 2), intervention=iv.patch("a", x(5, 5)))
    )
    custom = run(
        Additive(),
        x(1, 2),
        intervention=iv.zero("a"),
        metric=iv.metrics.custom("sum", lambda o: o.sum(), implementation_revision="v1"),
    )
    assert "CUSTOM_METRIC_UNVERIFIED" in codes(custom)
    assert "CUSTOM_METRIC_UNVERIFIED" not in codes(const)
    zero = run(Additive(), x(1, 2), intervention=iv.zero("a"))
    (lim,) = [lim for lim in zero.limitations if lim.code == "ZERO_ABLATION_MAY_BE_OOD"]
    assert lim.applies_to == (zero.effect.id,)


# ----------------------------------------------------------------- targeting


def test_call_index_targets_one_call_of_a_shared_module() -> None:
    model = TinyMLP().eval()
    first = iv.intervene(
        model, torch.ones(2, 4), intervention=iv.zero("shared", call_index=0), metric=SEL
    )
    second = iv.intervene(
        model, torch.ones(2, 4), intervention=iv.zero("shared", call_index=1), metric=SEL
    )
    assert first.value != second.value
    with pytest.raises(iv.InterventionNotAppliedError, match="ran 0 times"):
        iv.intervene(
            model, torch.ones(2, 4), intervention=iv.zero("shared", call_index=2), metric=SEL
        )
    assert hook_count(model) == 0


def test_structured_output_leaf_can_be_targeted() -> None:
    model = TinyTransformer().eval()
    tokens = torch.tensor([[1, 5, 9, 3]])
    r = iv.intervene(
        model,
        tokens,
        intervention=iv.zero("blocks.0.attn", output_path="[0]"),
        metric=iv.metrics.mean(),
    )
    assert r.value != 0.0
    with pytest.raises(iv.InterventionError, match="no tensor leaf"):
        iv.intervene(
            model,
            tokens,
            intervention=iv.zero("blocks.0.attn", output_path="[5]"),
            metric=iv.metrics.mean(),
        )
    assert hook_count(model) == 0


def test_constant_tensors_are_never_broadcast() -> None:
    model = Additive().eval()
    with pytest.raises(iv.InterventionError, match="no broadcasting"):
        iv.intervene(model, x(1, 2), intervention=iv.constant("a", torch.zeros(2, 1)), metric=SEL)
    with pytest.raises(iv.InterventionError, match="no broadcasting"):
        iv.intervene(
            model,
            x(1, 2),
            intervention=iv.constant("a", torch.zeros(1, 1, dtype=torch.float64)),
            metric=SEL,
        )
    assert hook_count(model) == 0


# ----------------------------------------------------------------- refusals and safety


def test_training_mode_is_refused() -> None:
    with pytest.raises(iv.StatefulComparisonError, match="training mode"):
        iv.intervene(Additive().train(), x(1, 2), intervention=iv.zero("a"), metric=SEL)


class Noisy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.a(x) + torch.rand(1)
        return out


def test_stochastic_models_are_refused_even_in_eval_mode() -> None:
    model = Noisy().eval()
    with pytest.raises(iv.StochasticComparisonError):
        iv.intervene(model, x(1, 2), intervention=iv.zero("a"), metric=SEL)
    assert hook_count(model) == 0


class Counting(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(2, 1)
        self.register_buffer("calls", torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.get_buffer("calls").add_(1)
        out: torch.Tensor = self.a(x) + self.get_buffer("calls")
        return out


def test_state_mutating_models_are_refused() -> None:
    with pytest.raises(iv.StatefulComparisonError, match="state changed"):
        iv.intervene(Counting().eval(), x(1, 2), intervention=iv.zero("a"), metric=SEL)


class Aliased(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        shared = nn.Linear(2, 1)
        self.left = shared
        self.right = shared

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.left(x)
        return out


def test_aliased_targets_are_refused() -> None:
    with pytest.raises(AliasSiteAmbiguityError):
        iv.intervene(Aliased().eval(), x(1, 2), intervention=iv.zero("left"), metric=SEL)


def test_foreign_hooks_are_still_refused_and_ours_are_recognised() -> None:
    model = Additive().eval()
    model.b.register_forward_hook(lambda m, a, o: None)
    with pytest.raises(ExternalForwardHooksError):
        iv.intervene(model, x(1, 2), intervention=iv.zero("a"), metric=SEL)
    assert hook_count(model) == 1  # only the user's hook remains


@pytest.mark.parametrize(
    "failing",
    [
        {"metric": iv.metrics.select([0, 9])},  # metric error
        {"intervention": iv.constant("a", torch.zeros(3))},  # replacement error
        {
            "metric": iv.metrics.custom(
                "boom",
                lambda o: (_ for _ in ()).throw(KeyboardInterrupt()),
                implementation_revision="v1",
            )
        },
    ],
)
def test_no_hooks_leak_on_any_failure(failing: dict[str, Any]) -> None:
    model = Additive().eval()
    kw: dict[str, Any] = {"intervention": iv.zero("a"), "metric": SEL} | failing
    with pytest.raises((iv.InterventionError, ValueError, KeyboardInterrupt)):
        iv.intervene(model, x(1, 2), **kw)
    assert hook_count(model) == 0
    assert torch.equal(model(x(1, 2)), torch.tensor([[3.0]]))


def test_model_is_unchanged_by_an_intervention_run() -> None:
    model = TinyMLP().eval()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    before = model(torch.ones(2, 4))
    iv.intervene(model, torch.ones(2, 4), intervention=iv.zero("projection"), metric=SEL)
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    assert torch.equal(model(torch.ones(2, 4)), before)
    assert hook_count(model) == 0


def test_container_rejects_intervention_provenance_without_its_spec() -> None:
    r = run(Additive(), x(1, 2), intervention=iv.zero("a"))
    fresh = TraceResult(TraceConfig((), (), "summary"))
    intervened = r.trace.origin(r.intervention_output)
    with pytest.raises(TraceIntegrityError, match="InterventionRecord"):
        fresh._add(intervened)


# ----------------------------------------------------------------- persistence


def test_results_round_trip_and_rebuild(tmp_path: Path) -> None:
    r = run(Additive(), x(3.0, 5.0), intervention=iv.patch("a", x(7.0, 1.0)), retention="cpu")
    r.trace.save(tmp_path / "t")
    loaded = iv.InterventionResult.from_trace(bnn.load_trace(tmp_path / "t"))
    assert loaded.effect.id == r.effect.id
    assert loaded.intervention.id == r.intervention.id
    assert loaded.value == 4.0
    assert [lim.id for lim in loaded.limitations] == [lim.id for lim in r.limitations]
    assert loaded.intervention.value is not None
    assert tensor_bytes(loaded.trace.tensor(loaded.intervention.value)) == tensor_bytes(
        torch.tensor([[7.0]])
    )


def test_tampered_patch_tensor_is_rejected_on_load(tmp_path: Path) -> None:
    r = run(Additive(), x(3.0, 5.0), intervention=iv.patch("a", x(7.0, 1.0)))
    r.trace.save(tmp_path / "t")
    tensors = torch.load(tmp_path / "t" / "tensors.pt", weights_only=True)
    key = next(iter(tensors))
    tensors[key] = tensors[key] + 1
    (tmp_path / "t" / "tensors.pt").unlink()
    torch.save(tensors, tmp_path / "t" / "tensors.pt")
    with pytest.raises(Exception, match="content digest"):
        bnn.load_trace(tmp_path / "t")


def test_handle_intervene_equals_intervene() -> None:
    a = bnn.instrument(Additive().eval()).intervene(x(3, 5), intervention=iv.zero("a"), metric=SEL)
    b = bnn.intervene(Additive().eval(), x(3, 5), intervention=iv.zero("a"), metric=SEL)
    assert a.effect.id == b.effect.id


# ----------------------------------------------------------------- claims


def _claim(
    relation: Relation, inputs: torch.Tensor, site: str = "a", metric: iv.metrics.Metric = SEL
) -> Any:
    return iv.make_claim(
        iv.zero(site), metric, relation, inputs, statement=f"{site} {relation.value} y"
    )


def test_declared_threshold_decides_necessity_claims() -> None:
    inputs = x(3.0, 5.0)
    spec_small = iv.threshold_spec(operation=InterventionOperation.ZERO, min_effect=1.0)
    spec_large = iv.threshold_spec(operation=InterventionOperation.ZERO, min_effect=10.0)
    nec = _claim(Relation.NECESSARY_FOR, inputs)
    inc = _claim(Relation.INCREASES, inputs)
    r = run(
        Additive(),
        inputs,
        intervention=iv.zero("a"),
        claims=[(nec, spec_small), (nec, spec_large), (inc, spec_small)],
    )
    outcomes = [res.outcome for res in r.claim_results]
    assert outcomes == [Outcome.SUPPORTS, Outcome.CONTRADICTS, Outcome.CONTRADICTS]
    supported = [res for res in r.claim_results if res.outcome is Outcome.SUPPORTS]
    assert Assessment.derive(nec, supported, iv.INTERVENTION_POLICY).verdict is Verdict.SUPPORTED
    assert all(
        ev.status is EvidenceStatus.INTERVENTIONAL for res in supported for ev in res.evidence
    )


def test_nonzero_effect_does_not_imply_support() -> None:
    inputs = x(3.0, 5.0)
    claim = _claim(Relation.NECESSARY_FOR, x(3.0, 5.0), site="p")
    spec = iv.threshold_spec(
        operation=InterventionOperation.ZERO, min_effect=6.0
    )  # "p carries all of y"
    r = run(Redundant(), inputs, intervention=iv.zero("p"), claims=[(claim, spec)])
    assert r.value == -3.0
    (result,) = r.claim_results
    assert result.outcome is Outcome.CONTRADICTS
    assert (
        Assessment.derive(claim, [result], iv.INTERVENTION_POLICY).verdict is Verdict.CONTRADICTED
    )


@pytest.mark.parametrize(
    "mismatch",
    [
        {"site": "b"},
        {"metric": iv.metrics.mean()},
        {"inputs": x(1.0, 1.0)},
    ],
)
def test_mismatched_claims_are_not_applicable(mismatch: dict[str, Any]) -> None:
    claim = _claim(
        Relation.NECESSARY_FOR,
        mismatch.get("inputs", x(3.0, 5.0)),
        site=mismatch.get("site", "a"),
        metric=mismatch.get("metric", SEL),
    )
    spec = iv.threshold_spec(operation=InterventionOperation.ZERO, min_effect=1.0)
    r = run(Additive(), x(3.0, 5.0), intervention=iv.zero("a"), claims=[(claim, spec)])
    assert [res.outcome for res in r.claim_results] == [Outcome.NOT_APPLICABLE]


def test_operation_mismatch_is_not_applicable() -> None:
    claim = _claim(Relation.NECESSARY_FOR, x(3.0, 5.0))
    spec = iv.threshold_spec(operation=InterventionOperation.CONSTANT, min_effect=1.0)
    r = run(Additive(), x(3.0, 5.0), intervention=iv.zero("a"), claims=[(claim, spec)])
    assert r.claim_results[0].outcome is Outcome.NOT_APPLICABLE


def test_sufficiency_cannot_be_tested_or_assessed() -> None:
    with pytest.raises(EvidenceRuleError):
        iv.threshold_spec(
            operation=InterventionOperation.ZERO,
            min_effect=1.0,
            relations=(Relation.SUFFICIENT_FOR,),
        )
    claim = _claim(Relation.SUFFICIENT_FOR, x(3.0, 5.0))
    with pytest.raises(EvidenceRuleError, match="names no protocol"):
        Assessment.derive(claim, [], iv.INTERVENTION_POLICY)


def test_protocol_registry_validates_policies() -> None:
    iv.check_policy(iv.INTERVENTION_POLICY)
    with pytest.raises(EvidenceRuleError, match="does not justify"):
        iv.check_policy(
            AssessmentPolicy(
                name="bad",
                version=1,
                requirements=(
                    PolicyRequirement(
                        relation=Relation.SUFFICIENT_FOR, protocols=("intervention_threshold",)
                    ),
                ),
            )
        )
    with pytest.raises(EvidenceRuleError, match="does not justify"):
        iv.check_policy(
            AssessmentPolicy(
                name="bad",
                version=1,
                requirements=(
                    PolicyRequirement(relation=Relation.NECESSARY_FOR, protocols=("vibes",)),
                ),
            )
        )


def test_claim_results_persist_with_the_comparison(tmp_path: Path) -> None:
    inputs = x(3.0, 5.0)
    claim = _claim(Relation.NECESSARY_FOR, inputs)
    spec = iv.threshold_spec(operation=InterventionOperation.ZERO, min_effect=1.0)
    r = run(Additive(), inputs, intervention=iv.zero("a"), claims=[(claim, spec)])
    r.trace.save(tmp_path / "t")
    loaded = iv.InterventionResult.from_trace(bnn.load_trace(tmp_path / "t"))
    assert [res.id for res in loaded.claim_results] == [res.id for res in r.claim_results]
    assert isinstance(loaded.trace.get(claim.id), type(claim))


def test_finite_sample_claims_use_the_sample_estimand() -> None:
    samples = [x(1.0, 0.0), x(2.0, 0.0)]
    r = iv.intervene_sample(Additive().eval(), samples, intervention=iv.zero("a"), metric=SEL)
    assert isinstance(r.effect, CausalEffect)
    claim = _claim(Relation.DECREASES, samples[0])
    assert (
        claim.estimand != r.effect.estimand
    )  # an instance claim cannot be decided by the sample mean


def test_every_baseline_pass_of_a_sample_comparison_is_clean() -> None:
    samples = [x(1.0, 0.0), x(2.0, 0.0)]
    r = iv.intervene_sample(Additive().eval(), samples, intervention=iv.zero("a"), metric=SEL)
    modes = [r.trace.origin(inp).execution.mode for inp in r.trace.inputs]
    assert modes == [ExecutionMode.CLEAN, ExecutionMode.INTERVENTION] * 2


class RandomWhenZero(nn.Module):
    """Consumes randomness only when its sub-module output is zero (i.e. only when intervened)."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(2, 1)
        with torch.no_grad():
            self.a.weight.fill_(1.0)
            self.a.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h: torch.Tensor = self.a(x)
        if bool((h == 0).all()):
            h = h + torch.rand(1)
        return h


def test_randomness_only_in_the_intervened_pass_is_refused() -> None:
    model = RandomWhenZero().eval()
    with pytest.raises(iv.StochasticComparisonError, match="intervention pass"):
        iv.intervene(model, x(1, 2), intervention=iv.zero("a"), metric=SEL)
    assert hook_count(model) == 0


class Caching(nn.Module):
    """Eval-mode module that records its last input in a buffer (output unaffected)."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(2, 1)
        self.register_buffer("last", torch.zeros(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.get_buffer("last").copy_(x)
        out: torch.Tensor = self.a(x)
        return out


def test_state_changes_that_do_not_affect_the_output_still_break_pairing() -> None:
    with pytest.raises(iv.StatefulComparisonError):
        iv.intervene(Caching().eval(), x(1, 2), intervention=iv.zero("a"), metric=SEL)


class InPlaceInput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x.add_(1.0)  # mutates the caller's input
        out: torch.Tensor = self.a(x)
        return out


def test_models_that_mutate_their_inputs_are_refused() -> None:
    model = InPlaceInput().eval()
    with pytest.raises(iv.StatefulComparisonError, match="inputs in place"):
        iv.intervene(model, x(1, 2), intervention=iv.zero("a"), metric=SEL)
    assert hook_count(model) == 0


class TogglesMode(nn.Module):
    """Switches itself to training mode after the first call (execution conditions drift)."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(2, 1)
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if self.calls == 1:
            self.training = True
        out: torch.Tensor = self.a(x)
        return out


def test_execution_condition_drift_is_refused() -> None:
    with pytest.raises(iv.StatefulComparisonError, match="execution conditions"):
        iv.intervene(TogglesMode().eval(), x(1, 2), intervention=iv.zero("a"), metric=SEL)


# ------------------------------------- caller metric identity (Phase 2 hardening)


def _first(o: torch.Tensor) -> float:
    return float(o[0, 0])


def _doubled(o: torch.Tensor) -> float:
    return 2 * float(o[0, 0])


def test_unversioned_custom_metrics_are_refused() -> None:
    with pytest.raises(TypeError):
        iv.metrics.custom("score", _first)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="implementation_revision"):
        iv.metrics.custom("score", _first, implementation_revision="")
    with pytest.raises(ValueError, match="implementation_revision"):
        iv.metrics.custom("score", _first, implementation_revision="has space")


def test_same_name_different_functions_have_distinct_declared_identity() -> None:
    a = iv.metrics.custom("score", _first, implementation_revision="git:aaa")
    b = iv.metrics.custom("score", _doubled, implementation_revision="git:bbb")
    c = iv.metrics.custom("score", _first, implementation_revision="git:aaa", config={"k": 1})
    assert len({a.spec, b.spec, c.spec}) == 3
    assert len({a.spec.target(), b.spec.target(), c.spec.target()}) == 3
    ra = run(Additive(), x(3, 5), intervention=iv.zero("a"), metric=a)
    rb = run(Additive(), x(3, 5), intervention=iv.zero("a"), metric=b)
    assert (ra.value, rb.value) == (-3.0, -6.0)
    assert ra.effect.metric != rb.effect.metric
    assert ra.effect.id != rb.effect.id
    decl = ra.effect.metric.declaration
    assert decl is not None
    assert decl.implementation_revision == "git:aaa"
    assert "CUSTOM_METRIC_UNVERIFIED" in codes(ra)


def test_claims_about_one_declared_revision_do_not_match_another() -> None:
    a = iv.metrics.custom("score", _first, implementation_revision="git:aaa")
    b = iv.metrics.custom("score", _doubled, implementation_revision="git:bbb")
    inputs = x(3.0, 5.0)
    spec = iv.threshold_spec(operation=InterventionOperation.ZERO, min_effect=1.0)
    claim_b = _claim(Relation.NECESSARY_FOR, inputs, metric=b)
    claim_a = _claim(Relation.NECESSARY_FOR, inputs, metric=a)
    claims = [(claim_b, spec), (claim_a, spec)]
    r = run(Additive(), inputs, intervention=iv.zero("a"), metric=a, claims=claims)
    assert [res.outcome for res in r.claim_results] == [Outcome.NOT_APPLICABLE, Outcome.SUPPORTS]


def test_declared_custom_metric_identity_persists(tmp_path: Path) -> None:
    m = iv.metrics.custom("score", _first, implementation_revision="git:aaa", config={"k": 1})
    r = run(Additive(), x(3, 5), intervention=iv.zero("a"), metric=m)
    r.trace.save(tmp_path / "t")
    loaded = iv.InterventionResult.from_trace(bnn.load_trace(tmp_path / "t"))
    assert loaded.effect == r.effect
    assert loaded.effect.metric.declaration == m.spec.declaration


def test_margin_metric_is_the_gap_to_the_best_other_class() -> None:
    import beyondnn as bnn
    from beyondnn._testing.audit_scenarios import WeightedSum

    out = torch.tensor([[1.0, 4.0, 3.0, -2.0]])
    m = bnn.interventions.metrics.margin([0, 1])
    assert m(out) == 1.0
    assert bnn.interventions.metrics.margin([0, 3])(out) == -6.0
    assert float(m.tensor(out)) == 1.0
    assert m.spec.name == "margin"
    with pytest.raises(bnn.interventions.metrics.MetricError):
        bnn.interventions.metrics.margin([0, 0])(torch.tensor([[1.0]]))
    # differentiable: usable as an attribution target and in faithfulness tests
    model = WeightedSum([5.0, 1.0, 1.0]).eval()

    class Two(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.inner(x)
            return torch.cat([y, -y], dim=1)

    two = Two().eval()
    x = torch.tensor([[1.0, 1.0, 1.0]])
    target = bnn.interventions.metrics.margin([0, 0])
    attr = bnn.attribute(two, x, target=target, method=bnn.attribution.gradient())
    test = bnn.faithfulness.comprehensiveness(
        target=target, min_drop=1.0, statement="x0 necessary", replacement=bnn.faithfulness.zero()
    )
    res = bnn.faithfulness.run(
        two, x, test=test, selection=bnn.faithfulness.top_k(attr, k=1), attributions=[attr]
    )
    assert res.outcome.value == "supports"
    bnn.compose(bnn.trace(two, x), attributions=[attr], faithfulness=[res])
