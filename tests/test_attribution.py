"""Phase 3: attribution ground truth, semantics, safety, persistence, claims (ADR-030).

Expected values and tolerances are the ones pre-registered in docs/PHASE_3_PLAN.md.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import beyondnn as bnn
import beyondnn.attribution as A
import beyondnn.interventions as iv
from beyondnn._testing.attribution_models import (
    Aliased,
    Irrelevant,
    Linear,
    Product,
    Saturating,
    Twice,
)
from beyondnn._testing.causal_models import Redundant
from beyondnn._testing.models import TinyMLP, TinyTransformer
from beyondnn.core.hooks import AliasSiteAmbiguityError
from beyondnn.core.trace import ExternalForwardHooksError, TraceIntegrityError, TraceResult
from beyondnn.explain import ExplainResponse
from beyondnn.interventions.metrics import MetricError
from beyondnn.protocols import check_policy
from beyondnn.schema import (
    Assessment,
    AssessmentPolicy,
    AttributionRecord,
    CausalEffect,
    ClaimTestResult,
    ClaimTestSpec,
    Estimand,
    EvidenceRef,
    EvidenceRuleError,
    EvidenceStatus,
    InterventionOperation,
    JsonMap,
    ModelDeclaration,
    Outcome,
    PolicyRequirement,
    RecordRef,
    Relation,
    Verdict,
)

SEL = iv.metrics.select([0, 0])
ATOL = 1e-5  # declared in docs/PHASE_3_PLAN.md (float32 forward/backward)


def x(a: float, b: float, **kw: Any) -> torch.Tensor:
    return torch.tensor([[a, b]], dtype=torch.float32, **kw)


def run(model: nn.Module, *inputs: Any, **kw: Any) -> A.AttributionResult:
    kw.setdefault("target", SEL)
    return A.attribute(model.eval(), *inputs, **kw)


def ig(n: int = 16, rule: str = "riemann_middle", baseline: A.Baseline | None = None) -> A.Method:
    return A.integrated_gradients(
        baseline=baseline if baseline is not None else A.zero_baseline(), n_steps=n, rule=rule
    )


def close(t: torch.Tensor, expected: Sequence[float], atol: float = ATOL) -> bool:
    return torch.allclose(
        t.double().flatten(), torch.tensor(expected, dtype=torch.float64), atol=atol, rtol=0
    )


def codes(r: A.AttributionResult) -> set[str]:
    return {lim.code for lim in r.limitations}


# ------------------------------------------------- ground truth (docs/PHASE_3_PLAN.md)


def test_linear_gradient_input_x_gradient_and_ig() -> None:
    assert close(run(Linear(), x(3, 5), method=A.gradient()).value, [2, 3])
    assert close(run(Linear(), x(3, 5), method=A.input_x_gradient()).value, [6, 15])
    for rule in ("riemann_left", "riemann_right", "riemann_middle", "trapezoid"):
        r = run(Linear(), x(3, 5), method=ig(4, rule))
        assert close(r.value, [6, 15]), rule
        assert r.completeness_delta is not None
        assert abs(r.completeness_delta) < ATOL


def test_irrelevant_input_gets_exactly_zero() -> None:
    for method, expected in (
        (A.gradient(), [4, 0]),
        (A.input_x_gradient(), [12, 0]),
        (ig(8), [12, 0]),
    ):
        value = run(Irrelevant(), x(3, 5), method=method).value
        assert close(value, expected)
        assert value[0, 1].item() == 0.0


def test_product_model_attribution_depends_on_input_and_baseline() -> None:
    assert close(run(Product(), x(3, 5), method=A.gradient()).value, [5, 3])
    assert close(run(Product(), x(3, 5), method=A.input_x_gradient()).value, [15, 15])
    for rule, n in (("riemann_middle", 3), ("trapezoid", 2)):  # exact: linear integrand
        assert close(run(Product(), x(3, 5), method=ig(n, rule)).value, [7.5, 7.5])
    shifted = run(Product(), x(3, 5), method=ig(4, baseline=A.baseline(x(1, 2))))
    assert close(shifted.value, [7, 6])  # [(x0-b0)(x1+b1)/2, (x1-b1)(x0+b0)/2]
    assert float(shifted.value.sum()) == pytest.approx(15 - 2, abs=ATOL)
    context = run(Product(), x(3, 0), method=A.gradient())
    assert close(context.value, [0, 3])
    assert close(run(Product(), x(3, 0), method=ig(4)).value, [0, 0])


@pytest.mark.parametrize(("rule", "sign"), [("riemann_left", -1), ("riemann_right", 1)])
def test_one_sided_rules_have_the_analytic_completeness_error(rule: str, sign: int) -> None:
    # Left/right sums of the linear integrand 15*alpha miss the integral by -+15/(2n)
    # per coordinate, so the completeness delta is exactly sign * x0*x1 / n.
    for n in (4, 10):
        r = run(Product(), x(3, 5), method=ig(n, rule))
        assert r.completeness_delta == pytest.approx(sign * 15 / n, abs=ATOL)


def test_saturating_model_separates_gradient_from_ig_within_the_declared_bound() -> None:
    sech2 = 1 / math.cosh(3) ** 2
    grad = run(Saturating(), x(3, 1), method=A.gradient()).value
    assert close(grad, [sech2, 1], atol=1e-6)
    ixg = run(Saturating(), x(3, 1), method=A.input_x_gradient()).value
    assert close(ixg, [3 * sech2, 1], atol=1e-6)
    for n in (16, 64, 256):
        value = run(Saturating(), x(3, 1), method=ig(n)).value
        bound = 2.25 / n**2 + 1e-5  # midpoint-rule bound, declared before running
        assert abs(value[0, 0].item() - math.tanh(3)) <= bound, n
        assert value[0, 1].item() == pytest.approx(1.0, abs=ATOL)
    assert grad[0, 0].item() < 0.01 < 0.99 < value[0, 0].item()  # saturation


# ------------------------------------------------- status, provenance, identity


def test_attribution_records_are_attributed_and_nothing_else_changes_status() -> None:
    r = run(Redundant(), x(3, 5), method=ig(4), sites=["**"])
    assert r.record.status is EvidenceStatus.ATTRIBUTED
    ref = EvidenceRef.to(r.record)
    assert ref.status is EvidenceStatus.ATTRIBUTED
    assert ref.estimand is None  # not causal evidence
    statuses = {type(rec).__name__: rec.status for rec in r.trace.records}
    assert statuses["InputRecord"] is EvidenceStatus.OBSERVED
    assert statuses["OutputRecord"] is EvidenceStatus.OBSERVED
    assert {rec.status for rec in r.trace.records} <= {
        None,
        EvidenceStatus.OBSERVED,
        EvidenceStatus.MEASURED,
        EvidenceStatus.ATTRIBUTED,
    }
    with pytest.raises(TypeError):
        AttributionRecord(status=EvidenceStatus.INTERVENTIONAL)  # type: ignore[call-arg]


def test_provenance_links_model_method_target_input_and_conditions() -> None:
    inputs = x(3, 5)
    declared = ModelDeclaration(implementation_revision="git:abc")
    r = run(Product(), inputs, method=ig(8), declared_model=declared)
    record = r.record
    prov = r.trace.origin(record)
    ref = r.trace.origin(r.trace.output)
    assert prov.model == ref.model
    assert prov.environment == ref.environment
    assert prov.declared_model == declared
    assert prov.execution.grad_enabled
    assert not prov.execution.training
    assert prov.method.name == "beyondnn:integrated_gradients"
    assert prov.method.params == JsonMap({"n_steps": 8, "rule": "riemann_middle"})
    assert record.sample_id == iv.sample_id(inputs)
    assert record.target == SEL.spec
    assert record.target_value == 15.0
    assert {d.kind for d in record.derived_from} == {"input", "output"}


def test_identity_includes_every_scientific_choice() -> None:
    base = run(Product(), x(3, 5), method=ig(8)).record.id
    variants = [
        run(Product(), x(3, 5), method=ig(9)).record.id,
        run(Product(), x(3, 5), method=ig(8, "trapezoid")).record.id,
        run(Product(), x(3, 5), method=ig(8, baseline=A.baseline(x(1, 2)))).record.id,
        run(Product(), x(3, 4), method=ig(8)).record.id,
        run(Product(), x(3, 5), method=ig(8), target=iv.metrics.mean()).record.id,
    ]
    assert base == run(Product(), x(3, 5), method=ig(8)).record.id  # deterministic
    assert len({base, *variants}) == 6


# ------------------------------------------------- targets and baselines


def test_target_is_explicit_and_changes_the_attribution() -> None:
    model = TinyMLP().eval()
    inputs = torch.linspace(-1, 1, 4).reshape(1, 4)
    a = run(model, inputs, method=A.gradient(), target=iv.metrics.select([0, 0]))
    b = run(model, inputs, method=A.gradient(), target=iv.metrics.select([0, 2]))
    d = run(model, inputs, method=A.gradient(), target=iv.metrics.difference([0, 0], [0, 2]))
    assert not torch.allclose(a.value, b.value)
    assert torch.allclose(d.value, a.value - b.value, atol=1e-6)
    assert a.record.target != b.record.target


@pytest.mark.parametrize(
    "target",
    [None, torch.zeros(()), "logit", 0],
)
def test_a_target_must_be_a_metric(target: Any) -> None:
    with pytest.raises(TypeError, match="scalar Metric"):
        A.attribute(Linear().eval(), x(3, 5), target=target, method=A.gradient())


def test_non_scalar_and_caller_targets_are_refused_never_summed() -> None:
    model = TinyMLP().eval()
    inputs = torch.ones(1, 4)
    with pytest.raises(MetricError, match="single element"):
        A.attribute(model, inputs, target=iv.metrics.select([0]), method=A.gradient())
    caller = iv.metrics.custom("s", lambda o: float(o.sum()), implementation_revision="v1")
    with pytest.raises(A.AttributionError, match="caller metrics"):
        A.attribute(model, inputs, target=caller, method=A.gradient())


def test_ig_needs_an_explicit_baseline_of_exactly_matching_shape_and_dtype() -> None:
    with pytest.raises(TypeError):
        A.integrated_gradients()  # type: ignore[call-arg]
    with pytest.raises(A.AttributionError, match="no broadcasting"):
        run(Product(), x(3, 5), method=ig(4, baseline=A.baseline(torch.zeros(2))))
    with pytest.raises(A.AttributionError, match="no broadcasting"):
        run(Product(), x(3, 5), method=ig(4, baseline=A.baseline(x(0, 0).double())))
    with pytest.raises(ValueError, match="finite"):
        A.baseline(x(math.nan, 0))
    with pytest.raises(A.AttributionError, match="layer attribution"):
        run(Product(), x(3, 5), method=ig(4, baseline=A.input_baseline(x(0, 0))))


def test_baselines_are_recorded_with_limitations() -> None:
    zero = run(Product(), x(3, 5), method=ig(4))
    assert zero.record.baseline is not None
    assert zero.record.baseline.kind.value == "zero"
    assert {"ATTRIBUTION_BASELINE_ASSUMPTION", "ATTRIBUTION_NUMERICAL_APPROXIMATION"} <= codes(zero)
    (lim,) = [lim for lim in zero.limitations if lim.code == "ATTRIBUTION_BASELINE_ASSUMPTION"]
    assert lim.applies_to == (zero.record.id,)
    assert lim.detail is not None
    assert "zero" in lim.detail
    tensor = run(Product(), x(3, 5), method=ig(4, baseline=A.baseline(x(1, 2))))
    assert tensor.record.baseline is not None
    assert tensor.record.baseline.value is not None
    stored = tensor.trace.tensor(tensor.record.baseline.value)
    assert torch.equal(stored, x(1, 2))
    plain = run(Product(), x(3, 5), method=A.gradient())
    assert plain.record.baseline is None
    assert not codes(plain) & {
        "ATTRIBUTION_BASELINE_ASSUMPTION",
        "ATTRIBUTION_NUMERICAL_APPROXIMATION",
    }


def test_invalid_method_configurations_are_refused() -> None:
    with pytest.raises(ValueError, match="n_steps"):
        ig(0)
    with pytest.raises(ValueError, match="rule"):
        ig(4, "riemann_trapezoid")  # a Captum rule name with different weights
    with pytest.raises(Exception, match="n_steps >= 2"):
        ig(1, "trapezoid")
    with pytest.raises(TypeError):
        A.attribute(Linear().eval(), x(1, 2), target=SEL, method="ig")  # type: ignore[arg-type]


# ------------------------------------------------- inputs, layers, calls, aliases


def test_integer_token_ids_are_not_differentiable_inputs() -> None:
    ids = torch.tensor([[1, 2, 3, 4]])
    target = iv.metrics.select([0, 3, 5])
    with pytest.raises(A.DiscreteInputError, match="token_embedding"):
        A.attribute(TinyTransformer().eval(), ids, target=target, method=A.gradient())


def test_integer_inputs_of_any_model_are_refused_for_input_attribution() -> None:
    with pytest.raises(A.DiscreteInputError, match="discrete"):
        run(Irrelevant(), torch.tensor([[1, 2]]), method=ig(4))


def test_embedding_layer_attribution_is_explicitly_not_token_attribution() -> None:
    ids = torch.tensor([[1, 2, 3, 4]])
    r = A.attribute(
        TinyTransformer().eval(),
        ids,
        target=iv.metrics.select([0, 3, 5]),
        method=A.integrated_gradients(baseline=A.input_baseline(torch.zeros_like(ids)), n_steps=16),
        at=A.layer("token_embedding"),
        reductions=[A.reduce("sum", (-1,))],
    )
    assert tuple(r.value.shape) == (1, 4, 16)
    assert {
        "DISCRETE_INPUT_ATTRIBUTED_VIA_REPRESENTATION",
        "LAYER_ATTRIBUTION_PARTIAL_COVERAGE",
    } <= codes(r)
    (reduction,) = r.reductions
    assert reduction.status is EvidenceStatus.ATTRIBUTED
    assert reduction.dims == (2,)
    assert reduction.scalar is None
    assert torch.allclose(r.reduced(reduction), r.value.double().sum(-1))
    assert r.record.baseline is not None
    assert r.record.baseline.input_path == "args[0]"
    assert r.completeness_delta is not None
    total = float(r.value.double().sum())
    base_value = r.record.diagnostics["baseline_target_value"]
    assert isinstance(base_value, float)
    assert r.completeness_delta == pytest.approx(total - (r.target_value - base_value), abs=1e-9)


def test_nothing_is_reduced_unless_declared() -> None:
    r = run(TinyMLP(), torch.ones(1, 4), method=A.gradient(), target=iv.metrics.select([0, 1]))
    assert r.reductions == ()
    full = run(
        Product(),
        x(3, -5),
        method=A.input_x_gradient(),
        reductions=[A.reduce("abs_sum", (0, 1)), A.reduce("l2", (-1,)), A.reduce("sum", [1])],
    )
    abs_sum, l2, total = full.reductions
    assert abs_sum.scalar == 30.0
    assert l2.scalar is None
    assert close(full.reduced(l2), [math.sqrt(2 * 15**2)])
    assert close(full.reduced(total), [-30])
    with pytest.raises(A.AttributionError, match="out of range"):
        run(Product(), x(3, 5), method=A.gradient(), reductions=[A.reduce("sum", (2,))])
    with pytest.raises(A.AttributionError, match="repeat"):
        run(Product(), x(3, 5), method=A.gradient(), reductions=[A.reduce("sum", (1, -1))])
    with pytest.raises(ValueError, match="reduction"):
        A.reduce("mean", (0,))


def test_repeated_module_calls_are_attributed_to_the_declared_call() -> None:
    first = run(Twice(), torch.tensor([[3.0]]), method=A.gradient(), at=A.layer("lin"))
    second = run(
        Twice(), torch.tensor([[3.0]]), method=A.gradient(), at=A.layer("lin", call_index=1)
    )
    assert close(first.value, [2.0])
    assert close(second.value, [1.0])
    for r, call in ((first, 0), (second, 1)):
        (activation,) = [
            r.trace.get(d.record_id) for d in r.record.derived_from if d.kind == "activation"
        ]
        assert r.record.call_index == call
        assert activation.call_index == call  # type: ignore[attr-defined]
    ixg = run(Twice(), torch.tensor([[3.0]]), method=A.input_x_gradient(), at=A.layer("lin"))
    assert close(ixg.value, [6.0 * 2.0])
    with pytest.raises(A.RepeatedCallError, match="did not run"):
        run(Twice(), torch.tensor([[3.0]]), method=A.gradient(), at=A.layer("lin", call_index=2))


def test_the_container_rejects_an_attribution_assigned_to_another_call() -> None:
    r = run(Twice(), torch.tensor([[3.0]]), method=A.gradient(), at=A.layer("lin"))
    wrong = dataclasses.replace(r.record, call_index=1)
    fresh = TraceResult(r.trace.config)
    for record in r.trace.records:
        if not isinstance(record, (AttributionRecord,)) and record.KIND != "limitation":
            fresh._add(record)
    with pytest.raises(TraceIntegrityError, match="site/call/pass"):
        fresh._add(wrong)


@pytest.mark.parametrize("path", ["first", "alias"])
def test_alias_paths_are_refused_before_any_attribution_runs(path: str) -> None:
    with pytest.raises(AliasSiteAmbiguityError, match="attribution to one path"):
        run(Aliased(), x(1, 2), method=A.gradient(), at=A.layer(path))


def test_layer_attribution_and_its_limitations() -> None:
    r = run(Redundant(), x(3, 5), method=A.input_x_gradient(), at=A.layer("p"))
    assert close(r.value, [3.0])
    assert "LAYER_ATTRIBUTION_PARTIAL_COVERAGE" in codes(r)
    assert "DISCRETE_INPUT_ATTRIBUTED_VIA_REPRESENTATION" not in codes(r)
    assert r.record.site.module == "p"
    with pytest.raises(ValueError, match="pattern"):
        A.layer("*")
    with pytest.raises(A.AttributionError, match="no tensor leaf"):
        run(Redundant(), x(3, 5), method=A.gradient(), at=A.layer("p", output_path="[0]"))


# ------------------------------------------------- autograd and model state


def _grads(model: nn.Module) -> list[torch.Tensor | None]:
    return [p.grad for p in model.parameters()]


def test_no_autograd_side_effects_on_caller_tensors_or_parameters() -> None:
    model = TinyMLP().eval()
    inputs = torch.linspace(-1, 1, 4).reshape(1, 4).requires_grad_(True)
    model(inputs).sum().backward()  # pre-existing grads on inputs and parameters
    before_params = [None if g is None else g.clone() for g in _grads(model)]
    grad_objects = _grads(model)
    input_grad = inputs.grad
    assert input_grad is not None
    input_grad_value = input_grad.clone()
    values = inputs.detach().clone()
    for method in (A.gradient(), A.input_x_gradient(), ig(4)):
        run(model, inputs, method=method, target=iv.metrics.select([0, 1]))
        run(
            model, inputs, method=method, target=iv.metrics.select([0, 1]), at=A.layer("projection")
        )
    assert all(a is b for a, b in zip(_grads(model), grad_objects, strict=True))
    assert all(
        b is not None and a is not None and torch.equal(a, b)
        for a, b in zip(_grads(model), before_params, strict=True)
    )
    assert inputs.grad is input_grad
    assert torch.equal(inputs.grad, input_grad_value)
    assert inputs.requires_grad
    assert torch.equal(inputs.detach(), values)
    fresh = TinyMLP().eval()
    plain = torch.ones(1, 4)
    run(fresh, plain, method=A.gradient(), target=iv.metrics.select([0, 1]))
    assert all(g is None for g in _grads(fresh))
    assert not plain.requires_grad
    assert plain.grad is None


class _WritesGrad(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.lin.weight.grad = torch.ones_like(self.lin.weight)
        out: torch.Tensor = self.lin(x)
        return out


@pytest.mark.parametrize("at", [A.input(), A.layer("lin")])
def test_parameter_grad_changes_are_restored_and_refused(at: A.At) -> None:
    model = _WritesGrad().eval()
    with pytest.raises(A.AutogradStateError, match="restored"):
        run(model, x(1, 2), method=A.gradient(), at=at)
    assert model.lin.weight.grad is None


class _Counting(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(2, 1)
        self.register_buffer("calls", torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.get_buffer("calls").add_(1)
        out: torch.Tensor = self.lin(x)
        return out


class _Noisy(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :1] + torch.rand(1, 1)


class _Drifting(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.count = 0  # plain Python state: invisible to the fingerprint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.count += 1
        return x[:, :1] * self.count


class _InPlace(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x.add_(1)
        out: torch.Tensor = self.lin(x)
        return out


def test_state_and_randomness_are_refused_not_averaged() -> None:
    with pytest.raises(A.StatefulAttributionError, match="state"):
        run(_Counting(), x(1, 2), method=A.gradient())
    with pytest.raises(A.StochasticAttributionError, match="random"):
        run(_Noisy(), x(1, 2), method=A.gradient())
    with pytest.raises(A.StochasticAttributionError, match="reproducible"):
        run(_Drifting(), x(1, 2), method=A.gradient())
    with pytest.raises(A.StatefulAttributionError, match="training"):
        A.attribute(Linear().train(), x(1, 2), target=SEL, method=A.gradient())
    model = Linear().eval()
    flags = [m.training for m in model.modules()]
    run(model, x(1, 2), method=A.gradient())
    assert [m.training for m in model.modules()] == flags


class _MarksInput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_leaf:
            x.requires_grad_(True)
        out: torch.Tensor = self.lin(x)
        return out


@pytest.mark.parametrize("method", [A.gradient(), ig(2)])
def test_a_change_to_caller_autograd_state_is_refused(method: A.Method) -> None:
    inputs = x(1, 2)
    with pytest.raises(A.AutogradStateError, match="autograd state"):
        run(_MarksInput(), inputs, method=method, at=A.layer("lin"))


class _GradModeState(nn.Module):
    """Updates a buffer only in grad mode: the traced (no_grad) pass looks clean."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(2, 1)
        self.register_buffer("seen", torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            self.get_buffer("seen").add_(1)
        out: torch.Tensor = self.lin(x)
        return out


def test_a_state_change_during_the_attribution_passes_is_refused() -> None:
    with pytest.raises(A.StatefulAttributionError, match="during attribution"):
        run(_GradModeState(), x(1, 2), method=A.gradient())


def test_in_place_input_modification_is_refused() -> None:
    inputs = x(1, 2)
    with pytest.raises(A.AutogradStateError, match="in place"):
        run(_InPlace(), inputs, method=A.gradient(), at=A.layer("lin"))


def hook_count(model: nn.Module) -> int:
    return sum(
        len(m._forward_hooks) + len(m._forward_pre_hooks) + len(m._backward_hooks)
        for m in model.modules()
    )


def test_foreign_forward_backward_and_tensor_hooks_are_refused() -> None:
    model = Linear().eval()
    model.lin.register_forward_hook(lambda m, a, o: None)
    with pytest.raises(ExternalForwardHooksError):
        run(model, x(1, 2), method=A.gradient())
    model = Linear().eval()
    model.lin.register_full_backward_hook(lambda m, gi, go: None)
    with pytest.raises(ExternalForwardHooksError, match="backward"):
        run(model, x(1, 2), method=A.gradient())
    model = Linear().eval()
    model.lin.weight.register_hook(lambda g: g * 2)  # type: ignore[no-untyped-call]
    with pytest.raises(ExternalForwardHooksError, match="parameter"):
        run(model, x(1, 2), method=A.gradient())


@pytest.mark.parametrize(
    "failing",
    [
        {"at": A.layer("lin", call_index=5)},
        {"at": A.layer("lin", output_path="[0]")},
        {"target": iv.metrics.select([0, 9])},
        {"reductions": [A.reduce("sum", (7,))]},
    ],
)
def test_no_hooks_leak_on_failure(failing: dict[str, Any]) -> None:
    model = Twice().eval()
    kw: dict[str, Any] = {"method": A.gradient(), "at": A.layer("lin")} | failing
    with pytest.raises((A.AttributionError, ValueError)):
        run(model, torch.tensor([[3.0]]), **kw)
    assert hook_count(model) == 0
    run(model, torch.tensor([[3.0]]), method=A.gradient(), at=A.layer("lin"))


# ------------------------------------------------- persistence


def test_attribution_round_trips_through_the_trace_format(tmp_path: Path) -> None:
    r = run(
        Product(),
        x(3, 5),
        method=ig(8, baseline=A.baseline(x(1, 2))),
        reductions=[A.reduce("abs_sum", (0, 1))],
    )
    r.trace.save(tmp_path / "t")
    loaded = A.AttributionResult.from_trace(bnn.load_trace(tmp_path / "t"))
    assert loaded.record == r.record
    assert torch.equal(loaded.value, r.value)
    assert loaded.reductions == r.reductions
    assert [lim.id for lim in loaded.limitations] == [lim.id for lim in r.limitations]
    assert sorted(p.name for p in (tmp_path / "t").iterdir()) == ["tensors.pt", "trace.json"]


def test_tampered_attribution_tensor_is_rejected_on_load(tmp_path: Path) -> None:
    r = run(Product(), x(3, 5), method=A.gradient())
    r.trace.save(tmp_path / "t")
    tensors = torch.load(tmp_path / "t" / "tensors.pt", weights_only=True)
    key = r.record.value.storage_key
    tensors[key] = tensors[key] + 1
    (tmp_path / "t" / "tensors.pt").unlink()
    torch.save(tensors, tmp_path / "t" / "tensors.pt")
    with pytest.raises(Exception, match="content digest"):
        bnn.load_trace(tmp_path / "t")


# ------------------------------------------------- claims: attribution is not causation


def _claim(
    inputs: torch.Tensor, units: tuple[int, ...] | None = None, at: A.At | None = None
) -> Any:
    return A.make_claim(at or A.input(), SEL, inputs, statement="x0 attributed", units=units)


def test_attribution_threshold_decides_only_attributed_to() -> None:
    method = ig(4)
    inputs = x(3, 5)
    low = A.threshold_spec(method, min_abs_attribution=7.0)
    high = A.threshold_spec(method, min_abs_attribution=8.0)
    claim = _claim(inputs, units=(0,))
    r = run(Product(), inputs, method=method, claims=[(claim, low), (claim, high)])
    assert [res.outcome for res in r.claim_results] == [Outcome.SUPPORTS, Outcome.CONTRADICTS]
    supports = [res for res in r.claim_results if res.outcome is Outcome.SUPPORTS]
    assert Assessment.derive(claim, supports, A.ATTRIBUTION_POLICY).verdict is Verdict.SUPPORTED
    assert all(ev.status is EvidenceStatus.ATTRIBUTED for ev in supports[0].evidence)
    assert low.applicable_relations == (Relation.ATTRIBUTED_TO,)


@pytest.mark.parametrize(
    "mismatch",
    [
        {"spec_method": ig(8)},  # declared configuration differs from the one run
        {"spec_method": ig(4, "trapezoid")},
        {"spec_method": ig(4, baseline=A.baseline(x(1, 1)))},  # declared baseline differs
        {"spec_method": A.gradient()},
        {"inputs": x(3, 4)},
        {"at": A.layer("lin")},
        {"units": (5,)},
    ],
)
def test_mismatched_attribution_claims_are_not_applicable(mismatch: dict[str, Any]) -> None:
    method = ig(4)
    spec = A.threshold_spec(mismatch.get("spec_method", method), min_abs_attribution=1.0)
    claim = _claim(mismatch.get("inputs", x(3, 5)), mismatch.get("units"), mismatch.get("at"))
    r = run(Linear(), x(3, 5), method=method, claims=[(claim, spec)])
    assert [res.outcome for res in r.claim_results] == [Outcome.NOT_APPLICABLE]


def test_the_registry_lets_attribution_justify_only_attributed_to() -> None:
    from beyondnn.protocols import ATTRIBUTION_THRESHOLD, PROTOCOLS

    assert PROTOCOLS[ATTRIBUTION_THRESHOLD] == frozenset({Relation.ATTRIBUTED_TO})
    check_policy(A.ATTRIBUTION_POLICY)


def test_call_index_mismatch_is_not_applicable() -> None:
    at1 = A.layer("lin", call_index=1)
    spec = A.threshold_spec(A.gradient(), at=A.layer("lin"), min_abs_attribution=0.5)
    claim = A.make_claim(at1, SEL, torch.tensor([[3.0]]), statement="call 1")
    r = run(Twice(), torch.tensor([[3.0]]), method=A.gradient(), at=at1, claims=[(claim, spec)])
    assert [res.outcome for res in r.claim_results] == [Outcome.NOT_APPLICABLE]


def test_attribution_can_never_support_a_causal_relation() -> None:
    r = run(Product(), x(3, 5), method=ig(4))
    necessity = iv.make_claim(iv.zero("lin"), SEL, Relation.NECESSARY_FOR, x(3, 5), statement="n")
    causal_spec = ClaimTestSpec(
        protocol="attribution_threshold",
        protocol_version=1,
        applicable_relations=(Relation.NECESSARY_FOR,),
        criteria=JsonMap({"min_abs_attribution": 1.0}),
    )
    with pytest.raises(EvidenceRuleError, match="INTERVENTIONAL or ESTIMATED_CAUSAL"):
        ClaimTestResult.for_claim(
            necessity,
            causal_spec,
            outcome=Outcome.SUPPORTS,
            evidence=(r.record,),
            provenance_id=r.record.provenance_id or "",
        )
    bad_policy = AssessmentPolicy(
        name="bad",
        version=1,
        requirements=(
            PolicyRequirement(
                relation=Relation.NECESSARY_FOR, protocols=("attribution_threshold",)
            ),
        ),
    )
    with pytest.raises(EvidenceRuleError, match="does not justify"):
        check_policy(bad_policy)
    spec = A.threshold_spec(ig(4), min_abs_attribution=1.0)
    r2 = run(Product(), x(3, 5), method=ig(4), claims=[(necessity, spec)])
    assert [res.outcome for res in r2.claim_results] == [Outcome.NOT_APPLICABLE]
    with pytest.raises(EvidenceRuleError, match="names no protocol"):
        Assessment.derive(necessity, r2.claim_results, A.ATTRIBUTION_POLICY)
    for relation in (Relation.NECESSARY_FOR, Relation.SUFFICIENT_FOR):
        assert relation not in {req.relation for req in A.ATTRIBUTION_POLICY.requirements}


def test_high_attribution_is_not_necessity_on_the_redundant_model() -> None:
    inputs = x(3, 5)
    attributed = run(
        Redundant(),
        inputs,
        method=ig(8, baseline=A.zero_baseline()),
        at=A.layer("p"),
        claims=[
            (
                A.make_claim(A.layer("p"), SEL, inputs, statement="p attributed"),
                A.threshold_spec(ig(8), at=A.layer("p"), min_abs_attribution=2.0),
            )
        ],
    )
    assert close(attributed.value, [3.0])  # substantial attribution to p
    assert [res.outcome for res in attributed.claim_results] == [Outcome.SUPPORTS]
    necessity = iv.make_claim(iv.zero("p"), SEL, Relation.NECESSARY_FOR, inputs, statement="p")
    ablated = iv.intervene(
        Redundant().eval(),
        inputs,
        intervention=iv.zero("p"),
        metric=SEL,
        claims=[
            (necessity, iv.threshold_spec(operation=InterventionOperation.ZERO, min_effect=6.0))
        ],
    )
    assert ablated.intervention_value == 3.0  # the output survives ablation of p
    assert [res.outcome for res in ablated.claim_results] == [Outcome.CONTRADICTS]
    # Each record keeps its own meaning; nothing merges them into a stronger status.
    assert attributed.record.status is EvidenceStatus.ATTRIBUTED
    assert ablated.effect.status is EvidenceStatus.INTERVENTIONAL
    with pytest.raises(EvidenceRuleError, match="cannot derive"):
        CausalEffect(
            interventions=ablated.effect.interventions,
            metric=SEL.spec,
            estimand=Estimand.instance("sha256:x"),
            baseline_value=6.0,
            intervention_value=3.0,
            effect=-3.0,
            provenance_id="provenance:" + "a" * 32,
            derived_from=(*ablated.effect.derived_from, RecordRef.to(attributed.record)),
        )


# ------------------------------------------------- explain integration and API


def test_attribution_can_appear_in_structured_why_only_when_run() -> None:
    r = run(Product(), x(3, 5), method=A.gradient())
    why = ExplainResponse.from_trace(r.trace).why
    assert why.attributions == (r.record,)
    lims = {lim.code for lim in why.limitations}
    assert "NO_ATTRIBUTION" not in lims
    assert "NO_CAUSAL_EVIDENCE" in lims
    plain = bnn.instrument(Linear().eval()).explain(x(1, 2))
    assert plain.why.attributions == ()
    assert "NO_ATTRIBUTION" in {lim.code for lim in plain.why.limitations}


def test_top_level_and_handle_entry_points() -> None:
    model = Linear().eval()
    a = bnn.attribute(model, x(3, 5), target=SEL, method=A.gradient())
    b = bnn.instrument(model).attribute(x(3, 5), target=SEL, method=A.gradient())
    assert a.record == b.record
    assert bnn.attribution is A


class _KernelDrift(nn.Module):
    """Deterministic, but the grad-enabled forward differs from the no-grad one by
    ``ulps`` rounding units of logits near 1000: what different kernels do on a trained
    CNN (Phase 5.5 model B; ADR-036). The margin logit0 - logit1 is 0.5."""

    def __init__(self, ulps: float) -> None:
        super().__init__()
        self.ulps = ulps

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        base = torch.cat([1000.5 + 0 * t[:, :1], 1000.0 + t[:, 1:2]], dim=1)
        if torch.is_grad_enabled():
            return base + torch.tensor([[self.ulps * 6.103515625e-05, 0.0]])
        return base


def test_rounding_level_kernel_differences_are_not_randomness() -> None:
    margin = iv.metrics.difference([0, 0], [0, 1])
    r = A.attribute(_KernelDrift(1).eval(), x(0, 0), target=margin, method=A.gradient())
    assert r.target_value == 0.5
    with pytest.raises(A.StochasticAttributionError, match="tolerance"):
        A.attribute(_KernelDrift(1000).eval(), x(0, 0), target=margin, method=A.gradient())
