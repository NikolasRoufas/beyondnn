"""Phase 3: Captum adapter and native/Captum cross-checks (skipped without Captum; the
optional-dependency contract is tested in test_attribution_optional.py).

Cross-check tolerances are the ones declared in docs/PHASE_3_PLAN.md before running:
IG rtol=1e-4, atol=1e-5; gradient and input x gradient atol=1e-6.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

import beyondnn.attribution as A
import beyondnn.attribution.captum as C
import beyondnn.interventions as iv
from beyondnn._testing.attribution_models import Product, Saturating, Twice
from beyondnn._testing.causal_models import Redundant
from beyondnn._testing.models import TinyCNN, TinyMLP, TinyTransformer

SEL = iv.metrics.select([0, 0])
EQUIVALENT_RULES = ("riemann_left", "riemann_right", "riemann_middle")


# ------------------------------------------------- with Captum

captum = pytest.importorskip("captum", reason="Captum not installed (optional extra)")


def run(model: torch.nn.Module, *inputs: Any, **kw: Any) -> A.AttributionResult:
    kw.setdefault("target", SEL)
    return A.attribute(model.eval(), *inputs, **kw)


def both(rule: str, n: int, baseline: A.Baseline) -> tuple[A.Method, A.Method]:
    return (
        A.integrated_gradients(baseline=baseline, n_steps=n, rule=rule),
        C.integrated_gradients(baseline=baseline, n_steps=n, rule=rule),
    )


CASES = [
    (Product, torch.tensor([[3.0, 5.0]]), SEL),
    (Saturating, torch.tensor([[3.0, 1.0]]), SEL),
    (TinyMLP, torch.linspace(-1, 1, 4).reshape(1, 4), iv.metrics.select([0, 1])),
    (TinyCNN, torch.linspace(-1, 1, 64).reshape(1, 1, 8, 8), iv.metrics.select([0, 2])),
]


@pytest.mark.parametrize("rule", EQUIVALENT_RULES)
@pytest.mark.parametrize(("model_cls", "inputs", "target"), CASES)
def test_native_and_captum_ig_agree_under_equivalent_settings(
    model_cls: Any, inputs: torch.Tensor, target: Any, rule: str
) -> None:
    native, adapted = both(rule, 16, A.zero_baseline())
    a = run(model_cls(), inputs, target=target, method=native)
    b = run(model_cls(), inputs, target=target, method=adapted)
    assert torch.allclose(a.value, b.value, rtol=1e-4, atol=1e-5)
    assert b.record.method.implementation == "captum"
    assert b.record.method.implementation_version == captum.__version__
    assert b.record.method.params["captum_class"] == "IntegratedGradients"
    assert a.record.id != b.record.id  # different implementations are different evidence


@pytest.mark.parametrize(("model_cls", "inputs", "target"), CASES)
def test_saliency_and_input_x_gradient_match_native(
    model_cls: Any, inputs: torch.Tensor, target: Any
) -> None:
    for native, adapted in (
        (A.gradient(), C.saliency()),
        (A.input_x_gradient(), C.input_x_gradient()),
    ):
        a = run(model_cls(), inputs, target=target, method=native)
        b = run(model_cls(), inputs, target=target, method=adapted)
        assert torch.allclose(a.value, b.value, rtol=0, atol=1e-6)
    signed = run(Product(), torch.tensor([[3.0, -5.0]]), method=C.saliency()).value
    assert signed[0, 0].item() == pytest.approx(-5.0)  # abs=False: the raw gradient


def test_captum_trapezoid_is_a_documented_convention_difference() -> None:
    n = 10
    captum_trap = run(
        Product(),
        torch.tensor([[3.0, 5.0]]),
        method=C.integrated_gradients(
            baseline=A.zero_baseline(), n_steps=n, rule="riemann_trapezoid"
        ),
    )
    native_trap = run(
        Product(),
        torch.tensor([[3.0, 5.0]]),
        method=A.integrated_gradients(baseline=A.zero_baseline(), n_steps=n, rule="trapezoid"),
    )
    expected = 15 * (0.5 - 1 / (2 * n))  # Captum weights 1/n, ends halved: sum (n-1)/n
    assert torch.allclose(captum_trap.value, torch.tensor([[expected, expected]]), atol=1e-5)
    assert torch.allclose(native_trap.value, torch.tensor([[7.5, 7.5]]), atol=1e-5)
    assert captum_trap.record.method.params["rule"] == "riemann_trapezoid"


def test_a_configuration_mismatch_is_visible_not_hidden() -> None:
    inputs = torch.tensor([[3.0, 1.0]])
    coarse = run(
        Saturating(), inputs, method=C.integrated_gradients(baseline=A.zero_baseline(), n_steps=2)
    )
    fine = run(
        Saturating(), inputs, method=A.integrated_gradients(baseline=A.zero_baseline(), n_steps=64)
    )
    assert coarse.record.method.params["n_steps"] == 2
    assert fine.record.method.params["n_steps"] == 64
    assert not torch.allclose(coarse.value, fine.value, atol=1e-3)  # really ran 2 steps
    matched = run(
        Saturating(), inputs, method=A.integrated_gradients(baseline=A.zero_baseline(), n_steps=2)
    )
    assert torch.allclose(coarse.value, matched.value, rtol=1e-4, atol=1e-5)


def test_captum_layer_ig_matches_native_with_an_input_baseline() -> None:
    ids = torch.tensor([[1, 2, 3, 4]])
    kw: dict[str, Any] = {"target": iv.metrics.select([0, 3, 5]), "at": A.layer("token_embedding")}
    base = A.input_baseline(torch.zeros_like(ids))
    native, adapted = both("riemann_middle", 16, base)
    a = run(TinyTransformer(), ids, method=native, **kw)
    b = run(TinyTransformer(), ids, method=adapted, **kw)
    assert torch.allclose(a.value, b.value, rtol=1e-4, atol=1e-5)
    assert b.record.method.params["captum_class"] == "LayerIntegratedGradients"
    assert "DISCRETE_INPUT_ATTRIBUTED_VIA_REPRESENTATION" in {lim.code for lim in b.limitations}
    redundant = run(
        Redundant(),
        torch.tensor([[3.0, 5.0]]),
        at=A.layer("p"),
        method=C.integrated_gradients(baseline=A.input_baseline(torch.zeros(1, 2)), n_steps=8),
    )
    assert torch.allclose(redundant.value, torch.tensor([[3.0]]), atol=1e-5)


def test_captum_refusals() -> None:
    method = C.integrated_gradients(baseline=A.input_baseline(torch.zeros(1, 1)), n_steps=4)
    with pytest.raises(A.RepeatedCallError, match="hook every call"):
        run(Twice(), torch.tensor([[3.0]]), method=method, at=A.layer("lin"))
    with pytest.raises(A.AttributionError, match="input_baseline"):
        run(
            Redundant(),
            torch.tensor([[3.0, 5.0]]),
            at=A.layer("p"),
            method=C.integrated_gradients(baseline=A.zero_baseline()),
        )
    with pytest.raises(A.AttributionError, match="IG only"):
        run(Redundant(), torch.tensor([[3.0, 5.0]]), at=A.layer("p"), method=C.saliency())
    with pytest.raises(A.AttributionError, match="batch"):
        run(Product(), torch.ones(2, 2), method=C.saliency(), target=SEL)
    with pytest.raises(ValueError, match="rule"):
        C.integrated_gradients(baseline=A.zero_baseline(), rule="trapezoid")


def test_captum_leaves_caller_tensors_and_hooks_untouched() -> None:
    model = TinyMLP().eval()
    inputs = torch.linspace(-1, 1, 4).reshape(1, 4)
    for method in (
        C.saliency(),
        C.input_x_gradient(),
        C.integrated_gradients(baseline=A.zero_baseline(), n_steps=4),
    ):
        run(model, inputs, method=method, target=iv.metrics.select([0, 1]))
    assert not inputs.requires_grad
    assert inputs.grad is None
    assert all(p.grad is None for p in model.parameters())
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
