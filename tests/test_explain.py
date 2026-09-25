"""M1.8: instrument() and the minimal honest INPUT -> WHY -> OUTPUT view."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import beyondnn as bnn
from beyondnn._testing.models import TinyCNN, TinyMLP, TinyTransformer
from beyondnn.core import trace as trace_module
from beyondnn.explain import EXPLANATION_LIMITATIONS, ExplainResponse, Instrumented, Why
from beyondnn.schema import (
    ActivationRecord,
    EvidenceStatus,
    ExecutionOccurrence,
    InputRecord,
    OutputRecord,
)

PHASE_1_STATUSES = {EvidenceStatus.OBSERVED, EvidenceStatus.MEASURED}


def _x() -> torch.Tensor:
    return torch.linspace(-1, 1, 8).reshape(2, 4)


def _tok() -> torch.Tensor:
    return torch.tensor([[1, 5, 9, 3, 0, 31]])


def hook_count(model: nn.Module) -> int:
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


# ----------------------------------------------------------------- instrument()


def test_instrument_returns_a_handle_to_the_same_unchanged_model() -> None:
    model = TinyMLP()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    children = list(model.named_children())
    y_before = model(_x())
    handle = bnn.instrument(model)
    assert isinstance(handle, Instrumented)
    assert handle.model is model
    assert not isinstance(handle, nn.Module)
    assert not callable(handle)
    assert list(model.named_children()) == children
    assert hook_count(model) == 0
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    handle.explain(_x(), sites=["**"])
    assert hook_count(model) == 0  # no permanent hooks
    assert torch.equal(model(_x()), y_before)
    with pytest.raises(dataclasses.FrozenInstanceError):
        handle.model = TinyMLP()  # type: ignore[misc]


def test_instrument_rejects_non_modules() -> None:
    with pytest.raises(TypeError):
        bnn.instrument(torch.ones(1))  # type: ignore[arg-type]


def _ids(t: bnn.TraceResult) -> list[str]:
    return [r.id for r in t.records if not isinstance(r, ExecutionOccurrence)]


def test_handle_trace_equals_direct_trace() -> None:
    kwargs: dict[str, Any] = {"sites": ["blocks.*.attn"], "input_sites": ["lm_head"]}
    via_handle = bnn.instrument(TinyTransformer()).trace(_tok(), **kwargs)
    direct = bnn.trace(TinyTransformer(), _tok(), **kwargs)
    assert _ids(via_handle) == _ids(direct)
    with bnn.instrument(TinyMLP()).recording(sites=["head"]) as ctx:
        ctx._model(_x())
    assert ctx.result.passes == 1


def test_explain_uses_exactly_one_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    entered = {"n": 0}
    original = trace_module.Recording.__enter__

    def counting(self: trace_module.Recording) -> trace_module.Recording:
        entered["n"] += 1
        return original(self)

    monkeypatch.setattr(trace_module.Recording, "__enter__", counting)
    response = bnn.instrument(TinyMLP()).explain(_x(), sites=["shared"])
    assert entered["n"] == 1
    assert response.trace.passes == 1


def test_explain_has_no_target_argument() -> None:
    with pytest.raises(TypeError):
        bnn.instrument(TinyMLP()).explain(_x(), target=1)  # type: ignore[call-arg]


# ----------------------------------------------------------------- INPUT -> WHY -> OUTPUT


def test_input_why_output_structure_and_statuses() -> None:
    response = bnn.instrument(TinyMLP()).explain(_x(), sites=["shared", "head"])
    assert isinstance(response.input, InputRecord)
    assert isinstance(response.output, OutputRecord)
    assert response.input.status is EvidenceStatus.OBSERVED
    assert response.output.status is EvidenceStatus.OBSERVED
    why = response.why
    assert isinstance(why, Why)
    assert why.activations
    assert all(isinstance(a, ActivationRecord) for a in why.activations)
    assert all(a.status is EvidenceStatus.MEASURED for a in why.activations)
    assert why.evidence_statuses == PHASE_1_STATUSES
    assert "measured" in Why.QUESTION
    assert "caused" in Why.NOT_ANSWERED


@pytest.mark.parametrize("build", [TinyMLP, TinyCNN, TinyTransformer])
def test_no_stronger_evidence_status_is_ever_produced(build: Any) -> None:
    make = {TinyMLP: _x, TinyCNN: lambda: torch.zeros(1, 1, 8, 8), TinyTransformer: _tok}[build]
    response = bnn.instrument(build()).explain(
        make(), sites=["**"], input_sites=["**"], retention="cpu"
    )
    statuses = {r.status for r in response.trace.records}
    assert statuses <= PHASE_1_STATUSES | {None}
    assert response.why.evidence_statuses <= PHASE_1_STATUSES


def test_explanation_limitations_are_always_present_and_structured() -> None:
    response = bnn.instrument(TinyMLP()).explain(_x(), sites=["**"])
    codes = [lim.code for lim in response.why.limitations]
    for code in EXPLANATION_LIMITATIONS:
        assert code in codes
    assert set(EXPLANATION_LIMITATIONS) == {
        "NO_ATTRIBUTION",
        "NO_CAUSAL_EVIDENCE",
        "NO_CLAIMS_TESTED",
    }
    assert "FUNCTIONAL_OPS_UNOBSERVED" in codes


def test_trace_limitations_are_preserved() -> None:
    response = bnn.instrument(TinyTransformer()).explain(
        _tok(), sites=["blocks"], model_kwargs={"return_dict": True}
    )
    codes = [lim.code for lim in response.why.limitations]
    for code in (
        "FUNCTIONAL_OPS_UNOBSERVED",
        "PARTIAL_SITE_COVERAGE",
        "SELECTED_SITE_NOT_EXECUTED",
        "NON_TENSOR_LEAVES_IGNORED",
    ):
        assert code in codes
    assert codes[: len(response.trace.limitations)] == [
        lim.code for lim in response.trace.limitations
    ]


def test_why_is_a_view_over_the_trace_not_a_copy() -> None:
    response = bnn.instrument(TinyMLP()).explain(_x(), sites=["shared"])
    for view, record in zip(response.why.activations, response.trace.activations, strict=True):
        assert view is record
    assert response.input is response.trace.input
    assert response.why.trace is response.trace


def test_responses_are_immutable_and_single_pass() -> None:
    response = bnn.instrument(TinyMLP()).explain(_x())
    with pytest.raises(dataclasses.FrozenInstanceError):
        response.trace = response.trace  # type: ignore[misc]
    model = TinyMLP()
    with bnn.recording(model) as ctx:
        model(_x())
        model(_x())
    with pytest.raises(ValueError, match="exactly one root invocation"):
        ExplainResponse.from_trace(ctx.result)


# ----------------------------------------------------------------- rendering and persistence


def test_render_is_deterministic_and_never_claims_causality() -> None:
    a = bnn.instrument(TinyMLP()).explain(_x(), sites=["**"]).render()
    b = bnn.instrument(TinyMLP()).explain(_x(), sites=["**"]).render()
    assert a == b
    assert "not a causal or attributed explanation" in a
    lowered = a.lower()
    for word in ("important", "decisive", "because", "reason", "supports"):
        assert word not in lowered


def test_response_survives_trace_save_and_load(tmp_path: Path) -> None:
    response = bnn.instrument(TinyTransformer()).explain(
        _tok(), sites=["blocks.*.attn"], retention="cpu"
    )
    response.trace.save(tmp_path / "t")
    rebuilt = ExplainResponse.from_trace(bnn.load_trace(tmp_path / "t"))
    assert [a.id for a in rebuilt.why.activations] == [a.id for a in response.why.activations]
    assert [lim.id for lim in rebuilt.why.limitations] == [
        lim.id for lim in response.why.limitations
    ]
    assert rebuilt.render() == response.render()
