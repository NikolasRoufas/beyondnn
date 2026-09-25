"""M1.6: trace(), recording(), TraceResult."""

from __future__ import annotations

import contextlib
import dataclasses
import gc
import math
import weakref
from typing import Any

import pytest
import torch
from torch import nn

import beyondnn as bnn
from beyondnn._testing.models import TinyCNN, TinyMLP, TinyTransformer
from beyondnn.core.hooks import AliasSiteAmbiguityError
from beyondnn.core.trace import (
    ActivationLookupError,
    OutOfPassExecutionError,
    RecordingError,
    TraceConfig,
    TraceIntegrityError,
    TraceResult,
)
from beyondnn.provenance import fingerprint_model
from beyondnn.schema import (
    ActivationRecord,
    ClaimRef,
    ClaimTestResult,
    Estimand,
    EvidenceRef,
    EvidenceStatus,
    ExecutionOccurrence,
    InputRecord,
    JsonMap,
    ModelDeclaration,
    Outcome,
    OutputRecord,
    ProvenanceRecord,
    Randomness,
    RecordRef,
    Site,
    SiteIO,
    TensorRef,
    TensorStats,
    TraceLimitation,
)


def _x() -> torch.Tensor:
    return torch.linspace(-1, 1, 8).reshape(2, 4)


def _img() -> torch.Tensor:
    return torch.linspace(-1, 1, 128).reshape(2, 1, 8, 8)


def _tok() -> torch.Tensor:
    return torch.tensor([[1, 5, 9, 3, 0, 31], [2, 2, 7, 30, 4, 8]])


def hook_count(model: nn.Module) -> int:
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


def codes(trace: TraceResult) -> list[str]:
    return [lim.code for lim in trace.limitations]


def scientific_ids(trace: TraceResult) -> list[str]:
    """Record ids excluding occurrences (whose timestamps legitimately differ)."""
    return [r.id for r in trace.records if not isinstance(r, ExecutionOccurrence)]


# ----------------------------------------------------------------- basics


def test_one_shot_trace_records_input_output_activations_and_provenance() -> None:
    model = TinyMLP()
    t = bnn.trace(model, _x(), sites=["shared"])
    assert isinstance(t, TraceResult)
    assert t.passes == 1
    assert t.input.status is EvidenceStatus.OBSERVED
    assert t.output.status is EvidenceStatus.OBSERVED
    assert [n.path for n in t.input.tensors] == ["args[0]"]
    assert [n.path for n in t.output.tensors] == ["output"]
    assert [(a.site.module, a.pass_index, a.call_index) for a in t.activations] == [
        ("shared", 0, 0),
        ("shared", 0, 1),
    ]
    assert all(a.status is EvidenceStatus.MEASURED for a in t.activations)
    (prov,) = t.provenance
    assert prov.model == fingerprint_model(model)
    assert prov.method.name == "forward_hook"
    assert prov.execution.device == "cpu"
    assert prov.execution.training is True
    assert prov.declared_model is None
    assert all(t.origin(r) == prov for r in (t.input, t.output, *t.activations))
    assert len(t.occurrences) == 1
    assert hook_count(model) == 0


def test_recording_multiple_passes_and_repeated_calls() -> None:
    model = TinyMLP()
    with bnn.recording(model, sites=["shared"]) as ctx:
        y = model(_x())
        model(_x())
    t = ctx.result
    assert y.shape == (2, 3)  # the live output is the caller's
    assert [(a.pass_index, a.call_index) for a in t.activations] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert [r.pass_index for r in t.inputs] == [0, 1]
    assert [r.pass_index for r in t.outputs] == [0, 1]
    assert t.inputs[0].id != t.inputs[1].id  # identical tensors, distinct passes
    assert len(t.provenance) == 1  # eval-free MLP: state unchanged across passes
    with pytest.raises(ActivationLookupError):
        _ = t.input
    assert t.activation("shared", pass_index=1, call_index=1).pass_index == 1


def test_trace_equals_a_one_pass_recording() -> None:
    kwargs: dict[str, Any] = {"sites": ["blocks.*.attn", "lm_head"], "input_sites": ["blocks.0"]}
    a = bnn.trace(TinyTransformer(), _tok(), **kwargs)
    model = TinyTransformer()
    with bnn.recording(model, **kwargs) as ctx:
        model(_tok())
    b = ctx.result
    assert scientific_ids(a) == scientific_ids(b)
    assert codes(a) == codes(b)


def test_model_kwargs_are_passed_through() -> None:
    t = bnn.trace(TinyMLP(), _x(), model_kwargs={"scale": 2.0})
    assert "NON_TENSOR_LEAVES_IGNORED" in codes(t)  # scale=2.0 is not tensor evidence


# ----------------------------------------------------------------- records


def test_input_and_output_activations_share_invocation_indices() -> None:
    t = bnn.trace(TinyMLP(), _x(), sites=["shared"], input_sites=["shared"])
    ins = [(a.pass_index, a.call_index) for a in t.activations if a.site.io is SiteIO.INPUT]
    outs = [(a.pass_index, a.call_index) for a in t.activations if a.site.io is SiteIO.OUTPUT]
    assert ins == outs == [(0, 0), (0, 1)]
    assert {a.site.output_path for a in t.activations if a.site.io is SiteIO.INPUT} == {"args[0]"}


def test_structured_outputs_become_one_record_per_tensor_leaf() -> None:
    t = bnn.trace(
        TinyTransformer(), _tok(), sites=["blocks.0.attn"], model_kwargs={"return_dict": True}
    )
    assert [a.site.output_path for a in t.activations] == ["[0]", "[1]"]
    assert [n.path for n in t.output.tensors] == ['output["logits"]', 'output["hidden_states"]']
    c = bnn.trace(TinyCNN().eval(), _img(), model_kwargs={"return_features": True})
    assert [n.path for n in c.output.tensors] == ["output[0]", "output[1]"]


def test_activations_follow_execution_order_not_pattern_order() -> None:
    t = bnn.trace(TinyCNN().eval(), _img(), sites=["classifier", "stem", "pool"])
    assert [a.site.module for a in t.activations] == ["stem", "pool", "classifier"]


def test_tensor_keyword_inputs_are_recorded_with_paths() -> None:
    class KW(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(2, 2)

        def forward(self, x: torch.Tensor, *, bias: torch.Tensor) -> torch.Tensor:
            out: torch.Tensor = self.lin(x) + bias
            return out

    t = bnn.trace(KW(), torch.ones(1, 2), model_kwargs={"bias": torch.zeros(2)})
    assert [n.path for n in t.input.tensors] == ["args[0]", 'kwargs["bias"]']
    assert "NON_TENSOR_LEAVES_IGNORED" not in codes(t)


# ----------------------------------------------------------------- retention


def test_retention_none_summary_cpu() -> None:
    none = bnn.trace(TinyMLP(), _x(), sites=["head"], retention="none")
    summary = bnn.trace(TinyMLP(), _x(), sites=["head"])
    cpu = bnn.trace(TinyMLP(), _x(), sites=["head"], retention="cpu")
    for t in (none, summary, cpu):
        assert t.activations[0].value.shape == (2, 3)
        assert t.activations[0].value.dtype == "float32"
    assert none.activations[0].value.stats is None
    assert summary.activations[0].value.stats is not None
    assert summary.activations[0].value.storage_key is None
    with pytest.raises(KeyError, match="not retained"):
        summary.tensor(summary.activations[0])
    ref = cpu.activations[0].value
    assert ref.storage_key == ref.content_digest
    values = cpu.tensor(cpu.activations[0])
    expected = TinyMLP()(_x())
    assert torch.equal(values, expected.detach())
    assert values.requires_grad is False
    values.add_(1)  # returned copies do not alter the trace
    assert torch.equal(cpu.tensor(cpu.activations[0]), expected.detach())


def test_summary_statistics_are_correct() -> None:
    t = bnn.trace(TinyMLP(), _x(), retention="cpu")
    x = _x().double()
    stats = t.input.tensors[0].ref.stats
    assert stats is not None
    assert math.isclose(stats.mean, x.mean().item())
    assert math.isclose(stats.std, x.std(correction=0).item())
    assert (stats.min, stats.max) == (x.min().item(), x.max().item())


@pytest.mark.parametrize("retention", ["none", "summary"])
def test_non_cpu_retention_keeps_no_tensors(retention: str) -> None:
    model = TinyMLP()
    seen: list[weakref.ref[torch.Tensor]] = []
    handle = model.head.register_forward_hook(lambda m, a, o: seen.append(weakref.ref(o)))
    t = bnn.trace(model, _x(), sites=["**"], retention=retention)
    handle.remove()
    gc.collect()
    assert seen
    assert all(r() is None for r in seen)
    assert t._tensors == {}


def test_cpu_retention_keeps_detached_independent_copies_only() -> None:
    model = TinyMLP()
    seen: list[weakref.ref[torch.Tensor]] = []
    handle = model.head.register_forward_hook(lambda m, a, o: seen.append(weakref.ref(o)))
    t = bnn.trace(model, _x(), sites=["**"], retention="cpu")
    handle.remove()
    gc.collect()
    assert all(r() is None for r in seen)  # the live output (with its graph) is gone
    for stored in t._tensors.values():
        assert stored.device.type == "cpu"
        assert stored.grad_fn is None
        assert not stored.requires_grad


def test_identical_contents_are_stored_once() -> None:
    t = bnn.trace(TinyMLP(), _x(), sites=["shared"], input_sites=["projection"], retention="cpu")
    projection_in = t.activation("projection", io=SiteIO.INPUT, output_path="args[0]")
    assert projection_in.value.storage_key == t.input.tensors[0].ref.storage_key


# ----------------------------------------------------------------- provenance


def test_declared_model_and_randomness_flow_into_provenance() -> None:
    declared = ModelDeclaration(config=JsonMap({"n_heads": 2}), implementation_revision="git:abc")
    t = bnn.trace(
        TinyTransformer(n_heads=2),
        _tok(),
        declared_model=declared,
        randomness=Randomness(declared_seed=0),
    )
    (prov,) = t.provenance
    assert prov.declared_model == declared
    assert prov.execution.randomness == Randomness(declared_seed=0)
    other = bnn.trace(
        TinyTransformer(n_heads=4),
        _tok(),
        declared_model=ModelDeclaration(config=JsonMap({"n_heads": 4})),
    )
    assert other.provenance[0].id != prov.id


def test_state_change_between_passes_gets_new_provenance() -> None:
    model = TinyCNN().train()
    before_pass_0 = fingerprint_model(model)
    with bnn.recording(model, sites=["stem.norm"]) as ctx:
        model(_img())
        after_pass_0 = fingerprint_model(model)
        model(_img())
    t = ctx.result
    assert before_pass_0.state_digest != after_pass_0.state_digest  # BatchNorm stats moved
    p0, p1 = t.origin(t.inputs[0]), t.origin(t.inputs[1])
    assert p0.model == before_pass_0
    assert p1.model == after_pass_0
    for a in t.activations:
        assert t.origin(a) == (p0 if a.pass_index == 0 else p1)


def test_unchanged_state_shares_provenance_and_mode_is_recorded() -> None:
    model = TinyCNN().eval()
    with bnn.recording(model, sites=["stem"]) as ctx:
        model(_img())
        with torch.no_grad():
            model(_img())
    t = ctx.result
    assert [p.execution.grad_enabled for p in t.provenance] == [True, False]
    assert all(not p.execution.training for p in t.provenance)


# ----------------------------------------------------------------- limitations


def test_functional_ops_limitation_is_always_present() -> None:
    t = bnn.trace(TinyTransformer(), _tok(), sites=["**"])
    assert "FUNCTIONAL_OPS_UNOBSERVED" in codes(t)
    assert "PARTIAL_SITE_COVERAGE" not in codes(t)  # every named module selected


def test_partial_coverage_limitation() -> None:
    t = bnn.trace(TinyMLP(), _x(), sites=["head"])
    (lim,) = [lim for lim in t.limitations if lim.code == "PARTIAL_SITE_COVERAGE"]
    assert lim.detail == "1 of 4 named modules selected for output"


def test_selected_but_never_executed_sites_are_reported() -> None:
    t = bnn.trace(TinyTransformer(), _tok(), sites=["blocks", "blocks.0"], input_sites=["blocks"])
    (lim,) = [lim for lim in t.limitations if lim.code == "SELECTED_SITE_NOT_EXECUTED"]
    assert lim.detail == "blocks (input), blocks (output)"
    assert {a.site.module for a in t.activations} == {"blocks.0"}


def test_no_explain_level_limitations_in_plain_traces() -> None:
    t = bnn.trace(TinyMLP(), _x(), sites=["**"])
    assert set(codes(t)) == {"FUNCTIONAL_OPS_UNOBSERVED"}


# ----------------------------------------------------------------- refusals and failures


def test_direct_submodule_execution_is_refused() -> None:
    model = TinyMLP()
    ctx = bnn.recording(model, sites=["shared"])
    with pytest.raises(OutOfPassExecutionError, match="outside any root invocation"), ctx:
        model.shared(torch.ones(1, 8))
    with pytest.raises(RecordingError, match="failed"):
        _ = ctx.result
    assert hook_count(model) == 0


def test_unselected_direct_submodule_calls_are_irrelevant() -> None:
    model = TinyMLP()
    with bnn.recording(model, sites=["head"]) as ctx:
        model.projection(torch.ones(1, 4))
        model(_x())
    assert ctx.result.passes == 1


class Aliased(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        shared = nn.Linear(2, 2)
        self.left = shared
        self.right = shared

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.right(self.left(x))
        return out


def test_alias_ambiguity_is_refused() -> None:
    model = Aliased()
    with pytest.raises(AliasSiteAmbiguityError):
        bnn.trace(model, torch.ones(1, 2), sites=["left"])
    assert hook_count(model) == 0
    bnn.trace(model, torch.ones(1, 2))  # root-only tracing of an aliasing model is fine


def test_failed_model_call_produces_no_result() -> None:
    model = TinyTransformer()
    ctx = bnn.recording(model, sites=["**"])
    with pytest.raises(ValueError, match="exceeds"), ctx:
        model(torch.zeros(1, 20, dtype=torch.int64))
    with pytest.raises(RecordingError, match="failed"):
        _ = ctx.result


def test_swallowed_model_failure_still_fails_the_recording() -> None:
    model = TinyTransformer()
    ctx = bnn.recording(model, sites=["lm_head"])

    def run() -> None:
        with ctx:
            model(_tok())
            with contextlib.suppress(ValueError):
                model(torch.zeros(1, 20, dtype=torch.int64))

    with pytest.raises(RecordingError, match="did not complete"):
        run()
    with pytest.raises(RecordingError):
        _ = ctx.result


def test_result_unavailable_before_exit_and_without_passes() -> None:
    model = TinyMLP()
    with bnn.recording(model) as ctx:
        with pytest.raises(RecordingError, match="only after"):
            _ = ctx.result
        model(_x())
    assert ctx.result.passes == 1
    empty = bnn.recording(model)
    with pytest.raises(RecordingError, match="no root invocation"), empty:
        pass


def test_argument_validation() -> None:
    model = TinyMLP()
    with pytest.raises(ValueError, match="root"):
        bnn.recording(model, sites=[""])
    with pytest.raises(TypeError):
        bnn.recording(model, sites="shared")
    with pytest.raises(ValueError, match="retention"):
        bnn.recording(model, retention="device")
    with pytest.raises(TypeError):
        bnn.recording(model, declared_model={"n_heads": 2})  # type: ignore[arg-type]


# ----------------------------------------------------------------- lookup


def test_activation_lookup_is_strict() -> None:
    t = bnn.trace(TinyMLP(), _x(), sites=["shared", "head"])
    with pytest.raises(ActivationLookupError, match="2 activations"):
        t.activation("shared")
    assert t.activation("shared", call_index=1).call_index == 1
    assert t.activation("head").site == Site(module="head")
    with pytest.raises(ActivationLookupError, match="no activation"):
        t.activation("projection")


# ----------------------------------------------------------------- model behaviour


@pytest.mark.parametrize(
    ("build", "make_x"), [(TinyMLP, _x), (TinyCNN, _img), (TinyTransformer, _tok)]
)
@pytest.mark.parametrize("mode", ["train", "eval"])
def test_tracing_does_not_change_model_behaviour(build: Any, make_x: Any, mode: str) -> None:
    base, traced = build(), build()
    getattr(base, mode)()
    getattr(traced, mode)()
    flags = [m.training for m in traced.modules()]
    y_a = base(make_x())
    torch.autograd.backward(y_a.float().square().mean())
    with bnn.recording(traced, sites=["**"], input_sites=["**"], retention="cpu") as ctx:
        y_b = traced(make_x())
        torch.autograd.backward(y_b.float().square().mean())
    assert ctx.result.passes == 1
    assert torch.equal(y_a, y_b)
    for (na, pa), (nb, pb) in zip(base.named_parameters(), traced.named_parameters(), strict=True):
        assert na == nb
        assert pa.grad is not None
        assert pb.grad is not None
        assert torch.equal(pa.grad, pb.grad)
    sa, sb = base.state_dict(), traced.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert [m.training for m in traced.modules()] == flags
    assert hook_count(traced) == 0


# ----------------------------------------------------------------- container integrity


def _prov() -> ProvenanceRecord:
    return bnn.trace(TinyMLP(), _x()).provenance[0]


def _container() -> tuple[TraceResult, ProvenanceRecord, InputRecord]:
    t = TraceResult(TraceConfig((), (), "summary"))
    prov = t._add(_prov())
    assert isinstance(prov, ProvenanceRecord)
    inp = InputRecord(pass_index=0, provenance_id=prov.id)
    t._add(inp)
    return t, prov, inp


def _nan_activation(prov: ProvenanceRecord) -> ActivationRecord:
    nan = float("nan")
    stats = TensorStats(numel=1, mean=nan, std=nan, min=nan, max=nan, l2_norm=nan)
    ref = TensorRef(shape=(1,), dtype="float32", device="cpu", stats=stats)
    return ActivationRecord(site=Site(module="m"), value=ref, provenance_id=prov.id)


def test_nan_records_are_deduplicated_by_id_not_equality() -> None:
    t, prov, _ = _container()
    a, b = _nan_activation(prov), _nan_activation(prov)
    assert a.id == b.id
    assert a != b  # NaN makes dataclass equality useless here
    t._add(a)
    assert t._add(b) is a
    assert len(t.activations) == 1


def test_dangling_or_wrong_provenance_is_rejected() -> None:
    t, _, inp = _container()
    with pytest.raises(TraceIntegrityError, match="does not name a ProvenanceRecord"):
        t._add(InputRecord(pass_index=1, provenance_id="provenance:" + "0" * 32))
    with pytest.raises(TraceIntegrityError, match="does not name a ProvenanceRecord"):
        t._add(InputRecord(pass_index=1, provenance_id=inp.id))


def test_forged_and_dangling_references_are_rejected() -> None:
    t, prov, inp = _container()
    forged = RecordRef(record_id=inp.id, kind="input", status=EvidenceStatus.MEASURED)
    with pytest.raises(TraceIntegrityError, match="does not match"):
        t._add(dataclasses.replace(_nan_activation(prov), derived_from=(forged,)))
    missing = RecordRef(record_id="input:" + "1" * 32, kind="input", status=EvidenceStatus.OBSERVED)
    with pytest.raises(TraceIntegrityError, match="not in trace"):
        t._add(dataclasses.replace(_nan_activation(prov), derived_from=(missing,)))
    with pytest.raises(TraceIntegrityError, match="applies_to"):
        t._add(TraceLimitation(code="NO_ATTRIBUTION", applies_to=("input:" + "2" * 32,)))


def test_forged_evidence_and_claim_references_are_rejected(mk: Any) -> None:
    t, prov, _ = _container()
    act = _nan_activation(prov)
    t._add(act)
    claim, spec = mk.claim(), mk.spec()
    t._add(claim)
    t._add(spec)
    forged_evidence = EvidenceRef(
        record_id=act.id, kind="activation", status=EvidenceStatus.ATTRIBUTED
    )
    bad = ClaimTestResult.for_claim(
        claim,
        spec,
        outcome=Outcome.INCONCLUSIVE,
        evidence=(forged_evidence,),
        provenance_id=prov.id,
    )
    with pytest.raises(TraceIntegrityError, match="EvidenceRef does not match"):
        t._add(bad)
    good = ClaimTestResult.for_claim(
        claim, spec, outcome=Outcome.INCONCLUSIVE, evidence=(act,), provenance_id=prov.id
    )
    forged_claim = ClaimRef(
        claim_id=claim.id, relation=claim.relation, estimand=Estimand.instance("x9")
    )
    with pytest.raises(TraceIntegrityError, match="ClaimRef does not match"):
        t._add(dataclasses.replace(good, claim=forged_claim))
    assert t._add(good) is good


def test_tampered_records_are_rejected() -> None:
    t, prov, _ = _container()
    act = _nan_activation(prov)
    object.__setattr__(act, "call_index", 5)  # bypass immutability
    with pytest.raises(TraceIntegrityError, match="does not match its content"):
        t._add(act)


def test_finalised_traces_are_read_only() -> None:
    t = bnn.trace(TinyMLP(), _x())
    with pytest.raises(TraceIntegrityError, match="finalised"):
        t._add(TraceLimitation(code="NO_ATTRIBUTION"))


def test_outputs_are_derived_from_their_pass_input() -> None:
    t = bnn.trace(TinyMLP(), _x(), sites=["head"])
    assert t.output.derived_from == (RecordRef.to(t.input),)
    assert t.activations[0].derived_from == (RecordRef.to(t.input),)
    assert isinstance(t.output, OutputRecord)


@pytest.mark.parametrize("retention", ["none", "summary", "cpu"])
def test_live_context_and_result_do_not_retain_live_tensors(retention: str) -> None:
    model = TinyTransformer()
    seen: list[weakref.ref[torch.Tensor]] = []

    def capture(module: nn.Module, args: Any, output: Any) -> None:
        seen.append(weakref.ref(args[0]))
        seen.append(weakref.ref(output[0]))

    handle = model.get_submodule("blocks.0.attn").register_forward_hook(capture)
    with bnn.recording(model, sites=["**"], input_sites=["**"], retention=retention) as ctx:
        y = model(_tok())
        torch.autograd.backward(y.square().mean())
        del y
    handle.remove()
    result = ctx.result  # context and result both still alive
    gc.collect()
    assert seen
    assert all(r() is None for r in seen)
    assert result.passes == 1


def test_unused_selected_module_is_reported_not_executed() -> None:
    class WithSpare(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.used = nn.Linear(2, 2)
            self.spare = nn.Linear(2, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out: torch.Tensor = self.used(x)
            return out

    t = bnn.trace(WithSpare(), torch.ones(1, 2), sites=["*"])
    (lim,) = [lim for lim in t.limitations if lim.code == "SELECTED_SITE_NOT_EXECUTED"]
    assert lim.detail == "spare (output)"
    assert [a.site.module for a in t.activations] == ["used"]


def test_evidence_record_with_non_provenance_provenance_is_rejected() -> None:
    t, prov, inp = _container()
    act = dataclasses.replace(_nan_activation(prov), provenance_id=inp.id)
    with pytest.raises(TraceIntegrityError, match="does not name a ProvenanceRecord"):
        t._add(act)


def test_evidence_ref_to_a_non_evidence_record_is_rejected(mk: Any) -> None:
    t, prov, _ = _container()
    claim, spec = mk.claim(), mk.spec()
    t._add(claim)
    t._add(spec)
    fake = EvidenceRef(record_id=prov.id, kind="provenance", status=EvidenceStatus.MEASURED)
    result = ClaimTestResult.for_claim(
        claim, spec, outcome=Outcome.INCONCLUSIVE, evidence=(fake,), provenance_id=prov.id
    )
    with pytest.raises(TraceIntegrityError):
        t._add(result)
