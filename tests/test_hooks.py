"""M1.5: HookSession lifecycle, indices, aliases, autograd, and retention."""

from __future__ import annotations

import contextlib
import gc
import weakref
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

from beyondnn._testing.models import TinyCNN, TinyMLP, TinyTransformer
from beyondnn.core.hooks import (
    AliasPolicy,
    AliasSiteAmbiguityError,
    HookEvent,
    HookSession,
    HookSessionError,
)
from beyondnn.core.sites import SiteResolutionError, UnmatchedPatternError
from beyondnn.schema import BaseRecord, Site, SiteIO


def hook_count(model: nn.Module) -> int:
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


class Recorder:
    """A sink that keeps only metadata (never values)."""

    def __init__(self) -> None:
        self.events: list[tuple[str, tuple[str, ...], int, int]] = []

    def __call__(self, event: HookEvent) -> None:
        self.events.append((event.io.value, event.paths, event.pass_index, event.call_index))


def _mlp_x() -> torch.Tensor:
    return torch.linspace(-1, 1, 8).reshape(2, 4)


def _tok() -> torch.Tensor:
    return torch.tensor([[1, 5, 9, 3, 0, 31], [2, 2, 7, 30, 4, 8]])


# ----------------------------------------------------------------- lifecycle


def test_hooks_exist_only_inside_the_session() -> None:
    model = TinyMLP()
    assert hook_count(model) == 0
    session = HookSession(model, sink=Recorder(), outputs=["shared"])
    assert hook_count(model) == 0  # nothing installed before entry
    with session:
        assert session.active
        # root: pass-start + pass-end; shared: guard + observe + output + cleanup
        assert hook_count(model) == session.installed_hooks == 6
    assert not session.active
    assert hook_count(model) == 0


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"outputs": ["shared"]}, 6),
        ({"inputs": ["shared"]}, 5),  # no output-observation hook
        ({"inputs": ["shared"], "outputs": ["shared"]}, 6),  # INPUT+OUTPUT share guard/cleanup
        ({"outputs": ["shared", "shared", "*.linear", "shared.**"]}, 10),  # shared + linear
        ({"outputs": [""]}, 6),  # root: bookkeeping pair + its own four
    ],
)
def test_physical_hooks_are_installed_once(kwargs: dict[str, Any], expected: int) -> None:
    model = TinyMLP()
    with HookSession(model, sink=Recorder(), **kwargs) as session:
        assert session.installed_hooks == expected
        assert hook_count(model) == expected


def test_cleanup_when_forward_raises() -> None:
    model = TinyTransformer()
    with (
        pytest.raises(ValueError, match="exceeds"),
        HookSession(model, sink=Recorder(), outputs=["**"]),
    ):
        model(torch.zeros(1, 20, dtype=torch.int64))
    assert hook_count(model) == 0
    assert model(_tok()).shape == (2, 6, 32)  # still usable


def test_cleanup_when_sink_raises() -> None:
    model = TinyMLP()

    def sink(event: HookEvent) -> None:
        raise RuntimeError("sink failed")

    with (
        pytest.raises(RuntimeError, match="sink failed"),
        HookSession(model, sink=sink, outputs=["shared"]),
    ):
        model(_mlp_x())
    assert hook_count(model) == 0


def test_cleanup_when_installation_fails_part_way(monkeypatch: pytest.MonkeyPatch) -> None:
    model = TinyMLP()
    original = nn.Module.register_forward_hook
    calls = {"n": 0}

    def flaky(self: nn.Module, *args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("installation failed")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(nn.Module, "register_forward_hook", flaky)
    session = HookSession(model, sink=Recorder(), outputs=["projection", "shared", "head"])
    with pytest.raises(RuntimeError, match="installation failed"):
        session.__enter__()
    assert calls["n"] == 3  # some hooks were installed before the failure...
    assert hook_count(model) == 0  # ...and all were removed
    assert not session.active


def test_unmatched_patterns_install_nothing() -> None:
    model = TinyMLP()
    with (
        pytest.raises(UnmatchedPatternError),
        HookSession(model, sink=Recorder(), outputs=["shared", "nope"]),
    ):
        pass
    assert hook_count(model) == 0


def test_sessions_are_single_use_and_exit_is_idempotent() -> None:
    model = TinyMLP()
    session = HookSession(model, sink=Recorder(), outputs=["shared"])
    with session:
        pass
    session.__exit__(None, None, None)  # second exit: harmless
    assert hook_count(model) == 0
    with pytest.raises(HookSessionError, match="only once"), session:
        pass
    assert hook_count(model) == 0


def test_repeated_sessions_do_not_leak_hooks() -> None:
    model = TinyTransformer()
    for _ in range(50):
        with HookSession(model, sink=Recorder(), outputs=["**"], inputs=["blocks.*"]):
            model(_tok())
    assert hook_count(model) == 0


def test_user_hooks_are_preserved_and_keep_working() -> None:
    model = TinyMLP()
    fired: list[str] = []
    model.shared.register_forward_pre_hook(lambda m, a: fired.append("user_pre"))
    model.head.register_forward_hook(lambda m, a, o: fired.append("user_post"))
    before = hook_count(model)
    with HookSession(model, sink=Recorder(), outputs=["shared", "head"], inputs=["shared"]):
        model(_mlp_x())
    assert hook_count(model) == before
    fired.clear()
    model(_mlp_x())
    assert fired == ["user_pre", "user_pre", "user_post"]


def test_argument_validation() -> None:
    model = TinyMLP()
    with pytest.raises(TypeError):
        HookSession(model, sink=Recorder(), outputs="shared")
    with pytest.raises(TypeError):
        HookSession(model, sink=None, outputs=["shared"])  # type: ignore[arg-type]
    with pytest.raises(SiteResolutionError, match="at least one"):
        HookSession(model, sink=Recorder())
    with pytest.raises(TypeError):
        HookSession(torch.ones(1), sink=Recorder(), outputs=["a"])  # type: ignore[arg-type]


# ----------------------------------------------------------------- ordering and indices


def test_nested_ordering_follows_execution() -> None:
    model = TinyCNN().eval()
    rec = Recorder()
    with HookSession(model, sink=rec, inputs=["stem", "stem.conv"], outputs=["stem", "stem.conv"]):
        model(torch.zeros(1, 1, 8, 8))
    assert [(io, p[0]) for io, p, _, _ in rec.events] == [
        ("input", "stem"),
        ("input", "stem.conv"),
        ("output", "stem.conv"),
        ("output", "stem"),
    ]


def test_repeated_calls_and_pass_indices() -> None:
    model = TinyMLP()
    rec = Recorder()
    with HookSession(model, sink=rec, outputs=["shared"]):
        model(_mlp_x())
        model(_mlp_x())
    assert [(p, c) for _, _, p, c in rec.events] == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_input_and_output_of_one_invocation_share_indices() -> None:
    model = TinyTransformer()
    rec = Recorder()
    with HookSession(model, sink=rec, inputs=["**"], outputs=["**"]):
        model(_tok())
    inputs = [(p, pi, c) for io, p, pi, c in rec.events if io == "input"]
    outputs = [(p, pi, c) for io, p, pi, c in rec.events if io == "output"]
    assert sorted(inputs) == sorted(outputs)
    assert len(inputs) == len(set(inputs))


def test_root_site() -> None:
    model = TinyMLP()
    rec = Recorder()
    with HookSession(model, sink=rec, inputs=[""], outputs=[""]):
        model(_mlp_x())
        model(_mlp_x())
    assert rec.events == [
        ("input", ("",), 0, 0),
        ("output", ("",), 0, 0),
        ("input", ("",), 1, 0),
        ("output", ("",), 1, 0),
    ]


def test_container_and_parameterless_sites() -> None:
    model = TinyCNN().eval()
    rec = Recorder()
    with HookSession(model, sink=rec, outputs=["stem", "pool", "gap"]):
        model(torch.zeros(1, 1, 8, 8))
    assert [p[0] for _, p, _, _ in rec.events] == ["stem", "pool", "gap"]


def test_modules_that_never_execute_emit_nothing() -> None:
    # ModuleList is iterated, not called: selecting it is valid but yields no events.
    model = TinyTransformer()
    rec = Recorder()
    with HookSession(model, sink=rec, outputs=["blocks", "blocks.0"]):
        model(_tok())
    assert [p[0] for _, p, _, _ in rec.events] == ["blocks.0"]


def test_submodule_only_invocation_is_out_of_pass() -> None:
    model = TinyMLP()
    rec = Recorder()
    h = torch.ones(1, 8)
    with HookSession(model, sink=rec, outputs=["shared"]):
        model.shared(h)
        model.shared(h)
        model(_mlp_x())
        model.shared(h)
    assert [(p, c) for _, _, p, c in rec.events] == [(-1, 0), (-1, 1), (0, 0), (0, 1), (-1, 2)]


def test_event_site_and_pass_flags() -> None:
    model = TinyMLP()
    seen: list[HookEvent] = []
    with HookSession(model, sink=seen.append, outputs=["shared.linear"]):
        model(_mlp_x())
    event = seen[0]
    assert event.site == Site(module="shared.linear", io=SiteIO.OUTPUT)
    assert event.path_specific
    assert event.in_root_pass
    assert not isinstance(event, BaseRecord)
    assert not hasattr(event, "id")


# ----------------------------------------------------------------- exceptions


class Flaky(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(2, 2)
        self.bad = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor, *, fail: bool = False) -> torch.Tensor:
        h = self.first(x)
        if fail:
            self.bad.register_forward_pre_hook(_raise_once)
        out: torch.Tensor = self.bad(h)
        return out


def _raise_once(module: nn.Module, args: Any) -> None:
    for handle_id in list(module._forward_pre_hooks):
        if module._forward_pre_hooks[handle_id] is _raise_once:
            del module._forward_pre_hooks[handle_id]
    raise RuntimeError("bad module failed")


def test_failed_root_invocation_consumes_its_pass_index() -> None:
    model = TinyTransformer()
    rec = Recorder()
    with HookSession(model, sink=rec, inputs=[""], outputs=["lm_head"]):
        with pytest.raises(ValueError, match="exceeds"):
            model(torch.zeros(1, 20, dtype=torch.int64))
        model(_tok())
    assert rec.events == [
        ("input", ("",), 0, 0),  # pass 0 started, then raised: no output
        ("input", ("",), 1, 0),
        ("output", ("lm_head",), 1, 0),
    ]


def test_failed_module_invocation_consumes_its_call_index() -> None:
    model = Flaky()
    rec = Recorder()
    x = torch.ones(1, 2)
    with HookSession(model, sink=rec, inputs=["bad"], outputs=["bad"]):
        with pytest.raises(RuntimeError, match="bad module failed"):
            model(x, fail=True)
        model(x)
    assert rec.events == [
        ("input", ("bad",), 0, 0),  # observed, then failed: no OUTPUT
        ("input", ("bad",), 1, 0),
        ("output", ("bad",), 1, 0),
    ]


def test_bookkeeping_survives_a_sink_error_caught_inside_the_session() -> None:
    model = TinyMLP()
    rec = Recorder()
    state = {"raise": True}

    def sink(event: HookEvent) -> None:
        if state["raise"]:
            state["raise"] = False
            raise RuntimeError("transient sink failure")
        rec(event)

    with HookSession(model, sink=sink, inputs=["shared"], outputs=["shared"]):
        with pytest.raises(RuntimeError, match="transient"):
            model(_mlp_x())
        model(_mlp_x())
    assert rec.events == [
        ("input", ("shared",), 1, 0),
        ("output", ("shared",), 1, 0),
        ("input", ("shared",), 1, 1),
        ("output", ("shared",), 1, 1),
    ]


def test_always_call_semantics_relied_on() -> None:
    # Compatibility: ordinary forward hooks skip failed calls; always_call hooks run once.
    class Boom(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("boom")

    calls: list[str] = []
    m = Boom()
    m.register_forward_hook(lambda mod, a, o: calls.append("ordinary"))
    m.register_forward_hook(lambda mod, a, o: calls.append("always"), always_call=True)
    with pytest.raises(RuntimeError):
        m(torch.ones(1))
    assert calls == ["always"]


# ----------------------------------------------------------------- reentrancy


class Recursive(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor, depth: int = 2) -> torch.Tensor:
        h: torch.Tensor = self.inner(x)
        return h if depth == 0 else self(h, depth - 1)


def test_recursive_calls_pair_inputs_and_outputs_by_stack() -> None:
    model = Recursive()
    rec = Recorder()
    with HookSession(model, sink=rec, inputs=[""], outputs=[""]):
        model(torch.ones(1, 2))
    assert rec.events == [
        ("input", ("",), 0, 0),
        ("input", ("",), 0, 1),
        ("input", ("",), 0, 2),
        ("output", ("",), 0, 2),
        ("output", ("",), 0, 1),
        ("output", ("",), 0, 0),
    ]


# ----------------------------------------------------------------- values untouched


def test_kwargs_are_observed_read_only_and_unchanged() -> None:
    model = TinyMLP()
    seen: list[HookEvent] = []
    with HookSession(model, sink=seen.append, inputs=[""]):
        y = model(_mlp_x(), scale=2.0)
    assert torch.equal(y, model(_mlp_x(), scale=2.0))
    (event,) = seen
    assert dict(event.kwargs or {}) == {"scale": 2.0}
    with pytest.raises(TypeError):
        event.kwargs["scale"] = 3.0  # type: ignore[index]


def test_extra_positional_args_are_observed_unchanged() -> None:
    model = TinyTransformer()
    seen: list[HookEvent] = []
    with HookSession(model, sink=seen.append, inputs=["blocks.0.attn"]):
        model(_tok())
    (event,) = seen
    assert event.args is not None
    assert len(event.args) == 2
    assert event.args[1] is model.get_buffer("causal_mask")


def test_structured_outputs_are_passed_unchanged() -> None:
    model = TinyTransformer()
    seen: list[HookEvent] = []
    with HookSession(model, sink=seen.append, outputs=["blocks.0.attn", ""]):
        result = model(_tok(), return_dict=True)
    attn, root = seen
    assert isinstance(attn.output, tuple)
    assert len(attn.output) == 2
    assert root.output is result
    assert isinstance(root.output, dict)

    cnn = TinyCNN().eval()
    seen.clear()
    with HookSession(cnn, sink=seen.append, outputs=[""]):
        out = cnn(torch.zeros(1, 1, 8, 8), return_features=True)
    assert seen[0].output is out
    assert isinstance(out, tuple)


def _loss_and_grads(
    model: nn.Module, x: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    y = model(x)
    loss = y.float().square().mean()
    torch.autograd.backward(loss)
    return loss.detach(), {
        n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
    }


@pytest.mark.parametrize(
    ("build", "make_x"),
    [
        (TinyMLP, _mlp_x),
        (TinyCNN, lambda: torch.linspace(-1, 1, 128).reshape(2, 1, 8, 8)),
        (TinyTransformer, _tok),
    ],
)
@pytest.mark.parametrize("mode", ["train", "eval"])
def test_outputs_loss_gradients_and_state_match_baseline(
    build: Callable[[], nn.Module], make_x: Callable[[], torch.Tensor], mode: str
) -> None:
    baseline, observed = build(), build()
    getattr(baseline, mode)()
    getattr(observed, mode)()
    flags = [m.training for m in observed.modules()]
    loss_a, grads_a = _loss_and_grads(baseline, make_x())
    count = {"n": 0}

    def sink(event: HookEvent) -> None:
        count["n"] += 1

    with HookSession(observed, sink=sink, inputs=["**"], outputs=["**", ""]):
        loss_b, grads_b = _loss_and_grads(observed, make_x())
    assert count["n"] > 0
    assert torch.equal(loss_a, loss_b)
    assert grads_a.keys() == grads_b.keys()
    assert all(torch.equal(grads_a[k], grads_b[k]) for k in grads_a)
    sa, sb = baseline.state_dict(), observed.state_dict()  # e.g. BatchNorm running stats
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert [m.training for m in observed.modules()] == flags


def test_session_does_not_retain_observed_tensors() -> None:
    model = TinyTransformer()
    refs: list[weakref.ref[torch.Tensor]] = []

    def sink(event: HookEvent) -> None:
        if isinstance(event.output, torch.Tensor):
            refs.append(weakref.ref(event.output))

    with HookSession(model, sink=sink, outputs=["**"]) as session:
        y = model(_tok())
        assert refs
        del y
        gc.collect()
        assert all(r() is None for r in refs)  # nothing alive while the session is active
    del session
    gc.collect()
    assert all(r() is None for r in refs)


def test_session_does_not_retain_graph_through_events() -> None:
    model = TinyMLP()
    grads: list[weakref.ref[torch.Tensor]] = []

    def sink(event: HookEvent) -> None:
        grads.append(weakref.ref(event.output))

    with HookSession(model, sink=sink, outputs=["shared.linear"]):
        loss = model(_mlp_x()).sum()
        del loss
        gc.collect()
        assert all(r() is None for r in grads)


# ----------------------------------------------------------------- aliases


class Aliased(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        shared = nn.Linear(2, 2)
        self.left = shared
        self.right = shared
        self.other = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.other(self.right(self.left(x)))
        return out


@pytest.mark.parametrize("patterns", [["left"], ["right"], ["left", "right"], ["*"], ["**"]])
def test_aliased_modules_are_refused_by_default(patterns: list[str]) -> None:
    model = Aliased()
    with (
        pytest.raises(AliasSiteAmbiguityError, match="several paths"),
        HookSession(model, sink=Recorder(), outputs=patterns),
    ):
        pass
    assert hook_count(model) == 0


def test_non_aliased_modules_of_an_aliasing_model_are_fine() -> None:
    model = Aliased()
    rec = Recorder()
    with HookSession(model, sink=rec, outputs=["other"]):
        model(torch.ones(1, 2))
    assert rec.events == [("output", ("other",), 0, 0)]


def test_alias_group_mode_observes_each_execution_once_without_a_path() -> None:
    model = Aliased()
    seen: list[HookEvent] = []
    with HookSession(
        model, sink=seen.append, outputs=["left", "right"], alias_policy=AliasPolicy.GROUP
    ) as session:
        assert session.installed_hooks == 2 + 4  # one physical module, hooked once
        model(torch.ones(1, 2))
    assert [(e.paths, e.path_specific, e.call_index) for e in seen] == [
        (("left", "right"), False, 0),
        (("left", "right"), False, 1),
    ]
    with pytest.raises(AliasSiteAmbiguityError):
        _ = seen[0].site


# ----------------------------------------------------------------- live resolution


def test_sites_are_resolved_against_the_live_model_on_entry() -> None:
    model = TinyMLP()
    session = HookSession(model, sink=Recorder(), outputs=["head"])
    replacement = nn.Linear(8, 3)
    model.head = replacement  # changed after construction, before entry
    with session:
        assert len(replacement._forward_hooks) == 2  # output + cleanup
    assert hook_count(model) == 0


def test_cleanup_when_a_base_exception_escapes_from_deep_inside() -> None:
    class Interrupt(BaseException):
        pass

    model = TinyMLP()

    def interrupting(module: nn.Module, args: Any) -> None:
        raise Interrupt

    user = model.head.register_forward_pre_hook(interrupting)
    with pytest.raises(Interrupt), HookSession(model, sink=Recorder(), outputs=["**", ""]):
        model(_mlp_x())
    user.remove()
    assert hook_count(model) == 0


def test_cleanup_when_the_sink_raises_for_an_input_event() -> None:
    model = TinyCNN()

    def sink(event: HookEvent) -> None:
        raise LookupError("sink rejected input")

    with pytest.raises(LookupError), HookSession(model, sink=sink, inputs=["stem.conv"]):
        model(torch.zeros(1, 1, 8, 8))
    assert hook_count(model) == 0


def test_user_hooks_on_the_root_survive() -> None:
    model = TinyTransformer()
    fired: list[str] = []
    model.register_forward_pre_hook(lambda m, a: fired.append("root_pre"))
    model.register_forward_hook(lambda m, a, o: fired.append("root_post"))
    with HookSession(model, sink=Recorder(), inputs=[""], outputs=[""]):
        model(_tok())
    fired.clear()
    model(_tok())
    assert fired == ["root_pre", "root_post"]
    assert hook_count(model) == 2


def test_alias_group_emits_one_event_per_execution_for_any_selection() -> None:
    model = Aliased()
    seen: list[HookEvent] = []
    with HookSession(model, sink=seen.append, outputs=["**"], alias_policy=AliasPolicy.GROUP):
        model(torch.ones(1, 2))
    grouped = [e for e in seen if not e.path_specific]
    assert len(grouped) == 2  # two executions (left, right), not four
    assert [e.call_index for e in grouped] == [0, 1]


# ----------------------------------------------------------------- M1.5 fix: invocation guard


class RecursiveWithFailingInner(nn.Module):
    """Outer call recurses; the first inner attempt is rejected by a user pre-hook."""

    def forward(self, x: torch.Tensor, depth: int = 0) -> torch.Tensor:
        if depth == 0:
            with contextlib.suppress(RuntimeError):
                self(x, depth=1)  # user pre-hook raises for depth == 1
            inner: torch.Tensor = self(x, depth=2)  # a later inner call succeeds
            return inner + 1
        return x * 2


def _reject_depth_1(module: nn.Module, args: Any, kwargs: dict[str, Any]) -> None:
    if kwargs.get("depth") == 1:
        raise RuntimeError("user pre-hook rejected the inner call")


@pytest.mark.parametrize("as_child", [False, True])
def test_user_pre_hook_failure_inside_recursion_keeps_outer_frame(as_child: bool) -> None:
    core = RecursiveWithFailingInner()
    model: nn.Module = nn.Sequential(core) if as_child else core
    path = "0" if as_child else ""
    user = core.register_forward_pre_hook(_reject_depth_1, with_kwargs=True)
    rec = Recorder()
    with HookSession(model, sink=rec, inputs=[path], outputs=[path]):
        y = model(torch.ones(1, 2))
        y2 = model(torch.ones(1, 2))
    user.remove()
    assert torch.equal(y, torch.full((1, 2), 3.0))
    assert torch.equal(y2, y)
    expected_pass = [
        ("input", (path,), 0, 0),  # outer
        # inner attempt: call 1 consumed by the guard, no INPUT event (rejected before observation)
        ("input", (path,), 0, 2),
        ("output", (path,), 0, 2),
        ("output", (path,), 0, 0),  # outer output still paired with call 0
    ]
    second = [(io, p, 1, c) for io, p, _, c in expected_pass]
    assert rec.events == expected_pass + second
    assert hook_count(model) == 0


def test_rejected_attempt_consumes_call_index_without_input_event() -> None:
    model = TinyMLP()
    state = {"reject": True}

    def reject_once(module: nn.Module, args: Any) -> None:
        if state["reject"]:
            state["reject"] = False
            raise RuntimeError("rejected")

    user = model.shared.register_forward_pre_hook(reject_once)
    rec = Recorder()
    with HookSession(model, sink=rec, inputs=["shared"]):
        with pytest.raises(RuntimeError, match="rejected"):
            model(_mlp_x())
        model.shared(torch.ones(1, 8))  # out of pass
        model(_mlp_x())
    user.remove()
    assert rec.events == [
        # pass 0: call 0 attempted and rejected -> nothing observed
        ("input", ("shared",), -1, 0),
        ("input", ("shared",), 1, 0),
        ("input", ("shared",), 1, 1),
    ]


def test_sessions_refuse_global_forward_pre_hooks() -> None:
    model = TinyMLP()
    handle = nn.modules.module.register_module_forward_pre_hook(lambda m, a: None)
    try:
        with (
            pytest.raises(HookSessionError, match="global module forward pre-hooks"),
            HookSession(model, sink=Recorder(), outputs=["shared"]),
        ):
            pass
    finally:
        handle.remove()
    assert hook_count(model) == 0
    with HookSession(model, sink=Recorder(), outputs=["shared"]):
        pass
