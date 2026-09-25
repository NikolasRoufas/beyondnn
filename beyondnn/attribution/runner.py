"""``attribute()``: attribution evidence with provenance (Phase 3; ADR-030).

One call produces one :class:`~beyondnn.core.trace.TraceResult`:

1. Guards: every module in eval mode; no forward/backward hooks that BeyondNN did
   not install (they could alter activations or gradients unseen); an alias
   module path is refused.
2. The attribution passes run *outside tracing* (they are the method's internals),
   on detached clones, using only :func:`torch.autograd.grad`.
3. Afterwards the run is refused, never reported, if it changed the model
   (fingerprint, train/eval flags), parameter ``.grad`` (restored first), the
   caller's tensors (values, ``requires_grad``, ``.grad``), hooks, or the CPU RNG
   state (randomness would make the attribution unreproducible).
4. One CLEAN traced *reference pass* on the input (under ``no_grad``) records the
   observed input and output and the MEASURED activation of the attributed module
   call; the target must have the same value there as in the attribution passes.
5. The ``AttributionRecord`` (ATTRIBUTED) derives from the reference output and the
   attributed input/activation; its provenance is the reference pass's model,
   environment and conditions with ``grad_enabled=True`` and the method identity.
   Declared reductions, claim tests and limitations are added to the same trace.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from beyondnn.core.hooks import AliasSiteAmbiguityError
from beyondnn.core.sites import resolve_sites
from beyondnn.core.tensors import walk
from beyondnn.core.trace import (
    ExternalForwardHooksError,
    Recording,
    TraceResult,
    _refuse_foreign_hooks,
)
from beyondnn.interventions.metrics import Metric
from beyondnn.interventions.runner import _retained_ref, sample_id
from beyondnn.provenance.fingerprint import fingerprint_model
from beyondnn.schema import (
    ActivationRecord,
    AttributionBaseline,
    AttributionMethodSpec,
    AttributionRecord,
    AttributionReduction,
    BaselineKind,
    Claim,
    ClaimTestResult,
    ClaimTestSpec,
    ExecutionContext,
    ExecutionMode,
    InputRecord,
    JsonMap,
    MethodIdentity,
    ModelDeclaration,
    ProvenanceRecord,
    RecordRef,
    TraceLimitation,
)

from . import native
from .native import AttributionError, make_space
from .spec import At, Method, Reduce

__all__ = [
    "AttributionResult",
    "AutogradStateError",
    "StatefulAttributionError",
    "StochasticAttributionError",
    "attribute",
]


class StochasticAttributionError(AttributionError):
    """Randomness was consumed or the target was not reproducible across passes."""


class StatefulAttributionError(AttributionError):
    """The model is in training mode or its state changed during attribution."""


class AutogradStateError(StatefulAttributionError):
    """Attribution would have left gradients or caller tensors modified (restored, refused)."""


# ------------------------------------------------------------------ state guards


def _hook_dicts(model: nn.Module) -> list[Any]:
    import torch.nn.modules.module as module_impl

    dicts: list[Any] = [
        getattr(module_impl, name)
        for name in (
            "_global_forward_hooks",
            "_global_forward_pre_hooks",
            "_global_backward_hooks",
            "_global_backward_pre_hooks",
        )
    ]
    for m in model.modules():
        dicts += [m._forward_hooks, m._forward_pre_hooks, m._backward_hooks, m._backward_pre_hooks]
    return dicts


def _refuse_hooks(model: nn.Module) -> None:
    _refuse_foreign_hooks(model, frozenset(), "before attribution")
    found = [
        name or "<root>"
        for name, m in model.named_modules()
        if m._backward_hooks or m._backward_pre_hooks
    ]
    import torch.nn.modules.module as module_impl

    if module_impl._global_backward_hooks or module_impl._global_backward_pre_hooks:
        found.append("<global backward hooks>")
    found += [
        f"parameter {n}" for n, p in model.named_parameters() if getattr(p, "_backward_hooks", None)
    ]
    if found:
        raise ExternalForwardHooksError(
            f"backward/gradient hooks are registered ({', '.join(found[:5])}); they could alter "
            "gradients without being represented in provenance; remove them before attribution"
        )


@dataclass
class _Snapshot:
    fingerprint: Any
    rng: torch.Tensor
    training: list[bool]
    params: list[tuple[nn.Parameter, torch.Tensor | None, torch.Tensor | None]]
    leaves: list[tuple[torch.Tensor, bool, torch.Tensor | None, torch.Tensor | None]]
    sample: str
    hooks: int


def _tensor_leaves(inputs: tuple[Any, ...], kwargs: dict[str, Any]) -> list[torch.Tensor]:
    return [leaf.tensor for leaf in walk([list(inputs), kwargs])[0]]


def _snapshot(model: nn.Module, inputs: tuple[Any, ...], kwargs: dict[str, Any]) -> _Snapshot:
    params = []
    for p in {id(p): p for p in model.parameters()}.values():
        g = p.grad
        params.append((p, g, None if g is None else g.detach().clone()))
    leaves = []
    for t in _tensor_leaves(inputs, kwargs):
        g = t.grad if t.is_leaf else None
        leaves.append((t, t.requires_grad, g, None if g is None else g.detach().clone()))
    return _Snapshot(
        fingerprint=fingerprint_model(model),
        rng=torch.get_rng_state(),
        training=[m.training for m in model.modules()],
        params=params,
        leaves=leaves,
        sample=sample_id(*inputs, model_kwargs=kwargs),
        hooks=sum(len(d) for d in _hook_dicts(model)),
    )


def _verify(
    snap: _Snapshot, model: nn.Module, inputs: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    changed = [
        p
        for p, g, v in snap.params
        if p.grad is not g or (g is not None and v is not None and not torch.equal(g, v))
    ]
    changed_ids = {id(p) for p in changed}
    for p, _g, v in snap.params:  # restore before refusing
        if id(p) in changed_ids:
            p.grad = None if v is None else v
    if changed:
        raise AutogradStateError(
            f"attribution modified {len(changed)} parameter .grad value(s); restored and refused"
        )
    for t, requires_grad, g, v in snap.leaves:
        now = t.grad if t.is_leaf else None
        if (
            t.requires_grad != requires_grad
            or now is not g
            or (g is not None and v is not None and not torch.equal(g, v))
        ):
            raise AutogradStateError("attribution modified a caller tensor's autograd state")
    if sample_id(*inputs, model_kwargs=kwargs) != snap.sample:
        raise AutogradStateError("attribution modified the caller's input values in place")
    if sum(len(d) for d in _hook_dicts(model)) != snap.hooks:
        raise StatefulAttributionError("hooks were left registered or removed during attribution")
    if [m.training for m in model.modules()] != snap.training:
        raise StatefulAttributionError("train/eval flags changed during attribution")
    if not torch.equal(snap.rng, torch.get_rng_state()):
        raise StochasticAttributionError(
            "the attribution passes consumed random numbers; the attribution would not be "
            "reproducible (and random runs are never averaged silently)"
        )
    if fingerprint_model(model) != snap.fingerprint:
        raise StatefulAttributionError(
            "the model state (parameters/buffers) changed during attribution"
        )


def _check_eval(model: nn.Module) -> None:
    training = [name or "<root>" for name, m in model.named_modules() if m.training]
    if training:
        raise StatefulAttributionError(
            f"modules in training mode {training[:5]}: dropout/batch statistics would make the "
            "attribution stochastic or state-dependent; call model.eval()"
        )


def _check_layer_site(model: nn.Module, at: At) -> None:
    (resolved,) = resolve_sites(model, [at.module])
    if resolved.aliases:
        raise AliasSiteAmbiguityError(
            f"module {at.module!r} is also registered as {list(resolved.aliases)}; an "
            "attribution to one path would silently be an attribution to all of them"
        )


# ------------------------------------------------------------------ result


@dataclass(frozen=True, slots=True, eq=False)
class AttributionResult:
    """An attribution's evidence: the trace (source of truth) and its ``AttributionRecord``."""

    trace: TraceResult
    record: AttributionRecord

    @classmethod
    def from_trace(cls, trace: TraceResult) -> AttributionResult:
        records = [r for r in trace.records if isinstance(r, AttributionRecord)]
        if len(records) != 1:
            raise AttributionError(f"expected one AttributionRecord, found {len(records)}")
        return cls(trace, records[0])

    @property
    def value(self) -> torch.Tensor:
        """The raw attribution tensor (same shape as the attributed tensor)."""
        return self.trace.tensor(self.record.value)

    @property
    def method(self) -> AttributionMethodSpec:
        return self.record.method

    @property
    def target_value(self) -> float:
        return self.record.target_value

    @property
    def completeness_delta(self) -> float | None:
        """IG only: ``sum(attribution) - (F(x) - F(x'))``; a numerical diagnostic."""
        delta = self.record.diagnostics.get("completeness_delta")
        return float(delta) if isinstance(delta, (int, float)) else None

    @property
    def reductions(self) -> tuple[AttributionReduction, ...]:
        return tuple(r for r in self.trace.records if isinstance(r, AttributionReduction))

    def reduced(self, reduction: AttributionReduction) -> torch.Tensor:
        return self.trace.tensor(reduction.value)

    @property
    def claim_results(self) -> tuple[ClaimTestResult, ...]:
        return tuple(r for r in self.trace.records if isinstance(r, ClaimTestResult))

    @property
    def limitations(self) -> tuple[TraceLimitation, ...]:
        return self.trace.limitations


# ------------------------------------------------------------------ attribute


def _effective_spec(method: Method, at: At) -> AttributionMethodSpec:
    spec = method.spec
    if method.is_captum and at.is_layer and method.captum_class == "IntegratedGradients":
        params = dict(spec.params.to_plain()) | {"captum_class": "LayerIntegratedGradients"}
        return AttributionMethodSpec(
            name=spec.name,
            implementation=spec.implementation,
            implementation_version=spec.implementation_version,
            params=JsonMap(params),
        )
    return spec


def _reduce(tensor: torch.Tensor, reduction: Reduce) -> tuple[tuple[int, ...], torch.Tensor]:
    ndim = tensor.dim()
    if any(not -ndim <= d < ndim for d in reduction.dims):
        raise AttributionError(f"reduction dims {reduction.dims} out of range for {ndim} dims")
    dims = tuple(sorted({d % ndim for d in reduction.dims}))
    if len(dims) != len(reduction.dims):
        raise AttributionError(f"reduction dims {reduction.dims} repeat a dimension")
    t = tensor.double()
    if reduction.reduction == "sum":
        out = t.sum(dim=dims)
    elif reduction.reduction == "abs_sum":
        out = t.abs().sum(dim=dims)
    else:
        out = t.pow(2).sum(dim=dims).sqrt()
    return dims, out


def reproducibility_tolerance(reference: float, output: Any) -> float:
    """How far the attribution passes' target may differ from the traced pass and still
    be the same computation (ADR-036): the larger of 1e-6 relative to the target and 16
    rounding units (``finfo(dtype).eps``) of the largest finite output magnitude.

    Grad-enabled and no-grad forwards may use different kernels, which differ by a few
    rounding units of the output; a target that is a difference of large logits (a
    margin) turns that into a large relative error. State drift and randomness change
    the output by far more and are still refused."""
    bound = max(1e-6 * abs(reference), 1e-9)
    for leaf in walk(output)[0]:
        t = leaf.tensor.detach()
        if not t.is_floating_point() or t.numel() == 0:
            continue
        finite = t[torch.isfinite(t)]
        if finite.numel():
            eps = torch.finfo(t.dtype).eps
            bound = max(bound, 16 * eps * float(finite.abs().max()))
    return bound


def attribute(
    model: nn.Module,
    *inputs: Any,
    target: Metric,
    method: Method,
    at: At | None = None,
    sites: Sequence[str] = (),
    retention: str = "summary",
    declared_model: ModelDeclaration | None = None,
    model_kwargs: dict[str, Any] | None = None,
    reductions: Sequence[Reduce] = (),
    claims: Sequence[tuple[Claim, ClaimTestSpec]] = (),
) -> AttributionResult:
    """Attribute the scalar ``target`` to ``at`` (default: positional input 0) with
    ``method``; see the module docstring. ``target`` must be a built-in metric
    (``metrics.select``/``difference``/``mean``) that selects exactly one element:
    outputs are never summed implicitly. ``claims`` (declared before running) are
    tested with ``attribution_threshold`` in the same trace."""
    from .claims import evaluate_claim

    if not isinstance(target, Metric):
        raise TypeError(
            "target must be a scalar Metric (e.g. interventions.metrics.select([0, k])); "
            "a tensor output is never reduced implicitly"
        )
    if target.tensor_function is None:
        raise AttributionError("caller metrics cannot be attribution targets in Phase 3")
    if not isinstance(method, Method):
        raise TypeError("method must come from beyondnn.attribution (or .captum) constructors")
    at = at if at is not None else At()
    if not isinstance(at, At):
        raise TypeError("at must come from attribution.input() or attribution.layer()")
    if isinstance(sites, str):
        raise TypeError("sites must be a sequence of patterns, not a str")
    kwargs = dict(model_kwargs or {})
    inputs = tuple(inputs)
    _check_eval(model)
    _refuse_hooks(model)
    if at.is_layer:
        _check_layer_site(model, at)
    spec = _effective_spec(method, at)

    snapshot = _snapshot(model, inputs, kwargs)
    try:
        space = make_space(model, inputs, kwargs, target, at)
        if method.is_captum:
            from . import captum as captum_adapter  # optional dependency, imported on use

            outcome = captum_adapter.run(method, space)
        else:
            outcome = native.run(method, space)
    except BaseException:
        # restore parameter gradients; the original error is the one reported
        with contextlib.suppress(AttributionError):
            _verify(snapshot, model, inputs, kwargs)
        raise
    _verify(snapshot, model, inputs, kwargs)
    attribution = (
        outcome.attribution.detach().to("cpu").clone(memory_format=torch.contiguous_format)
    )
    if not bool(torch.isfinite(attribution).all()):
        raise AttributionError("the attribution is not finite")
    sample = sample_id(*inputs, model_kwargs=kwargs)
    base_tensor = None if method.baseline is None else method.baseline.tensor

    selected = [at.module] if at.is_layer else []
    recording = Recording(
        model,
        sites=[*selected, *[s for s in sites if s not in selected]],
        retention=retention,
        declared_model=declared_model,
    )
    with torch.no_grad(), recording:
        output = model(*inputs, **kwargs)
        reference_value = target(output)
        tolerance = reproducibility_tolerance(reference_value, output)
        del output

        def finalize(trace: TraceResult) -> None:
            if not abs(reference_value - outcome.value) <= tolerance:
                raise StochasticAttributionError(
                    f"the target is {reference_value!r} in the traced pass but "
                    f"{outcome.value!r} in the attribution passes (tolerance {tolerance:.3g}); "
                    "not reproducible"
                )
            (out,) = trace.outputs
            pass_index = out.pass_index
            assert pass_index is not None
            reference = trace.origin(out)
            if reference.model != snapshot.fingerprint:
                raise StatefulAttributionError("the model changed before the reference pass")
            if at.is_layer:
                calls = [
                    a
                    for a in trace.activations
                    if a.site == at.site() and a.pass_index == out.pass_index
                ]
                if len(calls) != getattr(space, "calls", -1):
                    raise native.RepeatedCallError(
                        f"the traced pass saw {len(calls)} calls of {at.module!r}, the "
                        f"attribution passes {getattr(space, 'calls', '?')}"
                    )
                attributed: ActivationRecord | InputRecord = next(
                    a for a in calls if a.call_index == at.call_index
                )
            else:
                (attributed,) = trace.inputs
            provenance = ProvenanceRecord(
                model=reference.model,
                environment=reference.environment,
                execution=ExecutionContext(
                    mode=ExecutionMode.CLEAN,
                    device=reference.execution.device,
                    training=False,
                    grad_enabled=True,
                    randomness=reference.execution.randomness,
                ),
                method=MethodIdentity(
                    name=f"{spec.implementation}:{spec.name}",
                    version=spec.implementation_version,
                    params=spec.params,
                ),
                declared_model=reference.declared_model,
            )
            trace._add(provenance)
            value_ref = _retained_ref(attribution, reference.execution.device)
            trace._add_tensor(value_ref.storage_key or "", attribution)
            record_baseline = None
            diagnostics: dict[str, float] = {}
            if method.baseline is not None:
                b = method.baseline
                b_ref = None
                if base_tensor is not None:
                    b_ref = _retained_ref(base_tensor, "cpu")
                    trace._add_tensor(b_ref.storage_key or "", base_tensor)
                record_baseline = AttributionBaseline(
                    kind=b.kind,
                    value=b_ref,
                    input_path=None if b.input_index is None else f"args[{b.input_index}]",
                )
                assert outcome.baseline_value is not None
                total = float(attribution.double().sum())
                diagnostics = {
                    "baseline_target_value": outcome.baseline_value,
                    "attribution_sum": total,
                    "completeness_delta": total - (reference_value - outcome.baseline_value),
                }
            record = AttributionRecord(
                method=spec,
                target=target.spec,
                site=at.site(),
                call_index=at.call_index,
                pass_index=pass_index,
                sample_id=sample,
                baseline=record_baseline,
                value=value_ref,
                target_value=reference_value,
                diagnostics=JsonMap(diagnostics),
                provenance_id=provenance.id,
                derived_from=(RecordRef.to(out), RecordRef.to(attributed)),
            )
            stored = trace._add(record)
            assert isinstance(stored, AttributionRecord)
            for reduction in reductions:
                dims, reduced = _reduce(attribution, reduction)
                reduced = reduced.contiguous()
                r_ref = _retained_ref(reduced, "cpu")
                trace._add_tensor(r_ref.storage_key or "", reduced)
                trace._add(
                    AttributionReduction(
                        reduction=reduction.reduction,
                        dims=dims,
                        value=r_ref,
                        scalar=float(reduced) if reduced.dim() == 0 else None,
                        provenance_id=provenance.id,
                        derived_from=(RecordRef.to(stored),),
                    )
                )
            _limitations(trace, stored, method, at, inputs, kwargs)
            for claim, claim_spec in claims:
                trace._add(claim)
                trace._add(claim_spec)
                trace._add(evaluate_claim(claim, claim_spec, stored, attribution))

        recording._finalizers.append(finalize)
    if not torch.equal(snapshot.rng, torch.get_rng_state()):
        raise StochasticAttributionError("the reference pass consumed random numbers")
    if sample_id(*inputs, model_kwargs=kwargs) != sample:
        raise AutogradStateError("the reference pass modified the caller's inputs in place")
    return AttributionResult.from_trace(recording.result)


def _limitations(
    trace: TraceResult,
    record: AttributionRecord,
    method: Method,
    at: At,
    inputs: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    codes: list[tuple[str, str]] = []
    baseline = record.baseline
    if method.name == "integrated_gradients" and baseline is not None:
        detail: str
        if baseline.kind is BaselineKind.ZERO:
            detail = "zero baseline (in the attributed tensor's space)"
        elif baseline.kind is BaselineKind.TENSOR:
            detail = f"caller baseline {baseline.value.content_digest if baseline.value else ''}"
        else:
            digest = baseline.value.content_digest if baseline.value else ""
            detail = f"layer baseline = activation on input baseline {digest}"
        codes.append(("ATTRIBUTION_BASELINE_ASSUMPTION", detail))
        p = record.method.params
        codes.append(
            (
                "ATTRIBUTION_NUMERICAL_APPROXIMATION",
                f"{p.get('rule')}, n_steps={p.get('n_steps')}, completeness_delta="
                f"{record.diagnostics.get('completeness_delta')}",
            )
        )
    if at.is_layer:
        codes.append(
            (
                "LAYER_ATTRIBUTION_PARTIAL_COVERAGE",
                f"{at.module} call {at.call_index} leaf {at.output_path or '<value>'}",
            )
        )
        if any(not t.is_floating_point() for t in _tensor_leaves(inputs, kwargs)):
            codes.append(
                (
                    "DISCRETE_INPUT_ATTRIBUTED_VIA_REPRESENTATION",
                    f"attributed to the output of {at.module}",
                )
            )
    for code, detail in codes:
        trace._add(TraceLimitation(code=code, detail=detail, applies_to=(record.id,)))
