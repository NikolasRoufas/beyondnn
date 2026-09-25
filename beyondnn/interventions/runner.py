"""Paired baseline-vs-intervention comparisons (Phase 2; ADR-028).

Each comparison is ONE ``recording()`` of the model (no second tracing system):

* PATCH only: a CLEAN *source* pass on the source inputs; the source activation is
  captured, retained, and referenced by the ``InterventionRecord``;
* a CLEAN *baseline* pass on the inputs;
* an *intervention* pass on the same inputs, whose provenance says
  ``execution_mode = INTERVENTION`` and ``intervention_id`` = the record's id. A
  BeyondNN-owned forward hook replaces exactly one tensor leaf of one call of the
  module output, before observation, so the recorded (MEASURED) activation is the
  replaced one that downstream computation received.

All passes run under ``torch.no_grad()``. The comparison is refused, never
reported, if it cannot be paired:

* any module is in training mode (``StatefulComparisonError``);
* the model state fingerprint, the recorded execution conditions, or the inputs
  (modified in place) differ between the baseline and intervention passes
  (``StatefulComparisonError``);
* the CPU RNG state changes during the baseline or intervention pass
  (``StochasticComparisonError``: randomness was consumed, so the two passes may
  differ for reasons other than the intervention);
* the declared call of the target module did not run, or ran more than once with
  the same call index (``InterventionNotAppliedError``).

The effect is ``intervention_value - baseline_value`` of a scalar metric.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from beyondnn.core.samples import SampleIdentityError
from beyondnn.core.samples import sample_id as _sample_id
from beyondnn.core.trace import Recording, TraceError, TraceResult
from beyondnn.provenance.fingerprint import tensor_bytes
from beyondnn.schema import (
    CausalEffect,
    Claim,
    ClaimSource,
    ClaimSourceKind,
    ClaimTestResult,
    ClaimTestSpec,
    Estimand,
    EstimandScope,
    InterventionOperation,
    InterventionRecord,
    MethodIdentity,
    ModelDeclaration,
    OutputRecord,
    ProvenanceRecord,
    RecordRef,
    Relation,
    Site,
    SiteIO,
    Subject,
    TensorRef,
    TraceLimitation,
)

from .claims import evaluate_claim
from .metrics import Metric, get_leaf
from .spec import Intervention

__all__ = [
    "ComparisonFamily",
    "InterventionError",
    "InterventionNotAppliedError",
    "InterventionResult",
    "StatefulComparisonError",
    "StochasticComparisonError",
    "intervene",
    "intervene_sample",
    "make_claim",
    "sample_id",
]


class InterventionError(TraceError):
    """An intervention could not be applied or compared faithfully."""


class StochasticComparisonError(InterventionError):
    """Randomness was consumed; the baseline/intervention difference would be confounded."""


class StatefulComparisonError(InterventionError):
    """Model state differs between (or is mutated by) the paired passes."""


class InterventionNotAppliedError(InterventionError):
    """The declared target call did not run exactly once."""


_TOKEN = re.compile(r'\[(\d+|"(?:[^"\\]|\\.)*")\]')


def _replace_leaf(value: Any, path: str, new: torch.Tensor) -> Any:
    if path == "":
        return new
    match = _TOKEN.match(path)
    if match is None:
        raise InterventionError(f"invalid output_path {path!r}")
    key_text, rest = match.group(1), path[match.end() :]
    if isinstance(value, (tuple, list)):
        index = int(key_text)
        items = list(value)
        items[index] = _replace_leaf(items[index], rest, new)
        if isinstance(value, list):
            return items
        make = getattr(type(value), "_make", None)  # namedtuples keep their type
        return make(items) if make is not None else tuple(items)
    if isinstance(value, dict):
        key: Any = json.loads(key_text) if key_text.startswith('"') else int(key_text)
        replaced = dict(value)
        replaced[key] = _replace_leaf(value[key], rest, new)
        return type(value)(replaced) if type(value) is not dict else replaced
    raise InterventionError(f"cannot descend into {type(value).__name__} at {path!r}")


def sample_id(*inputs: Any, model_kwargs: dict[str, Any] | None = None) -> str:
    """Deterministic identity of one input sample: SHA-256 over tensor dtypes, shapes, and
    bytes plus JSON scalars, in positional/keyword order (``beyondnn.core.samples``)."""
    try:
        return _sample_id(*inputs, model_kwargs=model_kwargs)
    except SampleIdentityError as exc:
        raise InterventionError(str(exc)) from None


def make_claim(
    intervention: Intervention,
    metric: Metric,
    relation: Relation,
    *inputs: Any,
    statement: str,
    model_kwargs: dict[str, Any] | None = None,
    source: ClaimSource | None = None,
) -> Claim:
    """A claim about ``intervention``'s site/leaf and ``metric`` on exactly this input
    (INSTANCE estimand), declared before the comparison runs."""
    return Claim(
        statement=statement,
        relation=relation,
        subject=Subject(site=Site(module=intervention.site, output_path=intervention.output_path)),
        target=metric.spec.target(),
        estimand=Estimand.instance(sample_id(*inputs, model_kwargs=model_kwargs)),
        source=source or ClaimSource(kind=ClaimSourceKind.USER),
    )


@dataclass(frozen=True, slots=True, eq=False)
class InterventionResult:
    """A comparison's evidence: the trace (source of truth) and its main ``CausalEffect``."""

    trace: TraceResult
    effect: CausalEffect

    @classmethod
    def from_trace(cls, trace: TraceResult) -> InterventionResult:
        """Rebuild from a (e.g. loaded) comparison trace: the FINITE_SAMPLE effect if any,
        else its single INSTANCE effect."""
        effects = [r for r in trace.records if isinstance(r, CausalEffect)]
        finite = [e for e in effects if e.estimand.scope is EstimandScope.FINITE_SAMPLE]
        chosen = finite or effects
        if len(chosen) != 1:
            raise InterventionError(f"expected one main CausalEffect, found {len(chosen)}")
        return cls(trace, chosen[0])

    @property
    def intervention(self) -> InterventionRecord:
        record = self.trace.get(self.effect.interventions[0].record_id)
        assert isinstance(record, InterventionRecord)
        return record

    @property
    def baseline_value(self) -> float:
        return self.effect.baseline_value

    @property
    def intervention_value(self) -> float:
        return self.effect.intervention_value

    @property
    def value(self) -> float:
        """``intervention_value - baseline_value``."""
        return self.effect.effect

    @property
    def instance_effects(self) -> tuple[CausalEffect, ...]:
        return tuple(
            r
            for r in self.trace.records
            if isinstance(r, CausalEffect) and r.estimand.scope is EstimandScope.INSTANCE
        )

    def _outputs(self, effect: CausalEffect) -> tuple[OutputRecord, OutputRecord]:
        outs = [self.trace.get(r.record_id) for r in effect.derived_from if r.kind == "output"]
        records = [o for o in outs if isinstance(o, OutputRecord)]
        assert len(records) == 2
        a, b = sorted(records, key=lambda o: o.pass_index or 0)
        return a, b

    @property
    def baseline_output(self) -> OutputRecord:
        return self._outputs(self.instance_effects[0])[0]

    @property
    def intervention_output(self) -> OutputRecord:
        return self._outputs(self.instance_effects[0])[1]

    @property
    def claim_results(self) -> tuple[ClaimTestResult, ...]:
        return tuple(r for r in self.trace.records if isinstance(r, ClaimTestResult))

    @property
    def limitations(self) -> tuple[TraceLimitation, ...]:
        return self.trace.limitations


class _Replacer:
    """The BeyondNN-owned hook that performs one replacement."""

    def __init__(
        self, recording: Recording, spec: Intervention, value: torch.Tensor | None
    ) -> None:
        self.recording = recording
        self.spec = spec
        self.value = value
        self.applied = 0
        self.capture: torch.Tensor | None = None

    def _leaf(self, module: nn.Module, output: Any) -> torch.Tensor | None:
        _, call = self.recording._session.current_invocation(module)
        if call != self.spec.call_index:
            return None
        try:
            return get_leaf(output, self.spec.output_path)
        except ValueError as exc:
            raise InterventionError(str(exc)) from None

    def capture_hook(self, module: nn.Module, args: Any, kwargs: Any, output: Any) -> None:
        leaf = self._leaf(module, output)
        if leaf is not None:
            if self.capture is not None:
                raise InterventionNotAppliedError("the source call ran more than once")
            self.capture = leaf.detach().to("cpu").clone(memory_format=torch.contiguous_format)

    def perturb(self, leaf: torch.Tensor) -> torch.Tensor:
        """The replaced leaf: the operation's values at the replaced units (all units
        when ``units`` is None; the complement of ``units`` when ``retain``)."""
        op = self.spec.operation
        if op is InterventionOperation.ZERO:
            full = torch.zeros_like(leaf)
        elif op is InterventionOperation.CONSTANT and self.value is None:
            assert self.spec.constant is not None
            full = torch.full_like(leaf, self.spec.constant)
        else:
            assert self.value is not None
            if self.value.shape != leaf.shape or self.value.dtype != leaf.dtype:
                raise InterventionError(
                    f"replacement of shape {tuple(self.value.shape)} / {self.value.dtype} does not "
                    f"match the activation {tuple(leaf.shape)} / {leaf.dtype}; no broadcasting"
                )
            full = self.value.to(leaf.device)
        units = self.spec.units
        if units is None:
            return full
        if leaf.dim() == 0 or max(units) >= leaf.shape[-1]:
            raise InterventionError(
                f"units {list(units)} are out of range for a leaf of shape {tuple(leaf.shape)}"
            )
        mask = torch.zeros(leaf.shape[-1], dtype=torch.bool, device=leaf.device)
        mask[list(units)] = True
        if self.spec.retain:
            mask = ~mask
        return torch.where(mask, full, leaf)

    def replace_hook(self, module: nn.Module, args: Any, kwargs: Any, output: Any) -> Any:
        leaf = self._leaf(module, output)
        if leaf is None:
            return None
        new = self.perturb(leaf)
        self.applied += 1
        return _replace_leaf(output, self.spec.output_path, new)

    def perturb_inputs(self, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        """Model-input interventions: the positional inputs with ``args[i]`` replaced."""
        index = int(self.spec.output_path[len("args[") : -1])
        if index >= len(inputs) or not isinstance(inputs[index], torch.Tensor):
            raise InterventionError(f"positional input {index} is not a tensor input of this call")
        args = list(inputs)
        args[index] = self.perturb(inputs[index].detach())
        self.applied += 1
        return tuple(args)


def _retained_ref(tensor: torch.Tensor, device: str) -> TensorRef:
    digest = "sha256:" + hashlib.sha256(tensor_bytes(tensor)).hexdigest()
    return TensorRef(
        shape=tuple(tensor.shape),
        dtype=str(tensor.dtype).removeprefix("torch."),
        device=device,
        storage_key=digest,
        content_digest=digest,
    )


def _check_not_training(model: nn.Module) -> None:
    training = [name or "<root>" for name, m in model.named_modules() if m.training]
    if training:
        raise StatefulComparisonError(
            f"modules in training mode {training[:5]}: stochastic/stateful training behaviour "
            "(dropout, batch statistics) would confound the comparison; call model.eval()"
        )


def _unique_sites(target: str, sites: Sequence[str]) -> list[str]:
    if isinstance(sites, str):
        raise TypeError("sites must be a sequence of patterns, not a str")
    if target == "":  # a model-input intervention: the root input is always observed
        return list(dict.fromkeys(sites))
    return [target, *[s for s in sites if s != target]]


def _run_pass(
    recording: Recording, model: nn.Module, inputs: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[int, Any]:
    before = set(recording._passes)
    output = model(*inputs, **kwargs)
    (index,) = set(recording._passes) - before
    return index, output


def _baseline_pass(
    recording: Recording,
    model: nn.Module,
    metric: Metric,
    inputs: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[int, float]:
    rng = torch.get_rng_state()
    before = sample_id(*inputs, model_kwargs=kwargs)
    base_pass, base_out = _run_pass(recording, model, inputs, kwargs)
    if sample_id(*inputs, model_kwargs=kwargs) != before:
        raise StatefulComparisonError(
            "the baseline pass modified its inputs in place; the intervention pass would "
            "not see the same input"
        )
    base_value = metric(base_out)
    del base_out
    if not torch.equal(rng, torch.get_rng_state()):
        raise StochasticComparisonError(
            "the baseline pass consumed random numbers; its difference from the "
            "intervention pass would not be attributable to the intervention"
        )
    return base_pass, base_value


class _Experiment:
    def __init__(
        self, recording: Recording, model: nn.Module, spec: Intervention, metric: Metric
    ) -> None:
        self.recording = recording
        self.model = model
        self.spec = spec
        self.metric = metric
        self.module = model.get_submodule(spec.site)
        self.record: InterventionRecord | None = None
        self.value: torch.Tensor | None = None
        self.source_id: str | None = None

    @property
    def trace(self) -> TraceResult:
        return self.recording._trace

    def _pass(self, inputs: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[int, Any]:
        return _run_pass(self.recording, self.model, inputs, kwargs)

    def prepare(self) -> InterventionRecord:
        spec = self.spec
        site = (
            Site(module="", io=SiteIO.INPUT, output_path=spec.output_path)
            if spec.on_input
            else Site(module=spec.site, output_path=spec.output_path)
        )
        if spec.operation is InterventionOperation.PATCH:
            replacer = _Replacer(self.recording, spec, None)
            with self.recording._session.owned_forward_hook(self.module, replacer.capture_hook):
                source_pass, _ = self._pass(spec.source_inputs, spec.source_kwargs)
            if replacer.capture is None:
                raise InterventionNotAppliedError(
                    f"the source execution never produced call {spec.call_index} of {spec.site!r}"
                )
            source = self.trace.activation(
                spec.site,
                output_path=spec.output_path,
                pass_index=source_pass,
                call_index=spec.call_index,
            )
            if source.value.shape != tuple(replacer.capture.shape):
                raise InterventionError("captured source activation does not match its record")
            self.value = replacer.capture
            ref = _retained_ref(self.value, source.value.device)
            self.trace._add_tensor(ref.storage_key or "", self.value)
            self.source_id = sample_id(*spec.source_inputs, model_kwargs=spec.source_kwargs)
            return InterventionRecord(
                site=site,
                call_index=spec.call_index,
                operation=spec.operation,
                value=ref,
                source=RecordRef.to(source),
                units=spec.units,
                retain=spec.retain,
            )
        if spec.tensor is not None:
            self.value = spec.tensor
            ref = _retained_ref(self.value, "cpu")
            self.trace._add_tensor(ref.storage_key or "", self.value)
            return InterventionRecord(
                site=site,
                call_index=spec.call_index,
                operation=spec.operation,
                value=ref,
                units=spec.units,
                retain=spec.retain,
            )
        return InterventionRecord(
            site=site,
            call_index=spec.call_index,
            operation=spec.operation,
            constant=spec.constant,
            units=spec.units,
            retain=spec.retain,
        )

    def compare(
        self, inputs: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[int, float, int, float]:
        base_pass, base_value = self.baseline(inputs, kwargs)
        int_pass, int_value = self.intervened(inputs, kwargs, base_pass)
        return base_pass, base_value, int_pass, int_value

    def baseline(self, inputs: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[int, float]:
        """The CLEAN baseline pass (shared by every intervention on this input)."""
        return _baseline_pass(self.recording, self.model, self.metric, inputs, kwargs)

    def intervened(
        self, inputs: tuple[Any, ...], kwargs: dict[str, Any], base_pass: int
    ) -> tuple[int, float]:
        """One INTERVENTION pass on ``inputs``, checked against the baseline ``base_pass``."""
        assert self.record is not None
        rng = torch.get_rng_state()
        before = sample_id(*inputs, model_kwargs=kwargs)
        replacer = _Replacer(self.recording, self.spec, self.value)
        self.recording._next_intervention = self.record
        try:
            if self.spec.on_input:
                int_pass, int_out = self._pass(replacer.perturb_inputs(inputs), kwargs)
            else:
                with self.recording._session.owned_forward_hook(self.module, replacer.replace_hook):
                    int_pass, int_out = self._pass(inputs, kwargs)
        finally:
            self.recording._next_intervention = None
        int_value = self.metric(int_out)
        del int_out
        if sample_id(*inputs, model_kwargs=kwargs) != before:
            raise StatefulComparisonError("the intervention pass modified its inputs in place")
        if not torch.equal(rng, torch.get_rng_state()):
            raise StochasticComparisonError("the intervention pass consumed random numbers")
        if replacer.applied != 1:
            raise InterventionNotAppliedError(
                f"call {self.spec.call_index} of {self.spec.site!r} ran {replacer.applied} times "
                "in the intervention pass (expected exactly once)"
            )
        base_prov = self.trace.origin(self._input(base_pass))
        int_prov = self.trace.origin(self._input(int_pass))
        if int_prov.model != base_prov.model:
            raise StatefulComparisonError(
                "the model state changed between the baseline and intervention passes "
                "(e.g. buffers updated during forward); the passes are not paired"
            )
        b, i = base_prov.execution, int_prov.execution
        if (b.training, b.grad_enabled, b.device, b.randomness) != (
            i.training,
            i.grad_enabled,
            i.device,
            i.randomness,
        ):
            raise StatefulComparisonError(
                "execution conditions (training/grad/device/randomness) differ between the "
                "baseline and intervention passes"
            )
        return int_pass, int_value

    def _input(self, pass_index: int) -> Any:
        return next(r for r in self.trace.inputs if r.pass_index == pass_index)

    def _output(self, pass_index: int) -> OutputRecord:
        return next(r for r in self.trace.outputs if r.pass_index == pass_index)

    def effect_provenance(self, int_pass: int, method: str) -> ProvenanceRecord:
        intervened = self.trace.origin(self._input(int_pass))
        provenance = ProvenanceRecord(
            model=intervened.model,
            environment=intervened.environment,
            execution=intervened.execution,
            method=MethodIdentity(name=method, version="1"),
            declared_model=intervened.declared_model,
        )
        stored = self.trace._add(provenance)
        assert isinstance(stored, ProvenanceRecord)
        return stored

    def instance_effect(
        self, sample: str, base_pass: int, base_value: float, int_pass: int, int_value: float
    ) -> CausalEffect:
        assert self.record is not None
        provenance = self.effect_provenance(int_pass, "intervention_effect")
        effect = CausalEffect(
            interventions=(RecordRef.to(self.record),),
            metric=self.metric.spec,
            estimand=Estimand.instance(sample),
            baseline_value=base_value,
            intervention_value=int_value,
            effect=int_value - base_value,
            provenance_id=provenance.id,
            derived_from=(
                RecordRef.to(self._output(base_pass)),
                RecordRef.to(self._output(int_pass)),
            ),
        )
        stored = self.trace._add(effect)
        assert isinstance(stored, CausalEffect)
        return stored

    def limitations(self, effect: CausalEffect, target_samples: set[str]) -> None:
        op = self.spec.operation
        codes: list[str] = []
        if op is InterventionOperation.ZERO:
            codes.append("ZERO_ABLATION_MAY_BE_OOD")
        elif op is InterventionOperation.CONSTANT:
            codes.append("CONSTANT_REPLACEMENT_MAY_BE_OOD")
        elif self.source_id not in target_samples:
            codes.append("PATCH_SOURCE_CONTEXT_DIFFERS")
        if not self.metric.spec.builtin:
            codes.append("CUSTOM_METRIC_UNVERIFIED")
        for code in codes:
            self.trace._add(TraceLimitation(code=code, applies_to=(effect.id,)))


def intervene(
    model: nn.Module,
    *inputs: Any,
    intervention: Intervention,
    metric: Metric,
    sites: Sequence[str] = (),
    retention: str = "summary",
    declared_model: ModelDeclaration | None = None,
    model_kwargs: dict[str, Any] | None = None,
    claims: Sequence[tuple[Claim, ClaimTestSpec]] = (),
) -> InterventionResult:
    """Compare one input's baseline pass with an intervened pass (see module docstring).

    Returns an :class:`InterventionResult` whose INSTANCE ``CausalEffect`` is
    INTERVENTIONAL. ``claims`` (declared before running) are tested with
    ``intervention_threshold`` and their results stored in the same trace.
    """
    if not isinstance(intervention, Intervention):
        raise TypeError("intervention must be built with zero(), constant() or patch()")
    if not isinstance(metric, Metric):
        raise TypeError("metric must be a beyondnn.interventions.metrics Metric")
    _check_not_training(model)
    kwargs = dict(model_kwargs or {})
    sample = sample_id(*inputs, model_kwargs=kwargs)
    recording = Recording(
        model,
        sites=_unique_sites(intervention.site, sites),
        retention=retention,
        declared_model=declared_model,
    )
    experiment = _Experiment(recording, model, intervention, metric)
    with torch.no_grad(), recording:
        experiment.record = experiment.prepare()
        base_pass, base_value, int_pass, int_value = experiment.compare(tuple(inputs), kwargs)

        def finalize(trace: TraceResult) -> None:
            effect = experiment.instance_effect(sample, base_pass, base_value, int_pass, int_value)
            experiment.limitations(effect, {sample})
            _add_claims(trace, claims, effect, experiment)

        recording._finalizers.append(finalize)
    return InterventionResult.from_trace(recording.result)


def intervene_sample(
    model: nn.Module,
    samples: Sequence[Any],
    *,
    intervention: Intervention,
    metric: Metric,
    sites: Sequence[str] = (),
    retention: str = "summary",
    declared_model: ModelDeclaration | None = None,
    claims: Sequence[tuple[Claim, ClaimTestSpec]] = (),
) -> InterventionResult:
    """Exact finite-sample comparison: one paired comparison per sample (a tensor or a tuple
    of positional inputs), then their mean as a FINITE_SAMPLE effect (INTERVENTIONAL:
    it describes exactly these samples and nothing beyond them). ZERO/CONSTANT only."""
    if intervention.operation is InterventionOperation.PATCH:
        raise InterventionError("finite-sample patching is not supported in Phase 2")
    if not isinstance(metric, Metric):
        raise TypeError("metric must be a beyondnn.interventions.metrics Metric")
    items = [s if isinstance(s, tuple) else (s,) for s in samples]
    if not items:
        raise InterventionError("intervene_sample needs at least one sample")
    _check_not_training(model)
    ids = [sample_id(*s) for s in items]
    sample_set = "sample:" + hashlib.sha256("\n".join(ids).encode()).hexdigest()
    recording = Recording(
        model,
        sites=_unique_sites(intervention.site, sites),
        retention=retention,
        declared_model=declared_model,
    )
    experiment = _Experiment(recording, model, intervention, metric)
    with torch.no_grad(), recording:
        experiment.record = experiment.prepare()
        runs = [experiment.compare(s, {}) for s in items]

        def finalize(trace: TraceResult) -> None:
            parts = [experiment.instance_effect(i, *r) for i, r in zip(ids, runs, strict=True)]
            n = len(parts)
            base = sum(p.baseline_value for p in parts) / n
            inter = sum(p.intervention_value for p in parts) / n
            mean_effect = sum(p.effect for p in parts) / n
            assert experiment.record is not None
            provenance = experiment.effect_provenance(runs[-1][2], "intervention_effect_mean")
            aggregate = CausalEffect(
                interventions=(RecordRef.to(experiment.record),),
                metric=metric.spec,
                estimand=Estimand.finite_sample(sample_set, n, "mean"),
                baseline_value=base,
                intervention_value=inter,
                effect=mean_effect,
                provenance_id=provenance.id,
                derived_from=tuple(RecordRef.to(p) for p in parts),
            )
            stored = trace._add(aggregate)
            assert isinstance(stored, CausalEffect)
            experiment.limitations(stored, set(ids))
            _add_claims(trace, claims, stored, experiment)

        recording._finalizers.append(finalize)
    return InterventionResult.from_trace(recording.result)


def _add_claims(
    trace: TraceResult,
    claims: Sequence[tuple[Claim, ClaimTestSpec]],
    effect: CausalEffect,
    experiment: _Experiment,
) -> None:
    assert experiment.record is not None
    for claim, spec in claims:
        trace._add(claim)
        trace._add(spec)
        trace._add(
            evaluate_claim(
                claim, spec, effect, experiment.record, provenance_id=effect.provenance_id or ""
            )
        )


@dataclass(frozen=True, slots=True, eq=False)
class ComparisonFamily:
    """Several interventions compared with one shared baseline pass per input, all in ONE
    recording (``compare_family``). ``effects[g][j]`` is the INSTANCE ``CausalEffect`` of
    intervention ``j`` of group ``g``; ``records[g][j]`` its ``InterventionRecord``."""

    trace: TraceResult
    samples: tuple[str, ...]
    baseline_values: tuple[float, ...]
    baseline_passes: tuple[int, ...]
    intervention_passes: tuple[tuple[int, ...], ...]
    records: tuple[tuple[InterventionRecord, ...], ...]
    effects: tuple[tuple[CausalEffect, ...], ...]


def compare_family(
    model: nn.Module,
    groups: Sequence[tuple[tuple[Any, ...], dict[str, Any], Sequence[Intervention]]],
    metric: Metric,
    *,
    sites: Sequence[str] = (),
    retention: str = "summary",
    declared_model: ModelDeclaration | None = None,
    extend: Callable[[TraceResult, ComparisonFamily], None] | None = None,
) -> ComparisonFamily:
    """Paired comparisons for many interventions (Phase-5 orchestration; ADR-032).

    Each group is ``(inputs, model_kwargs, interventions)``: one CLEAN baseline pass on
    ``inputs``, then one INTERVENTION pass per intervention, each checked against that
    baseline with every Phase-2 pairing check. A group may have no interventions (its
    baseline is still recorded). ``extend(trace, family)`` runs inside the recording's
    finalisation, so callers can add records that reference the effects to the same
    trace. Nothing new is estimated: every effect is an exact INSTANCE effect.
    """
    if not isinstance(metric, Metric):
        raise TypeError("metric must be a beyondnn.interventions.metrics Metric")
    if isinstance(sites, str):
        raise TypeError("sites must be a sequence of patterns, not a str")
    _check_not_training(model)
    planned = []
    for inputs, kwargs, specs in groups:
        specs = tuple(specs)
        if not all(isinstance(s, Intervention) for s in specs):
            raise TypeError("interventions must be built with the beyondnn.interventions builders")
        planned.append((tuple(inputs), dict(kwargs), specs))
    if not planned:
        raise InterventionError("compare_family needs at least one group")
    targets = [s.site for _, _, specs in planned for s in specs if s.site]
    all_sites = list(dict.fromkeys([*targets, *[s for s in sites if s]]))
    recording = Recording(
        model, sites=all_sites, retention=retention, declared_model=declared_model
    )
    runs: list[tuple[str, int, float, list[tuple[_Experiment, int, float]]]] = []
    holder: list[ComparisonFamily] = []
    with torch.no_grad(), recording:
        for inputs, kwargs, specs in planned:
            sample = sample_id(*inputs, model_kwargs=kwargs)
            experiments = [_Experiment(recording, model, spec, metric) for spec in specs]
            for experiment in experiments:
                experiment.record = experiment.prepare()
            base_pass, base_value = _baseline_pass(recording, model, metric, inputs, kwargs)
            done = []
            for experiment in experiments:
                int_pass, int_value = experiment.intervened(inputs, kwargs, base_pass)
                done.append((experiment, int_pass, int_value))
            runs.append((sample, base_pass, base_value, done))

        def finalize(trace: TraceResult) -> None:
            effects = []
            for sample, base_pass, base_value, done in runs:
                row = []
                for experiment, int_pass, int_value in done:
                    effect = experiment.instance_effect(
                        sample, base_pass, base_value, int_pass, int_value
                    )
                    experiment.limitations(effect, {sample})
                    row.append(effect)
                effects.append(tuple(row))
            family = ComparisonFamily(
                trace=trace,
                samples=tuple(r[0] for r in runs),
                baseline_values=tuple(r[2] for r in runs),
                baseline_passes=tuple(r[1] for r in runs),
                intervention_passes=tuple(tuple(d[1] for d in r[3]) for r in runs),
                records=tuple(
                    tuple(d[0].record for d in r[3] if d[0].record is not None) for r in runs
                ),
                effects=tuple(effects),
            )
            holder.append(family)
            if extend is not None:
                extend(trace, family)

        recording._finalizers.append(finalize)
    recording.result  # noqa: B018 - raises if the recording failed
    return holder[0]
