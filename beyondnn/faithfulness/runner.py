"""Running faithfulness tests (Phase 5; ADR-033).

Every run is ONE recording built with the Phase-2 engine (``compare_family``): per
sample, a CLEAN baseline pass and one INTERVENTION pass per perturbation (the tested
selection, then each matched random control). Every perturbation is verified against
what was recorded (the perturbed input's exact ``sample_id``, or the retained
activation of the intervened site), and a perturbation that changed nothing is
reported as such. The records added to the same trace are:

* ``EvidenceSelection``: what was tested;
* the instantiated claim (from the declared template), its ``ClaimTestSpec`` and its
  ``ClaimTestResult`` (comprehensiveness/sufficiency), whose evidence is the
  INTERVENTIONAL effects;
* ``ProtocolResult``s for diagnostics (curves, counterexample and paired-control
  summaries);
* limitations (OOD replacement from Phase 2, site-relative sufficiency, selection ties).

No new evidence status is created, and no single score is computed.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

import beyondnn.interventions as iv
from beyondnn.attribution import AttributionResult, attribute
from beyondnn.core.samples import sample_id
from beyondnn.core.trace import TraceResult
from beyondnn.core.trace import trace as bnn_trace
from beyondnn.core.units import UnitError, unit_count, unit_mask, unit_values
from beyondnn.schema import (
    ActivationRecord,
    AspectOutcome,
    CheckOutcome,
    Claim,
    ClaimSource,
    ClaimSourceKind,
    ClaimTestResult,
    ClaimTestSpec,
    Estimand,
    EvidenceSelection,
    InterventionRecord,
    JsonMap,
    MethodIdentity,
    ModelDeclaration,
    Outcome,
    ProtocolResult,
    ProvenanceRecord,
    RecordRef,
    SelectionSource,
    Subject,
    TraceLimitation,
)

from .claims import evaluate
from .spec import (
    Controls,
    FaithfulnessError,
    Replacement,
    Selection,
    SelectionMismatchError,
    SelectionRule,
    TestTemplate,
)
from .stats import (
    control_fractions,
    quantile,
    sign_flip_p,
    stratified_subsets,
    uniform_permutations,
    uniform_subsets,
)

__all__ = [
    "CurveResult",
    "DatasetResult",
    "FaithfulnessResult",
    "PerturbationNotAppliedError",
    "curve",
    "run",
    "run_dataset",
]


class PerturbationNotAppliedError(FaithfulnessError):
    """The recorded execution does not show the requested perturbation."""


# ------------------------------------------------------------------ results


@dataclass(frozen=True, slots=True, eq=False)
class FaithfulnessResult:
    """One faithfulness claim test on one sample: its trace (source of truth) and the
    attribution results its selection came from (separate traces)."""

    trace: TraceResult
    attributions: tuple[AttributionResult, ...] = ()

    @property
    def result(self) -> ClaimTestResult:
        (r,) = [r for r in self.trace.records if isinstance(r, ClaimTestResult)]
        return r

    @property
    def claim(self) -> Claim:
        (c,) = [r for r in self.trace.records if isinstance(r, Claim)]
        return c

    @property
    def selection(self) -> EvidenceSelection:
        (s,) = [r for r in self.trace.records if isinstance(r, EvidenceSelection)]
        return s

    @property
    def outcome(self) -> Outcome:
        return self.result.outcome

    @property
    def drop(self) -> float:
        return float(self.result.statistics["drop"])  # type: ignore[arg-type]

    @property
    def statistics(self) -> JsonMap:
        return self.result.statistics

    @property
    def limitations(self) -> tuple[TraceLimitation, ...]:
        return self.trace.limitations


@dataclass(frozen=True, slots=True, eq=False)
class DatasetResult:
    """A claim template tested on each sample of a declared set, in one trace, plus the
    per-sample attribution results. Sample-level results stay separate; the summaries
    (``counterexample``, ``paired_control``) link to them."""

    trace: TraceResult
    attributions: tuple[AttributionResult, ...] = ()

    @property
    def results(self) -> tuple[ClaimTestResult, ...]:
        return tuple(r for r in self.trace.records if isinstance(r, ClaimTestResult))

    @property
    def summaries(self) -> tuple[ProtocolResult, ...]:
        return tuple(r for r in self.trace.records if isinstance(r, ProtocolResult))

    def summary(self, protocol: str) -> ProtocolResult:
        (s,) = [r for r in self.summaries if r.protocol == protocol]
        return s

    @property
    def counterexamples(self) -> tuple[str, ...]:
        found = self.summary("counterexample").measurements["counterexamples"]
        return tuple(found)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True, eq=False)
class CurveResult:
    """A removal or retention curve (``ProtocolResult``) and its trace."""

    trace: TraceResult
    attributions: tuple[AttributionResult, ...] = ()

    @property
    def protocol_result(self) -> ProtocolResult:
        (r,) = [r for r in self.trace.records if isinstance(r, ProtocolResult)]
        return r

    @property
    def points(self) -> tuple[int, ...]:
        return tuple(self.protocol_result.measurements["points"])  # type: ignore[arg-type]

    @property
    def drops(self) -> tuple[float, ...]:
        return tuple(self.protocol_result.measurements["drops"])  # type: ignore[arg-type]


# ------------------------------------------------------------------ perturbation plumbing


def _input_index(selection: Selection) -> int:
    return int(selection.site.output_path[len("args[") : -1])


def _intervention(
    selection: Selection, units: tuple[int, ...] | None, retain: bool, rep: Replacement
) -> iv.Intervention:
    axes = selection.unit_axes if units is not None else None
    if selection.is_input:
        index = _input_index(selection)
        if rep.tensor is None:
            return iv.zero_input(index, units=units, retain=retain, unit_axes=axes)
        return iv.constant_input(rep.tensor, index, units=units, retain=retain, unit_axes=axes)
    site = selection.site
    if rep.tensor is None:
        return iv.zero(
            site.module,
            output_path=site.output_path,
            call_index=selection.call_index,
            units=units,
            retain=retain,
            unit_axes=axes,
        )
    return iv.constant(
        site.module,
        rep.tensor,
        output_path=site.output_path,
        call_index=selection.call_index,
        units=units,
        retain=retain,
        unit_axes=axes,
    )


def _expected(
    original: torch.Tensor,
    units: tuple[int, ...] | None,
    retain: bool,
    rep: Replacement,
    unit_axes: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """The perturbed tensor, computed independently of the intervention engine."""
    replacement = torch.zeros_like(original) if rep.tensor is None else rep.tensor.to(original)
    if tuple(replacement.shape) != tuple(original.shape):
        raise FaithfulnessError(
            f"the replacement has shape {tuple(replacement.shape)}, the site "
            f"{tuple(original.shape)}; no broadcasting"
        )
    if units is None:
        return replacement.clone()
    if unit_axes is not None:
        mask = unit_mask(tuple(original.shape), unit_axes, units)
        if retain:
            mask = ~mask
        return torch.where(mask, replacement, original)
    out = original.clone()
    chosen = set(units)
    for u in range(original.shape[-1]):
        if (u in chosen) != retain:
            out[..., u] = replacement[..., u]
    return out


def _check_units(shape: tuple[int, ...], selection: Selection, where: str) -> None:
    n_units = selection.n_units
    if selection.unit_axes is not None:
        try:
            count = unit_count(shape, selection.unit_axes)
        except UnitError as exc:
            raise FaithfulnessError(f"{where}: {exc}") from None
        if count != n_units:
            raise SelectionMismatchError(
                f"{where} has {count} units over axes {selection.unit_axes}, the selection "
                f"{n_units}"
            )
        return
    if len(shape) == 0 or any(d != 1 for d in shape[:-1]):
        raise FaithfulnessError(
            f"{where} has shape {shape}; unit selection needs all non-last dimensions to be 1 "
            "unless unit_axes are declared (ADR-034)"
        )
    if shape[-1] != n_units:
        raise SelectionMismatchError(f"{where} has {shape[-1]} units, the selection {n_units}")


class _Verifier:
    """Checks every perturbation against the recorded execution; reports no-ops."""

    def __init__(
        self,
        trace: TraceResult,
        selection: Selection,
        rep: Replacement,
        inputs: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        self.trace = trace
        self.selection = selection
        self.rep = rep
        self.inputs = inputs
        self.kwargs = kwargs

    def _activation(self, pass_index: int) -> torch.Tensor:
        site = self.selection.site
        record = self.trace.activation(
            site.module,
            output_path=site.output_path,
            pass_index=pass_index,
            call_index=self.selection.call_index,
        )
        assert isinstance(record, ActivationRecord)
        return self.trace.tensor(record)

    def check(self, base_pass: int, record: InterventionRecord, int_pass: int) -> bool:
        """Raise unless pass ``int_pass`` shows exactly the perturbation; return no-op."""
        if self.selection.is_input:
            index = _input_index(self.selection)
            original = self.inputs[index]
            _check_units(tuple(original.shape), self.selection, "the input")
            perturbed = _expected(
                original.detach(), record.units, record.retain, self.rep, record.unit_axes
            )
            args = list(self.inputs)
            args[index] = perturbed
            expected_id = sample_id(*args, model_kwargs=self.kwargs)
            (seen,) = [i for i in self.trace.inputs if i.pass_index == int_pass]
            if seen.sample_id != expected_id:
                raise PerturbationNotAppliedError(
                    f"pass {int_pass} did not run on the requested perturbed input"
                )
            return bool(torch.equal(perturbed, original.detach()))
        base = self._activation(base_pass)
        _check_units(tuple(base.shape), self.selection, f"site {self.selection.site}")
        expected = _expected(base, record.units, record.retain, self.rep, record.unit_axes)
        if not torch.equal(self._activation(int_pass), expected):
            raise PerturbationNotAppliedError(
                f"the activation recorded in pass {int_pass} is not the requested perturbation"
            )
        return bool(torch.equal(expected, base))


def _provenance(
    trace: TraceResult, base_pass: int, name: str, params: dict[str, Any]
) -> ProvenanceRecord:
    (inp,) = [i for i in trace.inputs if i.pass_index == base_pass]
    ref = trace.origin(inp)
    record = ProvenanceRecord(
        model=ref.model,
        environment=ref.environment,
        execution=ref.execution,
        method=MethodIdentity(name=f"faithfulness:{name}", version="1", params=JsonMap(params)),
        declared_model=ref.declared_model,
    )
    stored = trace._add(record)
    assert isinstance(stored, ProvenanceRecord)
    return stored


def _selection_record(
    selection: Selection, sample: str, provenance: ProvenanceRecord, seed: int | None = None
) -> EvidenceSelection:
    return EvidenceSelection(
        site=selection.site,
        call_index=selection.call_index,
        sample_id=sample,
        source=selection.source,
        rule=selection.rule,
        order=selection.order,
        n_units=selection.n_units,
        k=selection.k,
        scores=selection.scores,
        source_record=selection.source_record,
        seed=seed,
        target=selection.target,
        unit_axes=selection.unit_axes,
        unit_reduction=selection.unit_reduction,
        eligible=selection.eligible,
        eligibility=selection.eligibility,
        provenance_id=provenance.id,
    )


def _tie_at_boundary(selection: Selection) -> bool:
    if selection.scores is None or selection.k is None or selection.k >= len(selection.order):
        return False
    key = [abs(s) if selection.rule == "abs_desc" else s for s in selection.scores]
    return key[selection.order[selection.k - 1]] == key[selection.order[selection.k]]


def _validate_selection(selection: Selection, test_target: Any, sample: str) -> Selection:
    if not isinstance(selection, Selection):
        raise TypeError("selection must come from faithfulness.top_k() / units()")
    if selection.k is None:
        raise FaithfulnessError("a claim test needs a selection with k (use top_k or units)")
    if selection.sample_id is not None and selection.sample_id != sample:
        raise SelectionMismatchError(
            f"the selection is about sample {selection.sample_id}, not the tested input "
            f"{sample}; evidence about another input is never used"
        )
    if selection.target is not None and selection.target != test_target:
        raise SelectionMismatchError("the selection was ranked for a different target")
    return selection


# ------------------------------------------------------------------ claim tests


def _claim(test: TestTemplate, selection: Selection, sample: str) -> Claim:
    source = (
        ClaimSource(kind=ClaimSourceKind.USER, detail="declared units; declared test template")
        if selection.source is SelectionSource.DECLARED
        else ClaimSource(
            kind=ClaimSourceKind.METHOD,
            detail=f"declared template instantiated with the top-{selection.k} "
            f"({selection.rule}) of {selection.source_record}",
        )
    )
    return Claim(
        statement=test.statement,
        relation=test.relation,
        subject=Subject(
            site=selection.site, units=selection.selected, unit_axes=selection.unit_axes
        ),
        target=test.target.spec.target(),
        estimand=Estimand.instance(sample),
        source=source,
    )


def _spec(
    test: TestTemplate, selection: Selection, magnitudes: list[float] | None = None
) -> ClaimTestSpec:
    declared_controls: dict[str, Any] | None = None
    if test.controls is not None:
        declared_controls = test.controls.identity()
        if magnitudes is not None:
            declared_controls["magnitudes"] = magnitudes
    return ClaimTestSpec(
        protocol=test.protocol,
        protocol_version=1,
        applicable_relations=(test.relation,),
        criteria=JsonMap(test.criteria),
        params=JsonMap(
            {
                "mode": test.mode,
                "level": "input" if selection.is_input else "internal",
                "call_index": selection.call_index,
                "k": selection.k,
                "n_units": selection.n_units,
                "replacement": test.replacement.identity(),
                "controls": declared_controls,
                "unit_axes": None if selection.unit_axes is None else list(selection.unit_axes),
            }
            # ADR-053: recorded only when declared, so specs without eligibility are unchanged
            | (
                {}
                if selection.eligible is None
                else {"eligible": list(selection.eligible), "eligibility": selection.eligibility}
            )
        ),
    )


def _plan(
    test: TestTemplate, selection: Selection, magnitudes: list[float] | None = None
) -> tuple[list[iv.Intervention], list[tuple[int, ...]]]:
    retain = test.mode == "retain"
    assert selection.k is not None
    if retain and selection.k == selection.n_units:
        raise FaithfulnessError("retaining every unit is vacuous: sufficiency is undefined here")
    control_sets: list[tuple[int, ...]] = []
    population = selection.population
    if test.controls is not None:
        if selection.k == len(population):
            raise FaithfulnessError(
                "matched random controls are degenerate when the selection covers every "
                "eligible unit"
            )
        if test.controls.match == "magnitude":
            if magnitudes is None:
                raise FaithfulnessError("magnitude-matched controls need the site's magnitudes")
            control_sets = stratified_subsets(
                magnitudes,
                selection.selected,
                test.controls.n,
                test.controls.seed,
                test.controls.strata,
                None if selection.eligible is None else population,
            )
        else:
            control_sets = uniform_subsets(
                selection.n_units,
                selection.k,
                test.controls.n,
                test.controls.seed,
                None if selection.eligible is None else population,
            )
    specs = [_intervention(selection, selection.selected, retain, test.replacement)]
    specs += [_intervention(selection, s, retain, test.replacement) for s in control_sets]
    return specs, control_sets


def _record_test(
    trace: TraceResult,
    family: iv.ComparisonFamily,
    group: int,
    test: TestTemplate,
    selection: Selection,
    verifier: _Verifier,
    magnitudes: list[float] | None = None,
) -> ClaimTestResult:
    sample = family.samples[group]
    base_pass = family.baseline_passes[group]
    records = family.records[group]
    effects = family.effects[group]
    no_ops = [
        verifier.check(base_pass, r, p)
        for r, p in zip(records, family.intervention_passes[group], strict=True)
    ]
    provenance = _provenance(trace, base_pass, test.protocol, {"k": selection.k})
    trace._add(_selection_record(selection, sample, provenance))
    claim = trace._add(_claim(test, selection, sample))
    spec = trace._add(_spec(test, selection, magnitudes))
    assert isinstance(claim, Claim)
    assert isinstance(spec, ClaimTestSpec)
    result = evaluate(
        claim,
        spec,
        effects[0],
        effects[1:],
        {r.id: r for r in records},
        no_op=no_ops[0],
        provenance_id=provenance.id,
    )
    stored = trace._add(result)
    assert isinstance(stored, ClaimTestResult)
    if test.mode == "retain" and not selection.is_input:
        trace._add(
            TraceLimitation(
                code="SITE_RELATIVE_SUFFICIENCY",
                detail=f"retention within {selection.site.module} only",
                applies_to=(stored.id,),
            )
        )
    if _tie_at_boundary(selection):
        trace._add(
            TraceLimitation(
                code="SELECTION_TIE_AT_BOUNDARY", detail=f"k={selection.k}", applies_to=(stored.id,)
            )
        )
    return stored


def _site_tensor(
    model: nn.Module, inputs: tuple[Any, ...], kwargs: dict[str, Any], selection: Selection
) -> torch.Tensor:
    """The clean site tensor: the positional input, or the module output captured by one
    public traced pass (``bnn.trace``, CPU retention)."""
    if selection.is_input:
        value = inputs[_input_index(selection)]
        assert isinstance(value, torch.Tensor)
        return value.detach()
    site = selection.site
    traced = bnn_trace(model, *inputs, model_kwargs=kwargs, sites=[site.module], retention="cpu")
    record = traced.activation(
        site.module, output_path=site.output_path, call_index=selection.call_index
    )
    return traced.tensor(record)


def _prepare(
    model: nn.Module,
    inputs: tuple[Any, ...],
    kwargs: dict[str, Any],
    selection: Selection,
    test: TestTemplate,
) -> list[float] | None:
    """Check the input site's units before any perturbation runs (internal sites are
    checked against the recorded activation) and, for magnitude-matched controls,
    return each unit's perturbation magnitude ||x_u - b_u||_2 (ADR-035)."""
    if selection.is_input:
        original = inputs[_input_index(selection)]
        if not isinstance(original, torch.Tensor):
            raise FaithfulnessError("the selected positional input is not a tensor")
        _check_units(tuple(original.shape), selection, "the input")
    if test.controls is None or test.controls.match != "magnitude":
        return None
    original = _site_tensor(model, inputs, kwargs, selection)
    _check_units(tuple(original.shape), selection, f"site {selection.site}")
    replacement = (
        torch.zeros_like(original)
        if test.replacement.tensor is None
        else test.replacement.tensor.to(original)
    )
    if tuple(replacement.shape) != tuple(original.shape):
        raise FaithfulnessError("the replacement does not have the site's exact shape")
    delta = original.double() - replacement.double()
    return [float(v) for v in unit_values(delta, selection.unit_axes, "l2").tolist()]


def _check_test(test: TestTemplate) -> None:
    if not isinstance(test, TestTemplate):
        raise TypeError("test must come from faithfulness.comprehensiveness() / sufficiency()")


def _sites(selection: Selection) -> list[str]:
    return [] if selection.is_input else [selection.site.module]


def run(
    model: nn.Module,
    *inputs: Any,
    test: TestTemplate,
    selection: Selection,
    model_kwargs: dict[str, Any] | None = None,
    declared_model: ModelDeclaration | None = None,
    attributions: Sequence[AttributionResult] = (),
) -> FaithfulnessResult:
    """Run ``test`` on one input for ``selection`` (see module docstring).

    ``attributions`` are the attribution results the selection came from (kept with the
    result so that composition can verify the selection).
    """
    _check_test(test)
    kwargs = dict(model_kwargs or {})
    inputs = tuple(inputs)
    sample = sample_id(*inputs, model_kwargs=kwargs)
    selection = _validate_selection(selection, test.target.spec, sample)
    magnitudes = _prepare(model, inputs, kwargs, selection, test)
    specs, _ = _plan(test, selection, magnitudes)

    def extend(trace: TraceResult, family: iv.ComparisonFamily) -> None:
        verifier = _Verifier(trace, selection, test.replacement, inputs, kwargs)
        _record_test(trace, family, 0, test, selection, verifier, magnitudes)

    family = iv.compare_family(
        model,
        [(inputs, kwargs, specs)],
        test.target,
        sites=_sites(selection),
        retention="cpu",
        declared_model=declared_model,
        extend=extend,
    )
    return FaithfulnessResult(family.trace, tuple(attributions))


# ------------------------------------------------------------------ dataset runs


def _select(
    model: nn.Module,
    rule: SelectionRule,
    inputs: tuple[Any, ...],
    kwargs: dict[str, Any],
    test: TestTemplate,
    declared_model: ModelDeclaration | None,
) -> tuple[Selection, AttributionResult | None]:
    from .spec import top_k

    if rule.fixed is not None:
        return rule.fixed, None
    assert rule.method is not None
    result = attribute(
        model,
        *inputs,
        target=test.target,
        method=rule.method,
        at=rule.at,
        model_kwargs=kwargs,
        declared_model=declared_model,
    )
    return top_k(result, k=rule.k, by=rule.by, unit_axes=rule.unit_axes, reduce=rule.reduce), result


def run_dataset(
    model: nn.Module,
    samples: Sequence[Any],
    *,
    test: TestTemplate,
    rule: SelectionRule,
    declared_model: ModelDeclaration | None = None,
    permutation_draws: int = 2000,
) -> DatasetResult:
    """Test the claim template on every declared sample (a tensor, or a tuple of
    positional inputs), in one trace. Attribution-based selections are computed
    explicitly per sample (``bnn.attribute``; separate traces). Adds the
    ``counterexample/v1`` summary (every CONTRADICTS kept) and, with controls, the
    ``paired_control/v1`` summary (per-sample paired differences, sign-flip p)."""
    _check_test(test)
    if not isinstance(rule, SelectionRule):
        raise TypeError("rule must come from faithfulness.selector() / fixed()")
    items = [s if isinstance(s, tuple) else (s,) for s in samples]
    if not items:
        raise FaithfulnessError("a dataset run needs at least one sample")
    ids = [sample_id(*s) for s in items]
    if len(set(ids)) != len(ids):
        raise FaithfulnessError("the declared samples contain duplicates")
    plans = []
    attributions = []
    for inputs, sample in zip(items, ids, strict=True):
        selection, attr = _select(model, rule, inputs, {}, test, declared_model)
        if attr is not None:
            attributions.append(attr)
        selection = _validate_selection(selection, test.target.spec, sample)
        magnitudes = _prepare(model, inputs, {}, selection, test)
        specs, _ = _plan(test, selection, magnitudes)
        plans.append((inputs, selection, specs, magnitudes))
    sites = sorted({s for _, sel, _, _ in plans for s in _sites(sel)})

    def extend(trace: TraceResult, family: iv.ComparisonFamily) -> None:
        results = []
        for group, (inputs, selection, _, magnitudes) in enumerate(plans):
            verifier = _Verifier(trace, selection, test.replacement, inputs, {})
            results.append(
                _record_test(trace, family, group, test, selection, verifier, magnitudes)
            )
        _dataset_summaries(trace, family, test, results, permutation_draws)

    family = iv.compare_family(
        model,
        [(inputs, {}, specs) for inputs, _, specs, _ in plans],
        test.target,
        sites=sites,
        retention="cpu",
        declared_model=declared_model,
        extend=extend,
    )
    return DatasetResult(family.trace, tuple(attributions))


def _dataset_summaries(
    trace: TraceResult,
    family: iv.ComparisonFamily,
    test: TestTemplate,
    results: list[ClaimTestResult],
    draws: int,
) -> None:
    provenance = _provenance(
        trace, family.baseline_passes[0], "dataset_summary", {"n_samples": len(results)}
    )
    outcomes = [r.outcome.value for r in results]
    counter = [
        s for s, r in zip(family.samples, results, strict=True) if r.outcome is Outcome.CONTRADICTS
    ]
    inconclusive = [
        s
        for s, r in zip(family.samples, results, strict=True)
        if r.outcome in (Outcome.INCONCLUSIVE, Outcome.NOT_APPLICABLE)
    ]
    held = sum(1 for r in results if r.outcome is Outcome.SUPPORTS)
    if counter:
        verdict = CheckOutcome.FAIL
    elif inconclusive:
        verdict = CheckOutcome.INDETERMINATE
    else:
        verdict = CheckOutcome.PASS
    refs = tuple(RecordRef.to(r) for r in results)
    trace._add(
        ProtocolResult(
            protocol="counterexample",
            protocol_version=1,
            params=JsonMap({"claim_protocol": test.protocol, "n_samples": len(results)}),
            criteria=JsonMap({"max_counterexamples": 0}),
            measurements=JsonMap(
                {
                    "samples": list(family.samples),
                    "results": [r.id for r in results],
                    "outcomes": outcomes,
                    "drops": [r.statistics["drop"] for r in results],
                    "counterexamples": counter,
                    "held": held,
                    "n": len(results),
                }
            ),
            outcomes=(
                AspectOutcome(
                    aspect="max_counterexamples",
                    outcome=verdict,
                    detail=f"held on {held} of {len(results)} declared samples; "
                    f"{len(counter)} counterexample(s); {len(inconclusive)} inconclusive",
                ),
            ),
            samples=tuple(family.samples),
            target=test.target.spec,
            provenance_id=provenance.id,
            derived_from=refs,
        )
    )
    if test.controls is None:
        return
    diffs = []
    for r in results:
        stats = r.statistics
        drops = stats.get("control_drops")
        if not isinstance(drops, tuple) or not drops:
            continue
        values = [float(d) for d in drops if isinstance(d, (int, float))]
        mean_control = sum(values) / len(values)
        drop = float(stats["drop"])  # type: ignore[arg-type]
        # positive = the tested selection did better than matched random sets
        diffs.append(drop - mean_control if test.mode == "remove" else mean_control - drop)
    measurements: dict[str, Any] = {
        "paired_differences": diffs,
        "n": len(diffs),
        "orientation": "positive means the selection beat the mean of its matched controls",
    }
    if diffs:
        measurements |= {
            "mean": sum(diffs) / len(diffs),
            "median": statistics.median(diffs),
            "sign_flip_p": sign_flip_p(diffs, draws, test.controls.seed),
            "permutation_draws": draws,
            "seed": test.controls.seed,
        }
    trace._add(
        ProtocolResult(
            protocol="paired_control",
            protocol_version=1,
            params=JsonMap({"claim_protocol": test.protocol, "controls": test.controls.identity()}),
            measurements=JsonMap(measurements),
            samples=tuple(family.samples),
            target=test.target.spec,
            provenance_id=provenance.id,
            derived_from=refs,
        )
    )


# ------------------------------------------------------------------ curves


def curve(
    model: nn.Module,
    *inputs: Any,
    ranking: Selection,
    target: iv.metrics.Metric,
    mode: str,
    replacement: Replacement,
    points: Sequence[int] | None = None,
    controls: Controls | None = None,
    model_kwargs: dict[str, Any] | None = None,
    declared_model: ModelDeclaration | None = None,
    attributions: Sequence[AttributionResult] = (),
) -> CurveResult:
    """A removal (``mode="remove"``: deletion / progressive ablation) or retention
    (``mode="retain"``: insertion / progressive retention) curve along ``ranking``.

    At point k the top-k units are removed (or only they are retained). The anchors
    (remove k=0, retain k=n) are unperturbed by definition (drop 0). Controls are
    seeded random rankings evaluated at the same points. ``aopc_mean_drop`` is the mean
    drop over the declared points (anchors included), as recorded in ``params``."""
    if mode not in ("remove", "retain"):
        raise ValueError("mode must be 'remove' or 'retain'")
    if not isinstance(ranking, Selection) or ranking.source is SelectionSource.DECLARED:
        raise TypeError("ranking must come from faithfulness.ranking(...)")
    if ranking.eligible is not None:
        raise FaithfulnessError(
            "curves over an eligibility-restricted ranking are not supported (ADR-053); "
            "use claim tests (comprehensiveness / sufficiency) with top_k(..., eligible=...)"
        )
    if not isinstance(replacement, Replacement):
        raise TypeError(
            "declare the curve's replacement explicitly (faithfulness.zero() or replacement())"
        )
    rep = replacement
    kwargs = dict(model_kwargs or {})
    inputs = tuple(inputs)
    sample = sample_id(*inputs, model_kwargs=kwargs)
    ranking = _validate_selection(ranking.with_k(1), target.spec, sample)
    n = ranking.n_units
    pts = list(range(n + 1)) if points is None else list(points)
    if pts != sorted(set(pts)) or not pts or pts[0] < 0 or pts[-1] > n:
        raise FaithfulnessError(f"points must be sorted, unique, within [0, {n}]")
    retain = mode == "retain"
    anchor = n if retain else 0

    def units_at(order: tuple[int, ...], k: int) -> tuple[int, ...] | None:
        if retain and k == 0:
            return None  # retain nothing: the whole leaf is replaced
        return tuple(sorted(order[:k]))

    orders = [ranking.order]
    if controls is not None and controls.match != "count":
        raise FaithfulnessError(
            "curve controls are random rankings; magnitude matching is defined for a fixed "
            "selection size only (claim tests)"
        )
    if controls is not None:
        orders += uniform_permutations(n, controls.n, controls.seed)
    plan: list[tuple[int, int]] = []  # (order index, k)
    specs = []
    for o, order in enumerate(orders):
        for k in pts:
            if k != anchor:
                plan.append((o, k))
                specs.append(_intervention(ranking, units_at(order, k), retain and k > 0, rep))

    def extend(trace: TraceResult, family: iv.ComparisonFamily) -> None:
        verifier = _Verifier(trace, ranking, rep, inputs, kwargs)
        base_pass = family.baseline_passes[0]
        drops: dict[tuple[int, int], float] = {}
        effect_ids: dict[tuple[int, int], str] = {}
        for (o, k), record, int_pass, effect in zip(
            plan, family.records[0], family.intervention_passes[0], family.effects[0], strict=True
        ):
            verifier.check(base_pass, record, int_pass)
            drops[(o, k)] = effect.baseline_value - effect.intervention_value
            effect_ids[(o, k)] = effect.id
        series = [[drops.get((o, k), 0.0) for k in pts] for o in range(len(orders))]
        selected = series[0]
        params = {
            "mode": mode,
            "level": "input" if ranking.is_input else "internal",
            "call_index": ranking.call_index,
            "points": pts,
            "n_units": n,
            "replacement": rep.identity(),
            "controls": None
            if controls is None
            else controls.identity() | {"strategy": "uniform_random_permutation"},
            "normalization": "aopc_mean_drop = arithmetic mean of drop over the declared "
            "points, anchors included; drop = F(x) - F(x_k)",
        }
        provenance = _provenance(trace, base_pass, f"{mode}_curve", {"points": len(pts)})
        as_ranking = ranking.with_k(None)  # keeps unit_axes / unit_reduction (ADR-034)
        trace._add(_selection_record(as_ranking, sample, provenance))
        measurements: dict[str, Any] = {
            "points": pts,
            "fractions": [k / n for k in pts],
            "drops": selected,
            "effects": [effect_ids.get((0, k)) for k in pts],
            "baseline_value": family.baseline_values[0],
            "aopc_mean_drop": sum(selected) / len(selected),
        }
        if controls is not None:
            ctrl = series[1:]
            worse = []
            for j in range(len(pts)):
                column = [c[j] for c in ctrl]
                worse.append(
                    control_fractions(selected[j], column)[
                        "fraction_below" if not retain else "fraction_above"
                    ]
                )
            control_aopc = [sum(c) / len(c) for c in ctrl]
            aopc = measurements["aopc_mean_drop"]
            measurements |= {
                "control_drops": ctrl,
                "control_mean": [sum(c[j] for c in ctrl) / len(ctrl) for j in range(len(pts))],
                "control_q05": [quantile([c[j] for c in ctrl], 0.05) for j in range(len(pts))],
                "control_q95": [quantile([c[j] for c in ctrl], 0.95) for j in range(len(pts))],
                "fraction_controls_worse": worse,
                "control_aopc_mean_drop": control_aopc,
                "aopc_fraction_controls_worse": control_fractions(aopc, control_aopc)[
                    "fraction_below" if not retain else "fraction_above"
                ],
            }
        refs = tuple(RecordRef.to(e) for e in family.effects[0])
        result = trace._add(
            ProtocolResult(
                protocol=f"{'removal' if not retain else 'retention'}_curve",
                protocol_version=1,
                params=JsonMap(params),
                measurements=JsonMap(measurements),
                samples=(sample,),
                target=target.spec,
                provenance_id=provenance.id,
                derived_from=refs,
            )
        )
        if retain and not ranking.is_input:
            trace._add(
                TraceLimitation(
                    code="SITE_RELATIVE_SUFFICIENCY",
                    detail=f"retention within {ranking.site.module} only",
                    applies_to=(result.id,),
                )
            )

    family = iv.compare_family(
        model,
        [(inputs, kwargs, specs)],
        target,
        sites=_sites(ranking),
        retention="cpu",
        declared_model=declared_model,
        extend=extend,
    )
    return CurveResult(family.trace, tuple(attributions))
