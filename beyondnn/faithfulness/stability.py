"""``stability/v1`` and method diagnostics (Phase 5; ADR-033).

Stability does not decide what is semantically irrelevant: the caller declares a
transformation (name, implementation revision, config, optional unit map). BeyondNN
runs the declared method on x and on g(x) and reports four aspects separately, never
combined:

* ``prediction``: |F(g(x)) - F(x)| (criterion ``max_prediction_change``);
* ``ranking``: Spearman correlation of the two ordinal rankings, after mapping g(x)'s
  units to x's through the unit map (criterion ``min_rank_correlation``);
* ``topk``: Jaccard overlap of the top-k sets (criterion ``min_topk_jaccard``);
* ``claim``: equality of a declared claim test's outcome on x and g(x) (criterion
  ``same_claim_outcome``; only if a test is given).

An aspect without a declared criterion is measured but INDETERMINATE; a requested
aspect that cannot be measured is NOT_APPLICABLE.

Diagnostics (not causal tests): ``method_agreement``, ``baseline_sensitivity`` (IG
under two declared baselines) and ``ig_step_sensitivity`` (IG at n and 2n steps).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

import beyondnn.interventions as iv
from beyondnn.attribution import AttributionResult, attribute
from beyondnn.attribution.spec import At, Method
from beyondnn.core.samples import sample_id
from beyondnn.core.trace import TraceResult
from beyondnn.schema import (
    AspectOutcome,
    CheckOutcome,
    JsonMap,
    ModelDeclaration,
    ProtocolResult,
    RecordRef,
    TraceLimitation,
)

from .runner import FaithfulnessResult, _provenance, run
from .spec import FaithfulnessError, SelectionMismatchError, TestTemplate, top_k, unit_scores
from .stats import jaccard, rank_order, spearman_of_orders

__all__ = [
    "DiagnosticResult",
    "Transformation",
    "baseline_sensitivity",
    "ig_step_sensitivity",
    "method_agreement",
    "stability",
    "transformation",
]

_TOKEN = re.compile(r"^\S+$")
_NAME = re.compile(r"^[A-Za-z0-9_.\-]+$")


@dataclass(frozen=True, slots=True, eq=False)
class Transformation:
    """A caller-declared transformation. The function is never serialised or inspected;
    ``unit_map[u]`` is the unit of x that unit ``u`` of g(x) corresponds to."""

    name: str
    function: Callable[..., Any]
    implementation_revision: str
    config: dict[str, Any] = field(default_factory=dict)
    unit_map: tuple[int, ...] | None = None

    def identity(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "implementation_revision": self.implementation_revision,
            "config": dict(self.config),
            "unit_map": None if self.unit_map is None else list(self.unit_map),
        }


def transformation(
    name: str,
    function: Callable[..., Any],
    *,
    implementation_revision: str,
    config: dict[str, Any] | None = None,
    unit_map: tuple[int, ...] | list[int] | None = None,
) -> Transformation:
    """Declare an invariance transformation (see module docstring)."""
    if not isinstance(name, str) or not _NAME.match(name):
        raise ValueError(f"invalid transformation name {name!r}")
    if not callable(function):
        raise TypeError("function must be callable")
    if not isinstance(implementation_revision, str) or not _TOKEN.match(implementation_revision):
        raise ValueError("implementation_revision must be a non-empty token")
    mapping = None if unit_map is None else tuple(unit_map)
    if mapping is not None and sorted(mapping) != list(range(len(mapping))):
        raise ValueError("unit_map must be a permutation of the units")
    JsonMap(dict(config or {}))  # must be JSON-compatible
    return Transformation(name, function, implementation_revision, dict(config or {}), mapping)


@dataclass(frozen=True, slots=True, eq=False)
class DiagnosticResult:
    """A diagnostic ``ProtocolResult`` in its (anchor) trace, plus the attribution and
    claim-test results it compared (separate traces)."""

    trace: TraceResult
    attributions: tuple[AttributionResult, ...] = ()
    tests: tuple[FaithfulnessResult, ...] = ()

    @property
    def protocol_result(self) -> ProtocolResult:
        (r,) = [r for r in self.trace.records if isinstance(r, ProtocolResult)]
        return r

    def outcome(self, aspect: str) -> CheckOutcome:
        for o in self.protocol_result.outcomes:
            if o.aspect == aspect:
                return o.outcome
        raise KeyError(aspect)

    @property
    def measurements(self) -> JsonMap:
        return self.protocol_result.measurements


def _check(
    value: float | None, criterion: float | None, *, higher_is_better: bool, aspect: str
) -> AspectOutcome:
    if value is None:
        return AspectOutcome(aspect=aspect, outcome=CheckOutcome.NOT_APPLICABLE)
    if criterion is None:
        return AspectOutcome(
            aspect=aspect,
            outcome=CheckOutcome.INDETERMINATE,
            detail="measured; no criterion declared",
        )
    ok = value >= criterion if higher_is_better else value <= criterion
    return AspectOutcome(
        aspect=aspect,
        outcome=CheckOutcome.PASS if ok else CheckOutcome.FAIL,
        detail=f"{value:.6g} vs {criterion:.6g}",
    )


_ASPECT_NAMES = {
    "prediction": "max_prediction_change",
    "ranking": "min_rank_correlation",
    "topk": "min_topk_jaccard",
    "claim": "same_claim_outcome",
}


def _scores(
    result: AttributionResult,
    unit_axes: tuple[int, ...] | None = None,
    reduce: str | None = None,
) -> list[float]:
    return list(unit_scores(result.value, unit_axes, reduce))


def _unit_params(unit_axes: tuple[int, ...] | None, reduce: str | None) -> dict[str, Any]:
    """Declared unit fields (ADR-034); absent for the Phase-5 vector rule, so legacy
    diagnostic records keep their identity."""
    if unit_axes is None:
        return {}
    return {"unit_axes": list(unit_axes), "unit_reduction": reduce}


def _anchor(
    model: nn.Module,
    samples: list[tuple[Any, ...]],
    target: iv.metrics.Metric,
    declared_model: ModelDeclaration | None,
    build: Callable[[TraceResult, iv.ComparisonFamily], None],
) -> TraceResult:
    """One CLEAN pass per sample (OBSERVED outputs, provenance) to anchor a diagnostic."""
    family = iv.compare_family(
        model, [(s, {}, []) for s in samples], target, declared_model=declared_model, extend=build
    )
    return family.trace


def stability(
    model: nn.Module,
    x: torch.Tensor,
    *,
    transformation: Transformation,
    method: Method,
    target: iv.metrics.Metric,
    k: int,
    at: At | None = None,
    max_prediction_change: float | None = None,
    min_rank_correlation: float | None = None,
    min_topk_jaccard: float | None = None,
    test: TestTemplate | None = None,
    declared_model: ModelDeclaration | None = None,
    unit_axes: tuple[int, ...] | None = None,
    reduce: str | None = None,
) -> DiagnosticResult:
    """``stability/v1`` (see module docstring). Every aspect is reported separately.
    ``unit_axes``/``reduce`` declare units as in :func:`faithfulness.ranking` (ADR-034)."""
    if not isinstance(transformation, Transformation):
        raise TypeError("transformation must come from faithfulness.transformation()")
    before = sample_id(x)
    transformed = transformation.function(x)
    if sample_id(x) != before:
        raise FaithfulnessError("the transformation modified its input in place")
    if (
        not isinstance(transformed, torch.Tensor)
        or transformed.shape != x.shape
        or (transformed.dtype != x.dtype)
    ):
        raise FaithfulnessError(
            "the declared transformation must return a tensor of the same shape and dtype "
            "(otherwise unit correspondence and the target are not defined)"
        )
    if torch.equal(transformed, x):
        raise FaithfulnessError("the transformation returned the identical input (a no-op)")
    a_x = attribute(model, x, target=target, method=method, at=at, declared_model=declared_model)
    a_g = attribute(
        model, transformed, target=target, method=method, at=at, declared_model=declared_model
    )
    scores_x, scores_g = _scores(a_x, unit_axes, reduce), _scores(a_g, unit_axes, reduce)
    n = len(scores_x)
    unit_map = transformation.unit_map if transformation.unit_map is not None else tuple(range(n))
    if len(unit_map) != n:
        raise FaithfulnessError(f"unit_map has {len(unit_map)} units, the site {n}")
    order_x = rank_order(scores_x, by="abs")
    order_g = tuple(unit_map[u] for u in rank_order(scores_g, by="abs"))
    if not 1 <= k <= n:
        raise FaithfulnessError(f"k must be in [1, {n}]")
    rho = spearman_of_orders(order_x, order_g) if n >= 2 else None
    jac = jaccard(order_x[:k], order_g[:k])
    change = abs(a_g.target_value - a_x.target_value)
    tests: tuple[FaithfulnessResult, ...] = ()
    same: float | None = None
    if test is not None:
        r_x = run(
            model,
            x,
            test=test,
            selection=top_k(a_x, k=k, unit_axes=unit_axes, reduce=reduce),
            attributions=[a_x],
            declared_model=declared_model,
        )
        r_g = run(
            model,
            transformed,
            test=test,
            selection=top_k(a_g, k=k, unit_axes=unit_axes, reduce=reduce),
            attributions=[a_g],
            declared_model=declared_model,
        )
        tests = (r_x, r_g)
        same = 1.0 if r_x.outcome is r_g.outcome else 0.0
    criteria = {
        name: value
        for name, value in (
            ("max_prediction_change", max_prediction_change),
            ("min_rank_correlation", min_rank_correlation),
            ("min_topk_jaccard", min_topk_jaccard),
            ("same_claim_outcome", 1.0 if test is not None else None),
        )
        if value is not None
    }
    outcomes = (
        _check(
            change, max_prediction_change, higher_is_better=False, aspect="max_prediction_change"
        ),
        _check(rho, min_rank_correlation, higher_is_better=True, aspect="min_rank_correlation"),
        _check(jac, min_topk_jaccard, higher_is_better=True, aspect="min_topk_jaccard"),
        _check(
            same,
            criteria.get("same_claim_outcome"),
            higher_is_better=True,
            aspect="same_claim_outcome",
        ),
    )
    measurements = {
        "prediction_x": a_x.target_value,
        "prediction_gx": a_g.target_value,
        "prediction_change": change,
        "ranking_x": list(order_x),
        "ranking_gx_mapped": list(order_g),
        "rank_correlation": rho,
        "topk_x": sorted(order_x[:k]),
        "topk_gx_mapped": sorted(order_g[:k]),
        "topk_jaccard": jac,
        "attribution_records": [a_x.record.id, a_g.record.id],
        "claim_outcomes": None if not tests else [t.outcome.value for t in tests],
        "claim_results": None if not tests else [t.result.id for t in tests],
    }

    def build(trace: TraceResult, family: iv.ComparisonFamily) -> None:
        provenance = _provenance(
            trace, family.baseline_passes[0], "stability", {"transformation": transformation.name}
        )
        outputs = tuple(RecordRef.to(o) for o in trace.outputs)
        result = trace._add(
            ProtocolResult(
                protocol="stability",
                protocol_version=1,
                params=JsonMap(
                    {
                        "transformation": transformation.identity(),
                        "k": k,
                        "method": a_x.record.method.name,
                        "ranking": "abs_desc, ties by lower index",
                    }
                    | _unit_params(unit_axes, reduce)
                ),
                criteria=JsonMap(criteria),
                measurements=JsonMap(measurements),
                outcomes=outcomes,
                samples=family.samples,
                target=target.spec,
                provenance_id=provenance.id,
                derived_from=outputs,
            )
        )
        trace._add(
            TraceLimitation(
                code="DECLARED_TRANSFORMATION_UNVERIFIED",
                detail=f"{transformation.name} ({transformation.implementation_revision})",
                applies_to=(result.id,),
            )
        )

    trace = _anchor(model, [(x,), (transformed,)], target, declared_model, build)
    return DiagnosticResult(trace, (a_x, a_g), tests)


def _same_context(a: AttributionResult, b: AttributionResult) -> None:
    ra, rb = a.record, b.record
    if ra.sample_id != rb.sample_id:
        raise SelectionMismatchError("the attributions are about different samples")
    if (ra.site, ra.call_index) != (rb.site, rb.call_index):
        raise SelectionMismatchError("the attributions are about different sites or calls")
    if ra.target != rb.target:
        raise SelectionMismatchError("the attributions have different targets")


def _compare(
    model: nn.Module,
    x: torch.Tensor,
    protocol: str,
    a: AttributionResult,
    b: AttributionResult,
    target: iv.metrics.Metric,
    k: int,
    params: dict[str, Any],
    criteria: dict[str, float | None],
    extra: dict[str, Any],
    declared_model: ModelDeclaration | None,
    unit_axes: tuple[int, ...] | None = None,
    reduce: str | None = None,
) -> DiagnosticResult:
    _same_context(a, b)
    if a.record.sample_id != sample_id(x):
        raise SelectionMismatchError("the attributions are not about this input")
    sa, sb = _scores(a, unit_axes, reduce), _scores(b, unit_axes, reduce)
    order_a, order_b = rank_order(sa, by="abs"), rank_order(sb, by="abs")
    n = len(sa)
    if not 1 <= k <= n:
        raise FaithfulnessError(f"k must be in [1, {n}]")
    rho = spearman_of_orders(order_a, order_b) if n >= 2 else None
    jac = jaccard(order_a[:k], order_b[:k])
    declared = {key: v for key, v in criteria.items() if v is not None}
    outcomes = [
        _check(
            rho,
            criteria.get("min_rank_correlation"),
            higher_is_better=True,
            aspect="min_rank_correlation",
        ),
        _check(
            jac, criteria.get("min_topk_jaccard"), higher_is_better=True, aspect="min_topk_jaccard"
        ),
    ]
    if "max_abs_difference" in criteria:
        outcomes.append(
            _check(
                extra.get("max_abs_difference"),
                criteria["max_abs_difference"],
                higher_is_better=False,
                aspect="max_abs_difference",
            )
        )
    measurements = {
        "ranking_a": list(order_a),
        "ranking_b": list(order_b),
        "rank_correlation": rho,
        "topk_jaccard": jac,
        "attribution_records": [a.record.id, b.record.id],
        "methods": [a.record.method.name, b.record.method.name],
    } | extra

    def build(trace: TraceResult, family: iv.ComparisonFamily) -> None:
        provenance = _provenance(trace, family.baseline_passes[0], protocol, {"k": k})
        trace._add(
            ProtocolResult(
                protocol=protocol,
                protocol_version=1,
                params=JsonMap(params | {"k": k} | _unit_params(unit_axes, reduce)),
                criteria=JsonMap(declared),
                measurements=JsonMap(measurements),
                outcomes=tuple(outcomes),
                samples=family.samples,
                target=target.spec,
                provenance_id=provenance.id,
                derived_from=tuple(RecordRef.to(o) for o in trace.outputs),
            )
        )

    trace = _anchor(model, [(x,)], target, declared_model, build)
    return DiagnosticResult(trace, (a, b))


def method_agreement(
    model: nn.Module,
    x: torch.Tensor,
    *,
    a: AttributionResult,
    b: AttributionResult,
    target: iv.metrics.Metric,
    k: int,
    min_rank_correlation: float | None = None,
    min_topk_jaccard: float | None = None,
    declared_model: ModelDeclaration | None = None,
    unit_axes: tuple[int, ...] | None = None,
    reduce: str | None = None,
) -> DiagnosticResult:
    """How two methods' rankings of the same units agree. Agreement is not correctness:
    two methods can agree and both miss the causal structure."""
    return _compare(
        model,
        x,
        "method_agreement",
        a,
        b,
        target,
        k,
        {},
        {"min_rank_correlation": min_rank_correlation, "min_topk_jaccard": min_topk_jaccard},
        {},
        declared_model,
        unit_axes,
        reduce,
    )


def baseline_sensitivity(
    model: nn.Module,
    x: torch.Tensor,
    *,
    a: AttributionResult,
    b: AttributionResult,
    target: iv.metrics.Metric,
    k: int,
    min_rank_correlation: float | None = None,
    min_topk_jaccard: float | None = None,
    declared_model: ModelDeclaration | None = None,
    unit_axes: tuple[int, ...] | None = None,
    reduce: str | None = None,
) -> DiagnosticResult:
    """IG rankings under two declared baselines (all other settings equal)."""
    ma, mb = a.record.method, b.record.method
    if ma.name != "integrated_gradients" or ma != mb:
        raise FaithfulnessError(
            "baseline_sensitivity compares IG runs that differ only in baseline"
        )
    if a.record.baseline == b.record.baseline:
        raise FaithfulnessError("the two baselines are identical")
    return _compare(
        model,
        x,
        "baseline_sensitivity",
        a,
        b,
        target,
        k,
        {"method": ma.name},
        {"min_rank_correlation": min_rank_correlation, "min_topk_jaccard": min_topk_jaccard},
        {},
        declared_model,
        unit_axes,
        reduce,
    )


def ig_step_sensitivity(
    model: nn.Module,
    x: torch.Tensor,
    *,
    a: AttributionResult,
    b: AttributionResult,
    target: iv.metrics.Metric,
    k: int,
    max_abs_difference: float | None = None,
    declared_model: ModelDeclaration | None = None,
    unit_axes: tuple[int, ...] | None = None,
    reduce: str | None = None,
) -> DiagnosticResult:
    """IG at two step counts (all other settings equal): max |difference| and the two
    completeness deltas. A numerical diagnostic, not a faithfulness test."""
    ma, mb = a.record.method, b.record.method
    same_rest = (ma.name, ma.implementation, ma.params.get("rule")) == (
        mb.name,
        mb.implementation,
        mb.params.get("rule"),
    )
    if ma.name != "integrated_gradients" or not same_rest or a.record.baseline != b.record.baseline:
        raise FaithfulnessError("ig_step_sensitivity compares IG runs differing only in n_steps")
    if ma.params.get("n_steps") == mb.params.get("n_steps"):
        raise FaithfulnessError("the two runs use the same number of steps")
    diff = float((a.value.double() - b.value.double()).abs().max())
    extra = {
        "max_abs_difference": diff,
        "n_steps": [ma.params.get("n_steps"), mb.params.get("n_steps")],
        "completeness_deltas": [a.completeness_delta, b.completeness_delta],
    }
    return _compare(
        model,
        x,
        "ig_step_sensitivity",
        a,
        b,
        target,
        k,
        {"method": ma.name},
        {"max_abs_difference": max_abs_difference},
        extra,
        declared_model,
        unit_axes,
        reduce,
    )
