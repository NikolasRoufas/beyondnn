"""Declarations for faithfulness tests (Phase 5; ADR-033): selections, replacements,
controls, and test templates. Everything scientifically relevant is explicit and is
recorded; nothing here runs a model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import torch

from beyondnn.attribution import AttributionResult
from beyondnn.attribution.spec import At, Method, tensor_digest
from beyondnn.core.units import UnitError, check_axes, unit_count, unit_values
from beyondnn.interventions.metrics import Metric
from beyondnn.schema import MetricSpec, Relation, SelectionSource, Site, SiteIO

from .stats import rank_order

__all__ = [
    "Controls",
    "Replacement",
    "Selection",
    "SelectionRule",
    "TestTemplate",
    "comprehensiveness",
    "controls",
    "ranking",
    "replacement",
    "selector",
    "sufficiency",
    "top_k",
    "units",
    "zero",
]


class FaithfulnessError(ValueError):
    """A faithfulness test cannot be run or recorded faithfully."""


class SelectionMismatchError(FaithfulnessError):
    """The selection is about another sample, site, call, or target than the test."""


# ------------------------------------------------------------------ selections


@dataclass(frozen=True, slots=True, eq=False)
class Selection:
    """A runtime selection (recorded as an ``EvidenceSelection`` when a test runs)."""

    site: Site
    call_index: int
    source: SelectionSource
    rule: str
    order: tuple[int, ...]
    n_units: int
    k: int | None
    sample_id: str | None = None
    scores: tuple[float, ...] | None = None
    source_record: str | None = None
    target: MetricSpec | None = None
    unit_axes: tuple[int, ...] | None = None
    unit_reduction: str | None = None

    @property
    def is_input(self) -> bool:
        return self.site.module == ""

    @property
    def selected(self) -> tuple[int, ...]:
        if self.k is None:
            raise FaithfulnessError("this selection is a ranking without k")
        return tuple(sorted(self.order[: self.k]))

    def with_k(self, k: int | None) -> Selection:
        if k is not None and not 1 <= k <= len(self.order):
            raise FaithfulnessError(f"k={k} must be in [1, {len(self.order)}]")
        return replace(self, k=k)


def unit_scores(
    value: torch.Tensor, unit_axes: tuple[int, ...] | None, reduce: str | None
) -> tuple[float, ...]:
    """Per-unit scores of an attribution tensor (ADR-034).

    ``unit_axes=None``: the Phase-5 vector rule (every non-last dimension must be 1).
    Otherwise units are the declared axes; if a unit spans more than one element, an
    explicit ``reduce`` (``sum``/``abs_sum``/``l2``) is required. Nothing is aggregated
    silently."""
    if unit_axes is None:
        if reduce is not None:
            raise FaithfulnessError("reduce applies only with declared unit_axes")
        if value.dim() == 0 or any(d != 1 for d in value.shape[:-1]):
            raise FaithfulnessError(
                f"selection needs an attribution whose non-last dimensions are 1 (got shape "
                f"{tuple(value.shape)}); declare unit_axes (and a reduce) for images, "
                "channels or sequences (ADR-034)"
            )
        return tuple(float(v) for v in value.double().flatten().tolist())
    try:
        axes = check_axes(unit_axes)
        n = unit_count(tuple(value.shape), axes)
    except UnitError as exc:
        raise FaithfulnessError(str(exc)) from None
    within = value.numel() // max(n, 1)
    if within > 1 and reduce is None:
        raise FaithfulnessError(
            f"each unit spans {within} elements of shape {tuple(value.shape)}; declare "
            "reduce='sum', 'abs_sum' or 'l2' (never aggregated implicitly)"
        )
    how = reduce if reduce is not None else "sum"
    return tuple(float(v) for v in unit_values(value, axes, how).tolist())


def ranking(
    result: AttributionResult,
    *,
    by: str = "abs",
    unit_axes: tuple[int, ...] | None = None,
    reduce: str | None = None,
) -> Selection:
    """The full ranking of the attributed units (descending |score| or signed score;
    ties broken by lower index), e.g. for curves."""
    if not isinstance(result, AttributionResult):
        raise TypeError("ranking() takes an AttributionResult")
    scores = unit_scores(result.value, unit_axes, reduce)
    axes = check_axes(unit_axes)
    record = result.record
    return Selection(
        site=record.site,
        call_index=record.call_index,
        source=SelectionSource.ATTRIBUTION,
        rule="abs_desc" if by == "abs" else "signed_desc",
        order=rank_order(scores, by=by),
        n_units=len(scores),
        k=None,
        sample_id=record.sample_id,
        scores=scores,
        source_record=record.id,
        target=record.target,
        unit_axes=axes,
        # the declared reduction is recorded whenever unit axes are declared, even when
        # each unit is one element: the scores were computed with it (ADR-034)
        unit_reduction=reduce if axes is not None else None,
    )


def top_k(
    result: AttributionResult,
    *,
    k: int,
    by: str = "abs",
    unit_axes: tuple[int, ...] | None = None,
    reduce: str | None = None,
) -> Selection:
    """The top-``k`` units of an attribution (see :func:`ranking`)."""
    return ranking(result, by=by, unit_axes=unit_axes, reduce=reduce).with_k(k)


def units(
    site: str | At,
    selected: tuple[int, ...] | list[int],
    *,
    n_units: int,
    call_index: int = 0,
    output_path: str = "",
    unit_axes: tuple[int, ...] | None = None,
) -> Selection:
    """A declared set of units. ``site`` is a module path, or an ``attribution.input(i)``
    / ``attribution.layer(...)`` spec; ``n_units`` is the number of units: the size of the
    last dimension, or, with declared ``unit_axes``, the size of their row-major sub-grid
    (ADR-034)."""
    if isinstance(site, At):
        where = site.site()
        call_index = site.call_index
    elif isinstance(site, str) and site:
        where = Site(module=site, io=SiteIO.OUTPUT, output_path=output_path)
    else:
        raise ValueError("site must be a module path or an attribution.input()/layer() spec")
    chosen = tuple(selected)
    if not chosen or len(set(chosen)) != len(chosen):
        raise FaithfulnessError("a declared selection needs distinct units")
    if any(not isinstance(u, int) or isinstance(u, bool) or not 0 <= u < n_units for u in chosen):
        raise FaithfulnessError(f"units must be ints in [0, {n_units})")
    try:
        axes = check_axes(unit_axes)
    except UnitError as exc:
        raise FaithfulnessError(str(exc)) from None
    return Selection(
        where,
        call_index,
        SelectionSource.DECLARED,
        "declared",
        chosen,
        n_units,
        len(chosen),
        unit_axes=axes,
    )


# ------------------------------------------------------------------ replacement and controls


@dataclass(frozen=True, slots=True, eq=False)
class Replacement:
    """The value that replaced units: zeros, or an explicit tensor of the site's exact
    shape (e.g. caller-computed means). Never a neutral "absence"."""

    tensor: torch.Tensor | None = None
    name: str | None = None

    def identity(self) -> dict[str, Any]:
        if self.tensor is None:
            return {"kind": "zero"}
        out: dict[str, Any] = {"kind": "tensor", "digest": tensor_digest(self.tensor)}
        if self.name is not None:  # declared label (ADR-048); absent keeps the v1 identity
            out["name"] = self.name
        return out


def zero() -> Replacement:
    return Replacement()


def replacement(tensor: torch.Tensor, *, name: str | None = None) -> Replacement:
    """An explicit replacement tensor. ``name`` (optional) labels it in the recorded
    identity, e.g. ``"resample"`` when the tensor differs per sample, so an audit plan
    can declare the role of that replacement (ADR-048)."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("replacement() takes a tensor")
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
        raise ValueError("the replacement must be finite")
    if name is not None and not (name.strip() and name == name.strip()):
        raise ValueError("a replacement name must be a non-empty, unpadded string")
    return Replacement(tensor.detach().to("cpu").clone(memory_format=torch.contiguous_format), name)


@dataclass(frozen=True, slots=True)
class Controls:
    """Matched random controls: ``n`` draws (same site, same size), seeded locally.

    ``match="count"`` (Phase 5): uniform random sets of the same size at the same site.
    ``match="magnitude"`` (ADR-035): additionally stratified on each unit's
    perturbation magnitude ``||x_u - b_u||_2`` into ``strata`` equal-count strata, so
    a control perturbs the input by similar amounts as the selection does."""

    n: int
    seed: int
    match: str = "count"
    strata: int = 4

    def identity(self) -> dict[str, Any]:
        if self.match == "count":
            return {
                "n": self.n,
                "seed": self.seed,
                "strategy": "uniform_without_replacement_same_site_same_size",
            }
        return {
            "n": self.n,
            "seed": self.seed,
            "strata": self.strata,
            "strategy": "perturbation_magnitude_stratified_same_site_same_size",
        }


def controls(n: int, *, seed: int, match: str = "count", strata: int = 4) -> Controls:
    """Matched random controls (see :class:`Controls`)."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be an int >= 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an int")
    if match not in ("count", "magnitude"):
        raise ValueError("match must be 'count' or 'magnitude'")
    if isinstance(strata, bool) or not isinstance(strata, int) or strata < 1:
        raise ValueError("strata must be an int >= 1")
    return Controls(n, seed, match, strata)


# ------------------------------------------------------------------ test templates


@dataclass(frozen=True, slots=True, eq=False)
class TestTemplate:
    """A declared claim-test template: protocol, target, replacement, controls, criteria,
    and the claim (relation + statement) it instantiates for each tested selection."""

    __test__ = False  # not a pytest class

    protocol: str
    target: Metric
    relation: Relation
    statement: str
    replacement: Replacement
    controls: Controls | None
    criteria: dict[str, float]

    @property
    def mode(self) -> str:
        return "remove" if self.protocol == "comprehensiveness" else "retain"


def _check_common(target: Metric, statement: str, rep: Replacement, ctrl: Controls | None) -> None:
    if not isinstance(target, Metric):
        raise TypeError("target must be a scalar Metric (beyondnn.interventions.metrics)")
    if not isinstance(statement, str) or not statement.strip():
        raise ValueError("a claim statement is required")
    if not isinstance(rep, Replacement):
        raise TypeError("replacement must come from faithfulness.zero() / replacement()")
    if ctrl is not None and not isinstance(ctrl, Controls):
        raise TypeError("controls must come from faithfulness.controls()")


def _number(value: float, name: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be > 0")
    return float(value)


def _fraction(value: float | None, name: str) -> dict[str, float]:
    if value is None:
        return {}
    v = _number(value, name, positive=False)
    if not 0 <= v <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return {name: v}


def comprehensiveness(
    *,
    target: Metric,
    min_drop: float,
    statement: str,
    replacement: Replacement,
    relation: Relation = Relation.NECESSARY_FOR,
    controls: Controls | None = None,
    min_fraction_below: float | None = None,
) -> TestTemplate:
    """``comprehensiveness/v1``: SUPPORTS iff removing the selection drops the target by
    at least ``min_drop`` (and, if declared, the drop exceeds that of at least
    ``min_fraction_below`` of the matched random controls)."""
    if not isinstance(replacement, Replacement):
        raise TypeError(
            "declare the replacement explicitly: faithfulness.zero() or "
            "faithfulness.replacement(...) (zero is not neutral; ADR-052)"
        )
    rep = replacement
    _check_common(target, statement, rep, controls)
    if relation not in (Relation.NECESSARY_FOR, Relation.DECREASES):
        raise ValueError("comprehensiveness/v1 tests NECESSARY_FOR or DECREASES claims only")
    if min_fraction_below is not None and controls is None:
        raise ValueError("min_fraction_below needs controls")
    criteria = {"min_drop": _number(min_drop, "min_drop", positive=True)}
    criteria |= _fraction(min_fraction_below, "min_fraction_below")
    return TestTemplate("comprehensiveness", target, relation, statement, rep, controls, criteria)


def sufficiency(
    *,
    target: Metric,
    max_drop: float,
    statement: str,
    replacement: Replacement,
    controls: Controls | None = None,
    min_fraction_above: float | None = None,
) -> TestTemplate:
    """``sufficiency/v1``: SUPPORTS iff retaining only the selection (the rest of the
    site replaced) drops the target by at most ``max_drop`` (and, if declared, at least
    ``min_fraction_above`` of matched random retained sets drop it by more)."""
    if not isinstance(replacement, Replacement):
        raise TypeError(
            "declare the replacement explicitly: faithfulness.zero() or "
            "faithfulness.replacement(...) (zero is not neutral; ADR-052)"
        )
    rep = replacement
    _check_common(target, statement, rep, controls)
    if min_fraction_above is not None and controls is None:
        raise ValueError("min_fraction_above needs controls")
    criteria = {"max_drop": _number(max_drop, "max_drop", positive=False)}
    if criteria["max_drop"] < 0:
        raise ValueError("max_drop must be >= 0")
    criteria |= _fraction(min_fraction_above, "min_fraction_above")
    return TestTemplate(
        "sufficiency", target, Relation.SUFFICIENT_FOR, statement, rep, controls, criteria
    )


# ------------------------------------------------------------------ per-sample selectors


@dataclass(frozen=True, slots=True, eq=False)
class SelectionRule:
    """How a dataset run selects units on each sample: the top-``k`` of an attribution
    ``method`` computed explicitly on that sample, or a fixed declared set."""

    method: Method | None
    at: At | None
    k: int
    by: str = "abs"
    fixed: Selection | None = None
    unit_axes: tuple[int, ...] | None = None
    reduce: str | None = None


def selector(
    method: Method,
    *,
    k: int,
    by: str = "abs",
    at: At | None = None,
    unit_axes: tuple[int, ...] | None = None,
    reduce: str | None = None,
) -> SelectionRule:
    """Per sample: run ``method`` explicitly (``bnn.attribute``), take its top-``k``
    units (declared ``unit_axes``/``reduce`` for non-vector sites, ADR-034)."""
    if not isinstance(method, Method):
        raise TypeError("method must come from beyondnn.attribution")
    return SelectionRule(method, at, k, by, None, check_axes(unit_axes), reduce)


def fixed(selection: Selection) -> SelectionRule:
    """Per sample: the same declared units (``faithfulness.units(...)``)."""
    if selection.source is not SelectionSource.DECLARED:
        raise FaithfulnessError("fixed() takes a declared selection (faithfulness.units)")
    assert selection.k is not None
    return SelectionRule(None, None, selection.k, "declared", selection)
