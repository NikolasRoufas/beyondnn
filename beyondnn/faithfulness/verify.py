"""Re-derivation of faithfulness records from raw records (used by composition; ADR-033).

Composition must not trust a result object because BeyondNN made it. These functions
recompute a recorded result from the records it rests on and raise on any difference:

* ``verify_claim_result``: the primary and control effects named in its statistics,
  their intervention records (units, retain, replacement), the controls re-drawn from
  the declared seed, and the no-op status re-derived from the recorded execution
  (the perturbed input's ``sample_id``, or the retained site activation), then the
  protocol evaluator again: the result id must be identical;
* ``verify_selection``: an attribution selection's ranking and scores re-derived from
  the attribution tensor;
* ``verify_protocol_result``: curves recomputed from their effects; stability and
  method diagnostics recomputed from the attribution tensors.

Nothing here runs a model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

from beyondnn.core.trace import TraceResult
from beyondnn.protocols import COMPREHENSIVENESS, SUFFICIENCY
from beyondnn.schema import (
    AttributionRecord,
    BaseRecord,
    CausalEffect,
    Claim,
    ClaimTestResult,
    ClaimTestSpec,
    EvidenceSelection,
    InterventionRecord,
    JsonMap,
    OutputRecord,
    ProtocolResult,
    SelectionSource,
)

from .claims import evaluate
from .stats import jaccard, rank_order, spearman_of_orders, uniform_subsets

__all__ = ["VerificationError", "verify_claim_result", "verify_protocol_result", "verify_selection"]

Lookup = Callable[[str], tuple[BaseRecord, TraceResult]]


class VerificationError(ValueError):
    """A faithfulness record does not follow from the records it rests on."""


def _int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VerificationError(f"expected an integer, got {value!r}")
    return value


def _get(lookup: Lookup, record_id: object, kind: type) -> tuple[Any, TraceResult]:
    if not isinstance(record_id, str):
        raise VerificationError(f"expected a record id, got {record_id!r}")
    try:
        record, trace = lookup(record_id)
    except KeyError:
        raise VerificationError(f"{record_id} is not among the composed records") from None
    if not isinstance(record, kind):
        raise VerificationError(f"{record_id} is not a {kind.__name__}")
    return record, trace


def _no_op(trace: TraceResult, effect: CausalEffect, record: InterventionRecord) -> bool:
    outputs = [trace.get(r.record_id) for r in effect.derived_from if r.kind == "output"]
    base, pert = sorted(
        (o for o in outputs if isinstance(o, OutputRecord)), key=lambda o: o.pass_index or 0
    )
    if record.on_input:
        (bi,) = [i for i in trace.inputs if i.pass_index == base.pass_index]
        (pi,) = [i for i in trace.inputs if i.pass_index == pert.pass_index]
        if bi.sample_id is None or pi.sample_id is None:
            raise VerificationError("input identities are missing; the no-op status is unknown")
        return bi.sample_id == pi.sample_id
    site = record.site

    def act(pass_index: int | None) -> torch.Tensor:
        a = trace.activation(
            site.module,
            output_path=site.output_path,
            pass_index=pass_index,
            call_index=record.call_index,
        )
        if a.value.storage_key is None:
            raise VerificationError("the intervened activation was not retained")
        return trace.tensor(a)

    return bool(torch.equal(act(base.pass_index), act(pert.pass_index)))


def verify_claim_result(result: ClaimTestResult, lookup: Lookup) -> None:
    """Re-derive a comprehensiveness/sufficiency result (see module docstring)."""
    claim, _ = _get(lookup, result.claim.claim_id, Claim)
    spec, _ = _get(lookup, result.spec.spec_id, ClaimTestSpec)
    if spec.protocol not in (COMPREHENSIVENESS, SUFFICIENCY):
        raise VerificationError(f"{spec.protocol} is not a faithfulness claim protocol")
    stats = result.statistics
    if "selected_effect" not in stats:  # NOT_APPLICABLE results cite nothing
        if result.evidence:
            raise VerificationError(f"{result.id} cites evidence without naming its effects")
        return
    primary, trace = _get(lookup, stats["selected_effect"], CausalEffect)
    control_ids = stats.get("control_effects") or ()
    controls = [_get(lookup, c, CausalEffect)[0] for c in control_ids]  # type: ignore[union-attr]
    records: dict[str, InterventionRecord] = {}
    for effect in (primary, *controls):
        rid = effect.interventions[0].record_id
        records[rid] = _get(lookup, rid, InterventionRecord)[0]
    main = records[primary.interventions[0].record_id]
    declared = spec.params.get("controls")
    if isinstance(declared, Mapping) and controls:
        expected = uniform_subsets(
            _int(spec.params["n_units"]),
            _int(spec.params["k"]),
            _int(declared["n"]),
            _int(declared["seed"]),
        )
        got = [records[c.interventions[0].record_id].units for c in controls]
        if got != expected:
            raise VerificationError(f"{result.id}: controls do not match the declared seed")
    expected_result = evaluate(
        claim,
        spec,
        primary,
        controls,
        records,
        no_op=_no_op(trace, primary, main),
        provenance_id=result.provenance_id or "",
    )
    if expected_result.id != result.id:
        raise VerificationError(
            f"{result.id} ({result.outcome.value}) does not follow from its effects "
            f"(re-derived {expected_result.outcome.value})"
        )


def _scores(attribution: AttributionRecord, trace: TraceResult) -> list[float]:
    return [float(v) for v in trace.tensor(attribution.value).double().flatten().tolist()]


def verify_selection(selection: EvidenceSelection, lookup: Lookup) -> None:
    if selection.source is not SelectionSource.ATTRIBUTION:
        return
    record, trace = _get(lookup, selection.source_record, AttributionRecord)
    scores = _scores(record, trace)
    by = "abs" if selection.rule == "abs_desc" else "signed"
    if (
        record.sample_id != selection.sample_id
        or (record.site, record.call_index) != (selection.site, selection.call_index)
        or record.target != selection.target
        or tuple(scores) != selection.scores
        or rank_order(scores, by=by) != selection.order
    ):
        raise VerificationError(f"{selection.id} does not re-derive from {record.id}")


def verify_protocol_result(result: ProtocolResult, lookup: Lookup) -> None:
    m = result.measurements
    if result.protocol in ("removal_curve", "retention_curve"):
        points = list(m["points"])  # type: ignore[arg-type]
        drops = []
        for eid in m["effects"]:  # type: ignore[union-attr]
            if eid is None:
                drops.append(0.0)
            else:
                effect, _ = _get(lookup, eid, CausalEffect)
                drops.append(effect.baseline_value - effect.intervention_value)
        if len(drops) != len(points) or tuple(drops) != tuple(m["drops"]):  # type: ignore[arg-type]
            raise VerificationError(f"{result.id}: the curve does not follow from its effects")
        if m["aopc_mean_drop"] != sum(drops) / len(drops):
            raise VerificationError(f"{result.id}: aopc_mean_drop is not the declared mean")
        return
    if result.protocol in (
        "stability",
        "method_agreement",
        "baseline_sensitivity",
        "ig_step_sensitivity",
    ):
        pair = m["attribution_records"]
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise VerificationError(f"{result.id}: expected two attribution records")
        a_id, b_id = pair
        a, ta = _get(lookup, a_id, AttributionRecord)
        b, tb = _get(lookup, b_id, AttributionRecord)
        sa, sb = _scores(a, ta), _scores(b, tb)
        k = _int(result.params["k"])
        order_a = rank_order(sa, by="abs")
        if result.protocol == "stability":
            declared = result.params["transformation"]
            if not isinstance(declared, JsonMap):
                raise VerificationError(f"{result.id}: the transformation is not declared")
            unit_map = declared.get("unit_map")
            mapping = (
                [int(u) for u in unit_map]  # type: ignore[arg-type,union-attr]
                if unit_map is not None
                else list(range(len(sb)))
            )
            order_b = tuple(mapping[u] for u in rank_order(sb, by="abs"))
            names = ("ranking_x", "ranking_gx_mapped")
            change = abs(b.target_value - a.target_value)
            if m["prediction_change"] != change:
                raise VerificationError(f"{result.id}: prediction change does not re-derive")
        else:
            order_b = rank_order(sb, by="abs")
            names = ("ranking_a", "ranking_b")
        rho = spearman_of_orders(order_a, order_b) if len(sa) >= 2 else None
        if (
            tuple(m[names[0]]) != order_a  # type: ignore[arg-type]
            or tuple(m[names[1]]) != order_b  # type: ignore[arg-type]
            or m["rank_correlation"] != rho
            or m["topk_jaccard"] != jaccard(order_a[:k], order_b[:k])
        ):
            raise VerificationError(f"{result.id}: rankings do not re-derive from attributions")
        if result.protocol == "ig_step_sensitivity":
            diff = float((ta.tensor(a.value).double() - tb.tensor(b.value).double()).abs().max())
            if m["max_abs_difference"] != diff:
                raise VerificationError(f"{result.id}: max_abs_difference does not re-derive")
        return
    raise VerificationError(
        f"{result.protocol} results cannot be composed into an instance-level explanation"
    )
