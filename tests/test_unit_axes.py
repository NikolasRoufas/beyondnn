"""Phase 5.5: declared unit axes (ADR-034) and perturbation-magnitude-matched controls
(ADR-035). Regression tests for the realistic failure recorded in
experiments/phase5_5/results/abstraction_probe.json: pixel, channel and token units
could not be selected, intervened on or tested with the Phase-5 last-axis rule.

Every expected value below is hand-computed from the fixed weights."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, TypeVar

import pytest
import torch
from torch import nn

import beyondnn as bnn
import beyondnn.attribution as A
import beyondnn.faithfulness as F
import beyondnn.interventions as iv
from beyondnn._testing.models import TinyTransformer
from beyondnn.core.units import UnitError, check_axes, unit_count, unit_mask, unit_values
from beyondnn.explain import EvidenceIntegrityError
from beyondnn.faithfulness import stats
from beyondnn.faithfulness.claims import evaluate
from beyondnn.schema import (
    CausalEffect,
    ClaimTestSpec,
    EvidenceSelection,
    InterventionRecord,
    Outcome,
    SchemaError,
    from_json,
    to_json,
)
from beyondnn.schema.codec import _expected_id

SEL = iv.metrics.select([0, 0])
IMG = torch.ones(1, 2, 3, 3)
PIXELS = (2, 3)
CHANNELS = (1,)
R = TypeVar("R")


def get(trace: Any, record_id: object, kind: type[R]) -> R:
    record = trace.get(record_id)
    assert isinstance(record, kind)
    return record


class _Scale(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        w0 = torch.arange(1.0, 10.0).reshape(3, 3)
        self.register_buffer("w", torch.stack([w0, 10 * w0]).unsqueeze(0))  # (1, 2, 3, 3)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        w = self.w
        assert isinstance(w, torch.Tensor)
        return t * w


class Pixels(nn.Module):
    """out = sum_{c,h,w} x[c,h,w] * w[c,h,w] with w[0] = 1..9 (row-major), w[1] = 10 * w[0].

    With x = 1: pixel unit u (row-major over H x W) contributes 11 * (u + 1); channel 0
    contributes 45, channel 1 contributes 450; the output is 495."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = _Scale()
        self.calls = 0

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        out: torch.Tensor = self.scale(t).sum(dim=(1, 2, 3)).reshape(1, 1)
        return out


def comp(min_drop: float, **kw: Any) -> F.TestTemplate:
    kw.setdefault("statement", "the selected units are necessary for the target")
    return F.comprehensiveness(target=SEL, min_drop=min_drop, **kw)


def grad_attr(model: nn.Module, x: torch.Tensor = IMG) -> A.AttributionResult:
    return A.attribute(model, x, target=SEL, method=A.gradient())


# ------------------------------------------------------------------ unit geometry


def test_units_are_row_major_over_the_declared_axes() -> None:
    shape = (1, 2, 3, 3)
    assert unit_count(shape, PIXELS) == 9
    assert unit_count(shape, CHANNELS) == 2
    assert unit_count(shape, None) == 3  # the Phase-5 last-axis rule
    m = unit_mask(shape, PIXELS, [5])  # row 1, column 2, every channel
    assert m.sum() == 2
    assert bool(m[0, 0, 1, 2])
    assert bool(m[0, 1, 1, 2])
    m = unit_mask(shape, CHANNELS, [1])
    assert m.sum() == 9
    assert bool(m[0, 1].all())
    assert not bool(m[0, 0].any())
    t = torch.arange(18.0).reshape(shape)
    assert unit_values(t, PIXELS, "sum").tolist() == [9.0 + 2 * u for u in range(9)]
    assert unit_values(-t, PIXELS, "abs_sum").tolist() == [9.0 + 2 * u for u in range(9)]
    assert unit_values(t, PIXELS, "l2").tolist() == [
        math.sqrt(u**2 + (9 + u) ** 2) for u in range(9)
    ]
    assert unit_values(t, CHANNELS, "sum").tolist() == [36.0, 117.0]


@pytest.mark.parametrize("axes", [(3, 2), (1, 1), (-1,), ()])
def test_invalid_unit_axes_are_refused(axes: tuple[int, ...]) -> None:
    with pytest.raises(UnitError):
        check_axes(axes)


def test_units_outside_the_grid_are_refused() -> None:
    with pytest.raises(UnitError, match="do not fit"):
        unit_count((1, 9), (2,))
    with pytest.raises(UnitError, match="out of range"):
        unit_mask((1, 2, 3, 3), PIXELS, [9])
    with pytest.raises(UnitError, match="reduction"):
        unit_values(torch.ones(2, 2), (1,), "mean")


# ------------------------------------------------------------------ interventions


def test_pixel_and_channel_interventions_are_exact() -> None:
    model = Pixels().eval()
    r = iv.intervene(
        model, IMG, intervention=iv.zero_input(units=(0, 4), unit_axes=PIXELS), metric=SEL
    )
    assert r.value == -11.0 * (1 + 5)
    assert r.intervention.unit_axes == PIXELS
    kept = iv.intervene(
        model,
        IMG,
        intervention=iv.zero_input(units=(0, 4), unit_axes=PIXELS, retain=True),
        metric=SEL,
    )
    assert kept.value == -(495.0 - 66.0)
    ch = iv.intervene(
        model, IMG, intervention=iv.zero("scale", units=(1,), unit_axes=CHANNELS), metric=SEL
    )
    assert ch.value == -450.0
    legacy = iv.intervene(model, IMG, intervention=iv.zero_input(units=(0,)), metric=SEL)
    assert legacy.value == -sum(float(c) * 11 for c in (1, 4, 7))  # last axis: column 0


def test_unit_axes_require_units_and_valid_ranges() -> None:
    with pytest.raises(ValueError, match="units"):
        iv.zero("scale", unit_axes=CHANNELS)
    with pytest.raises(iv.InterventionError, match="out of range"):
        iv.intervene(
            Pixels().eval(),
            IMG,
            intervention=iv.zero("scale", units=(2,), unit_axes=CHANNELS),
            metric=SEL,
        )


# ------------------------------------------------------------------ selections


def test_unit_scores_need_declared_axes_and_an_explicit_reduction() -> None:
    attr = grad_attr(Pixels().eval())
    with pytest.raises(F.FaithfulnessError, match="declare unit_axes"):
        F.top_k(attr, k=1)
    with pytest.raises(F.FaithfulnessError, match="reduce="):
        F.top_k(attr, k=1, unit_axes=PIXELS)
    with pytest.raises(F.FaithfulnessError, match="only with declared unit_axes"):
        F.ranking(attr, reduce="sum")
    s = F.ranking(attr, unit_axes=PIXELS, reduce="sum")
    assert s.scores == tuple(11.0 * (u + 1) for u in range(9))
    assert s.order == tuple(range(8, -1, -1))
    l2 = F.ranking(attr, unit_axes=PIXELS, reduce="l2")
    assert l2.scores == pytest.approx([math.sqrt(101) * (u + 1) for u in range(9)])
    c = F.top_k(attr, k=1, unit_axes=CHANNELS, reduce="sum")
    assert c.selected == (1,)
    assert c.n_units == 2


class _Negated(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner = Pixels()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = -self.inner(t)
        return out


def test_a_declared_reduction_is_recorded_even_for_single_element_units() -> None:
    model = _Negated().eval()
    one_channel = torch.ones(1, 1, 3, 3)  # broadcast over the two weight channels
    attr = grad_attr(model, one_channel)
    sel = F.top_k(attr, k=1, unit_axes=PIXELS, reduce="l2")
    assert sel.scores == tuple(11.0 * (u + 1) for u in range(9))  # |-11 (u + 1)|
    r = F.run(model, one_channel, test=comp(50.0), selection=sel, attributions=[attr])
    assert r.selection.unit_reduction == "l2"
    bnn.compose(bnn.trace(model, one_channel), attributions=[attr], faithfulness=[r])


def test_a_pixel_comprehensiveness_run_is_exact_recorded_and_composable() -> None:
    model = Pixels().eval()
    attr = grad_attr(model)
    sel = F.top_k(attr, k=2, unit_axes=PIXELS, reduce="sum")
    assert sel.selected == (7, 8)
    r = F.run(model, IMG, test=comp(150.0), selection=sel, attributions=[attr])
    assert r.drop == 11.0 * (8 + 9)
    assert r.outcome is Outcome.SUPPORTS
    assert r.selection.unit_axes == PIXELS
    assert r.selection.unit_reduction == "sum"
    assert r.claim.subject.unit_axes == PIXELS
    assert r.claim.subject.units == (7, 8)
    bnn.compose(bnn.trace(model, IMG), attributions=[attr], faithfulness=[r])


def test_internal_channel_sufficiency_is_exact() -> None:
    model = Pixels().eval()
    r = F.run(
        model,
        IMG,
        test=F.sufficiency(target=SEL, max_drop=50.0, statement="channel 1 suffices"),
        selection=F.units("scale", (1,), n_units=2, unit_axes=CHANNELS),
    )
    assert r.drop == 45.0
    assert r.outcome is Outcome.SUPPORTS


def test_input_units_are_checked_before_any_pass() -> None:
    model = Pixels().eval()
    for selection in (
        F.units(A.input(), (0,), n_units=3),  # the legacy rule cannot describe an image
        F.units(A.input(), (0,), n_units=10, unit_axes=PIXELS),
    ):
        with pytest.raises(F.FaithfulnessError):
            F.run(model, IMG, test=comp(1.0), selection=selection)
    assert model.calls == 0


def test_token_positions_of_a_transformer_are_units() -> None:
    model = TinyTransformer().eval()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    target = iv.metrics.select([0, 4, 5])
    attr = A.attribute(
        model, ids, target=target, method=A.gradient(), at=A.layer("token_embedding")
    )
    with pytest.raises(F.FaithfulnessError, match="unit_axes"):
        F.top_k(attr, k=1)
    sel = F.top_k(attr, k=2, unit_axes=CHANNELS, reduce="l2")
    assert sel.n_units == 5
    r = F.run(
        model,
        ids,
        test=F.comprehensiveness(target=target, min_drop=1e-9, statement="tokens"),
        selection=sel,
        attributions=[attr],
    )
    direct = iv.intervene(
        model,
        ids,
        intervention=iv.zero("token_embedding", units=sel.selected, unit_axes=CHANNELS),
        metric=target,
    )
    assert r.drop == pytest.approx(-direct.value, abs=1e-6)
    bnn.compose(bnn.trace(model, ids), attributions=[attr], faithfulness=[r])


def test_diagnostics_rank_declared_units() -> None:
    model = Pixels().eval()
    a = grad_attr(model)
    b = A.attribute(model, IMG, target=SEL, method=A.input_x_gradient())
    with pytest.raises(F.FaithfulnessError, match="unit_axes"):
        F.method_agreement(model, IMG, a=a, b=b, target=SEL, k=2)
    d = F.method_agreement(model, IMG, a=a, b=b, target=SEL, k=2, unit_axes=PIXELS, reduce="sum")
    assert d.measurements["topk_jaccard"] == 1.0
    assert d.protocol_result.params["unit_axes"] == (2, 3)
    bnn.compose(bnn.trace(model, IMG), attributions=[a, b], faithfulness=[d])


def test_curves_over_declared_units_record_and_verify_their_units() -> None:
    model = Pixels().eval()
    attr = grad_attr(model)
    c = F.curve(
        model,
        IMG,
        ranking=F.ranking(attr, unit_axes=PIXELS, reduce="sum"),
        target=SEL,
        mode="remove",
        points=[0, 1, 2],
        controls=F.controls(5, seed=0),
        attributions=[attr],
    )
    assert c.drops == (0.0, 99.0, 187.0)
    (selection,) = [r for r in c.trace.records if isinstance(r, EvidenceSelection)]
    assert selection.unit_axes == PIXELS
    assert selection.unit_reduction == "sum"
    assert selection.k is None
    bnn.compose(bnn.trace(model, IMG), attributions=[attr], faithfulness=[c])


# ------------------------------------------------------------------ magnitude-matched controls


def test_magnitude_strata_are_exact() -> None:
    assert stats.magnitude_strata([5, 1, 3, 3, 0, 2], 3) == [[0, 2], [3, 5], [1, 4]]
    assert stats.magnitude_strata([1, 2], 4) == [[1], [0]]
    mags = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0]
    draws = stats.stratified_subsets(mags, (0, 1, 6), 30, seed=5, strata=4)
    assert draws == stats.stratified_subsets(mags, (0, 1, 6), 30, seed=5, strata=4)
    assert draws != stats.stratified_subsets(mags, (0, 1, 6), 30, seed=6, strata=4)
    for d in draws:  # two units from stratum {0, 1}, one from {6, 7}
        assert sorted(u // 2 for u in d) == [0, 0, 3]


def _image() -> torch.Tensor:
    return torch.arange(1.0, 19.0).reshape(1, 2, 3, 3) / 4


def test_magnitude_matched_controls_are_recorded_reproducible_and_verified() -> None:
    model = Pixels().eval()
    x = _image()
    attr = A.attribute(model, x, target=SEL, method=A.input_x_gradient())
    sel = F.top_k(attr, k=2, unit_axes=PIXELS, reduce="sum")
    test = comp(1.0, controls=F.controls(20, seed=3, match="magnitude", strata=3))
    r = F.run(model, x, test=test, selection=sel, attributions=[attr])
    again = F.run(model, x, test=test, selection=sel, attributions=[attr])
    assert r.result.id == again.result.id
    spec = get(r.trace, r.result.spec.spec_id, ClaimTestSpec)
    declared: Any = spec.params["controls"]
    assert declared["strategy"] == "perturbation_magnitude_stratified_same_site_same_size"
    expected = unit_values(x, PIXELS, "l2").tolist()
    assert list(declared["magnitudes"]) == expected
    groups = stats.magnitude_strata(expected, 3)
    stratum = {u: i for i, g in enumerate(groups) for u in g}
    control_ids: Any = r.statistics["control_effects"]
    assert len(control_ids) == 20
    for cid in control_ids:
        effect = get(r.trace, cid, CausalEffect)
        units = get(r.trace, effect.interventions[0].record_id, InterventionRecord).units or ()
        assert sorted(stratum[u] for u in units) == sorted(stratum[u] for u in sel.selected)
    bnn.compose(bnn.trace(model, x), attributions=[attr], faithfulness=[r])
    count = F.run(model, x, test=comp(1.0, controls=F.controls(20, seed=3)), selection=sel)
    assert count.result.id != r.result.id


def test_magnitude_controls_are_refused_where_undefined() -> None:
    with pytest.raises(ValueError, match="match"):
        F.controls(5, seed=0, match="size")
    model = Pixels().eval()
    attr = grad_attr(model)
    with pytest.raises(F.FaithfulnessError, match="magnitude"):
        F.curve(
            model,
            IMG,
            ranking=F.ranking(attr, unit_axes=PIXELS, reduce="sum"),
            target=SEL,
            mode="remove",
            controls=F.controls(5, seed=0, match="magnitude"),
            attributions=[attr],
        )


# ------------------------------------------------------------------ forging and migration


def _forge(tmp_path: Path, trace: Any, kind: str, mutate: Any, version: int | None = None) -> Any:
    target = tmp_path / f"forged-{kind}-{len(list(tmp_path.iterdir()))}"
    trace.save(target)
    document = json.loads((target / "trace.json").read_text())
    renamed: dict[str, str] = {}
    for env in document["records"]:
        text = json.dumps(env["data"])
        for new, old in renamed.items():
            text = text.replace(new, old)
        env["data"] = json.loads(text)
        if env["kind"] == kind:
            mutate(env["data"])
            if version is not None:
                env["record_version"] = version
        new_id = env["id"]
        env["id"] = _expected_id(env["kind"], env["record_version"], env["data"])
        renamed[new_id] = env["id"]
    (target / "trace.json").write_text(json.dumps(document))
    return bnn.load_trace(target)


def _magnitude_run() -> tuple[Pixels, A.AttributionResult, F.FaithfulnessResult]:
    model = Pixels().eval()
    attr = A.attribute(model, _image(), target=SEL, method=A.input_x_gradient())
    r = F.run(
        model,
        _image(),
        test=comp(1.0, controls=F.controls(10, seed=1, match="magnitude")),
        selection=F.top_k(attr, k=2, unit_axes=PIXELS, reduce="sum"),
        attributions=[attr],
    )
    return model, attr, r


def test_forged_magnitudes_and_unit_fields_are_refused(tmp_path: Path) -> None:
    model, attr, r = _magnitude_run()

    def reverse(data: dict[str, Any]) -> None:
        data["params"]["controls"]["magnitudes"].reverse()

    def reduce_l2(data: dict[str, Any]) -> None:
        data["unit_reduction"] = "l2"

    for kind, mutate, message in (
        ("claim_test_spec", reverse, "magnitudes"),
        ("evidence_selection", reduce_l2, "re-derive"),
    ):
        forged = F.FaithfulnessResult(_forge(tmp_path, r.trace, kind, mutate), (attr,))
        with pytest.raises(EvidenceIntegrityError, match=message):
            bnn.compose(bnn.trace(model, _image()), attributions=[attr], faithfulness=[forged])


def test_rounding_level_magnitude_differences_are_accepted(tmp_path: Path) -> None:
    """Realistic failure (Phase 5.5, CNN channels): the magnitudes come from a separate
    traced pass whose conv kernels rounded 1 ulp differently from the family's pass."""
    model, attr, r = _magnitude_run()

    def nudge(data: dict[str, Any]) -> None:
        mags = data["params"]["controls"]["magnitudes"]
        data["params"]["controls"]["magnitudes"] = [m * (1 + 1e-7) for m in mags]

    def inflate(data: dict[str, Any]) -> None:
        mags = data["params"]["controls"]["magnitudes"]
        data["params"]["controls"]["magnitudes"] = [m * 1.01 for m in mags]

    ok = F.FaithfulnessResult(_forge(tmp_path, r.trace, "claim_test_spec", nudge), (attr,))
    bnn.compose(bnn.trace(model, _image()), attributions=[attr], faithfulness=[ok])
    bad = F.FaithfulnessResult(_forge(tmp_path, r.trace, "claim_test_spec", inflate), (attr,))
    with pytest.raises(EvidenceIntegrityError, match="magnitudes"):
        bnn.compose(bnn.trace(model, _image()), attributions=[attr], faithfulness=[bad])


def test_evaluation_requires_matching_unit_axes_for_claim_and_controls() -> None:
    model = Pixels().eval()
    r = F.run(
        model,
        IMG,
        test=comp(1.0, controls=F.controls(4, seed=0)),
        selection=F.units(A.input(), (4,), n_units=9, unit_axes=PIXELS),
    )
    assert r.outcome is Outcome.SUPPORTS
    spec = get(r.trace, r.result.spec.spec_id, ClaimTestSpec)
    effect = get(r.trace, r.statistics["selected_effect"], CausalEffect)
    control_ids: Any = r.statistics["control_effects"]
    controls = [get(r.trace, c, CausalEffect) for c in control_ids]
    records = {
        e.interventions[0].record_id: get(r.trace, e.interventions[0].record_id, InterventionRecord)
        for e in (effect, *controls)
    }

    def outcome(claim: Any, recs: dict[str, InterventionRecord]) -> Outcome:
        return evaluate(
            claim,
            spec,
            effect,
            controls,
            recs,
            no_op=False,
            provenance_id=r.result.provenance_id or "",
        ).outcome

    assert outcome(r.claim, records) is Outcome.SUPPORTS  # re-derives unchanged
    subject = dataclasses.replace(r.claim.subject, unit_axes=None)
    assert outcome(dataclasses.replace(r.claim, subject=subject), records) is Outcome.NOT_APPLICABLE
    cid = controls[0].interventions[0].record_id
    legacy_control = records | {cid: dataclasses.replace(records[cid], unit_axes=None)}
    assert outcome(r.claim, legacy_control) is Outcome.NOT_APPLICABLE


def test_attribution_thresholds_do_not_apply_to_declared_unit_axes() -> None:
    method = A.gradient()
    claim = A.make_claim(A.input(), SEL, IMG, statement="column 2", units=(2,))
    wide = dataclasses.replace(claim, subject=dataclasses.replace(claim.subject, unit_axes=PIXELS))
    spec = A.threshold_spec(method, min_abs_attribution=1.0)
    r = A.attribute(
        Pixels().eval(), IMG, target=SEL, method=method, claims=[(claim, spec), (wide, spec)]
    )
    assert [c.outcome for c in r.claim_results] == [Outcome.SUPPORTS, Outcome.NOT_APPLICABLE]


def test_unit_axes_schema_rules() -> None:
    claim = A.make_claim(A.input(), SEL, IMG, statement="all", units=None)
    with pytest.raises(SchemaError, match="requires units"):
        dataclasses.replace(claim.subject, unit_axes=PIXELS)
    with pytest.raises(SchemaError, match="sorted"):
        dataclasses.replace(claim.subject, units=(1,), unit_axes=(3, 2))


def test_phase5_claims_and_selections_migrate_without_invented_axes(tmp_path: Path) -> None:
    x = torch.ones(1, 4)
    lin = nn.Linear(4, 1).eval()
    attr = A.attribute(lin, x, target=SEL, method=A.gradient())
    r = F.run(lin, x, test=comp(1e-9), selection=F.top_k(attr, k=1), attributions=[attr])

    def v1_claim(data: dict[str, Any]) -> None:
        data["subject"].pop("unit_axes")

    def v1_selection(data: dict[str, Any]) -> None:
        data.pop("unit_axes")
        data.pop("unit_reduction")

    trace = _forge(tmp_path, r.trace, "claim", v1_claim, version=1)
    trace = _forge(tmp_path, trace, "evidence_selection", v1_selection, version=1)
    loaded = F.FaithfulnessResult(trace, (attr,))
    assert loaded.claim.subject.unit_axes is None
    assert loaded.selection.unit_axes is None
    assert loaded.selection.unit_reduction is None
    assert from_json(to_json(loaded.selection)) == loaded.selection
    bnn.compose(bnn.trace(lin, x), attributions=[attr], faithfulness=[loaded])
