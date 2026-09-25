"""Phase 5: faithfulness results in the structured WHY (composition; ADR-031/033).

Expected values are pre-registered in docs/PHASE_5_PLAN.md §17."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

import beyondnn as bnn
import beyondnn.attribution as A
import beyondnn.faithfulness as F
import beyondnn.interventions as iv
from beyondnn._testing import faithfulness_models as FM
from beyondnn._testing.causal_models import Additive, Interaction
from beyondnn.explain import (
    EvidenceIntegrityError,
    ModelMismatchError,
    SampleMismatchError,
    TargetMismatchError,
    UnsupportedScopeError,
)
from beyondnn.schema import (
    EvidenceStatus,
    ModelDeclaration,
    Outcome,
    Relation,
    Verdict,
    from_json,
    to_json,
)
from beyondnn.schema.codec import _expected_id

SEL = iv.metrics.select([0, 0])
ONES = torch.ones(1, 8)
ATOL = 1e-5


def x(*values: float) -> torch.Tensor:
    return torch.tensor([list(values)], dtype=torch.float32)


def ig(n: int = 64, baseline: A.Baseline | None = None) -> A.Method:
    return A.integrated_gradients(baseline=baseline or A.zero_baseline(), n_steps=n)


def comp(min_drop: float = 1.0, **kw: Any) -> F.TestTemplate:
    kw.setdefault("statement", "the selected units are necessary for the target")
    return F.comprehensiveness(target=SEL, min_drop=min_drop, **kw)


def suff(max_drop: float = 0.5, **kw: Any) -> F.TestTemplate:
    kw.setdefault("statement", "the selected units suffice for the target")
    return F.sufficiency(target=SEL, max_drop=max_drop, **kw)


def declared(*units: int, n: int, site: Any = None) -> F.Selection:
    return F.units(site if site is not None else A.input(), units, n_units=n)


def weighted8_attr() -> A.AttributionResult:
    return A.attribute(FM.Weighted8().eval(), ONES, target=SEL, method=ig(8))


def test_scenario_g_a_result_about_another_input_is_not_composed() -> None:
    model = Additive().eval()
    elsewhere = F.run(model, x(1, 1), test=comp(0.5), selection=declared(0, n=2))
    with pytest.raises(SampleMismatchError):
        bnn.compose(bnn.trace(model, x(3, 5)), faithfulness=[elsewhere])


def interaction_dataset() -> F.DatasetResult:
    samples = [x(3, 5), x(2, 2), x(1, 3), x(3, 0)]
    return F.run_dataset(
        Interaction().eval(), samples, test=comp(), rule=F.fixed(declared(0, n=1, site="a"))
    )


def test_scenario_k_metrics_disagree_and_both_stay_visible() -> None:
    model = FM.RedundantMax().eval()
    inputs = x(3, 3)
    c = F.run(model, inputs, test=comp(), selection=declared(0, n=2))
    s = F.run(model, inputs, test=suff(0.5), selection=declared(0, n=2))
    assert (c.outcome, s.outcome) == (Outcome.CONTRADICTS, Outcome.SUPPORTS)
    response = bnn.compose(
        bnn.trace(model, inputs),
        faithfulness=[c, s],
        policies=[F.COMPREHENSIVENESS_POLICY, F.SUFFICIENCY_POLICY],
    )
    verdicts = {v.claim.relation: [a.verdict for a in v.assessments] for v in response.why.claims}
    assert verdicts[Relation.NECESSARY_FOR] == [Verdict.CONTRADICTED]
    assert verdicts[Relation.SUFFICIENT_FOR] == [Verdict.SUPPORTED]
    text = response.render()
    assert "result: contradicts" in text
    assert "result: supports" in text


def test_results_round_trip_and_recompose_identically(tmp_path: Path) -> None:
    attr = weighted8_attr()
    model = FM.Weighted8().eval()
    r = F.run(
        model,
        ONES,
        test=comp(controls=F.controls(10, seed=0)),
        selection=F.top_k(attr, k=2),
        attributions=[attr],
    )
    c = F.curve(
        model, ONES, ranking=F.ranking(attr), target=SEL, mode="remove", attributions=[attr]
    )
    trace = bnn.trace(model, ONES)
    before = bnn.compose(trace, attributions=[attr], faithfulness=[r, c])
    for name, t in (("t", trace), ("a", attr.trace), ("r", r.trace), ("c", c.trace)):
        t.save(tmp_path / name)
    la = A.AttributionResult.from_trace(bnn.load_trace(tmp_path / "a"))
    after = bnn.compose(
        bnn.load_trace(tmp_path / "t"),
        attributions=[la],
        faithfulness=[
            F.FaithfulnessResult(bnn.load_trace(tmp_path / "r"), (la,)),
            F.CurveResult(bnn.load_trace(tmp_path / "c"), (la,)),
        ],
    )
    assert after.to_dict() == before.to_dict()
    assert after.render() == before.render()
    sel = r.selection
    assert from_json(to_json(sel)) == sel


def _forge(tmp_path: Path, trace: Any, kind: str, mutate: Any) -> Any:
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
        new_id = env["id"]
        env["id"] = _expected_id(env["kind"], env["record_version"], env["data"])
        renamed[new_id] = env["id"]
    (target / "trace.json").write_text(json.dumps(document))
    return bnn.load_trace(target)


def test_forged_faithfulness_results_are_refused(tmp_path: Path) -> None:
    model = FM.Weighted8().eval()
    r = F.run(model, ONES, test=comp(20.0), selection=declared(0, 1, n=8))
    assert r.outcome is Outcome.CONTRADICTS

    def flip(data: dict[str, Any]) -> None:
        data["outcome"] = "supports"

    forged = F.FaithfulnessResult(_forge(tmp_path, r.trace, "claim_test_result", flip))
    with pytest.raises(EvidenceIntegrityError, match="does not follow"):
        bnn.compose(bnn.trace(model, ONES), faithfulness=[forged])


def test_forged_curves_and_selections_are_refused(tmp_path: Path) -> None:
    attr = weighted8_attr()
    model = FM.Weighted8().eval()
    c = F.curve(
        model, ONES, ranking=F.ranking(attr), target=SEL, mode="remove", attributions=[attr]
    )

    def inflate(data: dict[str, Any]) -> None:
        data["measurements"]["aopc_mean_drop"] = 99.0

    forged = F.CurveResult(_forge(tmp_path, c.trace, "protocol_result", inflate), (attr,))
    with pytest.raises(EvidenceIntegrityError, match="aopc"):
        bnn.compose(bnn.trace(model, ONES), faithfulness=[forged])

    def reorder(data: dict[str, Any]) -> None:
        data["order"] = list(reversed(data["order"]))

    r = F.run(model, ONES, test=comp(), selection=F.top_k(attr, k=2), attributions=[attr])
    bad = F.FaithfulnessResult(_forge(tmp_path, r.trace, "evidence_selection", reorder), (attr,))
    with pytest.raises(EvidenceIntegrityError, match="re-derive"):
        bnn.compose(bnn.trace(model, ONES), faithfulness=[bad])


def test_composition_refuses_mismatched_context() -> None:
    model = FM.Weighted8().eval()
    r = F.run(model, ONES, test=comp(), selection=declared(0, n=8))
    with pytest.raises(TargetMismatchError):
        bnn.compose(
            bnn.trace(model, ONES),
            faithfulness=[r],
            attributions=[A.attribute(model, ONES, target=iv.metrics.mean(), method=A.gradient())],
        )
    with pytest.raises(ModelMismatchError, match="ModelDeclaration"):
        bnn.compose(
            bnn.trace(model, ONES, declared_model=ModelDeclaration(implementation_revision="v2")),
            faithfulness=[r],
        )
    with pytest.raises(UnsupportedScopeError, match="dataset-level"):
        bnn.compose(bnn.trace(Interaction().eval(), x(3, 5)), faithfulness=[interaction_dataset()])


def test_scenario_j_a_pre_adr_031_trace_cannot_anchor_faithfulness(tmp_path: Path) -> None:
    model = FM.Weighted8().eval()
    r = F.run(model, ONES, test=comp(), selection=declared(0, n=8))

    def drop_identity(data: dict[str, Any]) -> None:
        data["sample_id"] = None

    old = _forge(tmp_path, bnn.trace(model, ONES), "input", drop_identity)
    assert old.input.sample_id is None
    bnn.compose(old)  # measured-only still works
    with pytest.raises(SampleMismatchError, match="no exact sample identity"):
        bnn.compose(old, faithfulness=[r])


def test_coverage_names_exactly_the_protocols_that_ran() -> None:
    model = FM.RedundantMax().eval()
    trace = bnn.trace(model, x(3, 3))
    assert bnn.compose(trace).why.coverage.faithfulness_evaluated is False
    c = F.run(model, x(3, 3), test=comp(), selection=declared(0, n=2))
    why = bnn.compose(trace, faithfulness=[c]).why
    assert why.coverage.faithfulness_evaluated is True
    assert why.coverage.faithfulness_protocols == ("comprehensiveness",)
    assert "sufficiency" in why.coverage.not_evaluated
    assert "concept validation" in why.coverage.not_evaluated
    assert why.coverage.concepts_validated is False
    assert why.faithfulness_tests[0].result is c.result
    assert all(
        e.status is EvidenceStatus.INTERVENTIONAL
        for e in why.by_status(EvidenceStatus.INTERVENTIONAL)
    )


def test_rendered_faithfulness_makes_no_global_statement() -> None:
    model = FM.RedundantMax().eval()
    c = F.run(model, x(3, 3), test=comp(), selection=declared(0, n=2))
    s = F.run(model, x(3, 3), test=suff(0.5), selection=declared(0, n=2))
    text = bnn.compose(bnn.trace(model, x(3, 3)), faithfulness=[c, s]).render()
    lowered = text.lower()
    for phrase in (
        "is faithful",
        "explanation is",
        "faithfulness score",
        "confidence:",
        "the model used",
        "caused",
    ):
        assert phrase not in lowered, phrase
    assert "each bears only on its own claim" in text
