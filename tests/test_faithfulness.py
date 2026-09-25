"""Phase 5: faithfulness tests (ADR-033). Expected values are pre-registered in
docs/PHASE_5_PLAN.md §17 (written before any Phase-5 code ran)."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
from torch import nn

import beyondnn as bnn
import beyondnn.attribution as A
import beyondnn.faithfulness as F
import beyondnn.interventions as iv
from beyondnn._testing import faithfulness_models as FM
from beyondnn._testing.attribution_models import Product, Saturating
from beyondnn._testing.causal_models import Additive, Interaction
from beyondnn.core.samples import sample_id
from beyondnn.faithfulness import stats
from beyondnn.protocols import COMPREHENSIVENESS, PROTOCOLS, SUFFICIENCY
from beyondnn.schema import (
    AspectOutcome,
    CheckOutcome,
    EvidenceSelection,
    EvidenceStatus,
    Outcome,
    ProtocolResult,
    Relation,
    SchemaError,
    SelectionSource,
    Site,
    SiteIO,
)

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


def seq(value: object) -> tuple[Any, ...]:
    assert isinstance(value, tuple)
    return value


def weighted8_attr() -> A.AttributionResult:
    return A.attribute(FM.Weighted8().eval(), ONES, target=SEL, method=ig(8))


# ------------------------------------------------------------------ statistics (exact)


def test_statistics_are_exact_on_hand_computed_cases() -> None:
    assert stats.control_fractions(3.0, [1.0, 3.0, 5.0, 2.0]) == {
        "fraction_below": 0.5,
        "fraction_tied": 0.25,
        "fraction_above": 0.25,
    }
    assert stats.mc_p_value(3.0, [1.0, 3.0, 5.0, 2.0]) == (1 + 2) / 5
    assert stats.spearman_of_orders((0, 1, 2), (0, 1, 2)) == 1.0
    assert stats.spearman_of_orders((0, 1, 2), (2, 1, 0)) == -1.0
    assert stats.jaccard((0, 1), (1, 2)) == 1 / 3
    assert stats.quantile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert stats.rank_order([1.0, -3.0, 3.0], by="abs") == (1, 2, 0)  # tie -> lower index
    assert stats.rank_order([1.0, -3.0, 3.0], by="signed") == (2, 0, 1)
    assert stats.sign_flip_p([1.0, 1.0], 1000, 0) == pytest.approx(0.25, abs=0.05)


def test_random_draws_are_seeded_locally_and_never_touch_the_global_rng() -> None:
    rng = torch.get_rng_state()
    a = stats.uniform_subsets(8, 2, 20, seed=3)
    b = stats.uniform_subsets(8, 2, 20, seed=3)
    assert a == b
    assert a != stats.uniform_subsets(8, 2, 20, seed=4)
    assert all(len(s) == 2 and list(s) == sorted(set(s)) for s in a)
    assert stats.uniform_permutations(5, 3, 1) == stats.uniform_permutations(5, 3, 1)
    assert torch.equal(rng, torch.get_rng_state())


# ------------------------------------------------------------------ scenario A


def test_scenario_a_true_causal_selection_beats_matched_controls() -> None:
    attr = weighted8_attr()
    model = FM.Weighted8().eval()
    r = F.run(
        model,
        ONES,
        test=comp(controls=F.controls(200, seed=0)),
        selection=F.top_k(attr, k=2),
        attributions=[attr],
    )
    s = r.statistics
    assert r.outcome is Outcome.SUPPORTS
    assert s["drop"] == 12.0
    assert abs(float(s["fraction_below"]) - 27 / 28) <= 0.04  # type: ignore[arg-type]
    assert 0.006 <= float(s["mc_p_value"]) <= 0.066  # type: ignore[arg-type]
    assert s["n_controls"] == 200
    assert r.claim.subject.units == (0, 1)
    assert r.trace.passes == 1 + 1 + 200
    sufficient = F.run(
        model, ONES, test=suff(4.0, controls=F.controls(200, seed=0)), selection=F.top_k(attr, k=2)
    )
    assert sufficient.drop == 3.0
    assert sufficient.outcome is Outcome.SUPPORTS
    assert abs(float(sufficient.statistics["fraction_above"]) - 27 / 28) <= 0.04  # type: ignore[arg-type]


def test_controls_are_reproducible_and_recorded() -> None:
    attr = weighted8_attr()
    model = FM.Weighted8().eval()
    runs = [
        F.run(model, ONES, test=comp(controls=F.controls(30, seed=7)), selection=F.top_k(attr, k=2))
        for _ in range(2)
    ]
    assert runs[0].result.id == runs[1].result.id
    spec = next(r for r in runs[0].trace.records if r.KIND == "claim_test_spec")
    assert spec.params["controls"]["seed"] == 7  # type: ignore[attr-defined]
    other = F.run(
        model, ONES, test=comp(controls=F.controls(30, seed=8)), selection=F.top_k(attr, k=2)
    )
    assert other.result.id != runs[0].result.id


# ------------------------------------------------------------------ scenarios B and I (premise)


def test_scenario_b_two_plausible_rankings_only_one_tracks_causal_structure() -> None:
    model = FM.ProxyDistractor().eval()
    inputs = x(3, 1, 4)
    attr = A.attribute(model, inputs, target=SEL, method=ig())
    assert torch.allclose(attr.value, x(3, 1, 0), atol=ATOL)
    data = FM.proxy_dataset()
    y = data[:, 0] + data[:, 1]
    corr = [float(torch.corrcoef(torch.stack([data[:, i], y]))[0, 1]) for i in range(3)]
    assert corr[2] == pytest.approx(1.0)
    assert corr[2] > corr[0] > corr[1]  # the correlation method ranks the distractor first
    by_ig = F.run(model, inputs, test=comp(), selection=F.top_k(attr, k=1))
    by_corr = F.run(model, inputs, test=comp(), selection=declared(2, n=3))
    assert (by_ig.drop, by_ig.outcome) == (3.0, Outcome.SUPPORTS)
    assert (by_corr.drop, by_corr.outcome) == (0.0, Outcome.CONTRADICTS)


def test_scenario_i_misleading_gradient_is_exposed_and_the_disagreement_kept() -> None:
    model = FM.SaturatingPlus().eval()
    inputs = x(3, 1)
    grad = A.attribute(model, inputs, target=SEL, method=A.gradient())
    integrated = A.attribute(model, inputs, target=SEL, method=ig())
    assert F.top_k(grad, k=1).selected == (1,)
    assert F.top_k(integrated, k=1).selected == (0,)
    by_grad = F.run(model, inputs, test=comp(0.5), selection=F.top_k(grad, k=1))
    by_ig = F.run(model, inputs, test=comp(0.5), selection=F.top_k(integrated, k=1))
    assert by_grad.drop == pytest.approx(0.2, abs=ATOL)
    assert by_grad.outcome is Outcome.CONTRADICTS
    assert by_ig.drop == pytest.approx(math.tanh(12), abs=ATOL)
    assert by_ig.outcome is Outcome.SUPPORTS
    agree = F.method_agreement(
        model, inputs, a=grad, b=integrated, target=SEL, k=1, min_rank_correlation=0.5
    )
    assert agree.measurements["rank_correlation"] == -1.0
    assert agree.outcome("min_rank_correlation") is CheckOutcome.FAIL
    assert agree.outcome("min_topk_jaccard") is CheckOutcome.INDETERMINATE  # no criterion


# ------------------------------------------------------------------ scenarios C, D, K, L


def test_scenario_c_redundant_causes_are_not_individually_necessary() -> None:
    model = FM.RedundantMax().eval()
    single = F.run(model, x(3, 3), test=comp(), selection=declared(0, n=2))
    both = F.run(model, x(3, 3), test=comp(), selection=declared(0, 1, n=2))
    assert (single.drop, single.outcome) == (0.0, Outcome.CONTRADICTS)
    assert (both.drop, both.outcome) == (3.0, Outcome.SUPPORTS)


def test_scenario_d_relevant_units_can_be_jointly_but_not_singly_sufficient() -> None:
    model = FM.EqualSum4().eval()
    retained = F.run(model, torch.ones(1, 4), test=suff(0.5), selection=declared(0, n=4))
    removed = F.run(model, torch.ones(1, 4), test=comp(0.5), selection=declared(0, n=4))
    assert (retained.drop, retained.outcome) == (3.0, Outcome.CONTRADICTS)
    assert (removed.drop, removed.outcome) == (1.0, Outcome.SUPPORTS)


def test_scenario_l_readable_but_unused_internal_unit() -> None:
    model = FM.ProbeReadable().eval()
    inputs = x(3, 5)
    trace = bnn.trace(model, inputs, sites=["hidden"], retention="cpu")
    assert torch.equal(trace.tensor(trace.activations[0]), x(3, 5))  # h1 = x1: readable
    drops = {
        u: F.run(model, inputs, test=comp(), selection=declared(u, n=2, site="hidden")).drop
        for u in (0, 1)
    }
    assert drops == {0: 6.0, 1: 0.0}
    kept = F.run(model, inputs, test=suff(0.5), selection=declared(0, n=2, site="hidden"))
    assert (kept.drop, kept.outcome) == (0.0, Outcome.SUPPORTS)
    assert "SITE_RELATIVE_SUFFICIENCY" in {lim.code for lim in kept.limitations}
    lost = F.run(model, inputs, test=suff(0.5), selection=declared(1, n=2, site="hidden"))
    assert (lost.drop, lost.outcome) == (6.0, Outcome.CONTRADICTS)


# ------------------------------------------------------------------ scenario E (stability)


def _swap() -> F.Transformation:
    return F.transformation("swap01", lambda t: t[:, [1, 0]], implementation_revision="v1")


def test_scenario_e_prediction_stable_but_ranking_not_and_never_combined() -> None:
    result = F.stability(
        Additive().eval(),
        x(3, 5),
        transformation=_swap(),
        method=A.input_x_gradient(),
        target=SEL,
        k=1,
        max_prediction_change=1e-6,
        min_rank_correlation=0.5,
        min_topk_jaccard=0.5,
    )
    outcomes = {o.aspect: o.outcome for o in result.protocol_result.outcomes}
    assert outcomes == {
        "max_prediction_change": CheckOutcome.PASS,
        "min_rank_correlation": CheckOutcome.FAIL,
        "min_topk_jaccard": CheckOutcome.FAIL,
        "same_claim_outcome": CheckOutcome.NOT_APPLICABLE,
    }
    assert result.measurements["rank_correlation"] == -1.0
    assert "DECLARED_TRANSFORMATION_UNVERIFIED" in {lim.code for lim in result.trace.limitations}
    assert len(result.protocol_result.samples) == 2


def test_stability_claim_aspect_and_undeclared_criteria() -> None:
    result = F.stability(
        Additive().eval(),
        x(3, 5),
        transformation=_swap(),
        method=A.input_x_gradient(),
        target=SEL,
        k=1,
        test=comp(1.0),
    )
    outcomes = {o.aspect: o.outcome for o in result.protocol_result.outcomes}
    assert outcomes["max_prediction_change"] is CheckOutcome.INDETERMINATE
    assert outcomes["same_claim_outcome"] is CheckOutcome.PASS
    assert [t.outcome for t in result.tests] == [Outcome.SUPPORTS, Outcome.SUPPORTS]


@pytest.mark.parametrize(
    ("fn", "message"),
    [
        (lambda t: t[:, :1], "same shape"),
        (lambda t: t.clone(), "no-op"),
        (lambda t: t.mul_(2), "in place"),
    ],
)
def test_invalid_declared_transformations_are_refused(fn: Any, message: str) -> None:
    with pytest.raises(F.FaithfulnessError, match=message):
        F.stability(
            Additive().eval(),
            x(3, 5),
            transformation=F.transformation("t", fn, implementation_revision="v1"),
            method=A.input_x_gradient(),
            target=SEL,
            k=1,
        )
    with pytest.raises(ValueError, match="permutation"):
        F.transformation("t", fn, implementation_revision="v1", unit_map=(0, 0))


# ------------------------------------------------------------------ scenario F (counterexample)


def interaction_dataset() -> F.DatasetResult:
    samples = [x(3, 5), x(2, 2), x(1, 3), x(3, 0)]
    return F.run_dataset(
        Interaction().eval(), samples, test=comp(), rule=F.fixed(declared(0, n=1, site="a"))
    )


def test_scenario_f_counterexample_is_recorded_not_discarded() -> None:
    result = interaction_dataset()
    assert [r.outcome for r in result.results] == [Outcome.SUPPORTS] * 3 + [Outcome.CONTRADICTS]
    assert result.counterexamples == (sample_id(x(3, 0)),)
    summary = result.summary("counterexample")
    (outcome,) = summary.outcomes
    assert outcome.outcome is CheckOutcome.FAIL
    assert outcome.detail is not None
    assert "held on 3 of 4" in outcome.detail
    assert list(seq(summary.measurements["drops"])) == [15.0, 4.0, 3.0, 0.0]
    assert len(summary.derived_from) == 4


def test_dataset_runs_with_attribution_selection_and_paired_controls() -> None:
    samples = [ONES, torch.full((1, 8), 2.0), torch.arange(1.0, 9.0).reshape(1, 8)]
    result = F.run_dataset(
        FM.Weighted8().eval(),
        samples,
        test=comp(controls=F.controls(40, seed=0)),
        rule=F.selector(ig(8), k=2),
        permutation_draws=500,
    )
    assert len(result.attributions) == 3
    assert all(r.outcome is Outcome.SUPPORTS for r in result.results)
    paired = result.summary("paired_control")
    diffs = list(seq(paired.measurements["paired_differences"]))
    assert len(diffs) == 3
    assert all(d > 0 for d in diffs)
    assert paired.outcomes == ()  # descriptive statistics, not a verdict
    assert paired.measurements["seed"] == 0
    assert result.counterexamples == ()


# ------------------------------------------------------------------ scenario H (random method)


def test_scenario_h_random_rankings_do_not_beat_matched_controls() -> None:
    model = FM.Weighted8().eval()
    scores = []
    for seed in range(40):
        order = stats.uniform_permutations(8, 1, 1000 + seed)[0]
        r = F.run(
            model,
            ONES,
            test=comp(0.5, controls=F.controls(200, seed=seed)),
            selection=declared(*order[:2], n=8),
        )
        s = r.statistics
        scores.append(float(s["fraction_below"]) + 0.5 * float(s["fraction_tied"]))  # type: ignore[arg-type]
    mean = sum(scores) / len(scores)
    assert 0.40 <= mean <= 0.60, mean
    attr = weighted8_attr()
    best = F.run(
        model, ONES, test=comp(controls=F.controls(200, seed=0)), selection=F.top_k(attr, k=2)
    ).statistics
    assert float(best["fraction_below"]) + 0.5 * float(best["fraction_tied"]) >= 0.95  # type: ignore[arg-type]


# ------------------------------------------------------------------ scenario M (curves)


def test_scenario_m_removal_and_retention_curves_are_exact() -> None:
    attr = weighted8_attr()
    model = FM.Weighted8().eval()
    removal = F.curve(model, ONES, ranking=F.ranking(attr), target=SEL, mode="remove")
    retention = F.curve(model, ONES, ranking=F.ranking(attr), target=SEL, mode="retain")
    assert removal.drops == (0, 8, 12, 14, 15, 15, 15, 15, 15)
    assert retention.drops == (15, 7, 3, 1, 0, 0, 0, 0, 0)
    assert removal.protocol_result.measurements["aopc_mean_drop"] == pytest.approx(109 / 9)
    assert retention.protocol_result.measurements["aopc_mean_drop"] == pytest.approx(26 / 9)
    assert removal.points == tuple(range(9))
    assert seq(removal.protocol_result.measurements["effects"])[0] is None  # the k=0 anchor
    assert "normalization" in removal.protocol_result.params


def test_curves_with_random_ranking_controls() -> None:
    attr = weighted8_attr()
    c = F.curve(
        FM.Weighted8().eval(),
        ONES,
        ranking=F.ranking(attr),
        target=SEL,
        mode="remove",
        points=(0, 1, 2, 4),
        controls=F.controls(25, seed=0),
    )
    m = c.protocol_result.measurements
    assert len(seq(m["control_drops"])) == 25
    assert all(len(row) == 4 for row in seq(m["control_drops"]))
    assert float(m["aopc_fraction_controls_worse"]) >= 0.8  # type: ignore[arg-type]
    with pytest.raises(F.FaithfulnessError, match="points"):
        F.curve(
            FM.Weighted8().eval(),
            ONES,
            ranking=F.ranking(attr),
            target=SEL,
            mode="remove",
            points=(2, 1),
        )


# ------------------------------------------------------------------ scenario N and RQ9


def test_scenario_n_method_diagnostics() -> None:
    model = Product().eval()
    inputs = x(3, 5)
    a1 = A.attribute(model, inputs, target=SEL, method=ig(8))
    a2 = A.attribute(model, inputs, target=SEL, method=ig(8, A.baseline(x(2, 1))))
    assert torch.allclose(a2.value, x(3, 10), atol=ATOL)
    sensitivity = F.baseline_sensitivity(
        model, inputs, a=a1, b=a2, target=SEL, k=1, min_rank_correlation=0.5
    )
    assert sensitivity.measurements["rank_correlation"] == -1.0
    assert sensitivity.outcome("min_rank_correlation") is CheckOutcome.FAIL
    sat = Saturating().eval()
    s16 = A.attribute(sat, x(3, 1), target=SEL, method=ig(16))
    s32 = A.attribute(sat, x(3, 1), target=SEL, method=ig(32))
    steps = F.ig_step_sensitivity(
        sat, x(3, 1), a=s16, b=s32, target=SEL, k=1, max_abs_difference=2.25 / 16**2 + 1e-5
    )
    assert steps.outcome("max_abs_difference") is CheckOutcome.PASS
    with pytest.raises(F.FaithfulnessError, match="same number of steps"):
        F.ig_step_sensitivity(sat, x(3, 1), a=s16, b=s16, target=SEL, k=1)


def test_rq9_replacement_choice_alone_can_erase_the_signal() -> None:
    model = FM.Weighted8().eval()
    inputs = x(1, 1, 1, 1, 5, 5, 5, 5)
    zero = [F.run(model, inputs, test=comp(0.5), selection=declared(u, n=8)) for u in range(8)]
    mean = [
        F.run(
            model,
            inputs,
            test=comp(0.5, replacement=F.replacement(inputs.clone())),
            selection=declared(u, n=8),
        )
        for u in range(8)
    ]
    assert [r.drop for r in zero] == [8, 4, 2, 1, 0, 0, 0, 0]
    assert all(r.outcome is Outcome.INCONCLUSIVE for r in mean)  # a no-op, not "unimportant"
    assert all(r.statistics["no_op"] for r in mean)


# ------------------------------------------------------------------ refusals and guards


def test_scenario_g_evidence_about_another_input_is_refused() -> None:
    model = Additive().eval()
    attr = A.attribute(model, x(3, 5), target=SEL, method=ig())
    with pytest.raises(F.SelectionMismatchError, match="another input"):
        F.run(model, x(1, 1), test=comp(), selection=F.top_k(attr, k=1))


def test_selection_and_test_refusals() -> None:
    model = FM.Weighted8().eval()
    attr = weighted8_attr()
    other_target = iv.metrics.mean()
    with pytest.raises(F.SelectionMismatchError, match="different target"):
        F.run(
            model,
            ONES,
            test=F.comprehensiveness(target=other_target, min_drop=1.0, statement="s"),
            selection=F.top_k(attr, k=2),
        )
    with pytest.raises(F.FaithfulnessError, match="vacuous"):
        F.run(model, ONES, test=suff(), selection=declared(*range(8), n=8))
    with pytest.raises(F.FaithfulnessError, match="degenerate"):
        F.run(
            model,
            ONES,
            test=comp(controls=F.controls(5, seed=0)),
            selection=declared(*range(8), n=8),
        )
    with pytest.raises(F.FaithfulnessError, match="units"):
        F.units(A.input(), (), n_units=8)
    with pytest.raises(F.FaithfulnessError, match="units"):
        F.units(A.input(), (9,), n_units=8)
    with pytest.raises(F.FaithfulnessError, match="k="):
        F.top_k(attr, k=9)
    with pytest.raises(F.SelectionMismatchError, match="units"):
        F.run(model, ONES, test=comp(), selection=declared(0, n=5))
    with pytest.raises(iv.InterventionError, match="no broadcasting"):
        F.run(
            model,
            ONES,
            test=comp(replacement=F.replacement(torch.zeros(8))),
            selection=declared(0, n=8),
        )
    with pytest.raises(iv.StatefulComparisonError, match="training"):
        F.run(FM.Weighted8().train(), ONES, test=comp(), selection=declared(0, n=8))
    with pytest.raises(ValueError, match="NECESSARY_FOR or DECREASES"):
        F.comprehensiveness(
            target=SEL, min_drop=1.0, statement="s", relation=Relation.SUFFICIENT_FOR
        )
    with pytest.raises(ValueError, match="needs controls"):
        F.comprehensiveness(target=SEL, min_drop=1.0, statement="s", min_fraction_below=0.9)


def test_selections_over_non_vector_sites_are_refused() -> None:
    ids = torch.tensor([[1, 2, 3]])
    from beyondnn._testing.models import TinyTransformer

    attr = A.attribute(
        TinyTransformer().eval(),
        ids,
        target=iv.metrics.select([0, 2, 5]),
        method=A.gradient(),
        at=A.layer("token_embedding"),
    )
    with pytest.raises(F.FaithfulnessError, match="non-last dimensions"):
        F.top_k(attr, k=1)


class _Infinite(nn.Module):
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return (t.sum() / (t[:, :1] * 0)).reshape(1, 1)


def test_non_finite_targets_are_refused() -> None:
    with pytest.raises((SchemaError, ValueError)):
        F.run(_Infinite().eval(), x(1, 2), test=comp(), selection=declared(0, n=2))


def test_a_perturbation_that_did_not_happen_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    from beyondnn.interventions import runner

    monkeypatch.setattr(runner._Replacer, "perturb", lambda self, leaf: leaf)
    with pytest.raises(Exception, match="requested perturb"):
        F.run(FM.Weighted8().eval(), ONES, test=comp(), selection=declared(0, n=8))
    with pytest.raises(Exception, match="requested perturbation"):
        F.run(
            FM.ProbeReadable().eval(),
            x(3, 5),
            test=comp(),
            selection=declared(0, n=2, site="hidden"),
        )


def test_ties_at_the_selection_boundary_are_flagged() -> None:
    attr = A.attribute(Product().eval(), x(3, 5), target=SEL, method=ig(4))  # [7.5, 7.5]
    r = F.run(Product().eval(), x(3, 5), test=comp(), selection=F.top_k(attr, k=1))
    assert "SELECTION_TIE_AT_BOUNDARY" in {lim.code for lim in r.limitations}
    untied = F.run(
        FM.Weighted8().eval(), ONES, test=comp(), selection=F.top_k(weighted8_attr(), k=2)
    )
    assert "SELECTION_TIE_AT_BOUNDARY" not in {lim.code for lim in untied.limitations}


def test_protocol_boundaries_are_enforced() -> None:
    assert PROTOCOLS[COMPREHENSIVENESS] == frozenset({Relation.NECESSARY_FOR, Relation.DECREASES})
    assert PROTOCOLS[SUFFICIENCY] == frozenset({Relation.SUFFICIENT_FOR})
    assert Relation.NECESSARY_FOR not in F.claims.DECIDABLE[SUFFICIENCY]
    from beyondnn.protocols import check_policy
    from beyondnn.schema import AssessmentPolicy, EvidenceRuleError, PolicyRequirement

    with pytest.raises(EvidenceRuleError):
        check_policy(
            AssessmentPolicy(
                name="bad",
                version=1,
                requirements=(
                    PolicyRequirement(relation=Relation.NECESSARY_FOR, protocols=(SUFFICIENCY,)),
                ),
            )
        )


# ------------------------------------------------------------------ records, persistence


def test_raw_effects_stay_interventional_and_results_are_not_evidence() -> None:
    r = F.run(
        FM.Weighted8().eval(),
        ONES,
        test=comp(controls=F.controls(5, seed=0)),
        selection=declared(0, 1, n=8),
    )
    statuses = {rec.KIND: rec.status for rec in r.trace.records}
    assert statuses["causal_effect"] is EvidenceStatus.INTERVENTIONAL
    assert statuses["evidence_selection"] is None
    assert statuses["claim_test_result"] is None
    assert all(e.status is EvidenceStatus.INTERVENTIONAL for e in r.result.evidence)
    assert {rec.status for rec in r.trace.records} <= {
        None,
        EvidenceStatus.OBSERVED,
        EvidenceStatus.MEASURED,
        EvidenceStatus.INTERVENTIONAL,
    }


def test_schema_rules_of_the_new_records() -> None:
    site = Site(module="", io=SiteIO.INPUT, output_path="args[0]")
    kw: dict[str, Any] = {"site": site, "sample_id": "sha256:x", "provenance_id": "p"}
    with pytest.raises(SchemaError, match="rule"):
        EvidenceSelection(
            source=SelectionSource.DECLARED, rule="abs_desc", order=(0,), n_units=2, k=1, **kw
        )
    with pytest.raises(SchemaError, match="orders every unit"):
        EvidenceSelection(
            source=SelectionSource.RANDOM,
            rule="uniform_random_permutation",
            order=(0,),
            n_units=2,
            seed=1,
            **kw,
        )
    with pytest.raises(SchemaError, match="score of every unit"):
        EvidenceSelection(
            source=SelectionSource.ATTRIBUTION,
            rule="abs_desc",
            order=(0, 1),
            n_units=2,
            source_record="attribution:" + "a" * 32,
            **kw,
        )
    with pytest.raises(SchemaError, match="criterion"):
        ProtocolResult(
            protocol="stability",
            protocol_version=1,
            measurements={"a": 1},  # type: ignore[arg-type]
            outcomes=(AspectOutcome(aspect="a", outcome=CheckOutcome.PASS),),
            samples=("s",),
            provenance_id="p",
        )
