"""Phase 7.75 adversarial tests (request §31): unit eligibility (ADR-053), control
attainability and competitive-only failure (ADR-054), and their persistence."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import beyondnn as bnn
from beyondnn.faithfulness.spec import FaithfulnessError
from beyondnn.schema import EvidenceSelection, SchemaError, from_json, to_json

A, F, AU, iv = bnn.attribution, bnn.faithfulness, bnn.audits, bnn.interventions
SEL = iv.metrics.select([0, 0])


class Tokens(nn.Module):
    """y = sum over positions of w_p * x_p; position 0 and 5 play [CLS]/[SEP] (large w)."""

    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Identity()
        self.w = torch.tensor([5.0, 1.0, 0.5, 0.2, 0.1, 6.0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.emb(x) * self.w).sum(-1, keepdim=True)


CONTENT = (1, 2, 3, 4)
X = torch.ones(1, 6)


def _attr(model: nn.Module) -> Any:
    return A.attribute(model, X, target=SEL, method=A.gradient(), at=A.layer("emb"))


def test_special_token_eligible_vs_ineligible() -> None:
    model = Tokens().eval()
    attr = _attr(model)
    everything = F.top_k(attr, k=2)
    assert everything.selected == (0, 5)  # the model-control positions dominate
    content = F.top_k(attr, k=2, eligible=CONTENT, eligibility="content_tokens")
    assert content.selected == (1, 2)
    assert content.order == (1, 2, 3, 4)
    assert content.eligibility == "content_tokens"


def test_eligibility_is_declared_with_its_name() -> None:
    attr = _attr(Tokens().eval())
    with pytest.raises(FaithfulnessError, match="together"):
        F.top_k(attr, k=1, eligible=CONTENT)
    with pytest.raises(FaithfulnessError, match="together"):
        F.top_k(attr, k=1, eligibility="content_tokens")
    with pytest.raises(FaithfulnessError, match="in \\[0, 6\\)"):
        F.top_k(attr, k=1, eligible=(1, 9), eligibility="content_tokens")


def test_content_token_claim_with_special_token_selected_is_refused() -> None:
    with pytest.raises(FaithfulnessError, match="not all eligible"):
        F.units(A.layer("emb"), (0, 2), n_units=6, eligible=CONTENT, eligibility="content_tokens")


def test_padding_excluded_from_selections_and_controls() -> None:
    model = Tokens().eval()
    padded = (1, 2)  # positions 3, 4 play padding; 0, 5 special
    sel = F.top_k(_attr(model), k=1, eligible=padded, eligibility="content_tokens")
    test = F.comprehensiveness(
        target=SEL,
        min_drop=1e-9,
        replacement=F.zero(),
        controls=F.controls(10, seed=1),
        min_fraction_below=0.5,
        statement="top content token necessary",
    )
    r = F.run(model, X, test=test, selection=sel, attributions=[_attr(model)])
    control_units = {
        rec.units
        for rec in r.trace.records
        if type(rec).__name__ == "InterventionRecord" and rec.units != sel.selected
    }
    assert control_units
    assert all(set(u) <= set(padded) for u in control_units if u)
    assert r.selection.eligible == padded


def test_curves_refuse_eligibility_restricted_rankings() -> None:
    model = Tokens().eval()
    ranking = F.ranking(_attr(model), eligible=CONTENT, eligibility="content_tokens")
    with pytest.raises(FaithfulnessError, match="eligibility"):
        F.curve(model, X, ranking=ranking, target=SEL, mode="remove", replacement=F.zero())


def test_selection_record_v3_rules_and_round_trip() -> None:
    model = Tokens().eval()
    attr = _attr(model)
    sel = F.top_k(attr, k=1, eligible=CONTENT, eligibility="content_tokens")
    test = F.comprehensiveness(target=SEL, min_drop=1e-9, replacement=F.zero(), statement="s")
    r = F.run(model, X, test=test, selection=sel, attributions=[attr])
    rec = r.selection
    assert from_json(to_json(rec)) == rec
    kw = {f.name: getattr(rec, f.name) for f in dataclasses.fields(rec) if f.init}
    with pytest.raises(SchemaError, match="declared together"):
        EvidenceSelection(**(kw | {"eligibility": None}))
    with pytest.raises(SchemaError, match="only eligible units"):
        EvidenceSelection(**(kw | {"eligible": (2, 3, 4)}))


def _run_both(model: nn.Module) -> tuple[list[Any], list[Any], str]:
    attr = _attr(model)
    sid = AU.sample_id(X)
    results = []
    for eligible, name in ((None, None), (CONTENT, "content_tokens")):
        sel = F.top_k(attr, k=1, eligible=eligible, eligibility=name)
        test = F.comprehensiveness(
            target=SEL, min_drop=0.5, replacement=F.zero(), statement="top-1 necessary"
        )
        results.append(F.run(model, X, test=test, selection=sel, attributions=[attr]))
    return results, [attr], sid


def _plan(model: nn.Module, sid: str, eligibility: str | None) -> Any:
    return AU.plan(
        name="elig",
        checkpoint=AU.checkpoint_of(model),
        declared_model=None,
        samples=[sid],
        datasets=[],
        claims=[
            AU.claim(
                "top1",
                statement="s",
                relation="necessary_for",
                target=SEL,
                scope="instance",
                requirement="n",
                selection=AU.selection("emb", method="gradient", k=1, eligibility=eligibility),
            )
        ],
        requirements=[AU.requirement("n", policy=F.COMPREHENSIVENESS_POLICY, controls=False)],
        concepts=[],
        counterexamples=AU.counterexample_rule(
            max_counterexample_fraction=None,
            max_false_positive_rate=None,
            max_false_negative_rate=None,
        ),
        naive_auroc=None,
    )


def test_audit_never_mixes_eligibilities() -> None:
    model = Tokens().eval()
    results, attrs, sid = _run_both(model)
    for eligibility in (None, "content_tokens"):
        report = bnn.audit([*results, *attrs], plan=_plan(model, sid, eligibility))
        (group,) = report.claims[0].groups
        assert len(group.tests) == 1  # only the evidence with the claim's own eligibility
        codes = {f.code for f in report.claims[0].findings} | {f.code for f in group.findings}
        assert "eligibility_mismatch" in codes
    # content-only evidence alone cannot decide an all-token claim: it is never used for it
    report = bnn.audit([results[1], *attrs], plan=_plan(model, sid, None))
    (group,) = report.claims[0].groups
    assert group.tests == ()
    assert group.standing.value in ("unsupported", "not_evaluated")


def test_eligibility_survives_save_and_reload(tmp_path: Path) -> None:
    model = Tokens().eval()
    results, attrs, sid = _run_both(model)
    plan = _plan(model, sid, "content_tokens")
    paths = AU.save_evidence([*results, *attrs], tmp_path / "ev")
    (tmp_path / "plan.json").write_text(to_json(plan))
    reloaded = from_json((tmp_path / "plan.json").read_text())
    assert reloaded == plan
    live = bnn.audit([*results, *attrs], plan=plan)
    again = bnn.audit(AU.load_evidence(tmp_path / "ev"), plan=reloaded)
    assert again.to_json() == live.to_json()
    assert len(paths) == 3


# ------------------------------------------------------------------ ADR-054


class Heads(nn.Module):
    """Four 'heads' (units) of one site; the output depends on head 0 only."""

    def __init__(self) -> None:
        super().__init__()
        self.site = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.site(x)[:, :1] * 2.0


def _control_plan(model: nn.Module, sid: str) -> Any:
    return AU.plan(
        name="ctl",
        checkpoint=AU.checkpoint_of(model),
        declared_model=None,
        samples=[sid],
        datasets=[],
        claims=[
            AU.claim(
                "h0",
                statement="head 0 necessary",
                relation="necessary_for",
                target=SEL,
                scope="instance",
                requirement="n",
                subject=bnn.schema.Subject(site=bnn.schema.Site(module="site"), units=(0,)),
            )
        ],
        requirements=[AU.requirement("n", policy=F.COMPREHENSIVENESS_POLICY, controls=True)],
        concepts=[],
        counterexamples=AU.counterexample_rule(
            max_counterexample_fraction=None,
            max_false_positive_rate=None,
            max_false_negative_rate=None,
        ),
        naive_auroc=None,
    )


def test_overly_strict_competitive_control_is_inconclusive_not_contradicting() -> None:
    model = Heads().eval()
    x = torch.ones(1, 4)
    sel = F.units("site", (0,), n_units=4)
    test = F.comprehensiveness(
        target=SEL,
        min_drop=1.0,
        replacement=F.zero(),
        controls=F.controls(20, seed=0),
        min_fraction_below=0.95,
        statement="head 0 necessary",
    )
    r = F.run(model, x, test=test, selection=sel)
    assert r.outcome.value == "contradicts"  # the protocol outcome itself is unchanged
    report = bnn.audit([r], plan=_control_plan(model, AU.sample_id(x)))
    (group,) = report.claims[0].groups
    assert group.standing.value == "inconclusive"
    assert "control_criterion_unattainable" in {f.code for f in group.findings}


class EqualHeads(nn.Module):
    """Every head contributes equally: removing any one head has the same effect."""

    def __init__(self) -> None:
        super().__init__()
        self.site = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.site(x).sum(-1, keepdim=True) * 2.0


def test_competitive_failure_is_not_a_failure_of_the_effect() -> None:
    model = EqualHeads().eval()
    x = torch.ones(1, 4)
    sel = F.units("site", (0,), n_units=4)
    # attainable criterion (0.5 <= 1 - identical/20), absolute effect met (drop 2 >= 1),
    # but every random head does as well: contradicted *as declared*, with the effect present
    test = F.comprehensiveness(
        target=SEL,
        min_drop=1.0,
        replacement=F.zero(),
        controls=F.controls(20, seed=0),
        min_fraction_below=0.5,
        statement="head 0 necessary",
    )
    r = F.run(model, x, test=test, selection=sel)
    assert r.outcome.value == "contradicts"
    assert r.drop >= 1.0
    report = bnn.audit([r], plan=_control_plan(model, AU.sample_id(x)))
    (group,) = report.claims[0].groups
    assert group.standing.value == "contradicted"
    assert "effect_without_competitive_advantage" in {f.code for f in group.findings}
    assert "control_criterion_unattainable" not in {f.code for f in group.findings}


def test_replacement_identity_reaches_the_audit_axis() -> None:
    model = Tokens().eval()
    attr = _attr(model)
    rep = F.replacement(torch.zeros(1, 6), name="mask_embedding")
    assert rep.identity()["name"] == "mask_embedding"
    test = F.comprehensiveness(target=SEL, min_drop=0.5, replacement=rep, statement="s")
    r = F.run(model, X, test=test, selection=F.top_k(attr, k=1), attributions=[attr])
    report = bnn.audit([r, attr], plan=_plan(model, AU.sample_id(X), None))
    (group,) = report.claims[0].groups
    (entry,) = group.tests
    assert dict(entry.axes)["replacement"].startswith("tensor/mask_embedding:")


def test_audit_plan_v2_selection_claims_migrate_to_all_units() -> None:
    from beyondnn.schema import from_dict, to_dict
    from beyondnn.schema.codec import _expected_id

    plan = _plan(Tokens().eval(), AU.sample_id(X), "content_tokens")
    env = to_dict(plan)
    data = env["data"]
    for c in data["claims"]:
        del c["selection"]["eligibility"]
    env |= {"record_version": 2, "id": _expected_id("audit_plan", 2, data)}
    migrated = from_dict(env)
    assert migrated.claims[0].selection.eligibility is None  # v2 claims were about every unit
