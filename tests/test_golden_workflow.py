"""The permanent golden scientific workflow (pre-Phase-8 hardening).

If this test breaks, BeyondNN's scientific workflow changed. Public API only (no ``_testing``,
no private imports). It pins scientific invariants -- standings, finding codes, eligibility,
replacement identity, provenance, save / reload equivalence, WHY sections -- not formatting.

Model (fixed weights, deterministic): x in R^24, hidden = x (24 units), logits = [0, y] with
y = 3*h0 + 1*h1 (h2..h23 are never read). IG's top-2 is always {h0, h1}; replacing them by
their train mean removes the margin (PRIMARY supports). A deliberately extreme STRESS_TEST
replacement (every unit = 4.0) can push the margin up instead (a stress-test reversal).
The eligible-units claim excludes h0, so its top-2 is {h1, one never-read unit}.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

import beyondnn as bnn

A, F, C, AU, iv = bnn.attribution, bnn.faithfulness, bnn.concepts, bnn.audits, bnn.interventions
D = 24
ELIGIBLE = tuple(range(1, D))  # declared: unit 0 is not part of the eligible-units claim


class Golden(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(D, D, bias=False)
        self.out = nn.Linear(D, 2, bias=False)
        with torch.no_grad():
            self.hidden.weight.copy_(torch.eye(D))
            self.out.weight.zero_()
            self.out.weight[1, 0] = 3.0
            self.out.weight[1, 1] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.out(self.hidden(x))
        return out


def _data() -> torch.Tensor:
    return torch.randn(240, D, generator=torch.Generator().manual_seed(2026))


def build(outdir: Path) -> dict[str, Any]:
    """Run the full workflow, save everything, and return the in-process conclusions."""
    torch.manual_seed(0)
    model = Golden().eval()
    x = _data()
    samples = [x[i : i + 1] for i in range(200, 206)]
    trace = bnn.trace(model, samples[0], sites=["hidden"], retention="cpu")
    ig = A.integrated_gradients(baseline=A.zero_baseline(), n_steps=16)
    mean = F.replacement(x[:200].mean(0, keepdim=True), name="train_mean")
    extreme = F.replacement(torch.full((1, D), 4.0), name="extreme")
    faith, attrs, targets = [], [], {}
    for i, s in enumerate(samples):
        with torch.no_grad():
            pred = int(model(s).argmax())
            metric = iv.metrics.margin([0, pred])
            margin = metric(model(s))
        attr = bnn.attribute(model, s, target=metric, method=ig)
        attrs.append(attr)
        targets[AU.sample_id(s)] = metric
        for rep in (mean, extreme):
            test = F.comprehensiveness(
                target=metric,
                min_drop=0.5 * margin,
                replacement=rep,
                controls=F.controls(20, seed=i),
                min_fraction_below=0.8,
                statement="the IG top-2 inputs are necessary",
            )
            for eligible, name in ((None, None), (ELIGIBLE, "units_1_to_23")):
                sel = F.top_k(attr, k=2, eligible=eligible, eligibility=name)
                faith.append(F.run(model, s, test=test, selection=sel, attributions=[attr]))
    effect = bnn.intervene(
        model,
        samples[0],
        intervention=iv.zero("hidden"),
        metric=targets[AU.sample_id(samples[0])],
        claims=[
            (
                iv.make_claim(
                    iv.zero("hidden"),
                    targets[AU.sample_id(samples[0])],
                    bnn.Relation.NECESSARY_FOR,
                    samples[0],
                    statement="the hidden layer is necessary",
                ),
                iv.threshold_spec(operation=bnn.schema.InterventionOperation.ZERO, min_effect=0.1),
            )
        ],
    )
    splits = ["train"] * 120 + ["val"] * 40 + ["test"] * 40
    data = C.dataset(
        [x[i : i + 1] for i in range(200)],
        (x[:200, 0] > 0).long().tolist(),
        splits,
        name="x0 positive",
        label_source="x0 > 0",
    )
    concept = C.propose(C.neuron("hidden", 0), label="x0 is positive", definition="x0 > 0")
    enc = C.encoding_test(
        model,
        concept,
        data,
        controls=[C.random_neurons(20, seed=1), C.label_permutation(100, seed=2)],
        criteria=C.encoding_criteria(min_fraction_below=0.9),
    )
    with torch.no_grad():
        train_y = model(x[:120])[:, 1]
    use = C.use_test(
        model,
        concept,
        data,
        target=iv.metrics.select([0, 1]),
        relation="decreases",
        intervention=C.remove(C.zero()),
        controls=[C.random_neurons(20, seed=3)],
        criteria=C.use_criteria(  # ADR-055: in the target's own scale
            min_change=0.2 * float(train_y.std()), min_fraction_beyond_controls=0.9
        ),
    )
    validation = C.validate(concept, encoding=enc, use=[use])
    roles = [
        AU.role("replacement", "tensor/train_mean:*", "primary"),
        AU.role("replacement", "tensor/extreme:*", "stress_test"),
    ]

    def claim(name: str, eligibility: str | None) -> Any:
        return AU.claim(
            name,
            statement=f"the IG top-2 ({eligibility or 'all units'}) are necessary",
            relation="necessary_for",
            target=None,
            sample_targets=targets,
            scope="instance",
            requirement="necessity",
            selection=AU.selection(
                "input", method="integrated_gradients", k=2, eligibility=eligibility
            ),
            roles=roles,
        )

    plan = AU.plan(
        name="golden",
        checkpoint=AU.checkpoint_of(model),
        declared_model=None,
        samples=list(targets),
        datasets=[data.id],
        claims=[claim("top2_all", None), claim("top2_eligible", "units_1_to_23")],
        requirements=[
            AU.requirement("necessity", policy=F.COMPREHENSIVENESS_POLICY, controls=True)
        ],
        concepts=[AU.concept(concept.id, policy=C.POLICY_V1)],
        counterexamples=AU.counterexample_rule(
            max_counterexample_fraction=None,
            max_false_positive_rate=0.3,
            max_false_negative_rate=0.3,
        ),
        naive_auroc=None,
    )
    evidence = [*faith, *attrs, validation, effect]
    report = bnn.audit(evidence, plan=plan)
    why = bnn.compose(
        trace,
        attributions=[attrs[0]],
        interventions=[effect],
        concepts=[validation],
        audit=report,
        policies=[iv.INTERVENTION_POLICY],
    )
    outdir.mkdir(parents=True, exist_ok=True)
    AU.save_evidence(evidence, outdir / "evidence")
    trace.save(outdir / "reference")
    (outdir / "plan.json").write_text(bnn.schema.to_json(plan))
    report.save(outdir / "report.json")
    torch.save(model.state_dict(), outdir / "model.pt")
    return conclusions(report, why.render(), validation)


def conclusions(report: Any, why_text: str, validation: Any) -> dict[str, Any]:
    """The scientific conclusions a reload must reproduce (not formatting)."""
    out: dict[str, Any] = {"why_audit": "AUDIT" in why_text, "why_concepts": "CONCEPTS" in why_text}
    for name in ("top2_all", "top2_eligible"):
        c = report.claim(name)  # named lookup: report.claims is ordered by name, not plan
        out[name] = {
            "distribution": dict(c.distribution),
            "codes": sorted(
                {f.code for f in c.findings} | {f.code for g in c.groups for f in g.findings}
            ),
            "replacements": sorted(
                {dict(t.axes)["replacement"].split(":")[0] for g in c.groups for t in g.tests}
            ),
            "roles": sorted({t.role for g in c.groups for t in g.tests}),
        }
    out["concept"] = report.concepts[0].standing.value
    out["validation"] = validation.semantic_status.value
    out["checkpoint"] = report.plan.checkpoint
    out["samples"] = sorted(report.plan.samples)
    return out


def reload(outdir: Path) -> dict[str, Any]:
    """A fresh process: load everything from disk, verify, re-audit, re-compose."""
    model = Golden()
    model.load_state_dict(torch.load(outdir / "model.pt", weights_only=True))
    model.eval()
    plan = bnn.schema.from_json((outdir / "plan.json").read_text())
    assert isinstance(plan, bnn.schema.AuditPlan)
    paths = AU.load_evidence(outdir / "evidence")
    report = bnn.audit(paths, plan=plan, model=model)  # the checkpoint must match
    AU.verify_report(AU.load_report(outdir / "report.json"), paths, plan)
    validation = C.load_validation(paths)
    why = bnn.compose(bnn.load_trace(outdir / "reference"), concepts=[validation], audit=report)
    return conclusions(report, why.render(), validation)


def test_golden_scientific_workflow(tmp_path: Path) -> None:
    live = build(tmp_path)
    # standings and findings (the scientific content of the workflow)
    assert live["top2_all"]["distribution"] == {"supported": 6}
    assert live["top2_eligible"]["distribution"] == {"contradicted": 6}  # h0 is not eligible
    for name in ("top2_all", "top2_eligible"):
        codes = set(live[name]["codes"])
        assert "eligibility_mismatch" in codes  # the other claim's evidence is never used
        assert "stress_test_reverses" in codes  # visible, but it does not decide the standing
        assert "attribution_is_not_intervention" not in codes
        assert live[name]["roles"] == ["primary", "stress_test"]
        # replacement identity survives into the audit axis
        assert live[name]["replacements"] == ["tensor/extreme", "tensor/train_mean"]
    assert live["validation"] == "validated_concept"
    assert live["concept"] == "supported"
    assert live["why_audit"]
    assert live["why_concepts"]
    # save -> fresh process -> load -> verify -> audit -> WHY: the same conclusions
    code = (
        "import json, sys, pathlib\n"
        f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "import test_golden_workflow as G\n"
        "print(json.dumps(G.reload(pathlib.Path(sys.argv[1]))))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, check=True
    )
    assert json.loads(out.stdout.strip().splitlines()[-1]) == live


def test_saved_selections_keep_their_eligibility_and_provenance(tmp_path: Path) -> None:
    build(tmp_path)
    plan = bnn.schema.from_json((tmp_path / "plan.json").read_text())
    assert isinstance(plan, bnn.schema.AuditPlan)
    seen: set[Any] = set()
    for path in AU.load_evidence(tmp_path / "evidence"):
        t = bnn.load_trace(path)
        for rec in t.records:
            if isinstance(rec, bnn.schema.EvidenceSelection):
                seen.add(rec.eligibility)
                if rec.eligibility is not None:
                    assert set(rec.order) <= set(ELIGIBLE)
                    assert 0 not in rec.selected
                assert rec.sample_id in plan.samples
                assert t.origin(rec).model.state_digest == plan.checkpoint
    assert seen == {None, "units_1_to_23"}


def test_named_claim_lookup_is_independent_of_plan_order(tmp_path: Path) -> None:
    """``report.claims`` is ordered by claim name (not plan order): look claims up by name."""
    build(tmp_path)
    plan = bnn.schema.from_json((tmp_path / "plan.json").read_text())
    assert isinstance(plan, bnn.schema.AuditPlan)
    reordered = AU.plan(
        name=plan.name,
        checkpoint=plan.checkpoint,
        declared_model=plan.declared_model,
        samples=list(plan.samples),
        datasets=list(plan.datasets),
        claims=list(reversed(plan.claims)),
        requirements=list(plan.requirements),
        concepts=list(plan.concepts),
        counterexamples=plan.counterexamples,
        naive_auroc=plan.naive_auroc,
    )
    paths = AU.load_evidence(tmp_path / "evidence")
    a, b = bnn.audit(paths, plan=plan), bnn.audit(paths, plan=reordered)
    for name in ("top2_all", "top2_eligible"):
        assert a.claim(name).distribution == b.claim(name).distribution
    assert (
        [c.name for c in a.claims] == [c.name for c in b.claims] == sorted(c.name for c in a.claims)
    )


def test_corrupt_saved_evidence_is_refused_and_names_the_trace(tmp_path: Path) -> None:
    import pytest

    from beyondnn.core.persistence import TracePersistenceError

    build(tmp_path)
    paths = AU.load_evidence(tmp_path / "evidence")
    bad = paths[3]
    doc = bad / "trace.json"
    doc.write_text(doc.read_text().replace('"', "'", 1))
    plan = bnn.schema.from_json((tmp_path / "plan.json").read_text())
    assert isinstance(plan, bnn.schema.AuditPlan)
    with pytest.raises(TracePersistenceError, match=str(bad.name)):
        bnn.audit(paths, plan=plan)  # refused: never audited without the corrupt trace
