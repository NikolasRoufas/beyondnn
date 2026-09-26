# ruff: noqa: E501
"""Phase 7.5 simulated external-researcher workflow (plan §18). Run in a CLEAN environment with
only the built wheel (+ numpy): public API only, no ``beyondnn._testing``, no private imports.

The model is new to BeyondNN: a 2-layer MLP trained here on a synthetic task (y = x0 + x1 > 0).
Steps: model -> trace -> attribution -> intervention -> faithfulness -> concept evidence -> audit
-> WHY -> save -> (fresh process) load -> audit -> WHY. Every problem an outside user meets is
recorded in ``docs/PHASE_7_5_API_REVIEW.md`` (not here).

Usage: <clean-env python> experiments/phase7_5/researcher_workflow.py run <outdir>
       <clean-env python> experiments/phase7_5/researcher_workflow.py reload <outdir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn

import beyondnn as bnn

A, F, C, AU, iv = bnn.attribution, bnn.faithfulness, bnn.concepts, bnn.audits, bnn.interventions


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(8, 16)
        self.act = nn.ReLU()
        self.out = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.act(self.hidden(x)))


def make_model() -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(600, 8, generator=g)
    y = ((x[:, 0] + x[:, 1]) > 0).long()
    torch.manual_seed(0)
    model = Net()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(300):
        opt.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        opt.step()
    return model.eval(), x, y


def run(outdir: Path) -> None:
    model, x, _ = make_model()
    samples = [x[i : i + 1] for i in range(500, 510)]
    # 1. trace (README "What works today")
    trace = bnn.trace(model, samples[0], sites=["act"], retention="cpu")
    # 2. attribution (README "Attribution")
    with torch.no_grad():
        pred = int(model(samples[0]).argmax())
    target = iv.metrics.margin([0, pred])
    attr = bnn.attribute(
        model,
        samples[0],
        target=target,
        method=A.integrated_gradients(baseline=A.zero_baseline(), n_steps=32),
    )
    # 3. intervention with a declared claim (README "Controlled interventions")
    claim = iv.make_claim(
        iv.zero("act"),
        target,
        bnn.Relation.NECESSARY_FOR,
        samples[0],
        statement="the hidden layer is necessary for the prediction",
    )
    effect = bnn.intervene(
        model,
        samples[0],
        intervention=iv.zero("act"),
        metric=target,
        claims=[
            (
                claim,
                iv.threshold_spec(operation=bnn.schema.InterventionOperation.ZERO, min_effect=0.1),
            )
        ],
    )
    # 4. faithfulness on several samples (README "Faithfulness tests")
    faith, attrs, targets = [], [attr], {}
    for i, s in enumerate(samples):
        with torch.no_grad():
            p = int(model(s).argmax())
            m = iv.metrics.margin([0, p])
            margin = m(model(s))
        a = (
            attr
            if i == 0
            else bnn.attribute(
                model,
                s,
                target=m,
                method=A.integrated_gradients(baseline=A.zero_baseline(), n_steps=32),
            )
        )
        if i:
            attrs.append(a)
        targets[AU.sample_id(s)] = m
        for rep in (F.zero(), F.replacement(x[:500].mean(0, keepdim=True), name="train_mean")):
            test = F.comprehensiveness(
                target=m,
                min_drop=0.5 * margin,
                replacement=rep,
                controls=F.controls(20, seed=i),
                min_fraction_below=0.9,
                statement="the IG top-2 inputs are necessary",
            )
            faith.append(F.run(model, s, test=test, selection=F.top_k(a, k=2), attributions=[a]))
    # 5. concept evidence (README "Concepts")
    splits = ["train"] * 300 + ["val"] * 100 + ["test"] * 100
    data = C.dataset(
        [x[i : i + 1] for i in range(500)],
        (x[:500, 0] > 0).long().tolist(),
        splits,
        name="x0 positive",
        label_source="x0 > 0",
    )
    feature = C.fit_direction(model, data, site="act")
    concept = C.propose(feature, label="x0 is positive", definition="x0 > 0")
    enc = C.encoding_test(
        model,
        concept,
        data,
        controls=[
            C.random_directions(100, seed=1, distribution="covariance"),
            C.label_permutation(100, seed=2),
        ],
        criteria=C.encoding_criteria(min_fraction_below=0.95),
    )
    use = C.use_test(
        model,
        concept,
        data,
        target=iv.metrics.select([0, 1]),
        relation="decreases",
        intervention=C.remove(C.zero()),
        controls=[C.random_directions(50, seed=3, distribution="covariance")],
        criteria=C.use_criteria(min_change=0.1, min_fraction_beyond_controls=0.95),
    )
    validation = C.validate(concept, encoding=enc, use=[use])
    # 6. audit (docs/audit/plans.md)
    plan = AU.plan(
        name="external_researcher",
        checkpoint=AU.checkpoint_of(model),
        declared_model=None,
        samples=list(targets),
        datasets=[data.id],
        claims=[
            AU.claim(
                "ig_top2_necessary",
                statement="the IG top-2 inputs are necessary",
                relation="necessary_for",
                target=None,
                sample_targets=targets,
                scope="instance",
                requirement="necessity",
                selection=AU.selection("input", method="integrated_gradients", k=2),
                roles=[
                    AU.role("replacement", "zero", "primary"),
                    AU.role("replacement", "tensor/train_mean:*", "alternative"),
                ],
            )
        ],
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
    # 7. WHY for sample 0
    why = bnn.compose(
        trace,
        attributions=[attr],
        interventions=[effect],
        faithfulness=faith[:2],
        concepts=[validation],
        audit=report,
        policies=[iv.INTERVENTION_POLICY, F.COMPREHENSIVENESS_POLICY],
    )
    text = why.render()
    # 8. save (public helpers added in Phase 7.5 after this workflow first failed; API review F-1/F-2)
    outdir.mkdir(parents=True, exist_ok=True)
    AU.save_evidence(evidence, outdir / "evidence")
    trace.save(outdir / "reference")
    (outdir / "plan.json").write_text(bnn.schema.to_json(plan))
    report.save(outdir / "report.json")
    torch.save(model.state_dict(), outdir / "model.pt")
    (outdir / "run.json").write_text(
        json.dumps(
            {
                "claim": dict(report.claims[0].distribution),
                "concept": report.concepts[0].standing.value,
                "validation": validation.semantic_status.value,
                "why_has_audit": "AUDIT" in text,
                "why_has_concepts": "CONCEPTS" in text,
            }
        )
    )
    print("run ok", json.loads((outdir / "run.json").read_text()))


def reload(outdir: Path) -> None:
    model = Net()
    model.load_state_dict(torch.load(outdir / "model.pt", weights_only=True))
    model.eval()
    plan = bnn.schema.from_json((outdir / "plan.json").read_text())
    paths = AU.load_evidence(outdir / "evidence")
    report = bnn.audit(paths, plan=plan, model=model)
    AU.verify_report(AU.load_report(outdir / "report.json"), paths, plan)
    validation = C.load_validation(paths)
    reference = bnn.load_trace(outdir / "reference")
    why = bnn.compose(reference, concepts=[validation], audit=report)
    text = why.render()
    before = json.loads((outdir / "run.json").read_text())
    after = {
        "claim": dict(report.claims[0].distribution),
        "concept": report.concepts[0].standing.value,
        "validation": validation.semantic_status.value,
        "why_has_audit": "AUDIT" in text,
        "why_has_concepts": "CONCEPTS" in text,
    }
    print("reload ok", after, "same as before:", before == after)


if __name__ == "__main__":
    {"run": run, "reload": reload}[sys.argv[1]](Path(sys.argv[2]))
