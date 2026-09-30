"""Saved evidence is independently re-auditable.

Run: ``python examples/07_save_reload.py``. Records a faithfulness test, audits it, saves the
evidence, the plan and the report, then re-loads everything in a *fresh Python process*,
re-audits (refusing any other checkpoint), verifies the stored report, and renders the WHY.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from torch import nn

import beyondnn as bnn

A, F, iv, AU = bnn.attribution, bnn.faithfulness, bnn.interventions, bnn.audits


def model() -> nn.Module:
    m = nn.Sequential(nn.Linear(4, 2)).eval()
    with torch.no_grad():
        m[0].weight.copy_(torch.tensor([[0.0, 0, 0, 0], [3.0, 1.0, 0, 0]]))
        m[0].bias.zero_()
    return m


def record(out: Path) -> None:
    net, x = model(), torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    target = iv.metrics.margin([0, 1])
    attr = bnn.attribute(net, x, target=target, method=A.gradient())
    test = F.comprehensiveness(
        target=target,
        min_drop=1.0,
        replacement=F.zero(),
        statement="the top-1 input is necessary",
    )
    result = F.run(net, x, test=test, selection=F.top_k(attr, k=1), attributions=[attr])
    plan = AU.plan(
        name="persisted",
        checkpoint=AU.checkpoint_of(net),
        declared_model=None,
        samples=[AU.sample_id(x)],
        datasets=[],
        concepts=[],
        naive_auroc=None,
        claims=[
            AU.claim(
                "top1_necessary",
                statement="the top-1 input is necessary",
                relation="necessary_for",
                target=target,
                scope="instance",
                requirement="necessity",
                selection=AU.selection("input", method="gradient", k=1),
            )
        ],
        requirements=[
            AU.requirement("necessity", policy=F.COMPREHENSIVENESS_POLICY, controls=False)
        ],
        counterexamples=AU.counterexample_rule(
            max_counterexample_fraction=None,
            max_false_positive_rate=None,
            max_false_negative_rate=None,
        ),
    )
    report = bnn.audit([result, attr], plan=plan)
    AU.save_evidence([result, attr], out / "evidence")
    (out / "plan.json").write_text(bnn.schema.to_json(plan))
    report.save(out / "report.json")
    bnn.trace(net, x, sites=["0"]).save(out / "reference")
    print("recorded:", dict(report.claim("top1_necessary").distribution))


def reload(out: Path) -> None:
    plan = bnn.schema.from_json((out / "plan.json").read_text())
    assert isinstance(plan, bnn.schema.AuditPlan)
    paths = AU.load_evidence(out / "evidence")
    report = bnn.audit(paths, plan=plan, model=model())  # the checkpoint must match
    AU.verify_report(AU.load_report(out / "report.json"), paths, plan)  # never corrects
    why = bnn.compose(bnn.load_trace(out / "reference"), audit=report)
    print("reloaded:", dict(report.claim("top1_necessary").distribution))
    print(why.render().split("COVERAGE")[0])


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--reload":
        reload(Path(sys.argv[2]))
    else:
        directory = Path(tempfile.mkdtemp())
        record(directory)
        subprocess.run(
            [sys.executable, __file__, "--reload", str(directory)], check=True
        )  # a fresh Python process
