"""One declared claim, audited over a body of evidence.

Run: ``python examples/06_audit.py``. On ``y = tanh(4 x0) + 0.2 x1`` at x = (3, 1):

* the gradient ranks x1 first (tanh is saturated at x0 = 3), integrated gradients x0;
* each ranking's top-1 unit is tested for necessity (comprehensiveness) under two
  declared replacements: zero, and a reference point (2.5, 0.5);
* the plan asserts that "the top-1 unit is necessary" holds across replacements.

The audit separates the two methods without a score: the gradient claim is
CONTRADICTED; the integrated-gradients claim is ASSUMPTION_SENSITIVE, because removing x0
matters against zero but not against 2.5 (tanh(10) ~ tanh(12)). An attribution-only
audit of the same plan is UNSUPPORTED. The last part shows the AUDIT section of the WHY.
"""

from __future__ import annotations

import torch
from torch import nn

import beyondnn as bnn

A, F, AU, iv = bnn.attribution, bnn.faithfulness, bnn.audits, bnn.interventions


class Saturated(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.tanh(4 * x[:, 0]) + 0.2 * x[:, 1]).unsqueeze(1)


def build() -> tuple[bnn.audits.AuditReport, bnn.audits.AuditReport, str]:
    model, x = Saturated().eval(), torch.tensor([[3.0, 1.0]])
    target = iv.metrics.select([0, 0])
    methods = {
        "gradient": A.gradient(),
        "integrated_gradients": A.integrated_gradients(baseline=A.zero_baseline(), n_steps=64),
    }
    attributions = {n: A.attribute(model, x, target=target, method=m) for n, m in methods.items()}
    replacements = [F.zero(), F.replacement(torch.tensor([[2.5, 0.5]]))]
    tests = []
    for attribution in attributions.values():
        for replacement in replacements:
            test = F.comprehensiveness(
                target=target,
                min_drop=0.5,
                replacement=replacement,
                statement="the top-ranked input unit is necessary for y",
            )
            tests.append(
                F.run(
                    model,
                    x,
                    test=test,
                    selection=F.top_k(attribution, k=1),
                    attributions=[attribution],
                )
            )
    claims = [
        AU.claim(
            f"{name}_top1_necessary",
            statement=f"the {name} top-1 input unit is necessary for y",
            relation="necessary_for",
            target=target,
            scope="instance",
            requirement="necessity",
            selection=AU.selection("input", method=name, k=1),
            invariant_over=[AU.invariance("replacement", min_values=2)],
        )
        for name in methods
    ]
    plan = AU.plan(
        name="saturated_top1",
        checkpoint=AU.checkpoint_of(model),
        declared_model=None,
        samples=[AU.sample_id(x)],
        datasets=[],
        claims=claims,
        requirements=[
            AU.requirement("necessity", policy=F.COMPREHENSIVENESS_POLICY, controls=False)
        ],
        concepts=[],
        counterexamples=AU.counterexample_rule(
            max_counterexample_fraction=None,
            max_false_positive_rate=None,
            max_false_negative_rate=None,
        ),
        naive_auroc=None,
    )
    report = bnn.audit(tests, plan=plan)
    attribution_only = bnn.audit(list(attributions.values()), plan=plan)
    why = bnn.compose(bnn.trace(model, x), audit=report).render()
    return report, attribution_only, why


if __name__ == "__main__":
    report, attribution_only, why = build()
    print(report.render())
    print()
    for claim in attribution_only.claims:
        print("attribution only:", claim.name, dict(claim.distribution))
    print()
    print(why)
