"""Quickstart: trace -> evidence -> claim -> audit -> WHY on a model with two redundant paths.

Run: ``python examples/01_quickstart.py``. Attribution credits ``p``; the audit refuses to treat
that as causal evidence (UNSUPPORTED) and the intervention contradicts "p is necessary"
(zeroing ``p`` leaves ``y`` at 3, because ``q`` computes the same value).
"""

import torch
from torch import nn

import beyondnn as bnn

A, iv, AU = bnn.attribution, bnn.interventions, bnn.audits


class Redundant(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p = nn.Linear(2, 1, bias=False)
        self.q = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.p.weight.copy_(torch.tensor([[1.0, 0.0]]))
            self.q.weight.copy_(torch.tensor([[1.0, 0.0]]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.p(x) + self.q(x)


model, x = Redundant().eval(), torch.tensor([[3.0, 5.0]])
target = iv.metrics.select([0, 0])  # one explicit scalar target; outputs are never summed

# 1. Evidence: an attribution (ATTRIBUTED) and a controlled intervention (INTERVENTIONAL),
#    each with the claim it tests declared before it runs
ig = A.integrated_gradients(baseline=A.zero_baseline(), n_steps=16)
credit = A.make_claim(A.layer("p"), target, x, statement="p receives attribution for y")
attribution = bnn.attribute(
    model,
    x,
    target=target,
    at=A.layer("p"),
    method=ig,
    claims=[(credit, A.threshold_spec(ig, at=A.layer("p"), min_abs_attribution=2.0))],
)
claim = iv.make_claim(
    iv.zero("p"), target, bnn.Relation.NECESSARY_FOR, x, statement="p is necessary for y"
)
effect = bnn.intervene(
    model,
    x,
    intervention=iv.zero("p"),
    metric=target,
    claims=[
        (claim, iv.threshold_spec(operation=bnn.schema.InterventionOperation.ZERO, min_effect=6.0))
    ],
)
print(attribution.record.status, effect.effect.status, effect.value)

# 2. A plan, declared before the audit
plan = AU.plan(
    name="quickstart",
    checkpoint=AU.checkpoint_of(model),
    declared_model=None,
    samples=[AU.sample_id(x)],
    datasets=[],
    concepts=[],
    naive_auroc=None,
    claims=[
        AU.claim(
            "p_necessary",
            statement="p is necessary for y",
            relation="necessary_for",
            target=target,
            scope="instance",
            requirement="intervention",
            subject=bnn.schema.Subject(site=bnn.schema.Site(module="p")),
        )
    ],
    requirements=[AU.requirement("intervention", policy=iv.INTERVENTION_POLICY, controls=False)],
    counterexamples=AU.counterexample_rule(
        max_counterexample_fraction=None, max_false_positive_rate=None, max_false_negative_rate=None
    ),
)

# 3. Audit: attribution alone cannot support a causal claim; the intervention contradicts it
for evidence in ([attribution], [attribution, effect]):
    audited = bnn.audit(evidence, plan=plan).claim("p_necessary")
    print(dict(audited.distribution), sorted(f.code for f in audited.findings))

# 4. WHY: the recorded evidence, kept separate, with the audit attached
report = bnn.audit([attribution, effect], plan=plan)
why = bnn.compose(
    bnn.trace(model, x, sites=["p", "q"]),
    attributions=[attribution],
    interventions=[effect],
    audit=report,
)
print(why.render())
