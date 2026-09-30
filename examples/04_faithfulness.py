"""A token-level faithfulness claim with declared unit eligibility, named replacements in
declared roles, and matched random controls.

Run: ``python examples/04_faithfulness.py``. Position 0 plays a model-control token (like
[CLS]); the claim is about *content* tokens, so position 0 is never selected and random
controls are drawn from content tokens only. The PRIMARY replacement decides the standing;
the extreme STRESS_TEST replacement stays visible as a finding.
"""

import torch
from torch import nn

import beyondnn as bnn

A, F, iv, AU = bnn.attribution, bnn.faithfulness, bnn.interventions, bnn.audits


class TokenModel(nn.Module):
    """Logit = 4*e0 + 2*e3 + 1*e5 over 8 positions of a 1-d 'embedding'; position 0 is special."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Identity()
        self.register_buffer("w", torch.tensor([4.0, 0, 0, 2.0, 0, 1.0, 0, 0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([torch.zeros(len(x)), (self.embed(x) * self.w).sum(-1)], dim=1)


model = TokenModel().eval()
samples = [torch.ones(1, 8) + 0.1 * i for i in range(3)]
content = tuple(range(1, 8))  # declared by the caller: every position except the special one
mean = F.replacement(torch.full((1, 8), 0.05), name="train_mean")  # a named replacement
extreme = F.replacement(torch.full((1, 8), 10.0), name="extreme")  # deliberately extreme

results, attributions, targets = [], [], {}
for i, x in enumerate(samples):
    target = iv.metrics.margin([0, 1])  # predicted class minus the best other class
    attr = bnn.attribute(
        model,
        x,
        target=target,
        at=A.layer("embed"),
        method=A.integrated_gradients(baseline=A.zero_baseline(), n_steps=16),
    )
    attributions.append(attr)
    targets[AU.sample_id(x)] = target
    selection = F.top_k(attr, k=2, eligible=content, eligibility="content_tokens")
    for replacement in (mean, extreme):
        test = F.comprehensiveness(
            target=target,
            min_drop=0.3 * float(target(model(x))),
            replacement=replacement,
            controls=F.controls(20, seed=i),
            min_fraction_below=0.9,
            statement="the top-2 content tokens are necessary",
        )
        results.append(F.run(model, x, test=test, selection=selection, attributions=[attr]))

plan = AU.plan(
    name="content_tokens",
    checkpoint=AU.checkpoint_of(model),
    declared_model=None,
    samples=list(targets),
    datasets=[],
    concepts=[],
    naive_auroc=None,
    claims=[
        AU.claim(
            "top2_content_necessary",
            statement="the IG top-2 content tokens are necessary",
            relation="necessary_for",
            target=None,
            sample_targets=targets,
            scope="instance",
            requirement="necessity",
            selection=AU.selection(
                "embed", method="integrated_gradients", k=2, eligibility="content_tokens"
            ),
            roles=[
                AU.role("replacement", "tensor/train_mean:*", "primary"),
                AU.role("replacement", "tensor/extreme:*", "stress_test"),
            ],
        )
    ],
    requirements=[AU.requirement("necessity", policy=F.COMPREHENSIVENESS_POLICY, controls=True)],
    counterexamples=AU.counterexample_rule(
        max_counterexample_fraction=None, max_false_positive_rate=None, max_false_negative_rate=None
    ),
)

report = bnn.audit([*results, *attributions], plan=plan)
claim = report.claim("top2_content_necessary")  # look claims up by name
print(dict(claim.distribution))
print(sorted({f.code for g in claim.groups for f in g.findings}))
print(claim.groups[0].profile.describe())
print(results[0].selection.eligibility, results[0].selection.selected)
