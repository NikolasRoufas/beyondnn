"""Attribution and intervention answer different questions.

Run: ``python examples/03_intervention.py``. On the redundant-path model
``y = p(x) + q(x)`` (with ``p = q = x0``), the attribution to ``p`` is substantial,
while zeroing ``p`` leaves ``y`` at 3. The structured WHY shows both, side by side,
without merging them.
"""

from __future__ import annotations

import torch
from torch import nn

import beyondnn as bnn
from beyondnn.schema import InterventionOperation, Relation

A, iv = bnn.attribution, bnn.interventions


def _pick_x0() -> nn.Linear:
    layer = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, 0.0]]))
    return layer


class Redundant(nn.Module):
    """y = p(x) + q(x), where p and q both compute x0: two redundant paths."""

    def __init__(self) -> None:
        super().__init__()
        self.p = _pick_x0()
        self.q = _pick_x0()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.p(x) + self.q(x)
        return out


def build() -> bnn.explain.ExplainResponse:
    handle = bnn.instrument(Redundant().eval())
    x = torch.tensor([[3.0, 5.0]])
    target = iv.metrics.select([0, 0])
    trace = handle.trace(x, sites=["p", "q"])
    ig = A.integrated_gradients(baseline=A.zero_baseline(), n_steps=16)
    credit = A.make_claim(A.layer("p"), target, x, statement="p receives attribution for y")
    attr = handle.attribute(
        x,
        target=target,
        method=ig,
        at=A.layer("p"),
        claims=[(credit, A.threshold_spec(ig, at=A.layer("p"), min_abs_attribution=2.0))],
    )
    necessary = iv.make_claim(
        iv.zero("p"), target, Relation.NECESSARY_FOR, x, statement="p is necessary for y"
    )
    effect = handle.intervene(
        x,
        intervention=iv.zero("p"),
        metric=target,
        claims=[
            (necessary, iv.threshold_spec(operation=InterventionOperation.ZERO, min_effect=6.0))
        ],
    )
    return bnn.compose(
        trace,
        attributions=[attr],
        interventions=[effect],
        policies=[A.ATTRIBUTION_POLICY, iv.INTERVENTION_POLICY],
    )


if __name__ == "__main__":
    print(build().render())
