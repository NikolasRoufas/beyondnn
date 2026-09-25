"""Phase-4 flagship example: attribution and intervention answer different questions.

Run: ``python examples/phase4_redundant_path.py``. On the redundant-path model
``y = p(x) + q(x)`` (with ``p = q = x0``), the attribution to ``p`` is substantial,
while zeroing ``p`` leaves ``y`` at 3. The structured WHY shows both, side by side,
without merging them.
"""

from __future__ import annotations

import torch

import beyondnn as bnn
from beyondnn._testing.causal_models import Redundant
from beyondnn.schema import InterventionOperation, Relation

A, iv = bnn.attribution, bnn.interventions


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
