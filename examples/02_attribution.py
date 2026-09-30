"""Attribution is method-relative ATTRIBUTED evidence, never a cause.

Run: ``python examples/02_attribution.py``. On y = tanh(4 x0) + 0.2 x1 at x = (3, 1), tanh is
saturated: the gradient ranks x1 first, integrated gradients ranks x0 first. Both results are
ATTRIBUTED; which ranking picks a *necessary* unit is a separate, declared test
(see 04_faithfulness.py and 06_audit.py).
"""

from __future__ import annotations

import torch
from torch import nn

import beyondnn as bnn

A, iv = bnn.attribution, bnn.interventions


class Saturated(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.tanh(4 * x[:, 0]) + 0.2 * x[:, 1]).unsqueeze(1)


def main() -> None:
    model, x = Saturated().eval(), torch.tensor([[3.0, 1.0]])
    target = iv.metrics.select([0, 0])  # one explicit scalar; outputs are never summed
    for name, method in (
        ("gradient", A.gradient()),
        ("integrated gradients", A.integrated_gradients(baseline=A.zero_baseline(), n_steps=64)),
    ):
        result = bnn.attribute(model, x, target=target, method=method)
        print(f"{name}: scores={result.value.tolist()} status={result.record.status.value}")
        print(f"  completeness delta: {result.completeness_delta}")  # a numerical diagnostic
        print(f"  limitations: {[lim.code for lim in result.limitations]}")


if __name__ == "__main__":
    main()
