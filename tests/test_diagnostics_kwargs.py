"""Phase 5.5: method diagnostics and stability accept ``model_kwargs`` like ``run`` and
``curve`` do. Realistic failure: every diagnostic on BERT-tiny (which needs
``attention_mask``) was refused as "not about this input", because the attributions'
sample identity includes the keyword inputs (API review F-11)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import beyondnn as bnn
import beyondnn.attribution as A
import beyondnn.faithfulness as F
import beyondnn.interventions as iv

SEL = iv.metrics.select([0, 0])


class _Masked(nn.Module):
    """out = sum(w * x * mask), w = (1, 2, 3, 4); the mask is a keyword input."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("w", torch.tensor([[1.0, 2.0, 3.0, 4.0]]))

    def forward(self, t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        w = self.w
        assert isinstance(w, torch.Tensor)
        return (t * w * mask).sum(dim=1, keepdim=True)


X = torch.ones(1, 4)
KW = {"mask": torch.tensor([[1.0, 1.0, 0.0, 1.0]])}


def test_method_diagnostics_take_model_kwargs() -> None:
    model = _Masked().eval()
    a = A.attribute(model, X, target=SEL, method=A.gradient(), model_kwargs=KW)
    b = A.attribute(model, X, target=SEL, method=A.input_x_gradient(), model_kwargs=KW)
    with pytest.raises(F.SelectionMismatchError, match="not about this input"):
        F.method_agreement(model, X, a=a, b=b, target=SEL, k=2)
    d = F.method_agreement(model, X, a=a, b=b, target=SEL, k=2, model_kwargs=KW)
    assert d.measurements["ranking_a"] == (3, 1, 0, 2)  # the masked unit scores 0
    assert d.measurements["topk_jaccard"] == 1.0
    bnn.compose(bnn.trace(model, X, model_kwargs=KW), attributions=[a, b], faithfulness=[d])


def test_stability_takes_model_kwargs() -> None:
    model = _Masked().eval()
    flip = F.transformation(
        "reverse_features",
        lambda t: t.flip(1) * 1.5,
        implementation_revision="test-v1",
        unit_map=[3, 2, 1, 0],
    )
    s = F.stability(
        model,
        X,
        transformation=flip,
        method=A.gradient(),
        target=SEL,
        k=1,
        test=F.comprehensiveness(
            replacement=F.zero(), target=SEL, min_drop=1.0, statement="top unit necessary"
        ),
        model_kwargs=KW,
    )
    assert s.measurements["prediction_x"] == 7.0
    assert s.measurements["prediction_gx"] == 1.5 * 7.0  # the mask is not transformed
    assert s.measurements["claim_outcomes"] == ("supports", "supports")
    bnn.compose(bnn.trace(model, X, model_kwargs=KW), faithfulness=[s])
