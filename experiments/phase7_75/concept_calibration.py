"""Phase 7.75 concept-criterion calibration on constructed models (docs/PHASE_7_75_PLAN.md §4).

No Tracr / held-out model is used. ``CalModel(w, s)``: x ~ N(0, I_32), hidden = x (32 neurons),
output y = s * (w * h0 + h2). Truth by construction: the concept "x0 > 0" (neuron 0) is used iff
w > 0; "x1 > 0" (neuron 1) is decodable but never read; K0 = "x0 > 0" with permuted labels.
``s`` rescales the output's units without changing what is used.

Two use criteria are compared on every (w, s):
* ABS (Phase 7.5): min_change = 0.1 in target units;
* REL (Phase 7.75 policy, ADR-055): min_change = C_REL * SD of the clean target over the concept
  dataset's train split, computed before the use test (C_REL = 0.2, Cohen's "small" effect).

Usage (experiment env, from experiments/phase5_5): python ../phase7_75/concept_calibration.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

import beyondnn as bnn

C, iv = bnn.concepts, bnn.interventions
HERE = Path(__file__).resolve().parent
C_REL = 0.2
ABS = 0.1
WS = (0.0, 0.1, 0.25, 0.5, 1.0, 4.0)
SS = (0.01, 0.1, 1.0, 10.0, 100.0)
N = 600
SPLITS = ["train"] * 300 + ["val"] * 100 + ["test"] * 200
ENC = C.encoding_criteria(min_fraction_below=0.95)


class _Hidden(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clone()


class CalModel(nn.Module):
    def __init__(self, w: float, s: float) -> None:
        super().__init__()
        self.hidden = _Hidden()
        self.w, self.s = w, s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hidden(x)
        return (self.s * (self.w * h[:, 0] + h[:, 2])).unsqueeze(1)


def rel_min_change(model: nn.Module, data_x: torch.Tensor) -> float:
    """C_REL x SD of the clean target over the train split (declared before the use test)."""
    with torch.no_grad():
        y = model(data_x[: SPLITS.count("train")])[:, 0]
    return C_REL * float(y.std())


def run(w: float, s: float, x: torch.Tensor) -> dict[str, Any]:
    model = CalModel(w, s).eval()
    target = iv.metrics.select([0, 0])
    lab_used = (x[:, 0] > 0).long().tolist()
    lab_unused = (x[:, 1] > 0).long().tolist()
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(4242)).tolist()
    lab_perm = [lab_used[i] for i in perm]
    rows = list(x.split(1))
    out: dict[str, Any] = {"w": w, "s": s, "rel_min_change": rel_min_change(model, x)}
    for cname, labels, neuron, truth in (
        ("used", lab_used, 0, "used" if w > 0 else "not_used"),
        ("decodable_unused", lab_unused, 1, "not_used"),
        ("negative", lab_perm, 0, "not_used"),
    ):
        data = C.dataset(rows, labels, SPLITS, name=f"cal_{cname}", label_source="constructed")
        concept = C.propose(
            C.neuron("hidden", neuron), label=cname, definition=cname, label_source="dataset"
        )
        enc = C.encoding_test(
            model,
            concept,
            data,
            controls=[C.random_neurons(50, seed=1), C.label_permutation(200, seed=2)],
            criteria=ENC,
        )
        res: dict[str, Any] = {"truth": truth, "encoding": enc.outcome.value}
        for rule, mc in (("abs", ABS), ("rel", out["rel_min_change"])):
            use = C.use_test(
                model,
                concept,
                data,
                target=target,
                relation="decreases",
                intervention=C.remove(C.zero()),
                controls=[C.random_neurons(50, seed=3)],
                criteria=C.use_criteria(min_change=mc, min_fraction_beyond_controls=0.95),
            )
            val = C.validate(concept, encoding=enc, use=[use])
            res[rule] = {
                "use": use.outcome.value,
                "effect": use.mean_effect,
                "min_change": mc,
                "validation": val.semantic_status.value,
            }
        out[cname] = res
    return out


def main() -> None:
    torch.set_num_threads(1)
    x = torch.randn(N, 32, generator=torch.Generator().manual_seed(7750))
    grid = [run(w, s, x) for w in WS for s in SS]
    summary: dict[str, Any] = {}
    for rule in ("abs", "rel"):
        cells = {}
        for g in grid:
            for cname in ("used", "decodable_unused", "negative"):
                r = g[cname]
                cells[f"w={g['w']}|s={g['s']}|{cname}"] = (r["truth"], r[rule]["validation"])
        summary[rule] = cells
    scale_invariant = {
        rule: all(
            len({g[c][rule]["validation"] for g in grid if g["w"] == w}) == 1
            for w in WS
            for c in ("used", "decodable_unused", "negative")
        )
        for rule in ("abs", "rel")
    }
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "concept_calibration.json").write_text(
        json.dumps(
            {"c_rel": C_REL, "abs": ABS, "grid": grid, "scale_invariant": scale_invariant}, indent=1
        )
    )
    print("scale invariant:", scale_invariant)
    for g in grid:
        print(
            f"w={g['w']:<5} s={g['s']:<6}",
            " ".join(
                f"{c[:4]}:{g[c]['abs']['validation'][:4]}/{g[c]['rel']['validation'][:4]}"
                for c in ("used", "decodable_unused", "negative")
            ),
        )


if __name__ == "__main__":
    main()
