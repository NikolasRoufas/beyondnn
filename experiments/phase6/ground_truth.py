"""Phase 6 pre-registered ground-truth suite (docs/PHASE_6_PLAN.md §21), run with the
exact pre-registered settings. Writes results/ground_truth.json.

Usage: python experiments/phase6/ground_truth.py
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch

import beyondnn as bnn
from beyondnn._testing.concept_models import ConceptToy, concept_inputs
from beyondnn.concepts.verify import verify_validation

C, iv = bnn.concepts, bnn.interventions
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)
MODEL = ConceptToy().eval()
SPLITS = ["train"] * 300 + ["val"] * 100 + ["test"] * 200
ENC = C.encoding_criteria(min_fraction_below=0.95)
USE = C.use_criteria(min_change=0.25, min_fraction_beyond_controls=0.9)


def enc_controls() -> list[Any]:
    return [
        C.random_directions(200, seed=1, distribution="isotropic"),
        C.label_permutation(200, seed=2),
    ]


def dataset(x: torch.Tensor, labels: list[int], name: str, splits: list[str] = SPLITS) -> Any:
    return C.dataset(list(x.split(1)), labels, splits, name=name, label_source=name)


def unit(*idx: int) -> torch.Tensor:
    v = torch.zeros(12)
    for i in idx:
        v[i] = 1.0
    return v / v.norm()


def run(
    name: str,
    feature: Any,
    data: Any,
    target: int | None,
    label: str,
    *,
    generated: Any = None,
) -> dict[str, Any]:
    concept = (
        C.propose(feature, definition=label, generated=generated)
        if generated is not None
        else C.propose(feature, label=label, definition=label)
    )
    t0 = time.perf_counter()
    enc = C.encoding_test(MODEL, concept, data, controls=enc_controls(), criteria=ENC)
    out: dict[str, Any] = {
        "case": name,
        "feature": feature.record.basis.value,
        "label_source": concept.record.label_source.value,
        "proposal_status": concept.semantic_status.value,
        "encoding_outcome": enc.outcome.value,
        "auroc": enc.auroc,
        "controls": [
            {k: c[k] for k in ("kind", "superiority", "median", "q95")}
            for c in enc.result.statistics["controls"]  # type: ignore[union-attr]
        ],
        "false_positive_rate": enc.counterexamples.measurements["false_positive_rate"],
        "false_negative_rate": enc.counterexamples.measurements["false_negative_rate"],
    }
    if target is not None:
        ctl = (
            [C.random_neurons(50, seed=3)]
            if feature.record.basis.value == "neuron"
            else [C.random_directions(50, seed=3, distribution="isotropic")]
        )
        use = C.use_test(
            MODEL,
            concept,
            data,
            target=iv.metrics.select([0, target]),
            relation="decreases",
            intervention=C.remove(C.zero()),
            controls=ctl,
            criteria=USE,
        )
        val = C.validate(concept, encoding=enc, use=[use])
        verify_validation(val)
        out |= {
            "use_outcome": use.outcome.value,
            "mean_effect": use.mean_effect,
            "use_superiority": use.result.statistics["superiority"],
            "status": val.semantic_status.value,
            "unmet": list(val.unmet),
            "verified": True,
        }
    out["seconds"] = time.perf_counter() - t0
    print(json.dumps(out)[:260], flush=True)
    return out


def main() -> None:
    x = concept_inputs(600, seed=0)
    rows = []
    lab = lambda col: (x[:, col] > 0).long().tolist()  # noqa: E731
    # A: encoded and used
    d = dataset(x, lab(0), "gt-A")
    rows += [
        run("A", C.neuron("hidden", 0), d, 0, "x0 > 0"),
        run("A", C.direction("hidden", unit(0)), d, 0, "x0 > 0"),
    ]
    # B: decodable but unused
    d = dataset(x, lab(1), "gt-B")
    rows += [
        run("B", C.neuron("hidden", 1), d, 0, "x1 > 0"),
        run("B", C.direction("hidden", unit(1)), d, 0, "x1 > 0"),
    ]
    # C: distributed
    d = dataset(x, lab(2), "gt-C")
    rows += [
        run("C", C.neuron("hidden", 2), d, 1, "x2 > 0"),
        run("C", C.direction("hidden", unit(2, 3)), d, 1, "x2 > 0"),
    ]
    # D: redundant
    d = dataset(x, lab(4), "gt-D")
    rows += [
        run("D", C.neuron("hidden", 4), d, 2, "x4 > 0"),
        run("D", C.direction("hidden", unit(4, 5)), d, 2, "x4 > 0"),
    ]
    # E: correlated proxy (E1 correlated, E2 independent)
    x1 = concept_inputs(600, seed=10, proxy_noise=0.3)
    x2 = concept_inputs(600, seed=11)
    rows.append(
        run(
            "E1",
            C.neuron("hidden", 6),
            dataset(x1, (x1[:, 5] > 0).long().tolist(), "gt-E1"),
            3,
            "x5 > 0",
        )
    )
    rows.append(
        run(
            "E2",
            C.neuron("hidden", 6),
            dataset(x2, (x2[:, 5] > 0).long().tolist(), "gt-E2"),
            3,
            "x5 > 0",
        )
    )
    # F: random directions, small test split (n = 16)
    small_splits = ["train"] * 300 + ["val"] * 100 + ["test"] * 16
    xf = concept_inputs(416, seed=20)
    df = dataset(xf, (xf[:, 0] > 0).long().tolist(), "gt-F", small_splits)
    naive = passed = 0
    f_rows = []
    for s in range(50):
        g = torch.Generator().manual_seed(1000 + s)
        v = torch.randn(12, generator=g)
        r = run(f"F{s}", C.direction("hidden", v), df, None, "x0 > 0 (random direction)")
        naive += r["auroc"] >= 0.7
        passed += r["encoding_outcome"] == "supports"
        f_rows.append(r)
    rows.append({"case": "F", "naive_auroc_ge_0.7": naive, "controls_supports": passed, "n": 50})
    # G: permuted labels
    g = torch.Generator().manual_seed(30)
    perm = torch.randperm(600, generator=g).tolist()
    rows.append(
        run(
            "G",
            C.neuron("hidden", 0),
            dataset(x, [lab(0)[p] for p in perm], "gt-G"),
            0,
            "x0 > 0 (permuted)",
        )
    )
    # H: polysemantic
    rows.append(run("H", C.neuron("hidden", 7), dataset(x, lab(7), "gt-H"), 4, "x7 > 0"))
    # I: feature splitting
    d = dataset(x, lab(9), "gt-I")
    rows += [
        run("I", C.neuron("hidden", 8), d, 5, "x9 > 0"),
        run("I", C.direction("hidden", unit(8, 9)), d, 5, "x9 > 0"),
    ]
    # J: generated label (wrong on h1; right on h0)
    wrong = C.neuron("hidden", 1)
    right = C.neuron("hidden", 0)
    d = dataset(x, lab(0), "gt-J")
    gw = C.generated_label(MODEL, wrong, "x0 > 0", generator="toy-labeler", revision="v1")
    gr = C.generated_label(MODEL, right, "x0 > 0", generator="toy-labeler", revision="v1")
    rows.append(
        run("J-wrong", wrong, d, 0, "x0 > 0", generated=gw)
        | {"generated_status": gw.record.status.value}
    )
    rows.append(
        run("J-right", right, d, 0, "x0 > 0", generated=gr)
        | {"generated_status": gr.record.status.value}
    )
    out = {"rows": rows, "f_detail": f_rows, "torch": torch.__version__, "cases": "plan §21"}
    (RESULTS / "ground_truth.json").write_text(
        json.dumps(
            out,
            indent=1,
            default=lambda o: None if isinstance(o, float) and math.isnan(o) else str(o),
        )
    )


if __name__ == "__main__":
    main()
