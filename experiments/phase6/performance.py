"""Phase 6 performance (plan §24; request §46): one torch thread, idle machine.

Measures, per model: feature-activation extraction (one recording over the concept
dataset), the encoding test with 200 + 200 controls, random-direction control scaling
(n = 50/200/800), the use test with 50 controls, validation, and composition with full
re-derivation.

Usage (experiment env): python experiments/phase6/performance.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase5_5"))
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import realistic as R  # noqa: E402

import beyondnn as bnn  # noqa: E402
from beyondnn._testing.concept_models import ConceptToy, concept_inputs  # noqa: E402
from beyondnn.concepts.features import record_site  # noqa: E402

C, iv = bnn.concepts, bnn.interventions


def clock(fn: Any) -> tuple[float, Any]:
    t = time.perf_counter()
    out = fn()
    return time.perf_counter() - t, out


def measure(
    model: Any,
    data: Any,
    site: str,
    axis: int,
    pooling: str,
    target: Any,
    x_ref: Any,
    kw: dict[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_samples": len(data.record.samples),
        "n_test": len(data.record.indices("test")),
    }
    feature = C.fit_direction(model, data, site=site, axis=axis, pooling=pooling)
    concept = C.propose(feature, label="k", definition="k")
    all_idx = list(range(len(data.record.samples)))
    out["activation_extraction_s"], _ = clock(
        lambda: record_site(model, data, all_idx, feature.record.site)
    )
    crit = C.encoding_criteria(min_fraction_below=0.95)
    out["encoding_200_200_s"], enc = clock(
        lambda: C.encoding_test(
            model, concept, data, controls=R.enc_controls("covariance"), criteria=crit
        )
    )
    for n in (50, 200, 800):
        out[f"encoding_random_directions_{n}_s"], _ = clock(
            lambda n=n: C.encoding_test(
                model,
                concept,
                data,
                controls=[C.random_directions(n, seed=1, distribution="covariance")],
                criteria=crit,
            )
        )
    out["use_50_s"], use = clock(
        lambda: C.use_test(
            model,
            concept,
            data,
            target=target,
            relation="decreases",
            intervention=C.remove(C.zero()),
            controls=[C.random_directions(50, seed=13, distribution="covariance")],
            criteria=C.use_criteria(min_change=0.1, min_fraction_beyond_controls=0.95),
        )
    )
    out["use_evaluated_samples"] = use.result.statistics["n"]
    out["use_trace_records"] = len(use.trace.records)
    out["validate_s"], val = clock(lambda: C.validate(concept, encoding=enc, use=[use]))
    ref = bnn.trace(model, *x_ref, model_kwargs=kw, sites=[site], retention="cpu")
    out["compose_with_rederivation_s"], _ = clock(lambda: bnn.compose(ref, concepts=[val]))
    return out


def main() -> None:
    torch.set_num_threads(1)
    payload: dict[str, Any] = {
        "environment": common.environment(),
        "note": "single run each; one torch thread; idle machine",
    }
    toy = ConceptToy().eval()
    x = concept_inputs(600, seed=0)
    splits = ["train"] * 300 + ["val"] * 100 + ["test"] * 200
    data = C.dataset(
        [x[i : i + 1] for i in range(600)],
        (x[:, 0] > 0).long().tolist(),
        splits,
        name="toy",
        label_source="x0>0",
    )
    payload["toy"] = measure(
        toy, data, "hidden", 1, "none", iv.metrics.select([0, 0]), (x[:1],), {}
    )
    print("toy", payload["toy"], flush=True)
    for kind, build in (("A", R.model_a), ("C", R.model_c)):
        model, runs, _ = build()
        ds, _, _, site, axis, pooling, target, _ = runs[0]
        payload[kind] = measure(model, ds, site, axis, pooling, target, ds.inputs[0], ds.kwargs[0])
        print(kind, payload[kind], flush=True)
    (R.RESULTS / "performance.json").write_text(json.dumps(payload, indent=1, default=str))


if __name__ == "__main__":
    main()
