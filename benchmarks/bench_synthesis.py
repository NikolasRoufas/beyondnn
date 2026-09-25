"""Phase-4 evidence-synthesis cost (CPU; no model runs during what is timed).

Usage::

    python benchmarks/bench_synthesis.py [--quick] [--json out.json]

Evidence is computed once, outside the timed region (Phases 2-3 characterised
that cost). For four compositions on the redundant-path model it times, median /
p90 in ms: ``bnn.compose`` (validation, revalidation of claim results, assessment)
and ``render()``, and reports the number of source records and the rendered size.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

import beyondnn as bnn
from beyondnn._testing.causal_models import Redundant
from beyondnn.schema import InterventionOperation, Relation

A = bnn.attribution
iv = bnn.interventions


def timed(fn: Callable[[], Any], warmup: int, iterations: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1e3)
    samples.sort()
    return statistics.median(samples), samples[
        min(len(samples) - 1, round(0.9 * (len(samples) - 1)))
    ]


def evidence() -> dict[str, Any]:
    h = bnn.instrument(Redundant().eval())
    x = torch.tensor([[3.0, 5.0]])
    target = iv.metrics.select([0, 0])
    method = A.integrated_gradients(baseline=A.zero_baseline(), n_steps=16)
    credit = A.make_claim(A.layer("p"), target, x, statement="p receives attribution for y")
    necessary = iv.make_claim(
        iv.zero("p"), target, Relation.NECESSARY_FOR, x, statement="p is necessary for y"
    )
    return {
        "trace": h.trace(x, sites=["p", "q"]),
        "attr": h.attribute(x, target=target, method=method, at=A.layer("p")),
        "effect": h.intervene(x, intervention=iv.zero("p"), metric=target),
        "attr_claims": h.attribute(
            x,
            target=target,
            method=method,
            at=A.layer("p"),
            claims=[(credit, A.threshold_spec(method, at=A.layer("p"), min_abs_attribution=2.0))],
        ),
        "effect_claims": h.intervene(
            x,
            intervention=iv.zero("p"),
            metric=target,
            claims=[
                (necessary, iv.threshold_spec(operation=InterventionOperation.ZERO, min_effect=6.0))
            ],
        ),
    }


def run(warmup: int, iterations: int) -> dict[str, Any]:
    e = evidence()
    policies = [A.ATTRIBUTION_POLICY, iv.INTERVENTION_POLICY]
    cases: dict[str, dict[str, Any]] = {
        "measured only": {},
        "trace + attribution": {"attributions": [e["attr"]]},
        "trace + intervention": {"interventions": [e["effect"]]},
        "trace + attribution + intervention + claims": {
            "attributions": [e["attr_claims"]],
            "interventions": [e["effect_claims"]],
            "policies": policies,
        },
    }
    rows = []
    for name, kw in cases.items():
        response = bnn.compose(e["trace"], **kw)
        text = response.render()
        records = sum(len(t.records) for t in response.bundle.sources)  # type: ignore[union-attr]
        rows.append(
            {
                "composition": name,
                "records": records,
                "compose_ms": timed(
                    lambda kw=kw: bnn.compose(e["trace"], **kw), warmup, iterations
                ),
                "render_ms": timed(response.render, warmup, iterations),
                "render_chars": len(text),
            }
        )
    env = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "platform": f"{platform.system()} {platform.machine()}",
    }
    return {"environment": env, "warmup": warmup, "iterations": iterations, "results": rows}


def markdown(report: dict[str, Any]) -> str:
    env = report["environment"]
    lines = [
        f"Environment: Python {env['python']}, torch {env['torch']}, {env['platform']}; "
        f"warm-up {report['warmup']}, {report['iterations']} iterations; median / p90 ms. "
        "Evidence computed beforehand (not timed).",
        "",
        "| composition | source records | compose | render | rendered chars |",
        "|---|---|---|---|---|",
    ]

    def f(p: tuple[float, float]) -> str:
        return f"{p[0]:.3f} / {p[1]:.3f}"

    for r in report["results"]:
        lines.append(
            f"| {r['composition']} | {r['records']} | {f(r['compose_ms'])} "
            f"| {f(r['render_ms'])} | {r['render_chars']} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-4 synthesis cost")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    iterations, warmup = (3, 1) if args.quick else (args.iterations, args.warmup)
    report = run(warmup, iterations)
    print(markdown(report))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
