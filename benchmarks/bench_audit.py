"""Phase-7 audit cost (CPU; no model runs during what is timed).

Usage::

    python benchmarks/bench_audit.py [--quick] [--json out.json]

The evidence is computed once, outside the timed region: comprehensiveness tests with 20
matched controls under 3 replacements on N samples (N = 1, 4, 16) of a 32-unit model,
plus one concept body (two encoding tests). For each body it times, median / p90 in ms:
``bnn.audit`` (integrity, re-derivation, classification), ``to_json`` and
``verify_report`` (a full re-audit), and reports records, results and the report size.
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
from beyondnn._testing.audit_scenarios import (
    WeightedSum,
    _comp,
    _comp_req,
    _plan,
    _unit_claim,
    build,
)
from beyondnn.core.samples import sample_id

F, AU = bnn.faithfulness, bnn.audits


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


def body(n: int) -> tuple[list[Any], Any]:
    model = WeightedSum([5.0] + [0.01] * 31).eval()
    reps = [
        F.zero(),
        F.replacement(torch.full((1, 32), 0.1)),
        F.replacement(torch.full((1, 32), -0.5)),
    ]
    evidence, samples = [], []
    for i in range(n):
        x = torch.ones(1, 32)
        x[0, 0] = 2.0 - 0.1 * i
        x[0, 1] = float(i)
        samples.append(sample_id(x))
        evidence += [
            _comp(model, x, r, controls=F.controls(20, seed=3), min_fraction_below=0.9)
            for r in reps
        ]
    claim = _unit_claim(
        "c",
        "necessary_for",
        "comprehensiveness",
        invariant_over=[AU.invariance("replacement", min_values=3)],
    )
    return evidence, _plan(model, samples, [claim], [_comp_req(True)])


def run(warmup: int, iterations: int) -> dict[str, Any]:
    bodies = {f"comprehensiveness x3 replacements, {n} sample(s)": body(n) for n in (1, 4, 16)}
    concept = build("L")
    bodies["concept encoding x2 nulls (scenario L)"] = (concept.evidence, concept.plan)
    rows = []
    for name, (evidence, plan) in bodies.items():
        report = bnn.audit(evidence, plan=plan)
        text = report.to_json()
        rows.append(
            {
                "body": name,
                "records": report.evidence.records,
                "results": report.evidence.results,
                "audit_ms": timed(
                    lambda e=evidence, p=plan: bnn.audit(e, plan=p), warmup, iterations
                ),
                "to_json_ms": timed(report.to_json, warmup, iterations),
                "verify_report_ms": timed(
                    lambda r=report, e=evidence, p=plan: AU.verify_report(r, e, p),
                    warmup,
                    iterations,
                ),
                "report_bytes": len(text.encode()),
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
        "| evidence body | records | results | audit | to_json | verify_report | report bytes |",
        "|---|---|---|---|---|---|---|",
    ]

    def f(p: tuple[float, float]) -> str:
        return f"{p[0]:.1f} / {p[1]:.1f}"

    for r in report["results"]:
        lines.append(
            f"| {r['body']} | {r['records']} | {r['results']} | {f(r['audit_ms'])} "
            f"| {f(r['to_json_ms'])} | {f(r['verify_report_ms'])} | {r['report_bytes']} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-7 audit cost")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    iterations, warmup = (2, 0) if args.quick else (args.iterations, args.warmup)
    report = run(warmup, iterations)
    print(markdown(report))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
