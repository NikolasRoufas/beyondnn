"""Phase-5 faithfulness cost characterisation (CPU, no downloads).

Usage::

    python benchmarks/bench_faithfulness.py [--quick] [--json out.json]

Scaling of wall time (median / p90 ms) and trace size (records) with the number of
evidence units, curve points, random controls, and dataset samples, on a fixed-weight
linear model ``y = w . x`` with ``d`` input units (input-level, zero replacement), and
one internal-site case. Every perturbation is one INTERVENTION pass in one recording
(``compare_family``), so cost should be linear in perturbations.
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
from torch import nn

import beyondnn as bnn

F = bnn.faithfulness
A = bnn.attribution
iv = bnn.interventions
SEL = iv.metrics.select([0, 0])


def timed(fn: Callable[[], Any], warmup: int, iterations: int) -> tuple[tuple[float, float], Any]:
    for _ in range(warmup):
        fn()
    samples, last = [], None
    for _ in range(iterations):
        start = time.perf_counter()
        last = fn()
        samples.append((time.perf_counter() - start) * 1e3)
    samples.sort()
    p90 = samples[min(len(samples) - 1, round(0.9 * (len(samples) - 1)))]
    return (statistics.median(samples), p90), last


def linear(d: int) -> nn.Module:
    layer = nn.Linear(d, 1, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.arange(d, 0, -1, dtype=torch.float32).reshape(1, d))
    return layer.eval()


class Hidden(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            self.hidden.weight.copy_(torch.eye(d))
        self.readout = linear(d)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.readout(self.hidden(t))
        return out


def comp(controls: int) -> F.TestTemplate:
    return F.comprehensiveness(
        target=SEL,
        min_drop=1.0,
        statement="selected units necessary",
        controls=F.controls(controls, seed=0) if controls else None,
    )


def run(warmup: int, iterations: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    def row(case: str, param: str, value: int, fn: Callable[[], Any]) -> None:
        stat, result = timed(fn, warmup, iterations)
        rows.append(
            {
                "case": case,
                "param": param,
                "value": value,
                "ms": stat,
                "records": len(result.trace.records),
                "passes": result.trace.passes,
            }
        )

    for d in (8, 32, 128):
        model, x = linear(d), torch.ones(1, d)
        row(
            "comprehensiveness, 50 controls",
            "units d",
            d,
            lambda m=model, x=x, d=d: F.run(
                m, x, test=comp(50), selection=F.units(A.input(), (0,), n_units=d)
            ),
        )
    model, x = linear(16), torch.ones(1, 16)
    for n in (0, 25, 100, 400):
        row(
            "comprehensiveness, d=16",
            "controls N",
            n,
            lambda n=n: F.run(
                model, x, test=comp(n), selection=F.units(A.input(), (0, 1), n_units=16)
            ),
        )
    attr = A.attribute(model, x, target=SEL, method=A.gradient())
    ranking = F.ranking(attr)
    for pts in (5, 9, 17):
        points = [round(i * 16 / (pts - 1)) for i in range(pts)]
        row(
            "removal curve, 10 control rankings, d=16",
            "curve points",
            pts,
            lambda p=points: F.curve(
                model,
                x,
                ranking=ranking,
                target=SEL,
                mode="remove",
                points=p,
                controls=F.controls(10, seed=0),
            ),
        )
    for n_samples in (1, 4, 16):
        samples = [torch.full((1, 16), float(i + 1)) for i in range(n_samples)]
        row(
            "dataset comprehensiveness, 20 controls, d=16",
            "samples",
            n_samples,
            lambda s=samples: F.run_dataset(
                model,
                s,
                test=comp(20),
                rule=F.fixed(F.units(A.input(), (0,), n_units=16)),
                permutation_draws=500,
            ),
        )
    hidden, hx = Hidden(16).eval(), torch.ones(1, 16)
    row(
        "internal comprehensiveness, 50 controls, d=16",
        "units d",
        16,
        lambda: F.run(hidden, hx, test=comp(50), selection=F.units("hidden", (0,), n_units=16)),
    )
    env = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "platform": f"{platform.system()} {platform.machine()}",
        "torch_threads": torch.get_num_threads(),
    }
    return {"environment": env, "warmup": warmup, "iterations": iterations, "results": rows}


def markdown(report: dict[str, Any]) -> str:
    env = report["environment"]
    lines = [
        f"Environment: Python {env['python']}, torch {env['torch']}, {env['platform']}, "
        f"{env['torch_threads']} threads; warm-up {report['warmup']}, "
        f"{report['iterations']} iterations; median / p90 ms.",
        "",
        "| case | varied | value | passes | records | time |",
        "|---|---|---|---|---|---|",
    ]
    for r in report["results"]:
        lines.append(
            f"| {r['case']} | {r['param']} | {r['value']} | {r['passes']} | "
            f"{r['records']} | {r['ms'][0]:.1f} / {r['ms'][1]:.1f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-5 faithfulness cost")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    iterations, warmup = (1, 0) if args.quick else (args.iterations, args.warmup)
    report = run(warmup, iterations)
    print(markdown(report))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
