"""Phase-3 attribution overhead characterisation (CPU, no downloads).

Usage::

    python benchmarks/bench_attribution.py [--quick] [--json out.json]

Per model (eval mode, one input with leading dimension 1, one scalar target) it
times, median / p90 in ms after warm-up: a plain forward, ``bnn.trace``, and
``bnn.attribute`` with native gradient, input x gradient, and Integrated
Gradients (zero baseline, ``riemann_middle``) at 16 and 64 steps, plus Captum IG
at the same settings when Captum is installed. Every attribution includes its
guards (two model fingerprints), the traced reference pass, and record building.
TinyTransformer is attributed at the ``token_embedding`` output (its inputs are
token ids); IG there uses an all-zero input-id baseline.
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
from beyondnn._testing.attribution_models import Product
from beyondnn._testing.models import TinyCNN, TinyMLP, TinyTransformer

A = bnn.attribution
iv = bnn.interventions
STEPS = (16, 64)


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


def _captum() -> Any:
    try:
        import beyondnn.attribution.captum as captum_adapter

        captum_adapter.saliency()  # raises if Captum is missing or unverified
    except ImportError:
        return None
    return captum_adapter


def case(
    name: str,
    model: nn.Module,
    x: torch.Tensor,
    target: Any,
    at: Any,
    baseline: Any,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    model.eval()
    site = [at.module] if at.is_layer else []

    def attr(method: Any) -> Callable[[], Any]:
        return lambda: bnn.attribute(model, x, target=target, method=method, at=at)

    with torch.no_grad():
        row: dict[str, Any] = {
            "model": name,
            "attributed": at.module or "input",
            "forward_ms": timed(lambda: model(x), warmup, iterations),
            "trace_ms": timed(lambda: bnn.trace(model, x, sites=site), warmup, iterations),
        }
    row["gradient_ms"] = timed(attr(A.gradient()), warmup, iterations)
    row["input_x_gradient_ms"] = timed(attr(A.input_x_gradient()), warmup, iterations)
    captum_adapter = _captum()
    for n in STEPS:
        row[f"ig{n}_ms"] = timed(
            attr(A.integrated_gradients(baseline=baseline, n_steps=n)), warmup, iterations
        )
        row[f"captum_ig{n}_ms"] = (
            None
            if captum_adapter is None
            else timed(
                attr(captum_adapter.integrated_gradients(baseline=baseline, n_steps=n)),
                warmup,
                iterations,
            )
        )
    return row


def run(warmup: int, iterations: int) -> dict[str, Any]:
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 32, (1, 8), generator=g)
    rows = [
        case(
            "Product (analytic)",
            Product(),
            torch.tensor([[3.0, 5.0]]),
            iv.metrics.select([0, 0]),
            A.input(),
            A.zero_baseline(),
            warmup,
            iterations,
        ),
        case(
            "TinyMLP",
            TinyMLP(),
            torch.randn(1, 4, generator=g),
            iv.metrics.select([0, 1]),
            A.input(),
            A.zero_baseline(),
            warmup,
            iterations,
        ),
        case(
            "TinyCNN",
            TinyCNN(),
            torch.randn(1, 1, 8, 8, generator=g),
            iv.metrics.select([0, 2]),
            A.input(),
            A.zero_baseline(),
            warmup,
            iterations,
        ),
        case(
            "TinyTransformer",
            TinyTransformer(),
            ids,
            iv.metrics.select([0, 7, 5]),
            A.layer("token_embedding"),
            A.input_baseline(torch.zeros_like(ids)),
            warmup,
            iterations,
        ),
    ]
    env = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "captum": None,
        "platform": f"{platform.system()} {platform.machine()}",
        "torch_threads": torch.get_num_threads(),
    }
    if _captum() is not None:
        import captum

        env["captum"] = str(captum.__version__)
    config = "zero baseline (input ids 0 for TinyTransformer), rule riemann_middle"
    return {
        "environment": env,
        "warmup": warmup,
        "iterations": iterations,
        "method_config": config,
        "results": rows,
    }


def markdown(report: dict[str, Any]) -> str:
    env = report["environment"]
    lines = [
        f"Environment: Python {env['python']}, torch {env['torch']}, Captum {env['captum']}, "
        f"{env['platform']}, {env['torch_threads']} threads; warm-up {report['warmup']}, "
        f"{report['iterations']} iterations; median / p90 ms. IG: {report['method_config']}.",
        "",
        "| model | attributed | forward | trace | gradient | input x grad | IG 16 | IG 64 "
        "| Captum IG 16 | Captum IG 64 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    def f(p: tuple[float, float] | None) -> str:
        return "n/a" if p is None else f"{p[0]:.3f} / {p[1]:.3f}"

    for r in report["results"]:
        lines.append(
            f"| {r['model']} | {r['attributed']} | {f(r['forward_ms'])} | {f(r['trace_ms'])} "
            f"| {f(r['gradient_ms'])} | {f(r['input_x_gradient_ms'])} | {f(r['ig16_ms'])} "
            f"| {f(r['ig64_ms'])} | {f(r['captum_ig16_ms'])} | {f(r['captum_ig64_ms'])} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-3 attribution overhead")
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
