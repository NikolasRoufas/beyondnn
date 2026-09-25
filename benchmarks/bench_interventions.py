"""Phase-2 intervention overhead characterisation (CPU, no downloads).

Usage::

    python benchmarks/bench_interventions.py [--quick] [--json out.json]

For TinyMLP and TinyTransformer (eval mode) it times, median / p90 in ms after
warm-up: a plain forward, a baseline ``bnn.trace`` of the intervened site, a
zero-ablation comparison (``bnn.intervene``: baseline + intervention pass), and
an activation-patching comparison (source + baseline + intervention pass).
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
from beyondnn._testing.models import TinyMLP, TinyTransformer

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


def case(
    name: str,
    model: nn.Module,
    x: torch.Tensor,
    x_src: torch.Tensor,
    site: str,
    path: str,
    metric: Any,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        forward = timed(lambda: model(x), warmup, iterations)
        trace = timed(lambda: bnn.trace(model, x, sites=[site]), warmup, iterations)
    zero = timed(
        lambda: bnn.intervene(
            model, x, intervention=iv.zero(site, output_path=path), metric=metric
        ),
        warmup,
        iterations,
    )
    patch = timed(
        lambda: bnn.intervene(
            model, x, intervention=iv.patch(site, x_src, output_path=path), metric=metric
        ),
        warmup,
        iterations,
    )
    return {
        "model": name,
        "site": site,
        "forward_ms": forward,
        "trace_ms": trace,
        "zero_ms": zero,
        "patch_ms": patch,
    }


def run(warmup: int, iterations: int) -> dict[str, Any]:
    g = torch.Generator().manual_seed(0)
    rows = [
        case(
            "TinyMLP",
            TinyMLP(),
            torch.randn(32, 4, generator=g),
            torch.randn(32, 4, generator=g),
            "shared",
            "",
            iv.metrics.select([0, 0]),
            warmup,
            iterations,
        ),
        case(
            "TinyTransformer",
            TinyTransformer(),
            torch.randint(0, 32, (4, 8), generator=g),
            torch.randint(0, 32, (4, 8), generator=g),
            "blocks.0.attn",
            "[0]",
            iv.metrics.select([0, 0, 0]),
            warmup,
            iterations,
        ),
    ]
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
        "| model | site | forward | baseline trace | zero-ablation comparison | patch comparison |",
        "|---|---|---|---|---|---|",
    ]
    for r in report["results"]:

        def f(p: tuple[float, float]) -> str:
            return f"{p[0]:.3f} / {p[1]:.3f}"

        lines.append(
            f"| {r['model']} | {r['site']} | {f(r['forward_ms'])} | {f(r['trace_ms'])} "
            f"| {f(r['zero_ms'])} | {f(r['patch_ms'])} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-2 intervention overhead")
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
