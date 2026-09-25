"""Phase-1 tracing overhead characterisation (M1.9). CPU only, no downloads.

Usage::

    python benchmarks/bench_trace_overhead.py            # 30 measured iterations
    python benchmarks/bench_trace_overhead.py --quick    # 3 iterations (smoke test)
    python benchmarks/bench_trace_overhead.py --json out.json

For every model it times (median / p90 over the measured iterations, after
warm-up): a baseline forward, ``bnn.trace`` with retention none / summary / cpu
over representative sites, ``fingerprint_model`` alone, and save / load of the
cpu-retained trace. Everything runs under ``torch.no_grad()``. Sizes are exact
byte counts (retained tensor bytes, trace.json, tensors.pt); no tracemalloc-based
memory claims are made.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn

import beyondnn as bnn
from beyondnn._testing.models import TinyCNN, TinyMLP, TinyTransformer
from beyondnn.core.sites import resolve_sites
from beyondnn.provenance import fingerprint_model


def dense_1m() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(1000, 1000), nn.ReLU(), nn.Linear(1000, 2))


CASES: list[tuple[str, Callable[[], nn.Module], Callable[[], torch.Tensor], list[str]]] = [
    (
        "TinyMLP",
        TinyMLP,
        lambda: torch.randn(32, 4, generator=torch.Generator().manual_seed(0)),
        ["**"],
    ),
    (
        "TinyCNN",
        lambda: TinyCNN().eval(),
        lambda: torch.randn(8, 1, 8, 8, generator=torch.Generator().manual_seed(0)),
        ["**"],
    ),
    (
        "TinyTransformer",
        TinyTransformer,
        lambda: torch.randint(0, 32, (4, 8), generator=torch.Generator().manual_seed(0)),
        ["blocks.*.attn", "blocks.*.mlp", "lm_head"],
    ),
    (
        "Dense~1M",
        dense_1m,
        lambda: torch.randn(16, 1000, generator=torch.Generator().manual_seed(0)),
        ["**"],
    ),
]


def timed(fn: Callable[[], Any], warmup: int, iterations: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1e3)
    samples.sort()
    p90 = samples[min(len(samples) - 1, round(0.9 * (len(samples) - 1)))]
    return statistics.median(samples), p90


def environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "beyondnn": bnn.__version__,
        "platform": f"{platform.system()} {platform.machine()}",
        "torch_threads": torch.get_num_threads(),
    }


def bench_case(
    name: str,
    model: nn.Module,
    x: torch.Tensor,
    sites: list[str],
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    params = sum(p.numel() for p in model.parameters())
    modes = {
        retention: timed(
            lambda r=retention: bnn.trace(model, x, sites=sites, retention=r), warmup, iterations
        )
        for retention in ("none", "summary", "cpu")
    }
    trace = bnn.trace(model, x, sites=sites, retention="cpu")
    retained = sum(t.numel() * t.element_size() for t in trace._tensors.values())
    with tempfile.TemporaryDirectory() as tmp:
        counter = iter(range(10**6))
        save = timed(lambda: trace.save(Path(tmp) / f"t{next(counter)}"), 1, iterations)
        target = Path(tmp) / "final"
        trace.save(target)
        load = timed(lambda: bnn.load_trace(target), 1, iterations)
        json_bytes = (target / "trace.json").stat().st_size
        sidecar = target / "tensors.pt"
        sidecar_bytes = sidecar.stat().st_size if sidecar.exists() else 0
    return {
        "model": name,
        "parameters": params,
        "input_shape": list(x.shape),
        "selected_output_sites": len(resolve_sites(model, sites)),
        "activation_records": len(trace.activations),
        "records": len(trace.records),
        "baseline_ms": timed(lambda: model(x), warmup, iterations),
        "fingerprint_ms": timed(lambda: fingerprint_model(model), warmup, iterations),
        "trace_ms": modes,
        "save_ms": save,
        "load_ms": load,
        "retained_tensor_bytes": retained,
        "trace_json_bytes": json_bytes,
        "sidecar_bytes": sidecar_bytes,
    }


def run(warmup: int, iterations: int) -> dict[str, Any]:
    with torch.no_grad():
        rows = [
            bench_case(name, build(), make_x(), sites, warmup, iterations)
            for name, build, make_x, sites in CASES
        ]
    return {
        "environment": environment(),
        "warmup": warmup,
        "iterations": iterations,
        "results": rows,
    }


def markdown(report: dict[str, Any]) -> str:
    env = report["environment"]
    out = [
        f"Environment: Python {env['python']}, torch {env['torch']}, {env['platform']}, "
        f"{env['torch_threads']} torch threads; warm-up {report['warmup']}, "
        f"{report['iterations']} measured iterations; times are median / p90 in ms, under no_grad.",
        "",
        "| model | params | input | sites | act. records | baseline | fingerprint | trace none "
        "| trace summary | trace cpu | fingerprint share of summary overhead | save (cpu) "
        "| load (cpu) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report["results"]:

        def f(pair: tuple[float, float]) -> str:
            return f"{pair[0]:.3f} / {pair[1]:.3f}"

        overhead = r["trace_ms"]["summary"][0] - r["baseline_ms"][0]
        share = r["fingerprint_ms"][0] / overhead if overhead > 0 else float("nan")
        trace_ms = r["trace_ms"]
        out.append(
            f"| {r['model']} | {r['parameters']:,} | {tuple(r['input_shape'])} "
            f"| {r['selected_output_sites']} | {r['activation_records']} "
            f"| {f(r['baseline_ms'])} | {f(r['fingerprint_ms'])} | {f(trace_ms['none'])} "
            f"| {f(trace_ms['summary'])} | {f(trace_ms['cpu'])} "
            f"| {share:.0%} | {f(r['save_ms'])} | {f(r['load_ms'])} |"
        )
    out += [
        "",
        "| model | records | retained tensor bytes (cpu) | trace.json bytes | tensors.pt bytes |",
        "|---|---|---|---|---|",
    ]
    for r in report["results"]:
        out.append(
            f"| {r['model']} | {r['records']} | {r['retained_tensor_bytes']:,} "
            f"| {r['trace_json_bytes']:,} | {r['sidecar_bytes']:,} |"
        )
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="3 iterations, 1 warm-up")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--json", type=Path, help="also write the raw report as JSON")
    args = parser.parse_args()
    iterations, warmup = (3, 1) if args.quick else (args.iterations, args.warmup)
    report = run(warmup, iterations)
    print(markdown(report))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
