"""The M1.9 benchmark must stay runnable (one iteration, no timing assertions)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


def _load(name: str) -> Any:
    path = Path(__file__).resolve().parents[1] / "benchmarks" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_faithfulness_benchmark_runs() -> None:
    module = _load("bench_faithfulness")
    report = module.run(warmup=0, iterations=1)
    assert len(report["results"]) == 14
    assert "|" in module.markdown(report)


def test_synthesis_benchmark_runs() -> None:
    module = _load("bench_synthesis")
    report = module.run(warmup=0, iterations=1)
    assert len(report["results"]) == 4
    assert "|" in module.markdown(report)


def test_attribution_benchmark_runs() -> None:
    module = _load("bench_attribution")
    report = module.run(warmup=0, iterations=1)
    assert [r["model"] for r in report["results"]] == [
        "Product (analytic)",
        "TinyMLP",
        "TinyCNN",
        "TinyTransformer",
    ]
    assert "|" in module.markdown(report)


def test_intervention_benchmark_runs() -> None:
    module = _load("bench_interventions")
    report = module.run(warmup=0, iterations=1)
    assert {r["model"] for r in report["results"]} == {"TinyMLP", "TinyTransformer"}
    assert "|" in module.markdown(report)


def test_benchmark_runs() -> None:
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_trace_overhead.py"
    spec = importlib.util.spec_from_file_location("bench_trace_overhead", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.run(warmup=0, iterations=1)
    assert {r["model"] for r in report["results"]} == {
        "TinyMLP",
        "TinyCNN",
        "TinyTransformer",
        "Dense~1M",
    }
    assert "|" in module.markdown(report)


def test_audit_benchmark_runs() -> None:
    module = _load("bench_audit")
    report = module.run(warmup=0, iterations=1)
    assert len(report["results"]) == 4
    assert "|" in module.markdown(report)
