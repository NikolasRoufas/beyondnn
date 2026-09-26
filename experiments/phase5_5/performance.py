"""Phase 5.5 performance measurements (docs/PHASE_5_5_PLAN.md §16), run on an otherwise
idle machine with one torch thread: median of 5 repeats (after one warm-up) for a plain
forward, a trace, each attribution method, one intervention, one faithfulness test with
N=50 count controls (and magnitude controls), plus the record count of each trace.

Usage: python performance.py
"""

# ruff: noqa: B023  (every lambda is called immediately inside timed(); no late binding)
from __future__ import annotations

import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import common
import run_faithfulness as rf
import torch

import beyondnn as bnn

A, F, iv = bnn.attribution, bnn.faithfulness, bnn.interventions


def timed(fn: Any, repeats: int = 5) -> tuple[float, Any]:
    fn()
    times, out = [], None
    for _ in range(repeats):
        t = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t)
    return statistics.median(times), out


def records(result: Any) -> int:
    return len(result.trace.records)


def measure(kind: str) -> dict[str, Any]:
    build = {"A": rf.settings_A, "B": rf.settings_B, "C": rf.settings_C}[kind]
    model, samples, sets, _info, path = build()
    s = samples[0]
    kw = s.kwargs
    out: dict[str, Any] = {}
    with torch.no_grad():
        logits = model(*s.inputs, **kw)
    metric, *_rest, margin = common.margin_target(logits.logits if path else logits, path=path)

    def forward() -> Any:
        with torch.no_grad():
            return model(*s.inputs, **kw)

    out["forward_s"], _ = timed(forward)
    for setting in sets:
        site: dict[str, Any] = {}
        sites = [setting.module] if setting.module else []
        site["trace_s"], tr = timed(
            lambda: bnn.trace(model, *s.inputs, model_kwargs=kw, sites=sites)
        )
        site["trace_records"] = len(tr.records)
        attrs = {}
        for name, method in rf._methods(setting).items():
            site[f"attribute_{name}_s"], attrs[name] = timed(
                lambda method=method: A.attribute(
                    model,
                    *s.inputs,
                    target=metric,
                    method=method,
                    at=setting.at,
                    model_kwargs=kw,
                )
            )
            site[f"attribute_{name}_records"] = records(attrs[name])
        ax = setting.unit_axes
        sel = F.top_k(
            attrs["integrated_gradients"], k=1, unit_axes=ax, reduce="sum" if ax else None
        )
        spec = rf._single_unit(setting, sel.selected[0], F.zero())
        site["intervention_s"], r = timed(
            lambda: iv.intervene(
                model, *s.inputs, intervention=spec, metric=metric, model_kwargs=kw
            )
        )
        site["intervention_records"] = records(r)
        for match in ("count", "magnitude"):
            test = F.comprehensiveness(
                replacement=F.zero(),
                target=metric,
                min_drop=0.5 * margin,
                statement="perf",
                controls=F.controls(50, seed=0, match=match),
            )
            site[f"faithfulness_N50_{match}_s"], res = timed(
                lambda test=test: F.run(
                    model, *s.inputs, test=test, selection=sel, model_kwargs=kw
                ),
                repeats=3,
            )
            site[f"faithfulness_N50_{match}_records"] = records(res)
        for n_controls in (10, 50, 200):
            scaling = F.comprehensiveness(
                replacement=F.zero(),
                target=metric,
                min_drop=0.5 * margin,
                statement="perf",
                controls=F.controls(n_controls, seed=0),
            )
            site[f"control_scaling_N{n_controls}_s"], _ = timed(
                lambda scaling=scaling: F.run(
                    model, *s.inputs, test=scaling, selection=sel, model_kwargs=kw
                ),
                repeats=1,
            )
        with tempfile.TemporaryDirectory() as tmp:
            res.trace.save(Path(tmp) / "t")
            site["faithfulness_N50_trace_bytes_on_disk"] = sum(
                f.stat().st_size for f in (Path(tmp) / "t").rglob("*") if f.is_file()
            )
        site["compose_verify_s"], _ = timed(
            lambda: bnn.compose(
                bnn.trace(model, *s.inputs, model_kwargs=kw),
                attributions=[attrs["integrated_gradients"]],
                faithfulness=[res],
            ),
            repeats=3,
        )
        out[setting.site] = site
    return out


def main() -> None:
    torch.set_num_threads(1)
    payload: dict[str, Any] = {
        "environment": common.environment(),
        "note": (
            "median of 5 repeats (3 for faithfulness runs, 1 for control scaling) "
            "after a warm-up; one torch thread; idle machine"
        ),
        "input_sizes": {
            "A": "(1, 30)",
            "B": "(1, 1, 8, 8)",
            "C": "(1, T) token ids, first declared sentence",
        },
    }
    for kind in "ABC":
        payload[kind] = measure(kind)
        print(kind, payload[kind], flush=True)
    common.save("performance", payload)


if __name__ == "__main__":
    main()
