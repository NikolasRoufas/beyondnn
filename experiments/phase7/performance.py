"""Phase 7 performance and reload check (docs/PHASE_7_PLAN.md §32, §26, D10).

For the MLP (A/input), CNN (B/pixels) and transformer (C/tokens) evidence saved by
central.py (the first 3 samples of each model/site), in a fresh process:

* load (``load_trace`` of every saved trace directory);
* integrity checks (ids, references, provenance) alone;
* evidence build (integrity + re-derivation of every result + axes);
* aggregation (the audit rules over the built evidence);
* serialization (``to_json``) and report save;
* peak memory of the audit (tracemalloc) and on-disk sizes;
* equality of every per-sample group with the in-memory full audit saved by central.py
  (the reload check of D10).

One torch thread, single run each.

Usage (experiment env, from experiments/phase5_5): python ../phase7/performance.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase5_5"))
import common  # noqa: E402
import torch  # noqa: E402

import beyondnn as bnn  # noqa: E402
from beyondnn.audits.engine import _Auditor, _check_plan  # noqa: E402
from beyondnn.audits.evidence import EvidenceSet  # noqa: E402
from beyondnn.explain.bundle import _check_source  # noqa: E402
from beyondnn.schema import AuditPlan, from_json  # noqa: E402

ARTIFACTS = HERE / "artifacts"
RESULTS = HERE / "results"
SITES = {"MLP": ("A", "input"), "CNN": ("B", "pixels"), "transformer": ("C", "tokens")}


def size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def clock(fn: Any) -> tuple[float, Any]:
    t = time.perf_counter()
    out = fn()
    return time.perf_counter() - t, out


def restricted(plan: AuditPlan, samples: list[str]) -> AuditPlan:
    return AuditPlan(
        name=plan.name,
        checkpoint=plan.checkpoint,
        declared_model=plan.declared_model,
        samples=tuple(samples),
        datasets=plan.datasets,
        claims=plan.claims,
        requirements=plan.requirements,
        concepts=plan.concepts,
        counterexamples=plan.counterexamples,
        naive_auroc=plan.naive_auroc,
    )


def measure(model_id: str, site: str) -> dict[str, Any]:
    base = ARTIFACTS / f"{model_id}_{site}"
    dirs = sorted(p for sample in base.iterdir() for p in sample.iterdir() if p.is_dir())
    full_plan = from_json((ARTIFACTS / f"plan_{model_id}_{site}.json").read_text())
    assert isinstance(full_plan, AuditPlan)
    load_s, traces = clock(lambda: [bnn.load_trace(d) for d in dirs])
    samples = sorted(
        {
            r.claim.estimand.sample_id
            for t in traces
            for r in t.records
            if hasattr(r, "claim") and hasattr(r.claim, "estimand")
        }
        & set(full_plan.samples)
    )  # type: ignore[attr-defined]
    plan = restricted(full_plan, samples)
    _check_plan(plan)
    integrity_s, _ = clock(lambda: [_check_source(t) for t in traces])
    tracemalloc.start()
    build_s, ev = clock(lambda: EvidenceSet.build(traces))
    aggregate_s, report = clock(lambda: _Auditor(plan, ev).run())
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    serialize_s, text = clock(report.to_json)
    with tempfile.TemporaryDirectory() as tmp:
        save_s, _ = clock(lambda: report.save(Path(tmp) / "r.json"))
        report_bytes = (Path(tmp) / "r.json").stat().st_size
        reload_s, _ = clock(lambda: bnn.audits.load_report(Path(tmp) / "r.json"))
    # D10: every per-sample group equals the in-memory full audit's group for that sample
    full = json.loads((ARTIFACTS / f"central_{model_id}_{site}_report.json").read_text())
    mine = json.loads(text)
    mismatches = 0
    compared = 0
    for c_full, c_mine in zip(full["claims"], mine["claims"], strict=True):
        assert c_full["claim"] == c_mine["claim"]
        groups = {g["sample"]: g for g in c_full["groups"]}
        for g in c_mine["groups"]:
            compared += 1
            mismatches += groups[g["sample"]] != g
    return {
        "samples": len(samples),
        "trace_dirs": len(dirs),
        "records": ev.records,
        "results": len(ev.results),
        "trace_bytes_on_disk": size(base),
        "load_traces_s": load_s,
        "integrity_s": integrity_s,
        "build_s (integrity + re-derivation + axes)": build_s,
        "rederivation_s (build - integrity)": build_s - integrity_s,
        "aggregate_s": aggregate_s,
        "serialize_s": serialize_s,
        "report_save_s": save_s,
        "report_load_s": reload_s,
        "report_bytes": report_bytes,
        "audit_peak_memory_mb (tracemalloc)": peak / 1e6,
        "reload_groups_compared": compared,
        "reload_group_mismatches": mismatches,
        "per_result_ms (build + aggregate)": 1000
        * (build_s + aggregate_s)
        / max(1, len(ev.results)),
    }


def main() -> None:
    torch.set_num_threads(1)
    payload: dict[str, Any] = {
        "environment": common.environment(),
        "note": "single run each; one torch thread; fresh process; first 3 samples per site (D10)",
    }
    for name, (model_id, site) in SITES.items():
        payload[name] = measure(model_id, site)
        print(name, json.dumps(payload[name]), flush=True)
    full = {}
    for m in ("A", "B", "C"):
        path = RESULTS / f"central_{m}.json"
        if path.exists():
            data = json.loads(path.read_text())
            full[m] = {
                s: {
                    k: v[k]
                    for k in (
                        "n_samples",
                        "n_results",
                        "run_seconds",
                        "audit_seconds",
                        "verify_report_seconds",
                    )
                }
                for s, v in data["settings"].items()
            }
    payload["full_central_audits"] = full
    (RESULTS / "performance.json").write_text(json.dumps(payload, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
