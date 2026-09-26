"""Phase 7.5 end-to-end performance (request §33): small model (A MLP input), moderate model
(D BERT-base SST-2) and the external benchmark (InterpBench case 7, run separately in the
InterpBench env via ``--external``). Single runs, one torch thread, idle machine.

Measured: evidence generation, save_evidence, evidence load, audit (integrity + re-derivation
+ axes), aggregation (standings, profiles), uncertainty computation, serialisation, report size,
peak memory (tracemalloc of the audit; process max RSS), and the full workflow time.

Usage (experiment env, from experiments/phase5_5): python ../phase7_5/performance75.py A|D
       (InterpBench env, from experiments/phase7_5/artifacts): python ../performance75.py external
"""

from __future__ import annotations

import json
import resource
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
N_SAMPLES = {"A": 3, "D": 2, "external": 5}


def clock(fn: Any) -> tuple[float, Any]:
    t = time.perf_counter()
    out = fn()
    return time.perf_counter() - t, out


def measure(evidence: list[Any], plan: Any, gen_s: float) -> dict[str, Any]:
    import beyondnn as bnn
    from beyondnn.audits.engine import _Auditor
    from beyondnn.audits.evidence import EvidenceSet, collect_traces

    AU = bnn.audits
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "ev"
        save_s, paths = clock(lambda: AU.save_evidence(evidence, d))
        disk = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        load_s, traces = clock(lambda: collect_traces(paths))
        tracemalloc.start()
        build_s, ev = clock(lambda: EvidenceSet.build(traces))
        agg_s, report = clock(lambda: _Auditor(plan, ev).run())
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        values = [float(i % 7) for i in range(60)]
        unc_s, _ = clock(
            lambda: (
                AU.wilson(20, 60, quantity="q", unit="samples"),
                AU.bootstrap(values, quantity="q", unit="samples", seed=1),
                AU.paired_bootstrap(values, values[::-1], quantity="q", unit="samples", seed=1),
            )
        )
        ser_s, text = clock(report.to_json)
        verify_s, _ = clock(lambda: AU.verify_report(json.loads(text), paths, plan))
    groups = [g for c in report.claims for g in c.groups]
    return {
        "results": report.evidence.results,
        "records": report.evidence.records,
        "traces": report.evidence.traces,
        "evidence_generation_s": gen_s,
        "save_evidence_s": save_s,
        "evidence_bytes_on_disk": disk,
        "load_s": load_s,
        "audit_build_s (integrity + re-derivation + axes)": build_s,
        "aggregation_and_profiles_s": agg_s,
        "uncertainty_s (wilson + 2 bootstraps x 10k)": unc_s,
        "serialize_s": ser_s,
        "report_bytes": len(text.encode()),
        "verify_report_s": verify_s,
        "audit_peak_tracemalloc_mb": peak / 1e6,
        "process_max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
        "configurations_profiled": sum(g.profile.tested for g in groups if g.profile),
        "full_workflow_s": gen_s + save_s + load_s + build_s + agg_s + ser_s,
    }


def central(model_id: str) -> dict[str, Any]:
    sys.path.insert(0, str(HERE.parent / "phase5_5"))
    sys.path.insert(0, str(HERE))
    import central75 as C75
    import torch

    torch.set_num_threads(1)
    model, samples, settings, _info, path = C75.build(model_id, "heldout")
    setting = settings[0]
    t0 = time.perf_counter()
    results, attrs, metas, targets, methods = [], [], [], {}, {}
    for s in samples[: N_SAMPLES[model_id]]:
        res, a, meta = C75.run_sample(model, setting, s, path)
        results += res
        attrs += a
        metas.append(meta)
        targets[meta["sample_id"]] = res[0].claim.target
        methods = {"g": a[0].record.method.name, "ig": a[1].record.method.name, "r": "declared"}
    gen_s = time.perf_counter() - t0
    plan = C75.plan_for(model, setting, targets, methods, metas)
    return measure(results, plan, gen_s)  # as central75.main: the audit ingests the results


def external() -> dict[str, Any]:
    sys.path.insert(0, str(HERE))
    import external_interpbench as X
    import torch

    import beyondnn as bnn

    torch.set_num_threads(1)
    X.N_SAMPLES = N_SAMPLES["external"]
    t0 = time.perf_counter()
    # re-run the case pipeline on 5 samples, keeping the evidence for the measurement
    captured: dict[str, Any] = {}
    original = bnn.audit

    def capture(evidence: Any, *, plan: Any, model: Any = None) -> Any:
        if "plan" not in captured:
            captured["evidence"], captured["plan"] = list(evidence), plan
        return original(evidence, plan=plan, model=model)

    bnn.audit = capture  # measurement harness only: the case code calls bnn.audit
    try:
        X.run_case("7")
    finally:
        bnn.audit = original
    gen_s = time.perf_counter() - t0
    return measure(captured["evidence"], captured["plan"], gen_s)


def main() -> None:
    which = sys.argv[1]
    out = external() if which == "external" else central(which)
    path = RESULTS / "performance75.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[
        {
            "A": "small (MLP A/input)",
            "D": "moderate (BERT-base D)",
            "external": "external (InterpBench case 7)",
        }[which]
    ] = out | {"samples": N_SAMPLES[which]}
    path.write_text(json.dumps(data, indent=1, sort_keys=True))
    print(which, json.dumps(out))


if __name__ == "__main__":
    main()
