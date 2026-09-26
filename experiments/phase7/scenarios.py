"""Phase 7 falsification scenarios A-N (docs/PHASE_7_PLAN.md §29.3): run every scenario
and the tampered variant of A, compare with the pre-registered expectation, and write
results/scenarios.json. The same builders back tests/test_audit.py.

Usage: python experiments/phase7/scenarios.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import beyondnn as bnn
from beyondnn._testing.audit_scenarios import SCENARIOS
from beyondnn.core.trace import TraceResult
from beyondnn.schema import Assessment, ClaimTestResult, Outcome

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)


def forge(trace: TraceResult, outcome: Outcome) -> TraceResult:
    old = next(r for r in trace.records if isinstance(r, ClaimTestResult))
    forged = ClaimTestResult(
        claim=old.claim,
        spec=old.spec,
        outcome=outcome,
        evidence=old.evidence,
        statistics=old.statistics,
        provenance_id=old.provenance_id,
        derived_from=old.derived_from,
    )
    out = TraceResult(trace.config)
    for record in trace.records:
        if record.id == old.id:
            out._add(forged)
        elif not (
            isinstance(record, Assessment)
            or old.id in getattr(record, "applies_to", ())
            or any(d.record_id == old.id for d in record.derived_from)
        ):
            out._add(record)
    for key, tensor in trace._tensors.items():
        out._add_tensor(key, tensor)
    out._seal()
    return out


def observed(s: Any, report: Any) -> tuple[str, list[str], list[str]]:
    if s.claim is not None:
        c = report.claim(s.claim)
        standing = (
            c.standing.value if c.standing is not None else ",".join(k for k, _ in c.distribution)
        )
        findings = c.findings
    else:
        k = report.concept(s.extra["concept"])
        standing, findings = k.standing.value, k.findings
    codes = sorted({f.code for f in findings})
    axes = sorted({f.axis for f in findings if f.axis})
    return standing, codes, axes


def main() -> None:
    rows = []
    for name in sorted(SCENARIOS):
        t0 = time.perf_counter()
        s = SCENARIOS[name]()
        report = bnn.audit(s.evidence, plan=s.plan)
        standing, codes, axes = observed(s, report)
        match = standing == s.standing and set(s.codes) <= set(codes)
        if "axis" in s.extra:
            match = match and s.extra["axis"] in axes
        rows.append(
            {
                "scenario": name,
                "doc": (SCENARIOS[name].__doc__ or "").strip().splitlines()[0],
                "expected_standing": s.standing,
                "expected_codes": list(s.codes),
                "expected_axis": s.extra.get("axis"),
                "observed_standing": standing,
                "observed_codes": codes,
                "observed_axes": axes,
                "matches_preregistration": match,
                "seconds": time.perf_counter() - t0,
            }
        )
        print(name, "OK" if match else "MISMATCH", standing, codes, flush=True)
    a = SCENARIOS["A"]()
    tampered = [forge(a.evidence[0].trace, Outcome.CONTRADICTS), *a.evidence[1:]]
    report = bnn.audit(tampered, plan=a.plan)
    c = report.claim("a_unit_necessary")
    kinds = sorted({f"{f.kind.value}:{f.code}:{f.severity.value}" for f in c.findings})
    detected = any(k.startswith("integrity_failure") for k in kinds)
    rows.append(
        {
            "scenario": "A-tampered",
            "doc": (
                "A with one comprehensiveness result forged "
                "(SUPPORTS -> CONTRADICTS, ids consistent)"
            ),
            "expected": "INTEGRITY_FAILURE and exclusion",
            "observed_distribution": dict(c.distribution),
            "observed_findings": kinds,
            "excluded_results": report.evidence.results_excluded,
            "matches_preregistration": detected and report.evidence.results_excluded == 1,
        }
    )
    print("A-tampered", detected, kinds, flush=True)
    out = {
        "plan_section": "docs/PHASE_7_PLAN.md §29.3",
        "rows": rows,
        "all_match": all(r["matches_preregistration"] for r in rows),
    }
    (RESULTS / "scenarios.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    print("all match:", out["all_match"])


if __name__ == "__main__":
    main()
