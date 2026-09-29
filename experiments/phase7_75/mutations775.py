# ruff: noqa: E501
"""Phase 7.75 mutations (request §32): each mutation of the Phase-7.75 fixes must make the tests
fail. Runs against a scratch copy of the repository.

Usage: python experiments/phase7_75/mutations775.py <repo copy> <out.json>
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
SPEC, STATS, ENG, EVI = (
    "beyondnn/faithfulness/spec.py",
    "beyondnn/faithfulness/stats.py",
    "beyondnn/audits/engine.py",
    "beyondnn/audits/evidence.py",
)
SCH, SCHA, AN = (
    "beyondnn/schema/faithfulness.py",
    "beyondnn/schema/audit.py",
    "experiments/phase7_75/analysis775.py",
)
TDB, TDC = "experiments/phase7_75/td_bench.py", "experiments/phase7_75/td_concepts.py"
TESTS = [
    "tests/test_phase775.py",
    "tests/test_phase775_analysis.py",
    "tests/test_phase75_immutable.py",
    "tests/test_audit_75.py",
    "tests/test_unit_axes.py",
    "tests/test_faithfulness.py",
    "tests/test_audit.py",
]
MUTATIONS = [
    (
        "special tokens silently included despite content-token policy",
        ENG,
        "        if eligibility_checked and s.eligibility != sel.eligibility:\n            return False",
        "        pass",
    ),
    (
        "special tokens silently included in the ranking",
        SPEC,
        "        order = tuple(u for u in order if u in keep)",
        "        order = order",
    ),
    (
        "special tokens silently removed despite all-token policy",
        SPEC,
        "    order = rank_order(scores, by=by)\n    if allowed is not None:",
        "    order = rank_order(scores, by=by)[1:-1]\n    if allowed is not None:",
    ),
    (
        "controls drawn from ineligible units",
        STATS,
        "    pool = list(range(n_units)) if population is None else list(population)",
        "    pool = list(range(n_units))",
    ),
    (
        "independence flag dropped",
        AN,
        "    return truth_operation.strip().lower() != audit_operation.strip().lower()",
        "    return True",
    ),
    (
        "same-operation ground truth treated as independent",
        AN,
        "    return truth_operation.strip().lower() != audit_operation.strip().lower()",
        "    return truth_operation != audit_operation",
    ),
    (
        "concept threshold tuned (C_REL changed)",
        AN,
        "C_REL = 0.2  # ADR-055",
        "C_REL = 0.05  # ADR-055",
    ),
    ("concept threshold tuned in the experiment", TDC, "C_REL = 0.2\n", "C_REL = 0.05\n"),
    (
        "known-positive result used during calibration",
        AN,
        "def rel_min_change(train_target_values: Sequence[float], c: float = C_REL) -> float:",
        "def rel_min_change(train_target_values: Sequence[float], c: float = C_REL, heldout_effect: float = 0.0) -> float:",
    ),
    (
        "competitive-control failure promoted to causal contradiction (unattainable ignored)",
        EVI,
        "        if self.result.outcome is Outcome.CONTRADICTS and self.control_unattainable():\n            return Outcome.INCONCLUSIVE",
        "        if False:\n            return Outcome.INCONCLUSIVE",
    ),
    (
        "attainability computed without identical control sets",
        EVI,
        "        identical = sum(1 for u in units[1:] if u == units[0])",
        "        identical = 0",
    ),
    (
        "effect-without-competitive-advantage finding dropped",
        ENG,
        "            if e.outcome is Outcome.CONTRADICTS and e.absolute_criterion_met and e.controlled",
        "            if False",
    ),
    (
        "zero stress test promoted to PRIMARY",
        TDB,
        '    ("replacement", "zero", "stress_test"),',
        '    ("replacement", "zero", "primary"),',
    ),
    (
        "replacement identity lost (name dropped)",
        SPEC,
        '    return Replacement(tensor.detach().to("cpu").clone(memory_format=torch.contiguous_format), name)',
        '    return Replacement(tensor.detach().to("cpu").clone(memory_format=torch.contiguous_format), None)',
    ),
    (
        "OOD (stress-test) reversal finding ignored",
        ENG,
        '        (STRESS_TEST, "stress_test_reverses", S.INFORMATIONAL),\n',
        "",
    ),
    (
        "selection v2 -> v3 migration drops eligibility",
        SCH,
        '    return data | {"eligible": None, "eligibility": None}',
        "    return data",
    ),
    (
        "audit plan v2 -> v3 migration drops eligibility",
        SCHA,
        '        return c if sel is None else c | {"selection": sel | {"eligibility": None}}',
        "        return c",
    ),
    (
        "shortcut stratification removes samples",
        AN,
        "        cell = out[labels[0] if flag else labels[1]]",
        "        if not flag:\n            continue\n        cell = out[labels[0]]",
    ),
    (
        "Phase-7.5 artifacts overwritten",
        "experiments/phase7_5/results/performance75.json",
        "{",
        "{ ",
    ),
]
ONLY = set(sys.argv[3].split("|")) if len(sys.argv) > 3 else None
rows = []
for name, rel, old, new in MUTATIONS:
    if ONLY and name not in ONLY:
        continue
    p = ROOT / rel
    src = p.read_text()
    if old not in src:
        rows.append({"mutation": name, "result": "NOT APPLIED"})
        print(name, "NOT APPLIED", flush=True)
        continue
    p.write_text(src.replace(old, new, 1))
    try:
        r = subprocess.run(
            [
                "uv",
                "run",
                "--python",
                "3.14",
                "--extra",
                "dev",
                "--with",
                "numpy",
                "pytest",
                "-x",
                "-q",
                "-p",
                "no:cacheprovider",
                *TESTS,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        killed = r.returncode != 0
        failing = [ln for ln in r.stdout.splitlines() if ln.startswith("FAILED")][:1]
    finally:
        p.write_text(src)
    rows.append({"mutation": name, "result": "killed" if killed else "SURVIVED", "by": failing})
    print(name, "killed" if killed else "SURVIVED", failing, flush=True)
Path(sys.argv[2]).write_text(json.dumps(rows, indent=1))
