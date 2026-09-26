# ruff: noqa: E501, E741
"""Phase 7.5 mutations (plan §32): each mutation of the Phase-7.5 behaviour must make the
Phase-7.5 tests fail. Runs against a scratch copy of the repository.

Usage: python experiments/phase7_5/mutations75.py <repo copy> <out.json>
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
E, R, U = "beyondnn/audits/engine.py", "beyondnn/audits/report.py", "beyondnn/audits/uncertainty.py"
I, AN = "beyondnn/audits/__init__.py", "experiments/phase7_5/analysis75.py"
X = "experiments/phase7_5/external_interpbench.py"
TESTS = ["tests/test_audit_75.py", "tests/test_phase75_analysis.py", "tests/test_audit.py"]
MUTATIONS = [
    (
        "configuration count changed",
        E,
        "        tested=len(configs),",
        "        tested=len(configs) + 1,",
    ),
    (
        "support profile omits a contradiction",
        E,
        "    con = [cf for cf in configs if cf.outcome is Outcome.CONTRADICTS]",
        "    con: list[_Configuration] = []",
    ),
    (
        "PRIMARY changed to STRESS_TEST (role order)",
        E,
        "_ROLE_ORDER = {PRIMARY: 0, ALTERNATIVE: 1, STRESS_TEST: 2}",
        "_ROLE_ORDER = {PRIMARY: 2, ALTERNATIVE: 1, STRESS_TEST: 0}",
    ),
    (
        "standing uses every role, not PRIMARY",
        E,
        "        in_standing = [cf for cf in configs if not declared or cf.role == PRIMARY]",
        "        in_standing = list(configs)",
    ),
    (
        "alternative reversal not reported",
        E,
        '        (ALTERNATIVE, "alternative_reverses", S.QUALIFYING),\n        (STRESS_TEST, "stress_test_reverses", S.INFORMATIONAL),\n    ):\n        reversing = [\n            (cf.axes',
        '        (STRESS_TEST, "stress_test_reverses", S.INFORMATIONAL),\n    ):\n        reversing = [\n            (cf.axes',
    ),
    (
        "sample-specific role rules applied to every sample",
        E,
        "    applicable = [r for r in rules if r.sample is None or r.sample == sample]",
        "    applicable = list(rules)",
    ),
    (
        "configuration-level disagreement dropped",
        E,
        "    left, right = keyed(ga.profile), keyed(gb.profile)",
        "    return []\n    left, right = keyed(ga.profile), keyed(gb.profile)",
    ),
    (
        "wrong requirement attached to a claim",
        E,
        "        requirement = self.requirements[c.requirement]",
        "        requirement = next(iter(self.requirements.values()))",
    ),
    (
        "uncertainty resamples the wrong unit (unpaired)",
        U,
        "        stats.append(fn([fa[i] for i in idx]) - fn([fb[i] for i in idx]))",
        "        stats.append(fn([fa[i] for i in idx]) - fn([fb[rng.randrange(n)] for _ in idx]))",
    ),
    (
        "bootstrap seed ignored",
        U,
        "    rng = random.Random(seed)\n    n = len(values)",
        "    rng = random.Random()\n    n = len(values)",
    ),
    ("Wilson level ignored", U, "    z = _z(level)\n    p = k / n", "    z = 1.0\n    p = k / n"),
    (
        "claim intervals dropped",
        E,
        "            intervals = tuple(\n                wilson(",
        "            intervals = () and tuple(\n                wilson(",
    ),
    (
        "audit profile serialised incorrectly",
        R,
        "    if isinstance(value, AuditedClaim):",
        "    if isinstance(value, SensitivityProfile):\n        return None\n    if isinstance(value, AuditedClaim):",
    ),
    (
        "saved evidence changes the result",
        I,
        "    for i, trace in enumerate(traces_of(evidence)):",
        "    for i, trace in enumerate(traces_of(evidence)[:-1]):",
    ),
    (
        "wrong checkpoint accepted",
        E,
        "        if entry.checkpoint != self.plan.checkpoint:",
        "        if False:",
    ),
    (
        "unsupported protocol version accepted",
        "beyondnn/audits/evidence.py",
        "        if protocol in PROTOCOLS and entry.spec.protocol_version != PROTOCOL_VERSIONS[protocol]:",
        "        if False:",
    ),
    (
        "held-out case used as development case",
        X,
        'DEV_CASES = ("7", "13")',
        'DEV_CASES = ("7", "13", "2")',
    ),
    (
        "external ground truth mislabelled",
        AN,
        '            key = {"supported": "tp", "contradicted": "fn"}.get(standing, "pos_other")',
        '            key = {"supported": "fp", "contradicted": "tn"}.get(standing, "neg_other")',
    ),
    (
        "false-positive count dropped",
        AN,
        '            key = {"contradicted": "tn", "supported": "fp"}.get(standing, "neg_other")',
        '            key = {"contradicted": "tn", "supported": "neg_other"}.get(standing, "neg_other")',
    ),
    (
        "false-negative count dropped",
        AN,
        '            key = {"supported": "tp", "contradicted": "fn"}.get(standing, "pos_other")',
        '            key = {"supported": "tp", "contradicted": "pos_other"}.get(standing, "pos_other")',
    ),
    (
        "benchmark scope mixed (unreliable cases kept)",
        AN,
        '        if "error" in r or not r.get("benchmark_reliable", False):',
        '        if "error" in r:',
    ),
    (
        "ambiguous instances counted as clear",
        AN,
        '        if gt == AMBIGUOUS:\n            out["ambiguous"] += 1',
        '        if gt == "never":\n            out["ambiguous"] += 1',
    ),
    (
        "token perturbation mislabelled",
        AN,
        'NATIVE = {"zero": False, "mask": True,',
        'NATIVE = {"zero": False, "mask": False,',
    ),
    (
        "human rationale treated as causal ground truth",
        AN,
        '"human_is_ground_truth": False}',
        '"human_is_ground_truth": True}',
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
