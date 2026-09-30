# ruff: noqa: E501
"""Pre-Phase-8 targeted mutations for the invariants added in the hardening pass.

Usage: python experiments/phase7_75/mutations_pre8.py <repo copy> <out.json>
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
TESTS = ["tests/test_golden_workflow.py", "tests/test_migration_matrix.py"]
MUTATIONS = [
    (
        "report producer dropped",
        "beyondnn/audits/report.py",
        '            "producer": _producer(),\n',
        "",
    ),
    (
        "verify compares the producer too (old reports unverifiable)",
        "beyondnn/audits/__init__.py",
        '    envelope = ("producer", "format_version")',
        "    envelope: tuple[str, ...] = ()",
    ),
    (
        "trace load error does not name the trace",
        "beyondnn/core/persistence.py",
        '        raise TracePersistenceError(f"{directory}: {exc}") from exc',
        "        raise",
    ),
    (
        "migration invents eligibility for old selections",
        "beyondnn/schema/faithfulness.py",
        '    return data | {"eligible": None, "eligibility": None}',
        '    return data | {"eligible": None, "eligibility": None} if False else data | {"eligible": [0], "eligibility": "all"}',
    ),
]
rows = []
for name, rel, old, new in MUTATIONS:
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
