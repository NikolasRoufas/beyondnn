"""Phase-7 code mutations (plan §31): each must make tests/test_audit.py fail."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
E, V = "beyondnn/audits/engine.py", "beyondnn/audits/evidence.py"
I = "beyondnn/audits/__init__.py"
MUTATIONS = [
    (
        "precedence: contradicted before disagreement",
        E,
        "    if sup and con:\n        return Standing.MIXED if unexplained else Standing.ASSUMPTION_SENSITIVE\n    if verdict is Verdict.CONTRADICTED:\n        return Standing.CONTRADICTED\n",
        "    if verdict is Verdict.CONTRADICTED:\n        return Standing.CONTRADICTED\n    if sup and con:\n        return Standing.MIXED if unexplained else Standing.ASSUMPTION_SENSITIVE\n",
    ),
    (
        "axis explanation accepts multi-axis pairs",
        E,
        "            elif len(diff) == 1:",
        "            elif len(diff) >= 1:",
    ),
    (
        "unexplained disagreement treated as explained",
        E,
        "        return Standing.MIXED if unexplained else Standing.ASSUMPTION_SENSITIVE\n    if verdict",
        "        return Standing.ASSUMPTION_SENSITIVE\n    if verdict",
    ),
    (
        "invariance count ignored",
        E,
        "(len(tested) >= inv.min_values and not absent)",
        "(len(tested) >= 1 and not absent)",
    ),
    (
        "counterexample fraction cap ignored",
        E,
        "exceeded = cap is not None and fraction > cap",
        "exceeded = False",
    ),
    (
        "FP/FN caps ignored",
        E,
        "if cap is not None and fp > cap",
        "if cap is not None and fp > cap + 1",
    ),
    (
        "sample scope check dropped",
        E,
        "        if sample is not None and sample not in self.samples:",
        "        if False:",
    ),
    (
        "dataset scope check dropped",
        E,
        "            dataset not in self.datasets\n        ):",
        "            False\n        ):",
    ),
    (
        "checkpoint check dropped",
        E,
        "        if entry.checkpoint != self.plan.checkpoint:",
        "        if False:",
    ),
    (
        "attribution->causal overclaim not flagged",
        E,
        "            if attribution_results:\n",
        "            if False:\n",
    ),
    (
        "narrower estimand not flagged",
        E,
        "            if narrower and c.scope is not EstimandScope.INSTANCE:",
        "            if False:",
    ),
    (
        "missing controls not flagged",
        E,
        "            if supports and not any(e.controlled for e in supports):",
        "            if False:",
    ),
    (
        "missing evidence -> SUPPORTED",
        E,
        "    if verdict is Verdict.INCONCLUSIVE:\n        return Standing.INCONCLUSIVE\n    return Standing.NOT_EVALUATED",
        "    if verdict is Verdict.INCONCLUSIVE:\n        return Standing.INCONCLUSIVE\n    return Standing.SUPPORTED",
    ),
    (
        "missing evidence -> CONTRADICTED",
        E,
        "    if verdict is Verdict.INCONCLUSIVE:\n        return Standing.INCONCLUSIVE\n    return Standing.NOT_EVALUATED",
        "    if verdict is Verdict.INCONCLUSIVE:\n        return Standing.INCONCLUSIVE\n    return Standing.CONTRADICTED",
    ),
    (
        "INCONCLUSIVE evidence ignored",
        E,
        "        weak = [e for e in matched if e.outcome not in _DECISIVE]",
        "        weak: list[ResultEntry] = []",
    ),
    (
        "re-derivation dropped",
        V,
        "        protocol = entry.spec.protocol\n        if protocol not in PROTOCOLS:",
        "        return None\n        protocol = entry.spec.protocol\n        if protocol not in PROTOCOLS:",
    ),
    (
        "concept validation re-derivation dropped",
        V,
        "            encoding, uses = verify_validation_trace(trace, locate)",
        "            encoding, uses = None, ()",
    ),
    (
        "alternative thresholds ignored",
        V,
        "        if key in self._alternatives:\n            return self._alternatives[key]",
        "        return None",
    ),
    (
        "decodable_not_used not flagged",
        E,
        "            and not any(e.outcome is Outcome.SUPPORTS for e in uses)\n        ):",
        "            and False\n        ):",
    ),
    (
        "generated label promoted",
        E,
        "        if concept.label_source is LabelSource.GENERATED:",
        "        if False:",
    ),
    (
        "concept test disagreement ignored",
        E,
        "            if sup and con:\n                disagreement = True",
        "            if False:\n                disagreement = True",
    ),
    (
        "per-sample targets ignored",
        E,
        "    if not c.sample_targets:\n        return c.target",
        "    if True:\n        return c.target",
    ),
    ("verify_report never mismatches", I, "    if stored != expected:", "    if False:"),
    (
        "cross-claim protocol disagreement dropped",
        E,
        "                if pair != {Standing.SUPPORTED, Standing.CONTRADICTED}:",
        "                if True:",
    ),
    (
        "attribution/intervention disagreement dropped",
        E,
        "            if agree:\n",
        "            if False:\n",
    ),
]
rows = []
ONLY = set(sys.argv[3].split("|")) if len(sys.argv) > 3 else None
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
                "tests/test_audit.py",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        killed = r.returncode != 0
        failing = [l for l in r.stdout.splitlines() if l.startswith("FAILED")][:1]
    finally:
        p.write_text(src)
    rows.append({"mutation": name, "result": "killed" if killed else "SURVIVED", "by": failing})
    print(name, "killed" if killed else "SURVIVED", failing, flush=True)
ONLY = set(sys.argv[3].split("|")) if len(sys.argv) > 3 else None
Path(sys.argv[2]).write_text(json.dumps(rows, indent=1))
