"""TNNLS revision, Stage 3 (experiment S3-IBSPEC): InterpBench re-analysed against
specification-derived labels (artifact-only; no model is run).

Phase 7.5 scored PRIMARY standings against labels that kept only instances on which the
trained (LL) model's own resample ablation agreed with the specification. That filter makes the
agreement partly circular. Here two labels that do NOT use any intervention on the trained
network are used instead (both are reconstructed from the committed rows + case metadata):

* SPEC-INPUT: per (node, input), "necessary" iff the node is in the benchmark circuit AND the
  interchange of the corresponding node of the compiled high-level (Tracr) program changes the
  HL output. Every frozen row encodes it: rows labelled necessary / not_necessary agree with it,
  and every 'ambiguous' row is a non-circuit node (verified below), whose SPEC-INPUT label is
  not_necessary by construction (SIIT: non-circuit nodes are not part of the mechanism).
* SPEC-STRUCT: node-level circuit membership (``circuit_heads`` / ``circuit_mlps``).

Usage: python3 experiments/tnnls_revision/stage3/interpbench_specification_labels.py
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
P75 = ROOT / "experiments/phase7_5/results"
OUT = Path(__file__).resolve().parent / "results"
CONFIGS = (
    "tensor/resample|none|recorded",
    "tensor/mean|none|recorded",
    "zero|none|recorded",
)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 5), round(min(1.0, c + h), 5)]


def in_circuit(row: dict[str, Any], case: dict[str, Any]) -> bool:
    layer_s, part = row["claim"].split("_")[:2]
    layer = int(layer_s[1:])
    if part == "mlp":
        return layer in set(case["circuit_mlps"])
    return (layer, int(part[1:])) in {tuple(h) for h in case["circuit_heads"]}


def main() -> None:
    cases = dict(json.loads((P75 / "external_interpbench_heldout.json").read_text())["cases"])
    dv1 = json.loads((P75 / "external_interpbench_heldout_case124_dv1.json").read_text())
    cases["124"] = dv1["cases"]["124"]
    rows = []
    for cid, case in cases.items():
        if "error" in case or not case.get("benchmark_reliable", False):
            continue
        for r in case["rows"]:
            if r["kind"] == "selection":
                continue
            inc = in_circuit(r, case)
            if r["gt"] == "ambiguous" and inc:
                raise SystemExit("ambiguous circuit node: SPEC-INPUT label not recoverable")
            spec = "necessary" if r["gt"] == "necessary" else "not_necessary"
            rows.append(r | {"case": cid, "in_circuit": inc, "spec": spec})
    out: dict[str, Any] = {
        "kind": "artifact-only re-analysis; no intervention on the trained model defines a label",
        "cases": len({r["case"] for r in rows}),
        "node_instances": len(rows),
        "spec_input_labels": dict(Counter(r["spec"] for r in rows)),
        "frozen_label_x_membership": {
            f"{g}|in_circuit={m}": n
            for (g, m), n in Counter((r["gt"], r["in_circuit"]) for r in rows).items()
        },
    }
    prim = {}
    for spec in ("necessary", "not_necessary"):
        xs = [r for r in rows if r["spec"] == spec]
        st = Counter(r["standing"] for r in xs)
        prim[spec] = {"n": len(xs), "standings": dict(st)}
    k_fp, n_neg = prim["not_necessary"]["standings"].get("supported", 0), prim["not_necessary"]["n"]
    k_tp, n_pos = prim["necessary"]["standings"].get("supported", 0), prim["necessary"]["n"]
    out["primary_vs_spec_input"] = prim | {
        "tp_rate": [k_tp, n_pos, wilson(k_tp, n_pos)],
        "fp_rate": [k_fp, n_neg, wilson(k_fp, n_neg)],
    }
    fps = [r for r in rows if r["spec"] == "not_necessary" and r["standing"] == "supported"]
    out["primary_false_positives_vs_spec"] = {
        "n": len(fps),
        "all_non_circuit": all(not r["in_circuit"] for r in fps),
        "all_frozen_ambiguous_i.e._LL_resample_flips": all(r["gt"] == "ambiguous" for r in fps),
        "cases": dict(Counter(r["case"] for r in fps)),
        "other_configs_mean_zero": {
            "|".join(map(str, k)): v
            for k, v in Counter(
                tuple(r["configs"].get(c) for c in CONFIGS[1:]) for r in fps
            ).items()
        },
    }
    per = {}
    for cfg in CONFIGS:
        cell = {}
        for spec in ("necessary", "not_necessary"):
            xs = [r for r in rows if r["spec"] == spec and cfg in r["configs"]]
            k = sum(r["configs"][cfg] == "supports" for r in xs)
            cell[spec] = [k, len(xs), wilson(k, len(xs))]
        per[cfg] = cell
    out["per_configuration_vs_spec_input"] = per
    struct = {}
    for m in (True, False):
        xs = [r for r in rows if r["in_circuit"] == m]
        struct[f"in_circuit={m}"] = {
            "n": len(xs),
            "primary_supported": sum(r["standing"] == "supported" for r in xs),
            "nodes": len({(r["case"], r["claim"]) for r in xs}),
        }
    out["primary_vs_spec_struct"] = struct
    out["clustering_note"] = (
        "instances are nested in nodes and in 18 cases (40 inputs each); Wilson intervals are "
        "unclustered and descriptive"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "interpbench_specification_labels.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
