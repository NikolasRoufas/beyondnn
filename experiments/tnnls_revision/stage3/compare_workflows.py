"""TNNLS revision, Stage 3 (S3-CMP; protocol docs/tnnls_revision/protocols/P2_*): BeyondNN audit
versus conventional workflows on the same measurements, with injected workflow faults; plus the
cost of one PRIMARY IG necessity test with and without the BeyondNN layer.

Usage (Phase-5.5 env, from experiments/phase5_5): python ../tnnls_revision/stage3/compare_workflows.py OUT_DIR CKPT_DIR
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for sub in ("phase5_5", "phase7_5", "phase7_75"):
    sys.path.insert(0, str(ROOT / "experiments" / sub))
sys.path.insert(0, str(HERE.parent))

import central75 as C75  # noqa: E402
import central775 as C775  # noqa: E402
import torch  # noqa: E402
from captum.attr import IntegratedGradients  # noqa: E402
from run_central_configs import train_seeded  # noqa: E402

import beyondnn as bnn  # noqa: E402

A, F, AU = bnn.attribution, bnn.faithfulness, bnn.audits
N, N_SUB, N_AFF = 20, 5, 5
TESTS_PRIMARY = [
    ("comprehensiveness", "count"),
    ("sufficiency", "count"),
    ("comprehensiveness", "magnitude"),
    ("comprehensiveness", "none"),
]
TESTS_OTHER = [("comprehensiveness", "count"), ("sufficiency", "count")]


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()


def ckpt_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------ evidence + flat log


def generate(
    model: Any, setting: Any, sample: Any, filed_as: Any
) -> tuple[list[Any], list[Any], dict[str, Any], list[dict[str, Any]]]:
    """Frozen evidence for ``sample`` and the conventional log a careful researcher would write
    in the same loop (the loop order of ``central75.run_sample`` is replicated for labels)."""
    res, attrs, meta = C775.run_sample(model, setting, sample, "", None)
    reps = C75.named(setting.replacements(sample.index, sample.inputs))
    roles = C75.REPLACEMENT_ROLES[(setting.model_id, setting.site)]
    primary_rep = next(r for r, role in roles.items() if role == "primary")
    labels = []
    for rname in reps:
        for p, k in C75.ks_for(meta["n_units"]).items():
            for method in C75.METHODS:
                for test, match in TESTS_PRIMARY if rname == primary_rep else TESTS_OTHER:
                    labels.append((rname, p, k, method, test, match))
    assert len(labels) == len(res)
    log = []
    ck, ih = ckpt_hash(model), sha(sample.inputs[0])
    tau = C75.PRIMARY_T * meta["margin"]
    for (rname, p, k, method, test, match), r in zip(labels, res, strict=True):
        s = r.statistics
        log.append(
            {
                "kind": "intervention",
                "checkpoint_sha256": ck,
                "input_sha256": ih,
                "sample": filed_as.index,
                "method": method,
                "p": p,
                "k": k,
                "replacement": rname,
                "replacement_role": roles[rname],
                "test": test,
                "controls": match,
                "n_controls": int(s.get("n_controls", 0) or 0),
                "tau": tau,
                "drop": float(s["drop"]) if s.get("drop") is not None else None,
                "control_drops": list(s.get("control_drops") or []),
                "outcome": r.outcome.value,
            }
        )
    return res, attrs, meta, log


# ------------------------------------------------------------------ conventional checkers

N_UNITS = 30


def b0_conclusion(log: list[dict[str, Any]], sample: int) -> str | None:
    """Single-configuration report: PRIMARY IG comprehensiveness, k at 10%, count controls."""
    for e in log:
        if (
            e["sample"] == sample
            and e["kind"] == "intervention"
            and e["method"] == "integrated_gradients"
            and e["p"] == 0.10
            and e["replacement_role"] == "primary"
            and e["test"] == "comprehensiveness"
            and e["controls"] == "count"
        ):
            return str(e["outcome"])
    for e in log:
        if e["sample"] == sample and e["kind"] == "attribution":
            return "supports"  # "IG ranks these features highest, so they are necessary"
    return None


def b1_checks(
    log: list[dict[str, Any]], sample: int, declared_ckpt: str, declared_input: str
) -> list[str]:
    """A careful researcher's explicit checks on the flat log (c1-c7; protocol P2)."""
    flags = []
    mine = [e for e in log if e["sample"] == sample]
    iv = [e for e in mine if e["kind"] == "intervention"]
    if not iv:
        flags.append("c3_no_intervention_evidence")
        return flags
    if any(e["checkpoint_sha256"] != declared_ckpt for e in iv):
        flags.append("c1_checkpoint_mismatch")
    if any(e["input_sha256"] != declared_input for e in iv):
        flags.append("c2_input_mismatch")
    prim = [
        e
        for e in iv
        if e["method"] == "integrated_gradients"
        and e["p"] == 0.10
        and e["replacement_role"] == "primary"
        and e["test"] == "comprehensiveness"
    ]
    pc = [e for e in prim if e["controls"] == "count"]
    if not pc or pc[0]["n_controls"] == 0:
        flags.append("c4_primary_without_controls")
    for e in prim:
        if e["k"] != max(1, math.ceil(0.10 * N_UNITS)):
            flags.append("c5_wrong_k")
            break
    for e in iv:
        if e["test"] != "comprehensiveness" or e["drop"] is None:
            continue
        cd = e["control_drops"]
        ok = e["drop"] >= e["tau"] and (not cd or sum(d < e["drop"] for d in cd) / len(cd) >= 0.95)
        noop = e["drop"] == 0.0
        expected = "inconclusive" if noop else ("supports" if ok else "contradicts")
        if expected != e["outcome"] and not (
            expected == "contradicts" and e["outcome"] == "inconclusive"
        ):
            flags.append("c6_outcome_not_reproducible")
            break
    if pc and pc[0]["outcome"] == "supports":
        alt = [
            e
            for e in iv
            if e["method"] == "integrated_gradients"
            and e["test"] == "comprehensiveness"
            and e is not pc[0]
            and e["replacement_role"] != "stress_test"
            and e["controls"] in ("count", "magnitude")
            and e["outcome"] == "contradicts"
        ]
        if alt:
            flags.append("c7_alternative_contradicts")
    return flags


B1_LOC = sum(
    1
    for line in inspect.getsource(b1_checks).splitlines()
    if line.strip() and not line.strip().startswith(("#", '"""'))
)
PLAN_LOC = sum(
    1
    for line in inspect.getsource(C75.plan_for).splitlines()
    + inspect.getsource(C75.roles_for).splitlines()
    if line.strip() and not line.strip().startswith(("#", '"""'))
)


# ------------------------------------------------------------------ BeyondNN side


def audit_ig(results: list[Any], plan: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    report = bnn.audit(results, plan=plan)
    ig = next(c for c in report.claims if c.name == "ig_necessary")
    per = {
        g.sample: {"standing": g.standing.value, "codes": sorted({f.code for f in g.findings})}
        for g in ig.groups
    }
    meta = {
        "results": report.evidence.results,
        "results_excluded": report.evidence.results_excluded,
        "report_findings": sorted({f.code for f in report.findings})
        if hasattr(report, "findings")
        else [],
    }
    return per, meta


def main() -> None:
    out, ckpt_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    model, samples, settings, info, _ = C75.build("A", "heldout")
    setting = settings[0]
    assert setting.site == "input"
    study, subs = samples[:N], samples[N : N + N_SUB]
    model1, _, info1 = train_seeded("A", seed=1, ckpt_dir=ckpt_dir)
    declared_ck = ckpt_hash(model)
    clean: dict[int, tuple[list[Any], list[Any], dict[str, Any], list[dict[str, Any]]]] = {}
    for s in study:
        clean[s.index] = generate(model, setting, s, s)
    metas = [clean[s.index][2] for s in study]
    targets = {clean[s.index][2]["sample_id"]: clean[s.index][0][0].claim.target for s in study}
    methods = {
        "g": clean[study[0].index][1][0].record.method.name,
        "ig": clean[study[0].index][1][1].record.method.name,
        "r": "declared",
    }
    plan = C775.plan_for(model, setting, targets, methods, metas, None)
    sid = {s.index: clean[s.index][2]["sample_id"] for s in study}
    din = {s.index: sha(s.inputs[0]) for s in study}

    def all_results(src: dict[int, Any]) -> list[Any]:
        return [r for s in study for r in src[s.index][0]]

    def all_log(src: dict[int, Any]) -> list[dict[str, Any]]:
        return [e for s in study for e in src[s.index][3]]

    base_per, base_meta = audit_ig(all_results(clean), plan)
    affected = [s.index for s in study[:N_AFF]]
    scenarios: dict[str, Any] = {}

    def record(
        name: str,
        results: list[Any] | None,
        log: list[dict[str, Any]],
        aff: list[int],
        per: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        if per is None and results is not None:
            per, meta = audit_ig(results, plan)
        rows = []
        for s in study:
            i = s.index
            b = per.get(sid[i]) if per is not None else None
            clean_b = base_per[sid[i]]
            bnn_signal = (
                error is not None
                or b is None
                or b["standing"] in ("not_evaluated", "unsupported", "inconclusive", "mixed")
                or b != clean_b
            )
            if name == "F6_configuration_dependence":
                bnn_signal = (
                    b is not None
                    and b["standing"] == "supported"
                    and "alternative_reverses" in b["codes"]
                )
            rows.append(
                {
                    "sample": i,
                    "affected": i in aff,
                    "b0_conclusion": b0_conclusion(log, i),
                    "b1_flags": b1_checks(log, i, declared_ck, din[i]),
                    "bnn": b,
                    "bnn_signal": bnn_signal,
                }
            )

        def tally(flag: str) -> dict[str, int]:
            sig = (lambda r: bool(r["b1_flags"])) if flag == "b1" else (lambda r: r["bnn_signal"])
            return {
                "detected": sum(r["affected"] and sig(r) for r in rows),
                "missed": sum(r["affected"] and not sig(r) for r in rows),
                "false_alerts": sum((not r["affected"]) and sig(r) for r in rows),
            }

        scenarios[name] = {
            "affected": len(aff),
            "b0": {
                "detected": 0,
                "missed": len(aff),
                "false_alerts": 0,
                "note": "B0 has no checks: it reports its conclusion",
            },
            "b1": tally("b1"),
            "beyondnn": tally("bnn"),
            "beyondnn_error": error,
            "beyondnn_report": meta,
            "b1_flag_codes": dict(Counter(f for r in rows if r["affected"] for f in r["b1_flags"])),
            "bnn_standings_affected": dict(
                Counter((r["bnn"] or {}).get("standing", "none") for r in rows if r["affected"])
            ),
            "bnn_codes_affected": dict(
                Counter(c for r in rows if r["affected"] and r["bnn"] for c in r["bnn"]["codes"])
            ),
            "rows": rows,
        }
        print(
            name,
            json.dumps(
                {k: scenarios[name][k] for k in ("b1", "beyondnn", "bnn_standings_affected")}
            ),
            flush=True,
        )

    # F0 clean (false-alert reference: everything is unaffected)
    record("F0_clean", all_results(clean), all_log(clean), [], per=base_per, meta=base_meta)
    # F1 attribution only
    attrs_only = [a for s in study for a in clean[s.index][1]]
    f1_log = [
        {
            "kind": "attribution",
            "sample": s.index,
            "checkpoint_sha256": declared_ck,
            "input_sha256": din[s.index],
        }
        for s in study
    ]
    record("F1_attribution_as_necessity", attrs_only, f1_log, [s.index for s in study])
    # F2 wrong checkpoint
    f2 = dict(clean)
    for s in study[:N_AFF]:
        f2[s.index] = generate(model1, setting, s, s)
    record("F2_wrong_checkpoint", all_results(f2), all_log(f2), affected)
    # F3 wrong input filed under the declared sample
    f3 = dict(clean)
    for s, sub in zip(study[:N_AFF], subs, strict=True):
        f3[s.index] = generate(model, setting, sub, s)
    record("F3_wrong_input", all_results(f3), all_log(f3), affected)
    # F4 wrong selection size: PRIMARY-k results removed, k=5% filed as the PRIMARY claim
    f4_res, f4_log = [], []
    for s in study:
        res, _, meta, log = clean[s.index]
        pk = meta["ks"][str(C75.PRIMARY_P)]
        for r, e in zip(res, log, strict=True):
            if s.index in affected and e["k"] == pk:
                continue
            e2 = dict(e)
            if s.index in affected and e["p"] == 0.05:
                e2["p"] = 0.10  # mislabelled as the 10% claim
            f4_res.append(r)
            f4_log.append(e2)
    record("F4_wrong_selection_size", f4_res, f4_log, affected)
    # F5 omitted controls: only uncontrolled PRIMARY comprehensiveness results
    f5_res, f5_log = [], []
    for s in study:
        res, _, _, log = clean[s.index]
        for r, e in zip(res, log, strict=True):
            if e["controls"] == "none":
                f5_res.append(r)
                e2 = dict(e)
                e2["controls"] = "count"  # reported as if controlled
                f5_log.append(e2)
    f5_aff = [s.index for s in study if b0_conclusion(f5_log, s.index) == "supports"]
    record("F5_omitted_controls", f5_res, f5_log, f5_aff)
    # F6 configuration dependence (clean evidence)
    f6_aff = [
        s.index
        for s in study
        if base_per[sid[s.index]]["standing"] == "supported"
        and any(
            e["method"] == "integrated_gradients"
            and e["test"] == "comprehensiveness"
            and e["replacement_role"] == "alternative"
            and e["outcome"] == "contradicts"
            for e in clean[s.index][3]
        )
    ]
    f6_aff_bnn = [
        s.index
        for s in study
        if base_per[sid[s.index]]["standing"] == "supported"
        and "alternative_reverses" in base_per[sid[s.index]]["codes"]
    ]
    record(
        "F6_configuration_dependence",
        all_results(clean),
        all_log(clean),
        sorted(set(f6_aff) | set(f6_aff_bnn)),
        per=base_per,
        meta=base_meta,
    )
    scenarios["F6_configuration_dependence"]["affected_definitions"] = {
        "flat_log_rule": f6_aff,
        "beyondnn_roles": f6_aff_bnn,
    }
    # F7 tampered outcome (contradicts -> supports on the PRIMARY IG result)
    targets_t = [
        s.index for s in study if b0_conclusion(clean[s.index][3], s.index) == "contradicts"
    ][:N_AFF]
    f7_log = []
    tamper_ids = set()
    for s in study:
        res, _, _, log = clean[s.index]
        for r, e in zip(res, log, strict=True):
            e2 = dict(e)
            if (
                s.index in targets_t
                and e["method"] == "integrated_gradients"
                and e["p"] == 0.10
                and e["replacement_role"] == "primary"
                and e["test"] == "comprehensiveness"
                and e["controls"] == "count"
            ):
                e2["outcome"] = "supports"
                tamper_ids.add(r.result.id)
            f7_log.append(e2)
    ev_dir = out / "f7_evidence"
    AU.save_evidence(all_results(clean), ev_dir)
    edited = 0
    for tj in ev_dir.glob("trace*/trace.json"):
        doc = json.loads(tj.read_text())
        hit = False
        for rec in doc.get("records", []):
            if (
                rec.get("id") in tamper_ids
                and isinstance(rec.get("data"), dict)
                and "outcome" in rec["data"]
            ):
                rec["data"]["outcome"] = "supports"
                hit = True
        if hit:
            tj.write_text(json.dumps(doc, indent=2, sort_keys=True))
            edited += 1
    err = None
    per7 = meta7 = None
    try:
        per7, meta7 = audit_ig(AU.load_evidence(ev_dir), plan)
    except Exception as exc:  # the refusal itself is the detection
        err = f"{type(exc).__name__}: {str(exc)[:300]}"
    record(
        "F7_tampered_outcome",
        None,
        f7_log,
        targets_t,
        per=per7 if per7 is not None else {},
        meta=(meta7 or {}) | {"edited_trace_files": edited},
        error=err,
    )
    # --------------------------------------------------------------- cost: one PRIMARY IG test
    cost = []
    reps_by = {}
    for s in study:
        reps_by[s.index] = C75.named(setting.replacements(s.index, s.inputs))
    for s in study:
        x = s.inputs[0]
        with torch.no_grad():
            logits = model(x)
        metric, pred, runner, margin = __import__("common").margin_target(logits)
        k = max(1, math.ceil(0.10 * N_UNITS))
        rep = reps_by[s.index]["r1_zero_train_mean"]
        times_b, times_p = [], []
        drop_b = drop_p = None
        out_b = out_p = None
        for _ in range(3):
            t0 = time.perf_counter()
            attr = A.attribute(
                model,
                x,
                target=metric,
                method=C75.RF._methods(setting)["integrated_gradients"],
                at=setting.at,
            )
            ax = setting.unit_axes
            sel = F.top_k(
                attr,
                k=k,
                unit_axes=ax,
                reduce=C75.RF.REDUCE["integrated_gradients"] if ax else None,
            )
            tmpl = F.comprehensiveness(
                target=metric,
                min_drop=0.5 * margin,
                replacement=rep,
                controls=F.controls(50, seed=10000 + s.index),
                statement="ig top-k necessary",
                min_fraction_below=0.95,
            )
            r = F.run(model, x, test=tmpl, selection=sel, attributions=[attr])
            times_b.append(time.perf_counter() - t0)
            drop_b, out_b = float(r.drop), r.outcome.value
            t1 = time.perf_counter()

            def fwd(inp: torch.Tensor) -> torch.Tensor:
                o = model(inp)
                return o[:, pred] - o[:, runner]

            ig = IntegratedGradients(fwd)
            att = ig.attribute(
                x, baselines=torch.zeros_like(x), n_steps=32, method="riemann_middle"
            )
            top = torch.topk(att[0], k).indices.tolist()
            with torch.no_grad():
                base = float(fwd(x))
                xs = x.clone()
                xs[0, top] = 0.0
                d = base - float(fwd(xs))
                g = torch.Generator().manual_seed(10000 + s.index)
                cds = []
                for _c in range(50):
                    idx = torch.randperm(N_UNITS, generator=g)[:k].tolist()
                    xc = x.clone()
                    xc[0, idx] = 0.0
                    cds.append(base - float(fwd(xc)))
            ok = d >= 0.5 * margin and sum(c < d for c in cds) / 50 >= 0.95
            times_p.append(time.perf_counter() - t1)
            drop_p, out_p = d, ("supports" if ok else "contradicts")
        cost.append(
            {
                "sample": s.index,
                "beyondnn_s": statistics.median(times_b),
                "plain_s": statistics.median(times_p),
                "drop_beyondnn": drop_b,
                "drop_plain": drop_p,
                "selection_equal": sorted(sel.selected) == sorted(top),
                "outcome_beyondnn": out_b,
                "outcome_plain": out_p,
            }
        )
    summary = {
        "protocol": "docs/tnnls_revision/protocols/P2_COMPARATIVE_EVALUATION_PROTOCOL.md",
        "model": {
            "frozen_digest": info["checkpoint_state_digest"],
            "seed1_digest": info1["checkpoint_state_digest"],
        },
        "n_study_samples": N,
        "burden_loc": {"b1_checks": B1_LOC, "beyondnn_plan_and_roles_reused": PLAN_LOC},
        "scenarios": {
            k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in scenarios.items()
        },
        "cost": {
            "per_sample": cost,
            "median_beyondnn_s": statistics.median(c["beyondnn_s"] for c in cost),
            "median_plain_s": statistics.median(c["plain_s"] for c in cost),
            "max_abs_drop_difference": max(abs(c["drop_beyondnn"] - c["drop_plain"]) for c in cost),
            "selection_equal": sum(c["selection_equal"] for c in cost),
            "outcome_agreement": sum(c["outcome_beyondnn"] == c["outcome_plain"] for c in cost),
        },
        "environment": __import__("common").environment(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    (out / "rows.json").write_text(
        json.dumps({k: v["rows"] for k, v in scenarios.items()}, indent=1, default=str)
    )
    print(
        json.dumps(
            {k: summary[k] for k in ("burden_loc",)}
            | {"cost": {k: v for k, v in summary["cost"].items() if k != "per_sample"}},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
