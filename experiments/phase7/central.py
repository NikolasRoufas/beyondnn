"""Phase 7 central experiment (docs/PHASE_7_PLAN.md §29.1, deviations D9-D11).

For each model/site (A = MLP/breast_cancer input + net.1; B = CNN/digits pixels + relu2;
C = BERT-tiny/SST-2 tokens) and every Phase-5.5 held-out sample (60/60/40, seed 1234):

* approaches: gradient (G), integrated gradients (IG), seeded random declared units (R);
* for k at p in {5, 10, 20}% and each method: comprehensiveness (count controls, N=50,
  min_fraction_below 0.95) under r1-r3, sufficiency (count controls, min_fraction_above
  0.95) under r1-r3, comprehensiveness with magnitude-matched controls under r1, and
  comprehensiveness without controls under r1; threshold 0.5 x margin;
* one audit per model/site under a pre-registered plan (claims *_necessary /
  *_sufficient with declared invariances and alternative thresholds x0.5 / x1.5);
* the same plan audited over the attribution traces alone (attribution-only audit);
* the naive single-configuration SUPPORTS count (IG, r1, k at p=10%, count controls).

Traces of the first 3 samples per model/site are saved for the reload check (D10).

Usage (experiment env, from experiments/phase5_5):
    python ../phase7/central.py A|B|C
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase5_5"))
import common  # noqa: E402
import run_faithfulness as RF  # noqa: E402
import torch  # noqa: E402

import beyondnn as bnn  # noqa: E402
from beyondnn.core.samples import sample_id  # noqa: E402

A, F, AU = bnn.attribution, bnn.faithfulness, bnn.audits
RESULTS = HERE / "results"
ARTIFACTS = HERE / "artifacts"
RESULTS.mkdir(exist_ok=True)
ARTIFACTS.mkdir(exist_ok=True)

PS = (0.05, 0.10, 0.20)
N_CONTROLS = 50
PRIMARY_T = 0.5
FRACTION = 0.95
METHODS = ("gradient", "integrated_gradients", "random")
SAVE_FIRST = 3  # D10


def ks_for(n: int) -> dict[float, int]:
    return {p: max(1, math.ceil(p * n)) for p in PS}


def run_sample(
    model: Any, setting: Any, sample: Any, path: str
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    """All tests for one sample and site. Returns (test results, attribution results, meta)."""
    kw = sample.kwargs
    with torch.no_grad():
        out = model(*sample.inputs, **kw)
    logits = out.logits if path else out
    metric, pred, runner, margin = common.margin_target(logits, path=path)
    methods = RF._methods(setting)
    attrs = {
        name: A.attribute(
            model,
            *sample.inputs,
            target=metric,
            method=methods[name],
            at=setting.at,
            model_kwargs=kw,
        )
        for name in ("gradient", "integrated_gradients")
    }
    ax = setting.unit_axes
    n = F.ranking(
        attrs["gradient"], unit_axes=ax, reduce=RF.REDUCE["gradient"] if ax else None
    ).n_units
    perm = torch.randperm(
        n, generator=torch.Generator().manual_seed(RF.RANDOM_SEED_BASE + sample.index)
    ).tolist()
    ks = ks_for(n)
    results: list[Any] = []
    naive: dict[str, Any] = {}
    reps = setting.replacements(sample.index, sample.inputs)
    for rname, rep in reps.items():
        r1 = rname.startswith("r1")
        for p, k in ks.items():
            for method in METHODS:
                if method == "random":
                    selection = F.units(setting.at, tuple(perm[:k]), n_units=n, unit_axes=ax)
                    attributions: list[Any] = []
                else:
                    selection = F.top_k(
                        attrs[method], k=k, unit_axes=ax, reduce=RF.REDUCE[method] if ax else None
                    )
                    attributions = [attrs[method]]
                tests = [("comprehensiveness", "count"), ("sufficiency", "count")]
                if r1:
                    tests += [("comprehensiveness", "magnitude"), ("comprehensiveness", "none")]
                for name, match in tests:
                    controls = (
                        None
                        if match == "none"
                        else F.controls(N_CONTROLS, seed=10000 + sample.index, match=match)
                    )
                    if name == "comprehensiveness":
                        extra = {} if controls is None else {"min_fraction_below": FRACTION}
                        template = F.comprehensiveness(
                            target=metric,
                            min_drop=PRIMARY_T * margin,
                            statement=f"the {method} top-{k} units are necessary for the margin",
                            controls=controls,
                            replacement=rep,
                            **extra,
                        )
                    else:
                        template = F.sufficiency(
                            target=metric,
                            max_drop=PRIMARY_T * margin,
                            statement=f"the {method} top-{k} units suffice for the margin",
                            controls=controls,
                            replacement=rep,
                            min_fraction_above=FRACTION,
                        )
                    res = F.run(
                        model,
                        *sample.inputs,
                        test=template,
                        selection=selection,
                        attributions=attributions,
                        model_kwargs=kw,
                    )
                    results.append(res)
                    if (
                        method == "integrated_gradients"
                        and r1
                        and p == 0.10
                        and name == "comprehensiveness"
                        and match == "count"
                    ):
                        naive = {
                            "outcome": res.outcome.value,
                            "drop": res.drop,
                            "k": k,
                            "units": list(res.selection.selected),  # descriptive (NLP §30)
                        }
    meta = {
        "index": sample.index,
        "sample_id": sample_id(*sample.inputs, model_kwargs=kw or None),
        "pred": pred,
        "runner_up": runner,
        "margin": margin,
        "n_units": n,
        "ks": {str(p): k for p, k in ks.items()},
        "naive_ig_r1_p10_count": naive,
        "text": sample.text,
        "tokens": None,
    }
    return results, list(attrs.values()), meta


def plan_for(
    model: Any, setting: Any, targets: dict[str, Any], method_names: dict[str, str]
) -> Any:
    """Every sample's own margin target (pred vs runner-up, fixed from its clean pass) is
    declared per sample (deviation D13)."""
    site = "input" if setting.module is None else setting.module
    claims = []
    for tag, method in method_names.items():
        sel = AU.selection(_site(setting), method=method, k=None)
        claims.append(
            AU.claim(
                f"{tag}_necessary",
                statement=f"the {tag} top-k units at {site} are necessary for the margin",
                relation="necessary_for",
                target=None,
                sample_targets=targets,
                scope="instance",
                requirement="necessity_v1",
                selection=sel,
                invariant_over=[
                    AU.invariance("replacement", min_values=3),
                    AU.invariance("k", min_values=2),
                    AU.invariance("null", min_values=2),
                ],
            )
        )
        claims.append(
            AU.claim(
                f"{tag}_sufficient",
                statement=f"the {tag} top-k units at {site} suffice for the margin",
                relation="sufficient_for",
                target=None,
                sample_targets=targets,
                scope="instance",
                requirement="sufficiency_v1",
                selection=sel,
                invariant_over=[
                    AU.invariance("replacement", min_values=3),
                    AU.invariance("k", min_values=2),
                ],
            )
        )
    return AU.plan(
        name=f"central_{setting.model_id}_{site.replace('.', '_')}".lower(),
        checkpoint=AU.checkpoint_of(model),
        declared_model=None,
        samples=list(targets),
        datasets=[],
        claims=claims,
        requirements=[
            AU.requirement(
                "necessity_v1",
                policy=F.COMPREHENSIVENESS_POLICY,
                controls=True,
                alternatives=[
                    AU.alternative("comprehensiveness", "min_drop", factor=0.5),
                    AU.alternative("comprehensiveness", "min_drop", factor=1.5),
                ],
            ),
            AU.requirement(
                "sufficiency_v1",
                policy=F.SUFFICIENCY_POLICY,
                controls=True,
                alternatives=[
                    AU.alternative("sufficiency", "max_drop", factor=0.5),
                    AU.alternative("sufficiency", "max_drop", factor=1.5),
                ],
            ),
        ],
        concepts=[],
        counterexamples=AU.counterexample_rule(
            max_counterexample_fraction=None,
            max_false_positive_rate=None,
            max_false_negative_rate=None,
        ),
        naive_auroc=None,
    )


def _site(setting: Any) -> Any:
    from beyondnn.schema import Site, SiteIO

    if setting.module is None:
        return Site(module="", io=SiteIO.INPUT, output_path="args[0]")
    return Site(module=setting.module)


def summarise(report: Any, metas: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"claims": {}}
    by_sample = {m["sample_id"]: m for m in metas}
    for c in report.claims:
        axes: dict[str, int] = {}
        kinds: dict[str, int] = {}
        for g in c.groups:
            seen: set[tuple[str, str]] = set()
            for f in g.findings:
                seen.add((f.kind.value, f.axis or ""))
            for kind, axis in seen:
                kinds[kind] = kinds.get(kind, 0) + 1
                if kind == "assumption_sensitive":
                    axes[axis] = axes.get(axis, 0) + 1
        out["claims"][c.name] = {
            "distribution": dict(c.distribution),
            "samples_with_finding_kind": dict(sorted(kinds.items())),
            "samples_sensitive_by_axis": dict(sorted(axes.items())),
            "counterexamples": [by_sample[s]["index"] for s in c.counterexamples if s in by_sample],
            "per_sample": {
                str(by_sample[g.sample]["index"]): g.standing.value
                for g in c.groups
                if g.sample in by_sample
            },
        }
    disagreements: dict[str, int] = {}
    for c in report.claims:
        for g in c.groups:
            for f in g.findings:
                if f.kind.value == "protocol_disagreement":
                    disagreements[c.name] = disagreements.get(c.name, 0) + 1
    out["protocol_disagreement_samples"] = disagreements
    out["coverage"] = json.loads(json.dumps(report.to_dict()["coverage"]))
    out["evidence"] = json.loads(json.dumps(report.to_dict()["evidence"]))
    out["report_findings"] = [
        {"kind": f.kind.value, "code": f.code, "records": len(f.records)} for f in report.findings
    ]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["A", "B", "C"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, samples, settings, info, path = getattr(RF, f"settings_{args.model}")()
    if args.limit:
        samples = samples[: args.limit]
    payload: dict[str, Any] = {
        "environment": common.environment(),
        "model": info,
        "settings": {},
        "plan_section": "docs/PHASE_7_PLAN.md §29.1 (D9-D11)",
    }
    for setting in settings:
        t0 = time.perf_counter()
        results: list[Any] = []
        attributions: list[Any] = []
        metas: list[dict[str, Any]] = []
        targets: dict[str, Any] = {}
        method_names: dict[str, str] = {}
        for i, sample in enumerate(samples):
            res, attrs, meta = run_sample(model, setting, sample, path)
            results += res
            attributions += attrs
            metas.append(meta)
            targets[meta["sample_id"]] = res[0].claim.target
            assert all(r.claim.target == res[0].claim.target for r in res)
            method_names = {
                "g": attrs[0].record.method.name,
                "ig": attrs[1].record.method.name,
                "r": "declared",
            }
            if i < SAVE_FIRST:
                site_dir = ARTIFACTS / f"{setting.model_id}_{setting.site}" / str(sample.index)
                site_dir.mkdir(parents=True, exist_ok=True)
                for j, r in enumerate(res):
                    if not (site_dir / f"t{j:03d}").exists():
                        r.trace.save(site_dir / f"t{j:03d}")
                for j, a in enumerate(attrs):
                    if not (site_dir / f"a{j}").exists():
                        a.trace.save(site_dir / f"a{j}")
            print(setting.site, i, sample.index, meta["naive_ig_r1_p10_count"], flush=True)
        run_seconds = time.perf_counter() - t0
        sample_ids = [m["sample_id"] for m in metas]
        recorded = {r.claim.estimand.sample_id for r in results}
        assert recorded == set(sample_ids), "sample identities differ from the plan's"
        assert set(targets) == set(sample_ids)
        plan = plan_for(model, setting, targets, method_names)
        (ARTIFACTS / f"plan_{setting.model_id}_{setting.site}.json").write_text(
            bnn.schema.to_json(plan)
        )
        t1 = time.perf_counter()
        report = bnn.audit(results, plan=plan)
        audit_seconds = time.perf_counter() - t1
        t2 = time.perf_counter()
        AU.verify_report(report.to_dict(), results, plan)
        verify_seconds = time.perf_counter() - t2
        attribution_only = bnn.audit(attributions, plan=plan)
        suffix = f"_limit{args.limit}" if args.limit else ""
        report_path = ARTIFACTS / f"central_{setting.model_id}_{setting.site}{suffix}_report.json"
        if report_path.exists():
            report_path.unlink()  # a stale report from an earlier run of this script
        report.save(report_path)
        naive = sum(1 for m in metas if m["naive_ig_r1_p10_count"].get("outcome") == "supports")
        payload["settings"][setting.site] = {
            "n_samples": len(samples),
            "plan_id": plan.id,
            "n_results": len(results),
            "run_seconds": run_seconds,
            "audit_seconds": audit_seconds,
            "verify_report_seconds": verify_seconds,
            "verify_report": "passed",
            "naive_single_configuration_supports": naive,
            "audit": summarise(report, metas),
            "attribution_only": summarise(attribution_only, metas),
            "samples": metas,
        }
        print(
            setting.site,
            "naive",
            naive,
            json.dumps(
                {
                    c: v["distribution"]
                    for c, v in payload["settings"][setting.site]["audit"]["claims"].items()
                }
            ),
            flush=True,
        )
        del results, attributions, report, attribution_only
        gc.collect()
    name = f"central_{args.model}" + (f"_limit{args.limit}" if args.limit else "")
    (RESULTS / f"{name}.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True, default=str)
    )


if __name__ == "__main__":
    main()
