"""Phase 7 concept audit experiment (docs/PHASE_7_PLAN.md §29.2).

Re-runs the Phase-6 realistic concept tests (experiments/phase6/realistic.py settings,
unchanged: covariance- and isotropic-null encoding tests, use tests with zero and
train-mean references, POLICY_V1 validations) for A (K1-K4), B (K5, K6), C (K7-K9), keeps
every trace, and audits them under a plan declared before the tests run:

* each (concept, feature kind) asserted VALIDATED_CONCEPT under POLICY_V1, invariant over
  the null (2 values) and the replacement (2 values);
* per concept an ENCODES claim (finite sample: the test split; invariant over the null)
  and a DECREASES use claim (finite sample: the positive test subset; invariant over the
  replacement);
* two counterexample rules, both reported: strict (FP, FN <= 0.10) and lenient (<= 0.30);
* naive_auroc = 0.6.

A save -> load -> audit check is run on model A's traces (D10: disk).

Usage (experiment env, from experiments/phase5_5): python ../phase7/concepts.py A|B|C
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase5_5"))
sys.path.insert(0, str(HERE.parent / "phase6"))
import common  # noqa: E402
import realistic as R  # noqa: E402
import torch  # noqa: E402

import beyondnn as bnn  # noqa: E402
from beyondnn.concepts._core import sample_set_id  # noqa: E402
from beyondnn.schema import JsonMap, Subject, TargetSpec  # noqa: E402

C, AU = bnn.concepts, bnn.audits
RESULTS = HERE / "results"
ARTIFACTS = HERE / "artifacts"
RESULTS.mkdir(exist_ok=True)
ARTIFACTS.mkdir(exist_ok=True)
RULES = {
    "strict": AU.counterexample_rule(
        max_counterexample_fraction=None, max_false_positive_rate=0.10, max_false_negative_rate=0.10
    ),
    "lenient": AU.counterexample_rule(
        max_counterexample_fraction=None, max_false_positive_rate=0.30, max_false_negative_rate=0.30
    ),
}


def claims_for(tag: str, concept: Any, data: Any, target: Any) -> list[Any]:
    feature = concept.feature.record
    subject = Subject(site=feature.site, feature=feature.id)
    ds = data.record
    test = [ds.samples[i] for i in ds.indices("test")]
    positive = [ds.samples[i] for i in ds.indices("test", 1)]
    label = TargetSpec(
        metric="concept_label",
        params=JsonMap({"concept": concept.id, "dataset": ds.id, "split": "test"}),
    )
    return [
        AU.claim(
            f"{tag}_encodes",
            statement=f"{tag}: the feature encodes the concept on the held-out split",
            relation="encodes",
            target=label,
            scope="finite_sample",
            sample_set=sample_set_id(test),
            requirement="encoding",
            subject=subject,
            invariant_over=[AU.invariance("null", min_values=2)],
        ),
        AU.claim(
            f"{tag}_used",
            statement=f"{tag}: removing the feature decreases the target on concept samples",
            relation="decreases",
            target=target.spec.target(),
            scope="finite_sample",
            sample_set=sample_set_id(positive),
            requirement="use",
            subject=subject,
            invariant_over=[AU.invariance("replacement", min_values=2)],
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["A", "B", "C"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, runs, info = {"A": R.model_a, "B": R.model_b, "C": R.model_c}[args.model]()
    started = time.perf_counter()
    ecrit = C.encoding_criteria(min_fraction_below=0.95)
    ucrit = C.use_criteria(min_change=0.1, min_fraction_beyond_controls=0.95)
    evidence: list[Any] = []
    claims: list[Any] = []
    concepts: list[Any] = []
    datasets: set[str] = set()
    rows: dict[str, Any] = {}
    for ds, name, definition, site, axis, pooling, target, _desc in runs:
        datasets.add(ds.id)
        feats = {
            "direction": C.fit_direction(model, ds, site=site, axis=axis, pooling=pooling),
            "neuron": C.search_neurons(model, ds, site=site, axis=axis, pooling=pooling),
        }
        for kind, feature in feats.items():
            tag = f"{name.split()[0].lower()}_{kind}"
            concept = C.propose(feature, label=name, definition=definition, label_source="dataset")
            # the plan entries are declared before any test of this feature runs
            claims += claims_for(tag, concept, ds, target)
            concepts.append(
                AU.concept(
                    concept.id,
                    policy=C.POLICY_V1,
                    invariant_over=[
                        AU.invariance("null", min_values=2),
                        AU.invariance("replacement", min_values=2),
                    ],
                )
            )
            enc = C.encoding_test(
                model, concept, ds, controls=R.enc_controls("covariance"), criteria=ecrit
            )
            iso = C.encoding_test(
                model, concept, ds, controls=R.enc_controls("isotropic"), criteria=ecrit
            )
            ctl = (
                [C.random_neurons(50, seed=13)]
                if kind == "neuron"
                else [C.random_directions(50, seed=13, distribution="covariance")]
            )
            refs = {
                "zero": C.zero(),
                "train_mean": C.reference(R.mean_reference(model, ds, feature), name="train_mean"),
            }
            vals = []
            for ref in refs.values():
                use = C.use_test(
                    model,
                    concept,
                    ds,
                    target=target,
                    relation="decreases",
                    intervention=C.remove(ref),
                    controls=ctl,
                    criteria=ucrit,
                )
                vals.append(C.validate(concept, encoding=enc, use=[use]))
            evidence += [*vals, iso]
            rows[tag] = {
                "concept": concept.id,
                "feature": feature.id,
                "validations": [v.semantic_status.value for v in vals],
                "encoding": {"covariance": enc.outcome.value, "isotropic": iso.outcome.value},
                "auroc": enc.auroc,
                "fp": enc.counterexamples.measurements["false_positive_rate"],
                "fn": enc.counterexamples.measurements["false_negative_rate"],
            }
            print(tag, json.dumps(rows[tag]), flush=True)
    run_seconds = time.perf_counter() - started
    out: dict[str, Any] = {
        "environment": common.environment(),
        "model": info,
        "plan_section": "docs/PHASE_7_PLAN.md §29.2",
        "run_seconds": run_seconds,
        "tests": rows,
        "audits": {},
    }
    for rule_name, rule in RULES.items():
        plan = AU.plan(
            name=f"concepts_{args.model}_{rule_name}".lower(),
            checkpoint=AU.checkpoint_of(model),
            declared_model=None,
            samples=[],
            datasets=sorted(datasets),
            claims=claims,
            requirements=[
                AU.requirement("encoding", policy=C.ENCODING_POLICY, controls=True),
                AU.requirement("use", policy=C.USE_POLICY, controls=True),
            ],
            concepts=concepts,
            counterexamples=rule,
            naive_auroc=0.6,
        )
        t0 = time.perf_counter()
        report = bnn.audit(evidence, plan=plan)
        seconds = time.perf_counter() - t0
        AU.verify_report(report.to_dict(), evidence, plan)
        path = ARTIFACTS / f"concepts_{args.model}_{rule_name}_report.json"
        if path.exists():
            path.unlink()
        report.save(path)
        (ARTIFACTS / f"plan_concepts_{args.model}_{rule_name}.json").write_text(
            bnn.schema.to_json(plan)
        )
        by_id = {v["concept"]: k for k, v in rows.items()}
        out["audits"][rule_name] = {
            "plan_id": plan.id,
            "audit_seconds": seconds,
            "verify_report": "passed",
            "claims": {
                c.name: {
                    "standing": c.standing.value if c.standing else None,
                    "findings": sorted(
                        {
                            f"{f.kind.value}:{f.code}" + (f"@{f.axis}" if f.axis else "")
                            for f in c.findings
                        }
                    ),
                }
                for c in report.claims
            },
            "concepts": {
                by_id[k.concept.concept]: {
                    "standing": k.standing.value,
                    "validations": [s for _, s, _ in k.validations],
                    "findings": sorted(
                        {
                            f"{f.kind.value}:{f.code}" + (f"@{f.axis}" if f.axis else "")
                            for f in k.findings
                        }
                    ),
                    "false_positives": len(k.false_positives),
                    "false_negatives": len(k.false_negatives),
                }
                for k in report.concepts
            },
            "coverage": json.loads(json.dumps(report.to_dict()["coverage"])),
            "evidence": json.loads(json.dumps(report.to_dict()["evidence"])),
            "report_findings": [
                {"kind": f.kind.value, "code": f.code, "detail": f.detail[:300]}
                for f in report.findings
            ],
        }
        print(
            rule_name,
            json.dumps({k: v["standing"] for k, v in out["audits"][rule_name]["concepts"].items()}),
            flush=True,
        )
    if args.model == "A":  # D10: reload check on one model's concept evidence
        from beyondnn.audits.evidence import collect_traces

        base = ARTIFACTS / "concepts_A_traces"
        base.mkdir(exist_ok=True)
        paths = []
        for i, t in enumerate(collect_traces(evidence)):
            p = base / f"t{i:04d}"
            if not p.exists():
                t.save(p)
            paths.append(p)
        plan = AU.plan(
            name="concepts_a_strict",
            checkpoint=AU.checkpoint_of(model),
            declared_model=None,
            samples=[],
            datasets=sorted(datasets),
            claims=claims,
            requirements=[
                AU.requirement("encoding", policy=C.ENCODING_POLICY, controls=True),
                AU.requirement("use", policy=C.USE_POLICY, controls=True),
            ],
            concepts=concepts,
            counterexamples=RULES["strict"],
            naive_auroc=0.6,
        )
        live = bnn.audit(evidence, plan=plan).to_json()
        reloaded = bnn.audit(paths, plan=plan).to_json()
        out["reload_equal"] = live == reloaded
        out["reload_traces"] = len(paths)
        print("reload equal", out["reload_equal"], flush=True)
    out["total_seconds"] = time.perf_counter() - started
    (RESULTS / f"concepts_{args.model}.json").write_text(
        json.dumps(out, indent=1, sort_keys=True, default=str)
    )


if __name__ == "__main__":
    main()
