# ruff: noqa: E501
"""Phase 7.75 central re-runs (docs/PHASE_7_75_PLAN.md §5-§6).

Exactly the Phase-7.5 central design (``experiments/phase7_5/central75.py``: models, samples,
targets, replacements and roles, k from p = 5/10/20%, controls, thresholds), with:

* the Phase-7.75 audit (ADR-053/054) -- every model, all units (``--eligibility all``);
* for token sites, a second declared claim about content tokens (``--eligibility
  content_tokens``, ADR-053): eligible positions are those whose input id is not one of the
  tokenizer's special ids ([CLS], [SEP], [PAD]) -- a caller-supplied mask, never filtered
  silently. k is p x (number of eligible positions); IG / gradient rank and random units are
  drawn among eligible positions only; controls are drawn from them (ADR-053).

``--shard i/n`` runs samples i, i+n, ... ; per-sample standings depend only on each sample's
evidence (instance scope), so shards are audited separately and merged by ``merge``.

Usage (experiment env, from experiments/phase5_5):
    python ../phase7_75/central775.py A|B|C|D dev|heldout --eligibility all|content_tokens [--shard i/n]
    python ../phase7_75/central775.py merge <model> <split> <eligibility>
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
sys.path.insert(0, str(HERE.parent / "phase7_5"))

import central75 as C75  # noqa: E402
import common  # noqa: E402
import torch  # noqa: E402

import beyondnn as bnn  # noqa: E402
from beyondnn.core.samples import sample_id  # noqa: E402

A, F, AU = bnn.attribution, bnn.faithfulness, bnn.audits
RF = C75.RF
RESULTS = HERE / "results"
ELIGIBILITY = "content_tokens"


def eligible_positions(model_id: str, sample: Any) -> tuple[int, ...] | None:
    """Content-token positions: input ids that are not the tokenizer's special ids."""
    if model_id not in ("C", "D"):
        raise SystemExit("content-token eligibility applies to token sites (C, D) only")
    tok = _tokenizer(model_id)
    special = set(tok.all_special_ids)
    ids = sample.inputs[0][0].tolist()
    return tuple(i for i, t in enumerate(ids) if t not in special)


_TOK: dict[str, Any] = {}


def _tokenizer(model_id: str) -> Any:
    if model_id not in _TOK:
        from transformers import AutoTokenizer

        repo, rev = (
            (common.BERT_REPO, common.BERT_REVISION)
            if model_id == "C"
            else (C75.D_REPO, C75.D_REVISION)
        )
        _TOK[model_id] = AutoTokenizer.from_pretrained(repo, revision=rev)
    return _TOK[model_id]


def run_sample(
    model: Any, setting: Any, sample: Any, path: str, eligible: tuple[int, ...] | None
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    """``central75.run_sample`` with an optional declared eligibility (ADR-053)."""
    if eligible is None:
        return C75.run_sample(model, setting, sample, path)
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
    elig = list(eligible)
    perm = [
        elig[i]
        for i in torch.randperm(
            len(elig), generator=torch.Generator().manual_seed(RF.RANDOM_SEED_BASE + sample.index)
        ).tolist()
    ]
    ks = C75.ks_for(len(elig))
    roles = C75.REPLACEMENT_ROLES[(setting.model_id, setting.site)]
    reps = C75.named(setting.replacements(sample.index, sample.inputs))
    primary_rep = next(r for r, role in roles.items() if role == "primary")
    results: list[Any] = []
    single: dict[str, Any] = {}
    for rname, rep in reps.items():
        is_primary = rname == primary_rep
        for p, k in ks.items():
            for method in C75.METHODS:
                if method == "random":
                    selection = F.units(
                        setting.at,
                        tuple(perm[:k]),
                        n_units=n,
                        unit_axes=ax,
                        eligible=eligible,
                        eligibility=ELIGIBILITY,
                    )
                    attributions: list[Any] = []
                else:
                    selection = F.top_k(
                        attrs[method],
                        k=k,
                        unit_axes=ax,
                        reduce=RF.REDUCE[method] if ax else None,
                        eligible=eligible,
                        eligibility=ELIGIBILITY,
                    )
                    attributions = [attrs[method]]
                tests = [("comprehensiveness", "count"), ("sufficiency", "count")]
                if is_primary:
                    tests += [("comprehensiveness", "magnitude"), ("comprehensiveness", "none")]
                for name, match in tests:
                    controls = (
                        None
                        if match == "none"
                        else F.controls(C75.N_CONTROLS, seed=10000 + sample.index, match=match)
                    )
                    if name == "comprehensiveness":
                        extra = {} if controls is None else {"min_fraction_below": C75.FRACTION}
                        template = F.comprehensiveness(
                            target=metric,
                            min_drop=C75.PRIMARY_T * margin,
                            replacement=rep,
                            controls=controls,
                            statement=f"the {method} top-{k} content tokens are necessary for the margin",
                            **extra,
                        )
                    else:
                        template = F.sufficiency(
                            target=metric,
                            max_drop=C75.PRIMARY_T * margin,
                            replacement=rep,
                            controls=controls,
                            statement=f"the {method} top-{k} content tokens suffice for the margin",
                            min_fraction_above=C75.FRACTION,
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
                        and is_primary
                        and p == C75.PRIMARY_P
                        and name == "comprehensiveness"
                        and match == "count"
                    ):
                        single = {
                            "outcome": res.outcome.value,
                            "drop": res.drop,
                            "k": k,
                            "units": list(res.selection.selected),
                        }
    meta = {
        "index": sample.index,
        "sample_id": sample_id(*sample.inputs, model_kwargs=kw or None),
        "pred": pred,
        "runner_up": runner,
        "margin": margin,
        "n_units": n,
        "eligible": list(eligible),
        "ks": {str(p): k for p, k in ks.items()},
        "primary_single_configuration": single,
        "text": sample.text,
    }
    return results, list(attrs.values()), meta


def plan_for(
    model: Any,
    setting: Any,
    targets: dict[str, Any],
    methods: dict[str, str],
    metas: list[dict[str, Any]],
    eligibility: str | None,
) -> Any:
    plan = C75.plan_for(model, setting, targets, methods, metas)
    if eligibility is None:
        return plan
    # the same claims, about content tokens (ADR-053)
    from dataclasses import replace

    claims = tuple(
        replace(c, selection=replace(c.selection, eligibility=eligibility)) for c in plan.claims
    )
    return AU.plan(
        name=plan.name + "_content",
        checkpoint=plan.checkpoint,
        declared_model=plan.declared_model,
        samples=list(plan.samples),
        datasets=list(plan.datasets),
        claims=list(claims),
        requirements=list(plan.requirements),
        concepts=list(plan.concepts),
        counterexamples=plan.counterexamples,
        naive_auroc=plan.naive_auroc,
    )


def run(model_id: str, split: str, eligibility: str, shard: tuple[int, int] | None) -> None:
    torch.set_num_threads(1)
    model, samples, settings, info, path = C75.build(model_id, split)
    if shard is not None:
        i, n = shard
        samples = samples[i::n]
    elig_name = None if eligibility == "all" else ELIGIBILITY
    payload: dict[str, Any] = {
        "environment": common.environment(),
        "model": info,
        "split": split,
        "eligibility": eligibility,
        "shard": None if shard is None else f"{shard[0]}/{shard[1]}",
        "policy": "docs/PHASE_7_75_PLAN.md §5-§6 (Phase-7.5 central design; ADR-053/054)",
        "settings": {},
    }
    for setting in settings:
        t0 = time.perf_counter()
        results, metas, targets = [], [], {}
        methods: dict[str, str] = {}
        for sample in samples:
            eligible = None if elig_name is None else eligible_positions(model_id, sample)
            res, attrs, meta = run_sample(model, setting, sample, path, eligible)
            results += res
            metas.append(meta)
            targets[meta["sample_id"]] = res[0].claim.target
            methods = {
                "g": attrs[0].record.method.name,
                "ig": attrs[1].record.method.name,
                "r": "declared",
            }
        run_seconds = time.perf_counter() - t0
        plan = plan_for(model, setting, targets, methods, metas, elig_name)
        t1 = time.perf_counter()
        report = bnn.audit(results, plan=plan)
        audit_seconds = time.perf_counter() - t1
        AU.verify_report(report.to_dict(), results, plan)
        payload["settings"][setting.site] = {
            "n_samples": len(samples),
            "n_results": report.evidence.results,
            "results_excluded": report.evidence.results_excluded,
            "plan_id": plan.id,
            "run_seconds": run_seconds,
            "audit_seconds": audit_seconds,
            "verify_report": "passed",
            "audit": C75.summarise(report, metas),
            "samples": metas,
        }
    RESULTS.mkdir(exist_ok=True)
    tag = f"central775_{model_id}_{split}_{eligibility}" + (
        "" if shard is None else f"_shard{shard[0]}of{shard[1]}"
    )
    (RESULTS / f"{tag}.json").write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))
    print("done", tag, flush=True)


def merge(model_id: str, split: str, eligibility: str) -> None:
    """Merge shard files: per-sample standings are shard-independent (instance scope)."""
    parts = sorted(RESULTS.glob(f"central775_{model_id}_{split}_{eligibility}_shard*of*.json"))
    docs = [json.loads(p.read_text()) for p in parts]
    out = dict(docs[0]) | {
        "shard": f"merged from {len(docs)} shards",
        "shard_files": [p.name for p in parts],
    }
    out["settings"] = {}
    for site in docs[0]["settings"]:
        merged = {
            "n_samples": sum(d["settings"][site]["n_samples"] for d in docs),
            "n_results": sum(d["settings"][site]["n_results"] for d in docs),
            "results_excluded": sum(d["settings"][site]["results_excluded"] for d in docs),
            "plan_ids": [d["settings"][site]["plan_id"] for d in docs],
            "run_seconds": sum(d["settings"][site]["run_seconds"] for d in docs),
            "audit_seconds": sum(d["settings"][site]["audit_seconds"] for d in docs),
            "verify_report": "passed"
            if all(d["settings"][site]["verify_report"] == "passed" for d in docs)
            else "failed",
            "samples": [m for d in docs for m in d["settings"][site]["samples"]],
            "audit": {},
        }
        for claim in docs[0]["settings"][site]["audit"]:
            per_sample = [
                s for d in docs for s in d["settings"][site]["audit"][claim]["per_sample"]
            ]
            dist: dict[str, int] = {}
            for s in per_sample:
                dist[s["standing"]] = dist.get(s["standing"], 0) + 1
            merged["audit"][claim] = {
                "distribution": dict(sorted(dist.items())),
                "per_sample": per_sample,
            }
        out["settings"][site] = merged
    (RESULTS / f"central775_{model_id}_{split}_{eligibility}.json").write_text(
        json.dumps(out, indent=1, sort_keys=True, default=str)
    )
    print("merged", len(docs), "shards")


def main() -> None:
    if sys.argv[1] == "merge":
        merge(*sys.argv[2:5])
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["A", "B", "C", "D"])
    parser.add_argument("split", choices=["dev", "heldout"])
    parser.add_argument("--eligibility", choices=["all", "content_tokens"], required=True)
    parser.add_argument("--shard", default=None)
    args = parser.parse_args()
    shard = None if args.shard is None else tuple(int(v) for v in args.shard.split("/"))
    run(args.model, args.split, args.eligibility, shard)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
