# ruff: noqa: SIM115, B007, PT018
"""Phase 7.5 external known-mechanism validation on InterpBench (docs/PHASE_7_5_FROZEN_POLICY.md
§E1). Runs in the InterpBench environment (experiments/phase7_5/requirements_interpbench.txt).

For each case (official weights: HF cybershiptrooper/InterpBench@a1242a84; official task code:
FlyingPumba/circuits-benchmark@220791e4):

* samples: the first N_SAMPLES unique inputs of ``case.get_clean_data(max_samples=400,
  seed=42, unique_data=True)`` whose LL prediction at the target position (last position)
  equals the HL (Tracr) output; the interchange source of sample i is sample (i+1) mod N;
* ground truth per (node, sample): HL interchange of the corresponding HL node (Tracr model,
  benchmark correspondence) changes the HL argmax at the target position -> necessary;
  non-circuit nodes are not necessary (SIIT). An independent TransformerLens resample ablation
  of the LL model must agree, otherwise the instance is benchmark-AMBIGUOUS;
* BeyondNN evidence per node: comprehensiveness (min_drop = clean margin) under the resample
  (PRIMARY), mean (ALTERNATIVE) and zero (STRESS_TEST) replacements, without controls
  (PRIMARY null) and, for heads under resample, with count controls (ALTERNATIVE null); plus
  IG top-1 head selections per layer under the same configurations;
* one audit per case under the frozen roles; the confusion of PRIMARY standings against the
  ground truth, and the same for every single configuration treated as if it were primary.

Usage (InterpBench env, from experiments/phase7_5/artifacts):
    python ../external_interpbench.py dev|heldout
"""

from __future__ import annotations

import argparse
import importlib
import json
import pickle
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
sys.path.insert(0, str(ART / "cb"))

import torch  # noqa: E402
from transformer_lens import HookedTransformer, HookedTransformerConfig  # noqa: E402

import beyondnn as bnn  # noqa: E402
from beyondnn.core.samples import sample_id  # noqa: E402
from beyondnn.schema import Site, Subject  # noqa: E402

A, F, AU, iv = bnn.attribution, bnn.faithfulness, bnn.audits, bnn.interventions

# ------------------------------------------------------------------ frozen parameters
HF_REVISION = "a1242a84f8a4d07d1be9a8f5ec710415198016b2"
CB_COMMIT = "220791e4a9cd29957ed5edfa445e7c09152d7298"
DEV_CASES = ("7", "13")
N_SAMPLES = 40
DATA_MAX = 400
DATA_SEED = 42
N_CONTROLS = 20
CONTROL_SEED = 777
FRACTION = 0.95
IG_STEPS = 32
MAX_LAYERS = 4
MAX_D_MODEL = 64
MIN_CLEAN_AGREEMENT = 0.90  # below: benchmark-unreliable case (reported, not in the primary table)
ROLES = [
    ("replacement", "tensor/resample:*", "primary"),
    ("replacement", "tensor/mean:*", "alternative"),
    ("replacement", "zero", "stress_test"),
    ("null", "none", "primary"),
    ("null", "count@*", "alternative"),
    ("threshold", "{*}", "primary"),
    ("threshold", "*|alt:min_dropx0.5", "alternative"),
    ("threshold", "*|alt:min_dropx1.5", "stress_test"),
]


def eligible_cases() -> list[str]:
    """Frozen rule: numeric InterpBench case, circuits-benchmark case importable, categorical,
    ground-truth circuit with >= 1 attention head and >= 1 MLP, n_layers <= 4, d_model <= 64."""
    out = []
    for path in sorted(
        (ART / "interpbench").iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 10**9
    ):
        c = path.name
        if not c.isdigit():
            continue
        cfg = pickle.load(open(path / "ll_model_cfg.pkl", "rb"))
        if cfg["n_layers"] > MAX_LAYERS or cfg["d_model"] > MAX_D_MODEL:
            continue
        heads, mlps = circuit_nodes(c)
        if not heads or not mlps:
            continue
        try:
            case = load_case(c)
            if not case.is_categorical():
                continue
        except Exception:
            continue
        out.append(c)
    return out


def load_case(c: str) -> Any:
    mod = importlib.import_module(f"circuits_benchmark.benchmark.cases.case_{c}")
    return getattr(mod, f"Case{c}")()


def circuit_nodes(c: str) -> tuple[set[tuple[int, int]], set[int]]:
    edges = pickle.load(open(ART / "interpbench" / c / "edges.pkl", "rb"))
    heads: set[tuple[int, int]] = set()
    mlps: set[int] = set()
    for edge in edges:
        for name in edge:
            parts = name.split(".")
            if "hook_result[" in name:
                heads.add((int(parts[1]), int(name.split("[")[1].rstrip("]"))))
            elif name.endswith("hook_mlp_out"):
                mlps.add(int(parts[1]))
    return heads, mlps


def load_ll(c: str) -> HookedTransformer:
    d = ART / "interpbench" / c
    if not (d / "ll_model.pth").exists():
        subprocess.run(
            [
                "curl",
                "-sL",
                "-o",
                str(d / "ll_model.pth"),
                f"https://huggingface.co/cybershiptrooper/InterpBench/resolve/{HF_REVISION}/{c}/ll_model.pth",
            ],
            check=True,
        )
    cfg = pickle.load(open(d / "ll_model_cfg.pkl", "rb"))
    cfg["device"] = "cpu"
    model = HookedTransformer(HookedTransformerConfig(**cfg))
    result = model.load_state_dict(torch.load(d / "ll_model.pth", map_location="cpu"), strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    return model.eval()


# ------------------------------------------------------------------ ground truth


def hl_map(case: Any) -> dict[tuple[str, int | None], tuple[str, Any]]:
    """LL node (hook name, head or None) -> (HL hook name, HL index or None)."""
    corr = case.get_correspondence()
    out = {}
    for hl_node, ll_nodes in corr.items():
        for ll in ll_nodes:
            name = ll.name
            idx = ll.index
            head = None
            if "hook_result" in name:
                head = int(idx.as_index[2])
            key = (name.replace("mlp.hook_post", "hook_mlp_out"), head)
            out[key] = (hl_node.name, getattr(hl_node, "index", None))
    return out


def interchange(
    model: Any, name: str, index: Any, x: torch.Tensor, source: torch.Tensor
) -> torch.Tensor:
    """Output of ``model`` on x with activation ``name`` (restricted to ``index``) taken
    from the run on ``source`` (TransformerLens hooks; independent of BeyondNN)."""
    _, cache = model.run_with_cache(source, names_filter=lambda n: n == name)
    src = cache[name]

    def hook(act: torch.Tensor, hook: Any) -> torch.Tensor:
        out = act.clone()
        if index is None:
            out[:] = src
        else:
            out[index] = src[index]
        return out

    with torch.no_grad():
        return model.run_with_hooks(x, fwd_hooks=[(name, hook)])


def _index_of(head: int | None) -> Any:
    return None if head is None else (slice(None), slice(None), head)


# ------------------------------------------------------------------ one case


def run_case(c: str) -> dict[str, Any]:
    t_case = time.perf_counter()
    case = load_case(c)
    hl = case.get_hl_model(device="cpu").eval()
    ll = load_ll(c)
    cfg = ll.cfg
    data = case.get_clean_data(max_samples=DATA_MAX, seed=DATA_SEED, unique_data=True)
    xs_all = torch.stack([data[i][0] for i in range(len(data))])
    P = xs_all.shape[1] - 1
    with torch.no_grad():
        hl_pred = hl(xs_all)[:, P].argmax(-1)
        ll_pred = ll(xs_all)[:, P].argmax(-1)
    agree = hl_pred == ll_pred
    chosen = [i for i in range(len(xs_all)) if bool(agree[i])][:N_SAMPLES]
    xs = xs_all[chosen]
    n = len(xs)
    heads_c, mlps_c = circuit_nodes(c)
    mapping = hl_map(case)
    nodes: list[tuple[str, int, int | None]] = [
        (f"blocks.{L}.attn.hook_result", L, h)
        for L in range(cfg.n_layers)
        for h in range(cfg.n_heads)
    ] + [(f"blocks.{L}.hook_mlp_out", L, None) for L in range(cfg.n_layers)]
    sites = sorted({s for s, _, _ in nodes})
    # site activations for replacements (resample source, mean)
    acts: dict[str, torch.Tensor] = {}
    _, cache = ll.run_with_cache(xs, names_filter=lambda nm: nm in sites)
    for s in sites:
        acts[s] = cache[s].detach()
    samples = [sample_id(xs[i : i + 1]) for i in range(n)]
    targets: dict[str, Any] = {}
    metrics: list[Any] = []
    margins: list[float] = []
    with torch.no_grad():
        logits = ll(xs)[:, P]
    for i in range(n):
        top = torch.topk(logits[i], 2)
        pred, runner = int(top.indices[0]), int(top.indices[1])
        metric = iv.metrics.margin([0, P, pred])  # drop >= margin <=> the argmax changes
        metrics.append(metric)
        margins.append(float(top.values[0] - top.values[1]))
        assert runner != pred
        targets[samples[i]] = metric
    # ground truth
    gt: dict[tuple[str, int | None], list[str]] = {}
    for site, L, h in nodes:
        in_circuit = (L, h) in heads_c if h is not None else L in mlps_c
        labels = []
        for i in range(n):
            j = (i + 1) % n
            x, src = xs[i : i + 1], xs[j : j + 1]
            ll_name = site if h is not None else f"blocks.{L}.mlp.hook_post"
            ll_out = interchange(ll, ll_name, _index_of(h), x, src)[0, P].argmax(-1)
            ll_flip = int(ll_out) != int(ll_pred[chosen[i]])
            if in_circuit and (site, h) in mapping:
                hl_name, hl_index = mapping[(site, h)]
                idx = None if hl_index is None else hl_index.as_index
                hl_out = interchange(hl, hl_name, idx, x, src)[0, P].argmax(-1)
                hl_change = int(hl_out) != int(hl_pred[chosen[i]])
            else:
                hl_change = False
            if hl_change == ll_flip:
                labels.append("necessary" if hl_change else "not_necessary")
            else:
                labels.append("ambiguous")
        gt[(site, h)] = labels
    # BeyondNN evidence
    t_ev = time.perf_counter()
    evidence: list[Any] = []
    attributions: list[Any] = []
    ig = A.integrated_gradients(baseline=A.zero_baseline(), n_steps=IG_STEPS)
    ig_selected: dict[tuple[int, int], int] = {}
    for i in range(n):
        x = xs[i : i + 1]
        j = (i + 1) % n
        for site, L, h in nodes:
            shape = acts[site].shape
            units = (h,) if h is not None else tuple(range(shape[1]))
            axes = (2,) if h is not None else (1,)
            n_units = shape[2] if h is not None else shape[1]
            reps = {
                "resample": F.replacement(acts[site][j : j + 1], name="resample"),
                "mean": F.replacement(acts[site].mean(0, keepdim=True), name="mean"),
                "zero": F.zero(),
            }
            sel = F.units(A.layer(site), units, n_units=n_units, unit_axes=axes)
            for rname, rep in reps.items():
                test = F.comprehensiveness(
                    target=metrics[i],
                    min_drop=margins[i],
                    statement=f"{site}[{h}] necessary",
                    replacement=rep,
                )
                evidence.append(F.run(ll, x, test=test, selection=sel))
                if h is not None and rname == "resample":
                    ctest = F.comprehensiveness(
                        target=metrics[i],
                        min_drop=margins[i],
                        statement=f"{site}[{h}] necessary",
                        replacement=rep,
                        controls=F.controls(N_CONTROLS, seed=CONTROL_SEED + i),
                        min_fraction_below=FRACTION,
                    )
                    evidence.append(F.run(ll, x, test=ctest, selection=sel))
        for L in range(cfg.n_layers):
            site = f"blocks.{L}.attn.hook_result"
            attr = A.attribute(ll, x, target=metrics[i], method=ig, at=A.layer(site))
            attributions.append(attr)
            sel = F.top_k(attr, k=1, unit_axes=(2,), reduce="sum")
            ig_selected[(i, L)] = sel.selected[0]
            for rname, rep in {
                "resample": F.replacement(acts[site][j : j + 1], name="resample"),
                "mean": F.replacement(acts[site].mean(0, keepdim=True), name="mean"),
                "zero": F.zero(),
            }.items():
                test = F.comprehensiveness(
                    target=metrics[i],
                    min_drop=margins[i],
                    statement=f"IG top-1 head at {site}",
                    replacement=rep,
                )
                evidence.append(F.run(ll, x, test=test, selection=sel, attributions=[attr]))
                if rname == "resample":
                    ctest = F.comprehensiveness(
                        target=metrics[i],
                        min_drop=margins[i],
                        statement=f"IG top-1 head at {site}",
                        replacement=rep,
                        controls=F.controls(N_CONTROLS, seed=CONTROL_SEED + i),
                        min_fraction_below=FRACTION,
                    )
                    evidence.append(F.run(ll, x, test=ctest, selection=sel, attributions=[attr]))
    evidence_seconds = time.perf_counter() - t_ev
    roles = [AU.role(a, p, r) for a, p, r in ROLES]
    claims = []
    for site, L, h in nodes:
        shape = acts[site].shape
        units = (h,) if h is not None else tuple(range(shape[1]))
        axes = (2,) if h is not None else (1,)
        claims.append(
            AU.claim(
                f"l{L}_" + (f"h{h}" if h is not None else "mlp") + "_necessary",
                statement=f"{site}"
                + (f" head {h}" if h is not None else "")
                + " is necessary for the output at the last position",
                relation="necessary_for",
                target=None,
                sample_targets=targets,
                scope="instance",
                requirement="necessity",
                subject=Subject(site=Site(module=site), units=units, unit_axes=axes),
                roles=roles,
            )
        )
    for L in range(cfg.n_layers):
        claims.append(
            AU.claim(
                f"l{L}_ig_top1_necessary",
                statement=f"the IG top-1 head of layer {L} is necessary for the output",
                relation="necessary_for",
                target=None,
                sample_targets=targets,
                scope="instance",
                requirement="necessity",
                selection=AU.selection(
                    f"blocks.{L}.attn.hook_result", method="integrated_gradients", k=1
                ),
                roles=roles,
            )
        )
    plan = AU.plan(
        name=f"interpbench_case_{c}",
        checkpoint=AU.checkpoint_of(ll),
        declared_model=None,
        samples=samples,
        datasets=[],
        claims=claims,
        requirements=[
            AU.requirement(
                "necessity",
                policy=F.COMPREHENSIVENESS_POLICY,
                controls=False,
                alternatives=[
                    AU.alternative("comprehensiveness", "min_drop", factor=0.5),
                    AU.alternative("comprehensiveness", "min_drop", factor=1.5),
                ],
            )
        ],
        concepts=[],
        counterexamples=AU.counterexample_rule(
            max_counterexample_fraction=None,
            max_false_positive_rate=None,
            max_false_negative_rate=None,
        ),
        naive_auroc=None,
    )
    t_a = time.perf_counter()
    report = bnn.audit([*evidence, *attributions], plan=plan)
    audit_seconds = time.perf_counter() - t_a
    AU.verify_report(report.to_dict(), [*evidence, *attributions], plan)
    attribution_only = bnn.audit(attributions, plan=plan)
    (ART / f"interpbench_{c}_report.json").unlink(missing_ok=True)
    report.save(ART / f"interpbench_{c}_report.json")

    # ground truth for selection claims: the selected head's label
    def gt_of(claim: Any, i: int) -> str:
        if claim.claim.selection is not None:
            L = int(claim.claim.selection.site.module.split(".")[1])
            h = ig_selected[(i, L)]
            return gt[(f"blocks.{L}.attn.hook_result", h)][i]
        s = claim.claim.subject
        h = s.units[0] if s.unit_axes == (2,) else None
        return gt[(s.site.module, h)][i]

    rows = []
    for claim in report.claims:
        for i, g in enumerate(sorted(claim.groups, key=lambda g: samples.index(g.sample))):
            prof = g.profile
            per_config = {}
            if prof is not None:
                for cf in prof.configurations:
                    ax = dict(cf.axes)
                    key = "|".join(
                        [
                            ax.get("replacement", "-").split(":")[0],
                            ax.get("null", "-").split("@")[0],
                            cf.alternative or "recorded",
                        ]
                    )
                    per_config[key] = cf.outcome
            rows.append(
                {
                    "claim": claim.name,
                    "kind": "selection"
                    if claim.claim.selection
                    else ("head" if claim.claim.subject.unit_axes == (2,) else "mlp"),
                    "sample": i,
                    "gt": gt_of(claim, i),
                    "standing": g.standing.value,
                    "codes": sorted({f.code for f in g.findings}),
                    "configs": per_config,
                }
            )
    attr_only = {c2.name: dict(c2.distribution) for c2 in attribution_only.claims}
    return {
        "case": c,
        "task": case.get_task_description(),
        "n_samples": n,
        "target_position": P,
        "ll_hl_clean_agreement": float(agree.float().mean()),
        "n_candidates": len(xs_all),
        "benchmark_reliable": float(agree.float().mean()) >= MIN_CLEAN_AGREEMENT,
        "circuit_heads": sorted(heads_c),
        "circuit_mlps": sorted(mlps_c),
        "cfg": {"n_layers": cfg.n_layers, "n_heads": cfg.n_heads, "d_model": cfg.d_model},
        "plan_id": plan.id,
        "n_results": report.evidence.results,
        "results_excluded": report.evidence.results_excluded,
        "report_findings": [(f.kind.value, f.code, len(f.records)) for f in report.findings],
        "evidence_seconds": evidence_seconds,
        "audit_seconds": audit_seconds,
        "case_seconds": time.perf_counter() - t_case,
        "rows": rows,
        "attribution_only": attr_only,
        "ig_selected": {f"{i}:{L}": h for (i, L), h in ig_selected.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["dev", "heldout", "list"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    eligible = eligible_cases()
    if args.split == "list":
        print(json.dumps({"eligible": eligible, "dev": DEV_CASES}))
        return
    cases = [c for c in eligible if (c in DEV_CASES) == (args.split == "dev")]
    out: dict[str, Any] = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "beyondnn_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE
            ).stdout.strip(),
            "interpbench_revision": HF_REVISION,
            "circuits_benchmark_commit": CB_COMMIT,
        },
        "split": args.split,
        "eligible": eligible,
        "cases": {},
    }
    for c in cases:
        try:
            out["cases"][c] = run_case(c)
        except Exception as exc:  # reported, never hidden
            out["cases"][c] = {"case": c, "error": f"{type(exc).__name__}: {exc}"}
        print(
            c,
            {k: v for k, v in out["cases"][c].items() if k not in ("rows",)}.get("error") or "ok",
            flush=True,
        )
    (RESULTS / f"external_interpbench_{args.split}.json").write_text(
        json.dumps(out, indent=1, default=str)
    )


if __name__ == "__main__":
    main()
