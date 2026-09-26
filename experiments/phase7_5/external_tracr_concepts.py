# ruff: noqa: E501, RUF015
"""Phase 7.5 external concept-policy validation on Tracr programs (docs/PHASE_7_5_FROZEN_POLICY.md
§E2). Runs in the InterpBench environment.

Program ``frac_prevs(is_x)`` (circuits-benchmark cases 3 = development, 39 = held-out). The
variable ``is_x`` is written by MLP0 into a known residual dimension ('is_x_*') and read by the
layer-1 attention that computes the output, so its representation and use are known exactly
in the compiled (HL, Tracr) model and, by SIIT training, in the trained (LL) model.

Concept label (per input): the fraction of 'x' among the non-BOS tokens exceeds the median of
that fraction on the concept dataset's train split. Concepts:

* K+  (known positive): HL: the 'is_x' residual direction at blocks.0.hook_resid_post; LL: the
      direction fitted on the train split (fit_direction). Encoded and used.
* K-  (known decodable-but-unused; HL only): the 'tokens:a' residual direction, same label.
* K0  (known negative): the K+ feature with the labels permuted (seed 4242).

Each concept: encoding tests under the covariance (PRIMARY) and isotropic (ALTERNATIVE) nulls,
use tests (removal, zero reference; random covariance directions) on the output at the last
position, POLICY_V1 validations, and audits under the strict (0.10) and lenient (0.30) caps.

Usage (InterpBench env, from experiments/phase7_5/artifacts): python ../external_tracr_concepts.py dev|heldout
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
RESULTS = HERE / "results"
sys.path.insert(0, str(ART / "cb"))

import torch  # noqa: E402

import beyondnn as bnn  # noqa: E402

sys.path.insert(0, str(HERE))
from external_interpbench import HF_REVISION, load_ll  # noqa: E402

C, AU, iv = bnn.concepts, bnn.audits, bnn.interventions
CASES = {"dev": ("3",), "heldout": ("39",)}
DATA_MAX = 400
SPLIT_SEED = 1234
PERMUTATION_SEED = 4242
ENC = C.encoding_criteria(min_fraction_below=0.95)
USE = C.use_criteria(min_change=0.1, min_fraction_beyond_controls=0.95)
RULES = {
    "strict": (0.10, 0.10),
    "lenient": (0.30, 0.30),
}


def enc_controls(dist: str) -> list[Any]:
    return [C.random_directions(200, seed=11, distribution=dist), C.label_permutation(200, seed=12)]


def run_concept(
    model: Any, data: Any, feature: Any, name: str, target: Any, known: str
) -> dict[str, Any]:
    concept = C.propose(feature, label=name, definition=name, label_source="dataset")
    encs = {
        d: C.encoding_test(model, concept, data, controls=enc_controls(d), criteria=ENC)
        for d in ("covariance", "isotropic")
    }
    use = C.use_test(
        model,
        concept,
        data,
        target=target,
        relation="decreases",
        intervention=C.remove(C.zero()),
        controls=[C.random_directions(50, seed=13, distribution="covariance")],
        criteria=USE,
    )
    vals = {d: C.validate(concept, encoding=encs[d], use=[use]) for d in encs}
    return {
        "concept": concept,
        "encodings": encs,
        "use": use,
        "validations": vals,
        "known": known,
        "summary": {
            "known": known,
            "auroc": encs["covariance"].auroc,
            "encoding": {d: e.outcome.value for d, e in encs.items()},
            "control_superiority": {
                d: [round(float(c["superiority"]), 4) for c in e.result.statistics["controls"]]  # type: ignore[union-attr,index]
                for d, e in encs.items()
            },
            "use": use.outcome.value,
            "use_effect": use.mean_effect,
            "validation": {d: v.semantic_status.value for d, v in vals.items()},
            "fp": encs["covariance"].counterexamples.measurements["false_positive_rate"],
            "fn": encs["covariance"].counterexamples.measurements["false_negative_rate"],
        },
    }


def run_case(c: str) -> dict[str, Any]:
    mod = importlib.import_module(f"circuits_benchmark.benchmark.cases.case_{c}")
    case = getattr(mod, f"Case{c}")()
    hl = case.get_hl_model(device="cpu").eval()
    ll = load_ll(c)
    labels_res = list(hl.residual_stream_labels)
    data = case.get_clean_data(max_samples=DATA_MAX, seed=DATA_MAX and 42, unique_data=True)
    xs = torch.stack([data[i][0] for i in range(len(data))])
    n = len(xs)
    g = torch.Generator().manual_seed(SPLIT_SEED)
    order = torch.randperm(n, generator=g).tolist()
    n_train, n_val = n // 2, n // 5
    splits = [""] * n
    for rank, i in enumerate(order):
        splits[i] = "train" if rank < n_train else "val" if rank < n_train + n_val else "test"
    x_id = hl.tracr_input_encoder.encoding_map["x"]
    frac = (xs[:, 1:] == x_id).double().mean(1)
    train_median = float(frac[[i for i in range(n) if splits[i] == "train"]].median())
    labels = (frac > train_median).long().tolist()
    perm_labels = [
        labels[i]
        for i in torch.randperm(
            n, generator=torch.Generator().manual_seed(PERMUTATION_SEED)
        ).tolist()
    ]
    P = xs.shape[1] - 1
    target = iv.metrics.select([0, P, 0])
    site = "blocks.0.hook_resid_post"
    out: dict[str, Any] = {
        "case": c,
        "task": case.get_task_description(),
        "n": n,
        "train_median_fraction": train_median,
        "label_balance": sum(labels) / n,
        "residual_labels": labels_res,
        "models": {},
    }
    evidence: dict[str, list[Any]] = {}
    for mname, model in (("tracr_hl", hl), ("siit_ll", ll)):
        t0 = time.perf_counter()
        ds = C.dataset(
            list(xs.split(1)),
            labels,
            splits,
            name=f"case{c}-frac-x",
            label_source="frac_x > train median",
        )
        dperm = C.dataset(
            list(xs.split(1)),
            perm_labels,
            splits,
            name=f"case{c}-permuted",
            label_source="permuted (seed 4242)",
        )
        d_model = model.cfg.d_model
        runs = {}
        if mname == "tracr_hl":
            e_is_x = torch.zeros(d_model)
            e_is_x[[i for i, lab in enumerate(labels_res) if lab.startswith("is_x")][0]] = 1.0
            e_a = torch.zeros(d_model)
            e_a[labels_res.index("tokens:a")] = 1.0
            f_pos = C.direction(site, e_is_x, axis=2, pooling="mean")
            runs["K+"] = run_concept(model, ds, f_pos, "frac x high", target, "positive")
            runs["K-"] = run_concept(
                model,
                ds,
                C.direction(site, e_a, axis=2, pooling="mean"),
                "frac x high (tokens:a feature)",
                target,
                "decodable_unused",
            )
            runs["K0"] = run_concept(
                model,
                dperm,
                C.direction(site, e_is_x, axis=2, pooling="mean"),
                "permuted labels",
                target,
                "negative",
            )
        else:
            runs["K+"] = run_concept(
                model,
                ds,
                C.fit_direction(model, ds, site=site, axis=2, pooling="mean"),
                "frac x high",
                target,
                "positive",
            )
            runs["K0"] = run_concept(
                model,
                dperm,
                C.fit_direction(model, dperm, site=site, axis=2, pooling="mean"),
                "permuted labels",
                target,
                "negative",
            )
        ev = [
            v for r in runs.values() for v in (*r["validations"].values(), *r["encodings"].values())
        ]
        evidence[mname] = ev
        audits = {}
        for rule, (fp_cap, fn_cap) in RULES.items():
            roles = [
                AU.role("null", "*covariance*", "primary"),
                AU.role("null", "*isotropic*", "alternative"),
                AU.role("replacement", "*", "primary"),
                AU.role("dataset", "*", "primary"),
            ]
            plan = AU.plan(
                name=f"tracr_concepts_{c}_{mname}_{rule}",
                checkpoint=AU.checkpoint_of(model),
                declared_model=None,
                samples=[],
                datasets=sorted({ds.id, dperm.id}),
                claims=[],
                requirements=[],
                concepts=[
                    AU.concept(r["concept"].id, policy=C.POLICY_V1, roles=roles)
                    for r in runs.values()
                ],
                counterexamples=AU.counterexample_rule(
                    max_counterexample_fraction=None,
                    max_false_positive_rate=fp_cap,
                    max_false_negative_rate=fn_cap,
                ),
                naive_auroc=0.6,
            )
            report = bnn.audit(ev, plan=plan)
            AU.verify_report(report.to_dict(), ev, plan)
            by_id = {r["concept"].id: k for k, r in runs.items()}
            audits[rule] = {
                by_id[k.concept.concept]: {
                    "standing": k.standing.value,
                    "findings": sorted({f.code for f in k.findings}),
                }
                for k in report.concepts
            }
        out["models"][mname] = {
            "params": sum(p.numel() for p in model.parameters()),
            "concepts": {k: r["summary"] for k, r in runs.items()},
            "audits": audits,
            "seconds": time.perf_counter() - t0,
        }
        print(
            c,
            mname,
            json.dumps(
                {
                    k: (r["summary"]["validation"], r["summary"]["encoding"], r["summary"]["use"])
                    for k, r in runs.items()
                }
            ),
            flush=True,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["dev", "heldout"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    out: dict[str, Any] = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "interpbench_revision": HF_REVISION,
            "beyondnn_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE
            ).stdout.strip(),
        },
        "split": args.split,
        "cases": {},
    }
    for c in CASES[args.split]:
        out["cases"][c] = run_case(c)
    (RESULTS / f"external_tracr_concepts_{args.split}.json").write_text(
        json.dumps(out, indent=1, default=str)
    )


if __name__ == "__main__":
    main()
