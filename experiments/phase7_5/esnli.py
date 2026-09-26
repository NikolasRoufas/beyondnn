"""Phase 7.5 NLP task 2: e-SNLI human rationales vs attribution vs intervention evidence
(docs/PHASE_7_5_FROZEN_POLICY.md §N2). Held-out task and model.

Model: textattack/bert-base-uncased-snli (pinned). Data: e-SNLI (Camburu et al. 2018), the
authors' CSVs pinned at commit 7b585a3f. Label mapping: the label permutation with the highest
agreement on the first 500 dev rows (calibration only). Samples: test rows, gold label in
{entailment, neutral, contradiction}, <= 64 WordPieces, annotator 1 highlighted >= 1 word,
correctly classified; seeded permutation (1234), first 40.

Per sample, three selections of k_h units (k_h = the WordPieces of annotator 1's highlighted
words, in both sentences): HUMAN (declared units), IG top-k_h, RANDOM k_h (seed 30000 + i).
Tests: comprehensiveness and sufficiency with count controls (N = 50, 0.95) under [MASK]
(PRIMARY), [PAD] (ALTERNATIVE) and zero (STRESS_TEST) embedding replacement; threshold
0.5 x margin (metric: margin of the predicted class); alternatives x0.5 / x1.5.

Human rationales are a PLAUSIBILITY reference only: agreement (token F1 / IoU of IG top-k_h
with the human units; inter-annotator IoU) is reported separately from the audit of the same
tokens, and no human rationale is treated as causal ground truth.

Usage (experiment env, from experiments/phase5_5): python ../phase7_5/esnli.py [--limit N] [--quiet]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase5_5"))
import common  # noqa: E402
import torch  # noqa: E402

import beyondnn as bnn  # noqa: E402
from beyondnn.core.samples import sample_id  # noqa: E402
from beyondnn.schema import Site  # noqa: E402

A, F, AU, iv = bnn.attribution, bnn.faithfulness, bnn.audits, bnn.interventions
RESULTS = HERE / "results"
ARTIFACTS = HERE / "artifacts"
REPO = "textattack/bert-base-uncased-snli"
REVISION = "d4ef8a69a50bc95cc074514f4b798c67f572163a"
ESNLI_COMMIT = "7b585a3f077fdea899780eb0473940522ae44a2e"
GOLD = ("entailment", "neutral", "contradiction")
CALIBRATION_ROWS = 500
MAX_PIECES = 64
N_SAMPLES = 40
SEED = 1234
N_CONTROLS = 50
FRACTION = 0.95
T = 0.5
IG_STEPS = 32
SITE = "bert.embeddings.word_embeddings"
PATH = '["logits"]'
BOOT_SEED = 7501
REPLACEMENT_ROLES = {"mask": "primary", "pad": "alternative", "zero": "stress_test"}


def rows(split: str) -> list[dict[str, str]]:
    path = ARTIFACTS / "esnli" / f"esnli_{split}.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def highlighted(row: dict[str, str], annotator: int) -> tuple[set[int], set[int]]:
    def parse(v: str) -> set[int]:
        v = (v or "").strip()
        return (
            set()
            if v in ("", "{}")
            else {int(x) for x in v.split(",") if x.strip().lstrip("-").isdigit()}
        )

    return parse(row.get(f"Sentence1_Highlighted_{annotator}", "")), parse(
        row.get(f"Sentence2_Highlighted_{annotator}", "")
    )


def word_spans(sentence: str) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for word in sentence.split(" "):
        spans.append((pos, pos + len(word)))
        pos += len(word) + 1
    return spans


def units_for(enc: Any, premise: str, hypothesis: str, h1: set[int], h2: set[int]) -> list[int]:
    """WordPiece positions overlapping a highlighted word (both sentences)."""
    offsets = enc["offset_mapping"][0].tolist()
    seq_ids = enc.sequence_ids(0)
    spans = {0: word_spans(premise), 1: word_spans(hypothesis)}
    marked = {0: h1, 1: h2}
    out = []
    for pos, ((a, b), sid) in enumerate(zip(offsets, seq_ids, strict=True)):
        if sid is None or a == b:
            continue
        for w in marked[sid]:
            if w < len(spans[sid]):
                wa, wb = spans[sid][w]
                if a < wb and b > wa:
                    out.append(pos)
                    break
    return sorted(set(out))


def load() -> tuple[Any, Any, list[int], dict[str, Any]]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(REPO, revision=REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(REPO, revision=REVISION).eval()
    calib = [r for r in rows("dev") if r["gold_label"] in GOLD][:CALIBRATION_ROWS]
    preds = []
    for r in calib:
        enc = tok(r["Sentence1"], r["Sentence2"], return_tensors="pt")
        with torch.no_grad():
            preds.append(int(model(**enc).logits.argmax(-1)))
    best = max(
        itertools.permutations(range(3)),
        key=lambda perm: sum(
            p == perm[GOLD.index(r["gold_label"])] for p, r in zip(preds, calib, strict=True)
        ),
    )
    acc = sum(
        p == best[GOLD.index(r["gold_label"])] for p, r in zip(preds, calib, strict=True)
    ) / len(calib)
    info = {
        "repo": REPO,
        "revision": REVISION,
        "parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint_state_digest": common.state_digest(model),
        "gold_to_model_label": dict(zip(GOLD, best, strict=True)),
        "calibration_accuracy": acc,
        "esnli_commit": ESNLI_COMMIT,
    }
    return model, tok, list(best), info


def select_samples(model: Any, tok: Any, mapping: list[int], limit: int) -> list[dict[str, Any]]:
    eligible = []
    for r in rows("test"):
        if r["gold_label"] not in GOLD:
            continue
        h1, h2 = highlighted(r, 1)
        if not (h1 or h2):
            continue
        enc = tok(r["Sentence1"], r["Sentence2"], return_tensors="pt", return_offsets_mapping=True)
        if enc["input_ids"].shape[1] > MAX_PIECES:
            continue
        eligible.append((r, enc, h1, h2))
    order = torch.randperm(len(eligible), generator=torch.Generator().manual_seed(SEED)).tolist()
    out = []
    for i in order:
        r, enc, h1, h2 = eligible[i]
        kwargs = {"attention_mask": enc["attention_mask"], "token_type_ids": enc["token_type_ids"]}
        with torch.no_grad():
            pred = int(model(enc["input_ids"], **kwargs).logits.argmax(-1))
        if pred != mapping[GOLD.index(r["gold_label"])]:
            continue
        units = units_for(enc, r["Sentence1"], r["Sentence2"], h1, h2)
        if not units or len(units) >= enc["input_ids"].shape[1] - 1:
            continue
        out.append(
            {
                "row": r,
                "input_ids": enc["input_ids"],
                "kwargs": kwargs,
                "pred": pred,
                "human": units,
                "h_other": [
                    units_for(enc, r["Sentence1"], r["Sentence2"], *highlighted(r, a))
                    for a in (2, 3)
                ],
            }
        )
        if len(out) == limit:
            break
    return out


def iou(a: set[int], b: set[int]) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def f1(pred: set[int], gold: set[int]) -> float:
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    return 0.0 if tp == 0 else 2 * tp / (len(pred) + len(gold))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=N_SAMPLES)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    t0 = time.perf_counter()
    model, tok, mapping, info = load()
    samples = select_samples(model, tok, mapping, args.limit)
    emb = model.bert.embeddings.word_embeddings.weight.detach()
    ig = A.integrated_gradients(baseline=A.zero_baseline(), n_steps=IG_STEPS)
    evidence: dict[str, list[Any]] = {"human": [], "ig": [], "random": []}
    attributions, metas, targets = [], [], {}
    for i, s in enumerate(samples):
        x, kw = s["input_ids"], s["kwargs"]
        n_units = x.shape[1]
        metric = iv.metrics.margin([0, s["pred"]], path=PATH)
        with torch.no_grad():
            margin = metric(model(x, **kw))
        sid = sample_id(x, model_kwargs=kw)
        targets[sid] = metric.spec.target()
        attr = A.attribute(model, x, target=metric, method=ig, at=A.layer(SITE), model_kwargs=kw)
        attributions.append(attr)
        k = len(s["human"])
        perm = torch.randperm(n_units, generator=torch.Generator().manual_seed(30000 + i)).tolist()
        selections = {
            "human": F.units(A.layer(SITE), tuple(s["human"]), n_units=n_units, unit_axes=(1,)),
            "ig": F.top_k(attr, k=k, unit_axes=(1,), reduce="sum"),
            "random": F.units(
                A.layer(SITE), tuple(sorted(perm[:k])), n_units=n_units, unit_axes=(1,)
            ),
        }
        reps = {
            "mask": F.replacement(
                emb[tok.mask_token_id].expand(1, n_units, -1).clone(), name="mask"
            ),
            "pad": F.replacement(emb[tok.pad_token_id].expand(1, n_units, -1).clone(), name="pad"),
            "zero": F.zero(),
        }
        outcomes: dict[str, Any] = {}
        for approach, sel in selections.items():
            for rname, rep in reps.items():
                for test_name in ("comprehensiveness", "sufficiency"):
                    controls = F.controls(N_CONTROLS, seed=10000 + i)
                    if test_name == "comprehensiveness":
                        template = F.comprehensiveness(
                            target=metric,
                            min_drop=T * margin,
                            replacement=rep,
                            controls=controls,
                            min_fraction_below=FRACTION,
                            statement=f"the {approach} units are necessary",
                        )
                    else:
                        template = F.sufficiency(
                            target=metric,
                            max_drop=T * margin,
                            replacement=rep,
                            controls=controls,
                            min_fraction_above=FRACTION,
                            statement=f"the {approach} units suffice",
                        )
                    res = F.run(
                        model,
                        x,
                        test=template,
                        selection=sel,
                        attributions=[attr] if approach == "ig" else [],
                        model_kwargs=kw,
                    )
                    evidence[approach].append(res)
                    outcomes[f"{approach}|{rname}|{test_name}"] = res.outcome.value
        human = set(s["human"])
        ig_units = set(selections["ig"].selected)
        metas.append(
            {
                "index": i,
                "pairID": s["row"]["pairID"],
                "gold": s["row"]["gold_label"],
                "pred": s["pred"],
                "n_units": n_units,
                "k_human": k,
                "margin": margin,
                "sample_id": sid,
                "plausibility": {
                    "ig_token_f1": f1(ig_units, human),
                    "ig_iou": iou(ig_units, human),
                    "random_token_f1": f1(set(selections["random"].selected), human),
                    "inter_annotator_iou": [iou(human, set(o)) for o in s["h_other"] if o],
                },
                "outcomes": outcomes,
            }
        )
        if not args.quiet:
            print(i, k, n_units, flush=True)
    run_seconds = time.perf_counter() - t0
    roles = [
        AU.role("replacement", "tensor/mask:*", "primary"),
        AU.role("replacement", "tensor/pad:*", "alternative"),
        AU.role("replacement", "zero", "stress_test"),
        AU.role("null", "count@*", "primary"),
        AU.role("threshold", "{*}", "primary"),
        AU.role("threshold", "*|alt:*x0.5", "alternative"),
        AU.role("threshold", "*|alt:*x1.5", "stress_test"),
    ]
    audits = {}
    for approach, method in (
        ("human", "declared"),
        ("ig", "integrated_gradients"),
        ("random", "declared"),
    ):
        sel = AU.selection(Site(module=SITE), method=method, k=None)
        claims = [
            AU.claim(
                f"{approach}_necessary",
                statement=f"the {approach} units are necessary for the prediction margin",
                relation="necessary_for",
                target=None,
                sample_targets=targets,
                scope="instance",
                requirement="necessity",
                selection=sel,
                invariant_over=[AU.invariance("replacement", min_values=3)],
                roles=roles,
            ),
            AU.claim(
                f"{approach}_sufficient",
                statement=f"the {approach} units suffice for the prediction margin",
                relation="sufficient_for",
                target=None,
                sample_targets=targets,
                scope="instance",
                requirement="sufficiency",
                selection=sel,
                invariant_over=[AU.invariance("replacement", min_values=3)],
                roles=roles,
            ),
        ]
        plan = AU.plan(
            name=f"esnli_{approach}",
            checkpoint=AU.checkpoint_of(model),
            declared_model=None,
            samples=list(targets),
            datasets=[],
            claims=claims,
            requirements=[
                AU.requirement(
                    "necessity",
                    policy=F.COMPREHENSIVENESS_POLICY,
                    controls=True,
                    alternatives=[
                        AU.alternative("comprehensiveness", "min_drop", factor=0.5),
                        AU.alternative("comprehensiveness", "min_drop", factor=1.5),
                    ],
                ),
                AU.requirement(
                    "sufficiency",
                    policy=F.SUFFICIENCY_POLICY,
                    controls=True,
                    alternatives=[
                        AU.alternative("sufficiency", "max_drop", factor=0.5),
                        AU.alternative("sufficiency", "max_drop", factor=1.5),
                    ],
                ),
            ],
            concepts=[],
            naive_auroc=None,
            counterexamples=AU.counterexample_rule(
                max_counterexample_fraction=None,
                max_false_positive_rate=None,
                max_false_negative_rate=None,
            ),
        )
        body = [*evidence[approach], *(attributions if approach == "ig" else [])]
        report = bnn.audit(body, plan=plan)
        AU.verify_report(report.to_dict(), body, plan)
        tag = f"esnli_{approach}" + (f"_limit{args.limit}" if args.limit != N_SAMPLES else "")
        (ARTIFACTS / f"{tag}_report.json").unlink(missing_ok=True)
        report.save(ARTIFACTS / f"{tag}_report.json")
        audits[approach] = {
            c.name: {
                "distribution": dict(c.distribution),
                "intervals": [iv_.describe() for iv_ in c.intervals],
                "per_sample": [
                    g.standing.value
                    for g in sorted(c.groups, key=lambda g: list(targets).index(g.sample))
                ],
                "findings": [(f.kind.value, f.code, len(f.samples)) for f in c.findings],
            }
            for c in report.claims
        }
    f1s = [m["plausibility"]["ig_token_f1"] for m in metas]
    rf1s = [m["plausibility"]["random_token_f1"] for m in metas]
    out = {
        "environment": common.environment(),
        "model": info,
        "n_samples": len(samples),
        "run_seconds": run_seconds,
        "audits": audits,
        "samples": metas,
        "plausibility": {
            "ig_token_f1": AU.bootstrap(
                f1s, quantity="IG top-k_h token F1 vs annotator 1", unit="samples", seed=BOOT_SEED
            ).describe()
            if len(f1s) > 1
            else None,
            "ig_minus_random_f1": AU.paired_bootstrap(
                f1s, rf1s, quantity="token F1: IG minus random", unit="samples", seed=BOOT_SEED
            ).describe()
            if len(f1s) > 1
            else None,
        },
    }
    tag = "esnli" + (f"_limit{args.limit}" if args.limit != N_SAMPLES else "")
    (RESULTS / f"{tag}.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    print("done", tag)


if __name__ == "__main__":
    main()
