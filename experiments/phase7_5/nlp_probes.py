# ruff: noqa: E501
"""Phase 7.5 token-level OOD, special tokens, padding and leakage probes
(docs/PHASE_7_5_FROZEN_POLICY.md §N3-§N4). Direct model measurements (HF forward passes),
outside BeyondNN's intervention protocols, declared as such.

Models and samples:
* C: BERT-tiny SST-2 (Phase 5.5 checkpoint; development), its 40 Phase-5.5 samples;
* D: BERT-base SST-2 (textattack; held-out model), its 40 held-out samples (central75 D);
* E: BERT-base SNLI (textattack; held-out model), the 40 e-SNLI samples (esnli.py).
Reference sets for OOD distances: 200 other inputs from the same source (not samples, not
calibration rows).

Per sample and every non-special position j, strategies: zero / [MASK] / [PAD] / [UNK] embedding
at the word-embedding output, and deletion of token j. Recorded: |delta margin| (predicted class
vs best other), prediction flip, KL(p_clean || p_perturbed), cosine distance of the final-layer
[CLS] vector to the clean one, and an OOD percentile: the rank of the perturbed [CLS]'s nearest-
neighbour distance to the reference set among the reference set's leave-one-out nearest-
neighbour distances (100 = further than every clean reference input).

Special tokens: [CLS], each [SEP] replaced by [MASK] / zero. Padding: right padding to 64 with and
without the attention mask. Leakage: SST-2 exact duplicates in GLUE train; SNLI hypothesis-only
accuracy (empty premise; [MASK] premise) on 500 e-SNLI test rows; word-order shuffles.

Usage (experiment env, from experiments/phase7_5/../phase5_5): python ../phase7_5/nlp_probes.py C|D|E [--limit N] [--quiet]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase5_5"))
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import torch  # noqa: E402

import beyondnn as bnn  # noqa: E402

AU = bnn.audits
RESULTS = HERE / "results"
STRATEGIES = ("zero", "mask", "pad", "unk", "delete")
N_REFERENCE = 200
PAD_TO = 64
N_SHUFFLES = 5
BOOT_SEED = 7502
HYPOTHESIS_ONLY_ROWS = 500


def load(
    model_id: str, limit: int
) -> tuple[Any, Any, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """(model, tokenizer, samples, reference inputs, info); inputs are dicts of HF kwargs."""
    if model_id == "C":
        import run_faithfulness as RF

        model, tok, info = common.bert()
        pool, _ = RF.c_samples(model, tok, 10_000)
        chosen = pool[:40][:limit]
        samples = [{"text": s.text, "enc": tok(s.text, return_tensors="pt")} for s in chosen]
        used = {s.text for s in pool[:80]}
        rows = [r for r in common.sst2_validation() if r["sentence"] not in used]
        ref = [{"enc": tok(r["sentence"], return_tensors="pt")} for r in rows[-N_REFERENCE:]]
        return model, tok, samples, ref, info
    if model_id == "D":
        import central75

        model, chosen, _, info, _ = central75.build("D", "heldout")
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(central75.D_REPO, revision=central75.D_REVISION)
        samples = [
            {"text": s.text, "enc": tok(s.text, return_tensors="pt")} for s in chosen[:limit]
        ]
        used = {s.text for s in chosen}
        rows = [
            r
            for r in common.sst2_validation()[central75.D_CALIBRATION_ROWS :]
            if r["sentence"] not in used
        ]
        ref = [{"enc": tok(r["sentence"], return_tensors="pt")} for r in rows[-N_REFERENCE:]]
        return model, tok, samples, ref, info
    import esnli

    model, tok, mapping, info = esnli.load()
    chosen = esnli.select_samples(model, tok, mapping, 40)[:limit]
    samples = [
        {
            "text": s["row"]["Sentence1"] + " || " + s["row"]["Sentence2"],
            "enc": tok(s["row"]["Sentence1"], s["row"]["Sentence2"], return_tensors="pt"),
            "row": s["row"],
        }
        for s in chosen
    ]
    used = {s["row"]["pairID"] for s in chosen}
    rows = [
        r for r in esnli.rows("test") if r["gold_label"] in esnli.GOLD and r["pairID"] not in used
    ]
    ref = [
        {"enc": tok(r["Sentence1"], r["Sentence2"], return_tensors="pt")}
        for r in rows[-N_REFERENCE:]
    ]
    info["mapping"] = mapping
    return model, tok, samples, ref, info


def forward(
    model: Any, enc: dict[str, torch.Tensor], replace: tuple[int, torch.Tensor] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """(probabilities, final-layer [CLS] vector); ``replace`` = (position, embedding row)."""
    handle = None
    if replace is not None:
        pos, vec = replace

        def hook(_m: Any, _i: Any, out: torch.Tensor) -> torch.Tensor:
            out = out.clone()
            out[:, pos] = vec
            return out

        handle = model.bert.embeddings.word_embeddings.register_forward_hook(hook)
    try:
        with torch.no_grad():
            o = model(
                **{k: v for k, v in enc.items() if k != "offset_mapping"}, output_hidden_states=True
            )
    finally:
        if handle is not None:
            handle.remove()
    return o.logits[0].softmax(-1), o.hidden_states[-1][0, 0]


def margin_of(probs: torch.Tensor, cls: int) -> float:
    logits = probs.log()
    others = torch.cat([logits[:cls], logits[cls + 1 :]])
    return float(logits[cls] - others.max())


def delete(enc: dict[str, torch.Tensor], pos: int) -> dict[str, torch.Tensor]:
    return {
        k: torch.cat([v[:, :pos], v[:, pos + 1 :]], dim=1)
        for k, v in enc.items()
        if k != "offset_mapping"
    }


def nn_distance(v: torch.Tensor, ref: torch.Tensor) -> float:
    return float((ref - v).norm(dim=1).min())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["C", "D", "E"])
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, tok, samples, ref, info = load(args.model, args.limit)
    emb = model.bert.embeddings.word_embeddings.weight.detach()
    vec = {
        "zero": torch.zeros(emb.shape[1]),
        "mask": emb[tok.mask_token_id],
        "pad": emb[tok.pad_token_id],
        "unk": emb[tok.unk_token_id],
    }
    ref_cls = torch.stack([forward(model, r["enc"])[1] for r in ref])
    loo = sorted(
        float(torch.cat([ref_cls[:i], ref_cls[i + 1 :]]).sub(ref_cls[i]).norm(dim=1).min())
        for i in range(len(ref_cls))
    )

    def percentile(d: float) -> float:
        return 100.0 * sum(x < d for x in loo) / len(loo)

    special = {tok.cls_token_id, tok.sep_token_id, tok.pad_token_id}
    per_sample = []
    for s in samples:
        enc = s["enc"]
        ids = enc["input_ids"][0].tolist()
        p0, c0 = forward(model, enc)
        pred = int(p0.argmax())
        m0 = margin_of(p0, pred)
        stats: dict[str, dict[str, list[float]]] = {
            st: {"dmargin": [], "flip": [], "kl": [], "cos": [], "ood": []} for st in STRATEGIES
        }
        for j, t in enumerate(ids):
            if t in special:
                continue
            for st in STRATEGIES:
                p, c = (
                    forward(model, delete(enc, j))
                    if st == "delete"
                    else forward(model, enc, (j, vec[st]))
                )
                stats[st]["dmargin"].append(abs(margin_of(p, pred) - m0))
                stats[st]["flip"].append(float(int(p.argmax()) != pred))
                stats[st]["kl"].append(float((p0 * (p0.log() - p.log())).sum()))
                stats[st]["cos"].append(
                    float(1 - torch.nn.functional.cosine_similarity(c, c0, dim=0))
                )
                stats[st]["ood"].append(percentile(nn_distance(c, ref_cls)))
        spec_effects = {}
        for name, positions in (
            ("cls", [0]),
            ("sep", [j for j, t in enumerate(ids) if t == tok.sep_token_id]),
        ):
            for st in ("mask", "zero"):
                d = [
                    abs(margin_of(forward(model, enc, (j, vec[st]))[0], pred) - m0)
                    for j in positions
                ]
                f = [
                    float(int(forward(model, enc, (j, vec[st]))[0].argmax()) != pred)
                    for j in positions
                ]
                spec_effects[f"{name}|{st}"] = {"dmargin": sum(d) / len(d), "flip": sum(f) / len(f)}
        t = len(ids)
        padded = (
            tok.pad(dict(enc), padding="max_length", max_length=PAD_TO, return_tensors="pt")
            if t < PAD_TO
            else enc
        )
        pm, _ = forward(model, padded)
        nomask = {k: v for k, v in padded.items()}
        nomask["attention_mask"] = torch.ones_like(padded["attention_mask"])
        pn, _ = forward(model, nomask)
        shuffles = []
        words = s["text"].split(" ")
        for sh in range(N_SHUFFLES):
            g = torch.Generator().manual_seed(9000 + sh)
            order = torch.randperm(len(words), generator=g).tolist()
            text = " ".join(words[i] for i in order)
            enc_s = (
                tok(*text.split(" || "), return_tensors="pt")
                if " || " in text
                else tok(text, return_tensors="pt")
            )
            shuffles.append(float(int(forward(model, enc_s)[0].argmax()) == pred))
        per_sample.append(
            {
                "text": s["text"],
                "tokens": t,
                "pred": pred,
                "margin": m0,
                "strategies": {
                    st: {k: (sum(v) / len(v) if v else None) for k, v in d.items()}
                    for st, d in stats.items()
                },
                "special": spec_effects,
                "padding": {
                    "masked_max_abs_prob_diff": float((pm - p0).abs().max()),
                    "unmasked_flip": float(int(pn.argmax()) != pred),
                    "unmasked_dmargin": abs(margin_of(pn, pred) - m0),
                },
                "shuffle_same_prediction": sum(shuffles) / len(shuffles),
            }
        )
        if not args.quiet:
            print(len(per_sample), flush=True)
    summary: dict[str, Any] = {}
    for st in STRATEGIES:
        for key in ("dmargin", "flip", "kl", "cos", "ood"):
            vals = [
                r["strategies"][st][key] for r in per_sample if r["strategies"][st][key] is not None
            ]
            if len(vals) > 1:
                summary[f"{st}|{key}"] = AU.bootstrap(
                    vals,
                    quantity=f"{st} {key} (per-sample mean over positions)",
                    unit="samples",
                    seed=BOOT_SEED,
                ).describe()
    for a, b in (("zero", "mask"), ("pad", "mask"), ("delete", "mask")):
        va = [r["strategies"][a]["ood"] for r in per_sample]
        vb = [r["strategies"][b]["ood"] for r in per_sample]
        if len(va) > 1:
            summary[f"ood_{a}_minus_{b}"] = AU.paired_bootstrap(
                va, vb, quantity=f"OOD percentile: {a} minus {b}", unit="samples", seed=BOOT_SEED
            ).describe()
    for key in ("cls|mask", "cls|zero", "sep|mask", "sep|zero"):
        va = [r["special"][key]["dmargin"] for r in per_sample]
        vb = [r["strategies"][key.split("|")[1]]["dmargin"] for r in per_sample]
        if len(va) > 1:
            summary[f"special_{key}_minus_interior"] = AU.paired_bootstrap(
                va,
                vb,
                quantity=f"|delta margin| {key} minus mean interior token",
                unit="samples",
                seed=BOOT_SEED,
            ).describe()
    leakage: dict[str, Any] = {}
    if args.model in ("C", "D"):
        import urllib.request

        import pyarrow.parquet as pq

        path = HERE / "artifacts" / f"sst2_train_{common.GLUE_REVISION[:12]}.parquet"
        if not path.exists():
            urllib.request.urlretrieve(
                common.SST2_URL.replace("validation-00000-of-00001", "train-00000-of-00001"), path
            )
        train = {r["sentence"].strip().lower() for r in pq.read_table(path).to_pylist()}
        dup = [s["text"] for s in samples if s["text"].strip().lower() in train]
        leakage = {
            "sst2_train_rows": len(train),
            "samples_exactly_in_train": len(dup),
            "examples": dup[:5],
        }
    else:
        import esnli

        rows = [r for r in esnli.rows("test") if r["gold_label"] in esnli.GOLD][
            :HYPOTHESIS_ONLY_ROWS
        ]
        mapping = info["mapping"]
        full = empty = masked = 0
        for r in rows:
            gold = mapping[esnli.GOLD.index(r["gold_label"])]
            full += (
                int(
                    forward(model, tok(r["Sentence1"], r["Sentence2"], return_tensors="pt"))[
                        0
                    ].argmax()
                )
                == gold
            )
            empty += (
                int(forward(model, tok("", r["Sentence2"], return_tensors="pt"))[0].argmax())
                == gold
            )
            n_premise = len(tok.tokenize(r["Sentence1"]))
            masked += (
                int(
                    forward(
                        model,
                        tok(
                            " ".join([tok.mask_token] * n_premise),
                            r["Sentence2"],
                            return_tensors="pt",
                        ),
                    )[0].argmax()
                )
                == gold
            )
        n = len(rows)
        leakage = {
            "rows": n,
            "full_accuracy": AU.wilson(
                full, n, quantity="accuracy (premise + hypothesis)", unit="e-SNLI test rows"
            ).describe(),
            "empty_premise_accuracy": AU.wilson(
                empty, n, quantity="accuracy (empty premise)", unit="e-SNLI test rows"
            ).describe(),
            "masked_premise_accuracy": AU.wilson(
                masked, n, quantity="accuracy ([MASK] premise)", unit="e-SNLI test rows"
            ).describe(),
            "chance": 1 / 3,
        }
    out = {
        "environment": common.environment(),
        "model": {k: v for k, v in info.items() if k != "mapping"},
        "model_id": args.model,
        "n_samples": len(per_sample),
        "n_reference": len(ref),
        "summary": summary,
        "leakage": leakage,
        "samples": per_sample,
    }
    tag = f"nlp_probes_{args.model}" + (f"_limit{args.limit}" if args.limit != 40 else "")
    (RESULTS / f"{tag}.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    print("done", tag)


if __name__ == "__main__":
    main()
