"""Phase 7 NLP case study (docs/PHASE_7_PLAN.md §30): descriptive cross-tabulations of the
BERT-tiny / SST-2 per-sample audit standings (central_C.json). No hypothesis is tested.

* special tokens: does the IG top-k selection (p = 10%) include [CLS] or [SEP]?
* sentence length: tertiles of the token count;
* negation: does the sentence contain a Phase-6 negation token?
* top-1 position: is the IG top-1 token the first, the last, or an interior token?
* token-removal OOD proxy: replacement sensitivity (zero vs [MASK] vs [PAD] embedding).

The IG attribution is recomputed (deterministic) only to read the top-1 position; its top-k
at p = 10% must equal the units recorded by central.py (checked).

Usage (experiment env, from experiments/phase5_5): python ../phase7/nlp.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase5_5"))
sys.path.insert(0, str(HERE.parent / "phase6"))
import common  # noqa: E402
import run_faithfulness as RF  # noqa: E402
import torch  # noqa: E402
from realistic import NEGATION  # noqa: E402

import beyondnn as bnn  # noqa: E402

A, F = bnn.attribution, bnn.faithfulness
RESULTS = HERE / "results"
CLAIMS = ("ig_necessary", "ig_sufficient", "g_necessary", "r_necessary")


def table(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for claim in CLAIMS:
        cells: dict[str, dict[str, int]] = {}
        for r in rows:
            cell = cells.setdefault(str(r[key]), {})
            standing = r["standing"][claim]
            cell[standing] = cell.get(standing, 0) + 1
        out[claim] = {k: dict(sorted(v.items())) for k, v in sorted(cells.items())}
    return out


def main() -> None:
    torch.set_num_threads(1)
    data = json.loads((RESULTS / "central_C.json").read_text())
    setting = data["settings"]["tokens"]
    audit = setting["audit"]["claims"]
    model, samples, settings, _info, path = RF.settings_C()
    _, tok, _ = common.bert()
    st = settings[0]
    by_index = {s.index: s for s in samples}
    rows = []
    mismatched = 0
    lengths = sorted(m["n_units"] for m in setting["samples"])
    cut1, cut2 = lengths[len(lengths) // 3], lengths[2 * len(lengths) // 3]
    for meta in setting["samples"]:
        sample = by_index[meta["index"]]
        ids = sample.inputs[0][0].tolist()
        tokens = tok.convert_ids_to_tokens(ids)
        with torch.no_grad():
            logits = model(*sample.inputs, **sample.kwargs).logits
        metric, *_ = common.margin_target(logits, path=path)
        attr = A.attribute(
            model,
            *sample.inputs,
            target=metric,
            method=RF._methods(st)["integrated_gradients"],
            at=st.at,
            model_kwargs=sample.kwargs,
        )
        ranking = F.ranking(attr, unit_axes=st.unit_axes, reduce=RF.REDUCE["integrated_gradients"])
        k = meta["naive_ig_r1_p10_count"]["k"]
        units = sorted(ranking.order[:k])
        mismatched += units != meta["naive_ig_r1_p10_count"]["units"]
        top1 = ranking.order[0]
        t = len(tokens)
        special = {0, t - 1}
        words = {w.lower() for w in tokens}
        rows.append(
            {
                "index": meta["index"],
                "tokens": t,
                "text": meta["text"],
                "selection_has_special_token": bool(special & set(units)),
                "top1_position": "first" if top1 == 0 else "last" if top1 == t - 1 else "interior",
                "top1_token": tokens[top1],
                "length_tertile": "short" if t <= cut1 else "medium" if t <= cut2 else "long",
                "has_negation": bool(words & set(NEGATION)) or any("n't" in w for w in words),
                "standing": {c: audit[c]["per_sample"][str(meta["index"])] for c in CLAIMS},
            }
        )
    out = {
        "environment": common.environment(),
        "plan_section": "docs/PHASE_7_PLAN.md §30 (descriptive)",
        "n": len(rows),
        "recomputed_selection_mismatches": mismatched,
        "length_cuts": [cut1, cut2],
        "by_special_token_in_selection": table(rows, "selection_has_special_token"),
        "by_length_tertile": table(rows, "length_tertile"),
        "by_negation": table(rows, "has_negation"),
        "by_top1_position": table(rows, "top1_position"),
        "top1_tokens": sorted({r["top1_token"] for r in rows}),
        "replacement_sensitive_samples": {
            c: audit[c]["samples_sensitive_by_axis"].get("replacement", 0) for c in CLAIMS
        },
        "rows": rows,
        "documented_not_tested": {
            "padding": "batch size 1; attention masks are all ones; padding effects not evaluated",
            "label_leakage": (
                "third-party fine-tune on SST-2 train; validation split used; "
                "training data not verifiable"
            ),
            "lexical_shortcuts": "see concepts_C.json (K7 positive words, K9 negation)",
        },
    }
    (RESULTS / "nlp.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "n",
                    "recomputed_selection_mismatches",
                    "by_special_token_in_selection",
                    "by_top1_position",
                )
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
