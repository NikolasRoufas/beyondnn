"""Phase 7.75 NLI shortcut stratification (docs/PHASE_7_75_PLAN.md §8). Analysis only.

For the 40 Phase-7.5 e-SNLI samples (same model, tokenizer, label mapping, selection), record
whether the SNLI model predicts the same label with an EMPTY premise, then report the Phase-7.5
PRIMARY necessity standings of the human / IG / random selections per stratum. No sample is
removed; the Phase-7.5 result file is read, never written.

Usage (experiment env, from experiments/phase5_5): python ../phase7_75/nli_strata.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase5_5"))
sys.path.insert(0, str(HERE.parent / "phase7_5"))
sys.path.insert(0, str(HERE))

import analysis775 as AN  # noqa: E402
import esnli  # noqa: E402


def main() -> None:
    torch.set_num_threads(1)
    model, tok, mapping, info = esnli.load()
    samples = esnli.select_samples(model, tok, mapping, esnli.N_SAMPLES)
    p75 = json.loads((HERE.parent / "phase7_5" / "results" / "esnli.json").read_text())
    ids = [str(s["pairID"]) for s in p75["samples"]]
    assert [s["row"]["pairID"] for s in samples] == ids, "sample selection must reproduce Phase 7.5"
    predictable = []
    for s in samples:
        enc = tok("", s["row"]["Sentence2"], return_tensors="pt")
        with torch.no_grad():
            empty = int(model(**enc).logits.argmax(-1))
        predictable.append(empty == s["pred"])
    labels = ("empty_premise_same_prediction", "needs_premise")
    strata = {
        approach: AN.stratify(
            p75["audits"][approach][f"{approach}_necessary"]["per_sample"],
            predictable,
            labels=labels,
        )
        for approach in ("human", "ig", "random")
    }
    out = {
        "policy": "docs/PHASE_7_75_PLAN.md §8 (analysis only; no sample removed)",
        "model": info,
        "n": len(samples),
        "empty_premise_same_prediction": sum(predictable),
        "per_sample": [
            {"pairID": s["row"]["pairID"], "gold": s["row"]["gold_label"], "empty_premise_same": p}
            for s, p in zip(samples, predictable, strict=True)
        ],
        "necessity_by_stratum": strata,
    }
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "nli_strata.json").write_text(json.dumps(out, indent=1, default=str))
    print(
        json.dumps(
            {k: out[k] for k in ("n", "empty_premise_same_prediction", "necessity_by_stratum")},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
