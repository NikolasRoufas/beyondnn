# Checkpoint robustness (Stage 4, S4-SEED)

- **Protocol:** P1, committed at `140bb2e` before execution.
- **Script:** `experiments/tnnls_revision/run_central_configs.py --seed S`.
- **Aggregation:** `experiments/tnnls_revision/stage4/aggregate.py` → `experiments/tnnls_revision/results/aggregate.json`.
- **Execution:** `interstellar`, Linux x86-64, CPU, one torch thread per process, 2026-11-03.

## 1. What was replicated, and what was not

- **Replicated:**
  - MLP (A) and CNN (B), each retrained with seeds 0–4. The code is `common.train` with the seed as its only change.
  - The full frozen central design, re-run on each checkpoint's held-out samples. The samples are drawn by the frozen rule (correctly classified test inputs, in seeded order), so the sample *sets* differ between checkpoints.
- **Not replicated:**
  - BERT-tiny (C) and BERT-base (D) are fixed public fine-tunes; only one checkpoint each exists in this study.
  - Intervention and control seeds were not varied. Varying them would not be model replication.
- **Cross-platform observation:** Linux seed 0 differs from the frozen macOS seed 0. Identical code and seed produced a **different checkpoint digest on Linux x86-64 than on macOS arm64**, for both A (`a9e2877b…` vs `4c19a8da…`) and B (`12f9082e…` vs `d799d1e8…`). Training is not bit-reproducible across platforms. The frozen checkpoints were therefore rebuilt on macOS for Stage 2 (digests verified), and Linux seed 0 is reported as a sixth checkpoint.

## 2. Results

Each cell gives IG necessity PRIMARY-SUPPORTED / of which ALTERNATIVE-reversed / of which reversed with the selected units fixed / random selection SUPPORTED. The denominator is the site's held-out sample count.

| site (n) | frozen (macOS) | seed 0 (Linux) | seed 1 | seed 2 | seed 3 | seed 4 | range of IG support |
|---|---|---|---|---|---|---|---|
| MLP input (51) | 8 / 7 / 3 / 1 | 8 / 7 / 3 / 1 | 8 / 8 / 4 / 0 | 7 / 6 / 5 / 0 | 4 / 4 / 3 / 1 | 7 / 6 / 4 / 1 | 4–8 |
| MLP hidden (51) | 4 / 2 / 2 / 0 | 4 / 2 / 2 / 0 | 2 / 1 / 0 / 0 | 2 / 2 / 2 / 0 | 2 / 2 / 0 / 0 | 2 / 2 / 0 / 0 | 2–4 |
| CNN pixels (60) | 14 / 13 / 10 / 1 | 14 / 13 / 10 / 1 | 22 / 20 / 15 / 2 | 20 / 19 / 12 / 3 | 17 / 16 / 13 / 0 | 12 / 12 / 10 / 1 | 12–22 |
| CNN relu2 (60) | 1 / 1 / 1 / 0 | 1 / 1 / 1 / 0 | 2 / 1 / 1 / 1 | 3 / 3 / 2 / 1 | 4 / 3 / 3 / 0 | 4 / 3 / 3 / 0 | 1–4 |

Test accuracy per checkpoint:
- MLP 0.974 on all six.
- CNN 0.958–0.967.

## 3. Pre-stated hypotheses

| hypothesis | result |
|---|---|
| (i) IG support > random support on every checkpoint and site | **holds** on all 24 (checkpoint, site) pairs (seeds 0–4 × 4 sites, plus frozen) |
| (ii) on every checkpoint with ≥ 3 IG supports, at least half have an ALTERNATIVE reversal | **holds**: the reversed share is 75–100% wherever there are ≥ 3 supports |

## 4. Interpretation

1. **The qualitative findings are stable across independently trained checkpoints:**
   - IG necessity support is rare (MLP input 8–16% of samples, CNN pixels 20–37%, hidden sites ≤ 8%);
   - it exceeds random support;
   - it almost always carries a qualifying ALTERNATIVE reversal.
2. **The fixed-subject share replicates for the CNN pixel site:**
   - With the selected units held fixed, a declared alternative replacement or null reverses 10–15 of 12–22 supports, i.e. 60–83%.
   - The MLP input site is lower and more variable: 3–5 of 4–8.
3. **The absolute counts vary across checkpoints:** CNN pixels 12–22 of 60. A single-checkpoint count, such as the frozen 14/60, is therefore not a stable estimate of a model-class rate, and the paper must not present it as one.
4. **CNN relu2:** every IG necessity test at this site carries `control_criterion_unattainable` on every checkpoint (60/60 samples). With few units per selection, too many control draws coincide with the selection, so the site's count-null tests are largely uninformative. Its 1–4 supports come from the tests that remain decisive. This is a property of the design at that site and must be stated alongside its numbers.
5. **Gradient versus IG at CNN pixels:** gradient top-k is supported on 37–41 of 60 samples on every checkpoint, against 12–22 for IG. This replicates the manuscript's single-checkpoint observation (37 vs 14).

## 5. Fresh, never-inspected samples (protocol P3)

- **Run:** CNN (frozen checkpoint) and BERT-tiny re-run on samples that no earlier phase used.
- **Supports:**
  - CNN pixels: 19/60 IG supports, all 19 reversed (16 with the units fixed); random 1.
  - CNN relu2: 2/60; random 1.
  - BERT-tiny: 9/40 IG supports, 6 reversed (5 only through changes of k); random 0.
- **Predictions:** all five pre-stated predictions hold (CONFIGURATION_SENSITIVITY_ANALYSIS §5).

## 6. Dependence and statistics

- Samples are nested in checkpoints, and checkpoints are not pooled.
- The between-checkpoint spread above is the relevant uncertainty for model-class statements. Five checkpoints per model give a range, not an interval estimate.
- No significance test was run.

## 7. What remains unsupported

- No checkpoint replication for the BERT models.
- No architecture or scale generalisation beyond these four models.
- The held-out sample sets differ across checkpoints (correctness-conditional selection). Paired, same-sample comparisons across checkpoints were not made.
