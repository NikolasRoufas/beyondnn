# Configuration-sensitivity analysis (Stage 2)

- **Protocol:** P1, committed at `140bb2e` before execution.
- **Fresh-sample confirmation:** P3, committed at `d023b8c` before execution.
- **Scripts:**
  - `experiments/tnnls_revision/run_central_configs.py` re-runs the frozen design and exports per-configuration records;
  - `stage2/decompose.py`;
  - `stage4/aggregate.py`;
  - `figures/make_tables.py` and `figures/make_figures.py`.
- **Data:** `experiments/tnnls_revision/results/runs/` holds compact per-run outputs, including `configs.csv.gz`, every tested configuration of every necessity claim per sample. `results/aggregate.json` holds the aggregate.
- **Manuscript:** Table III and Fig. 3.

## 1. Why the committed artifacts were insufficient

- **What the frozen JSON stores:** only role × outcome counts per sample.
- **What the `reversal_axes` field merges:** ALTERNATIVE and STRESS_TEST findings (Stage 1, R01).
- **Where the axis values live:** only in the audit reports, which were never committed.

The decomposition therefore required regenerating evidence. Artifact-only analysis could bound the effects but could not isolate them.

## 2. Exact re-execution (prerequisite)

- **Design:** the frozen central design was re-run unchanged on `interstellar` (Linux x86-64, CPU).
- **Checkpoints:**
  - MLP and CNN: the frozen checkpoints, rebuilt on macOS, with digests verified;
  - BERT models: the pinned HF revisions.
- **Result:** every per-sample standing, finding code and outcome profile of all six claims at all six sites equals the frozen macOS results. **0 differences.**
- **Consequence:** the decomposition below concerns exactly the 52 samples behind 47/52.

## 3. Definitions

- **Unit:** a (site, held-out sample) whose IG necessity claim is PRIMARY-SUPPORTED. There are 52 such units, nested in four checkpoints.
- **Changed axes:** the axes on which a configuration differs from the PRIMARY one (replacement, k, null, threshold).
- **Fixed subject:** k unchanged. The IG ranking does not depend on the replacement, null or threshold, so the selected units are identical.
  - **A. Replacement only:** fixed selection, replacement changed.
  - **B. Null only:** fixed selection, control rule changed (count → magnitude-matched).
  - **C. Threshold only:** fixed selection, criterion re-evaluated (τ × 0.5).
- **D. Subject change:** k changed. This alters the units and therefore the claim.
- **E. Multi-axis:** ≥ 2 axes changed.
- **Reversal:** the configuration's outcome is `contradicts` while PRIMARY `supports`. `inconclusive` is a separate category, never counted as a reversal.

## 4. Results: held-out samples (the 52 behind 47/52)

| site | n | supported | reversed (any ALT) | reversed, units fixed | A: replacement only | B: null only | C: threshold only | D: k only (single axis) | reversed *only* by k changes |
|---|---|---|---|---|---|---|---|---|---|
| MLP input | 51 | 8 | 7 | 3 | 3 | 1 | 0 | 5 | 4 |
| MLP hidden | 51 | 4 | 2 | 2 | 2 | 0 | 0 | 2 | 0 |
| CNN pixels | 60 | 14 | 13 | 10 | 9 | 7 | 0 | 7 | 3 |
| CNN relu2 | 60 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 0 |
| BERT-tiny | 40 | 14 | 13 | 3 | 0 | 3 | 0 | 11 | 10 |
| BERT-base | 40 | 11 | 11 | 4 | 4 | 1 | 0 | 10 | 7 |
| **total** | 302 | **52** | **47** | **23** | **19** | **12** | **0** | **36** | **24** |

**E (multi-axis interactions):**
- No unit was reversed with fixed units *only* through a multi-axis configuration (0 at every site).
- Every fixed-subject reversal has a single-axis explanation.
- Multi-axis configurations that include the threshold do reverse, but always together with a reversing single axis.

**C (threshold):**
- τ × 0.5 reversed no supported claim, at any site, in any run.
- This is exactly what the threshold-asymmetry property predicts (lowering τ cannot turn *supports* into *contradicts*).
- The threshold ALTERNATIVE is therefore structurally unable to qualify a supported claim, which a reader of "47/52" could not have known.

**Standing:** all 47 reversed units keep the standing SUPPORTED, with the qualifying finding `alternative_reverses` (role invariance).

## 5. Results: fresh, never-inspected samples (P3)

| site | n | supported | reversed | units fixed | replacement only | null only | threshold only | k only (single) | only by k changes |
|---|---|---|---|---|---|---|---|---|---|
| CNN pixels | 60 | 19 | 19 | 16 | 10 | 12 | 0 | 15 | 3 |
| CNN relu2 | 60 | 2 | 2 | 1 | 1 | 0 | 0 | 1 | 1 |
| BERT-tiny | 40 | 9 | 6 | 1 | 0 | 1 | 0 | 4 | 5 |

- The fresh samples are disjoint from the dev and held-out sets (verified: 0 overlap).
- Random-selection supports: CNN pixels 1, CNN relu2 1, BERT-tiny 0.

**Pre-stated predictions:**

| id | prediction | result |
|---|---|---|
| F-i | IG support > random support at every site | 19 > 1; 2 > 1; 9 > 0: **holds** |
| F-ii | ≥ 50% of supports reversed where there are ≥ 3 supports | 19/19; 6/9: **holds** |
| F-iii | CNN pixels: ≥ 50% of supports reversed with units fixed | 16/19 = 84%: **holds** |
| F-iv | BERT-tiny: ≥ 50% of reversals only by k changes | 5/6 = 83%: **holds** |
| F-v | τ × 0.5 reverses no supported claim | 0 everywhere: **holds** |

## 6. Robustness across checkpoints

From the seed runs (`ROBUSTNESS_REPORT.md`): the share of supports reversed with the units fixed is 60–83% on CNN pixels across five checkpoints and the frozen one, and 0–100% on the sparse MLP sites.

## 7. Interpretation

1. **The 47/52 headline conflated two different phenomena:**
   - In **23** of the 47 reversals, a declared alternative *intervention or control rule* reverses the conclusion about the *same* units. This is genuine configuration dependence of a fixed claim: replacement in 19 units, null in 12, with overlap.
   - In **24**, only a change of k reverses the conclusion. k changes which units are claimed, so this is a dependence of the *conclusion on the claim's own definition*, not instability of a fixed test.
2. **The split is model-dependent.**
   - On CNN pixels, fixed-subject reversals dominate: 10 of 13 held-out, 16 of 19 fresh, and 60–83% across checkpoints.
   - On the token models, subject change dominates: BERT-tiny 10 of 13 held-out and 5 of 6 fresh; BERT-base 7 of 11.
   - On the token models, IG necessity often holds only at the declared k.
3. **The decomposition is conditional on the declared family.** The count can only grow with more alternatives, and it is not an instability rate.
4. **Uncertainty and dependence:**
   - Samples are nested in four checkpoints, and per-site counts are small.
   - No pooled interval is reported.
   - Cross-checkpoint ranges are in Table V.

## 8. Recommendation on the headline

- **Keep 47/52 as the headline only together with its decomposition.** The single number must not be presented as fixed-subject intervention sensitivity.
- **Recommended manuscript wording:** the abstract and §VI-A now use it.

> "On six held-out sites, 47 of 52 PRIMARY-supported IG necessity claims had at least one contradicting declared alternative. In an exact re-execution, 23 of the 47 reversals keep the selected units fixed (a change of replacement or control rule reverses the conclusion about the same units), and 24 arise only from changing the selection size, hence the claim's subject. On never-inspected samples of two models the same pattern recurs."

- **Avoid:**
  - "47 of 52 conclusions are unstable under the intervention";
  - "IG is unreliable";
  - "47/52 shows replacement sensitivity".
