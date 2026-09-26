# Phase 7.5 Frozen Evaluation Policy

- **Status:** frozen. It is committed before any held-out Phase-7.5 evaluation ran.
- **Changes after the freeze** are appended under "Deviations" in this format: original rule / problem / reason / replacement rule / whether affected results become exploratory. Nothing above that section is edited.
- **Code:** the experiment code is committed with this document (`experiments/phase7_5/*.py`); its module constants are part of the policy.
- **Framework:** BeyondNN at the freeze commit (ADR-044 to ADR-051).

**What was seen before freezing (development evidence):**
- Phase 5.5/6/7 results on models A, B, C (the Phase-5.5 samples) and the Phase-6 concepts;
- InterpBench cases 7 and 13 (`results/external_interpbench_dev.json`) and Tracr case 3 (`results/external_tracr_concepts_dev.json`);
- code smoke runs with 1 sample (outputs deleted unread);
- model loading, the D label-mapping calibration code, and forward-pass timing.

Development findings that shaped this policy:
- **The margin metric.** A fixed runner-up margin under-detects argmax changes in multi-class outputs. Hence the builtin `metrics.margin` (ADR-051) for the new experiments.
- **Case reliability.** Case 13's trained model agrees with its Tracr model on 19% of clean inputs. Hence the case-reliability rule.

## 1. Standing rules

- **Standings and findings:** Phase 7 §22 with D1 (ADR-044), refined by roles (ADR-048). With declared roles, the standing and verdict use PRIMARY configurations only.
- **Reversals:** ALTERNATIVE reversals give `alternative_reverses` (QUALIFYING); STRESS_TEST reversals give `stress_test_reverses` (INFORMATIONAL).
- **Undeclared configurations** are listed and never part of a standing. With no PRIMARY configuration tested, the finding is `primary_untested`.
- **No score** (ADR-007). Counts and fractions are descriptive only.
- **Configuration-level disagreement** (`configuration_level_disagreement`) is always reported alongside the standing.

## 2. Required evidence, controls, thresholds

| setting | requirement | controls | PRIMARY threshold | alternatives |
|---|---|---|---|---|
| InterpBench (E1) | comprehensiveness | not required (PRIMARY null `none`; count@0.95, N=20, is ALTERNATIVE for heads) | min_drop = clean margin (`metrics.margin`: "the argmax changes") | ×0.5 ALTERNATIVE, ×1.5 STRESS_TEST |
| central (C1: A, B, C, D) | comprehensiveness / sufficiency | required; PRIMARY null count@0.95 (N=50) | 0.5 × clean margin (A, B, C: the Phase-7 `difference(pred, runner-up)`; D: identical for 2 classes) | ×0.5 ALTERNATIVE, ×1.5 STRESS_TEST |
| e-SNLI (N2) | comprehensiveness / sufficiency | required; count@0.95 (N=50) | 0.5 × clean margin (`metrics.margin`) | ×0.5 ALTERNATIVE, ×1.5 STRESS_TEST |
| concepts (E2) | POLICY_V1 | encoding: random directions (200) + label permutation (200) at 0.95; use: random directions (50, covariance) at 0.95, min_change 0.1 | — | — |

## 3. Roles (declared per setting)

**E1, InterpBench:**
- replacement: PRIMARY `tensor/resample:*` (the interchange source sample, i.e. the benchmark's own semantics); ALTERNATIVE `tensor/mean:*`; STRESS_TEST `zero`;
- null: PRIMARY `none`; ALTERNATIVE `count@*`;
- threshold: PRIMARY recorded; ALTERNATIVE ×0.5; STRESS_TEST ×1.5.

**C1, central:**

| site | PRIMARY replacement | ALTERNATIVE | STRESS_TEST |
|---|---|---|---|
| A/input | zero (= train mean, standardised) | resample train row | train min |
| A/net.1 | train mean | resample train row | zero |
| B/pixels | train-mean image | resample train image | zero |
| B/relu2 | train-mean map | resample train map | zero |
| C/tokens, D/tokens | [MASK] embedding | [PAD] embedding | zero |

- k: PRIMARY is the k given by p = 10% on each sample; ALTERNATIVE is the k of p = 5% and 20% (per-sample rules; when k coincides, the PRIMARY value wins).
- null: PRIMARY count@0.95; ALTERNATIVE magnitude@0.95; STRESS_TEST none.
- threshold: as E1.

**N2, e-SNLI:** replacement PRIMARY [MASK]; ALTERNATIVE [PAD]; STRESS_TEST zero. Null PRIMARY count@0.95. Threshold as E1.

**E2, concepts:** null PRIMARY covariance; ALTERNATIVE isotropic. Every replacement and dataset is PRIMARY.

**Rationale:**
- Mean / [MASK] / resample are the literature's less-OOD choices (Hase et al. 2021; causal scrubbing; InterpBench's SIIT resample ablation).
- Zero is the classic OOD stress test.
- The count null is the Phase-7 policy (D9).

## 4. Counterexample caps

- **Concepts:** both caps are reported, strict (FP and FN ≤ 0.10) and lenient (≤ 0.30). They are not tuned; they are the Phase-7 values.
- **Per-sample claims:** no cap is declared, so counterexample identities are always listed.

## 5. Invariances

- **Central:** necessary claims are invariant over replacement (3), k (2), null (2); sufficient claims over replacement (3), k (2).
- **e-SNLI:** replacement (3).
- **E1:** none declared, because roles carry the structure.

## 6. Uncertainty

- **Methods:**
  - Wilson 95% intervals for every proportion (unit: the declared samples or the declared claim instances, stated per table);
  - percentile bootstrap (B = 10,000) for means;
  - paired bootstrap for within-sample differences.
- **Seeds:** 7501 (e-SNLI), 7502 (probes), 7503 (external tables), 7504 (central tables).
- **Confirmatory interval hypotheses** (§9: N2, N3-D, N3-E, N4) use Bonferroni-adjusted 98.75% intervals (4 tests; a conservative version of Holm). Everything else is descriptive and unadjusted.

## 7. Sample selection

- **E1:** per case, the first 40 inputs of `case.get_clean_data(max_samples=400, seed=42, unique_data=True)` whose trained-model prediction at the last position equals the Tracr model's. Interchange source of sample i is sample (i+1) mod 40.
- **E2:** the unique inputs of `get_clean_data(max_samples=400, seed=42, unique_data=True)`, split 50% / 20% / 30% by a seeded permutation (1234). Label: fraction of 'x' > the train median.
- **C1 dev:** the Phase-5.5 samples. **C1 held-out:** the next samples in the same seeded order (A/B: up to 60 from the correctly classified test pool; C: 40 of the eligible SST-2 validation sentences).
- **C1 D:** SST-2 validation rows 200 onwards (rows 0–199 calibrate the label mapping only), 8–24 WordPieces, correctly classified, seeded permutation (1234), first 40.
- **N2:** e-SNLI test rows with gold label in {entailment, neutral, contradiction}, ≤ 64 WordPieces, annotator-1 highlight present, correctly classified (mapping calibrated on the first 500 dev rows), seeded permutation (1234), first 40.
- **N3/N4 probes:** the C, D and N2 samples. Reference sets: 200 other rows from the same source.

## 8. Benchmark selection and reliability

- **E1 eligibility:** numeric InterpBench case (HF revision `a1242a84`); the circuits-benchmark case imports (commit `220791e4`); categorical; the ground-truth circuit contains ≥ 1 attention head and ≥ 1 MLP; n_layers ≤ 4; d_model ≤ 64.
  - This gives 20 cases: **development** 7 and 13; **held-out** 2, 11, 14, 21, 25, 44, 45, 58, 63, 67, 71, 82, 93, 103, 110, 111, 124, 129.
- **Reliability:** a case whose clean trained-vs-Tracr agreement at the last position is < 0.90 over the candidate inputs is **benchmark-unreliable**. It is reported, and excluded from the primary confusion tables.
- **Ground truth per (node, sample):**
  - necessary if the Tracr-model interchange of the corresponding node changes the Tracr argmax;
  - not necessary for non-circuit nodes (SIIT);
  - **ambiguous** if an independent TransformerLens resample ablation of the trained model disagrees with that label.

  Ambiguous instances are counted separately, never in TP/FP/TN/FN.
- **E2:** development case 3; held-out case 39 (same program family, different size and sequence length). **This is a weak held-out**, and is reported as such.
- **MIB, RAVEL, CausalGym:** not used (`PHASE_7_5_LITERATURE.md` §3).

## 9. Pre-registered hypotheses (held-out)

**External, E1** (held-out reliable cases, node claims, non-ambiguous instances; PRIMARY standing):
- **EH1 (RQ1):** SUPPORTED on ≥ 90% of ground-truth-necessary instances.
- **EH2 (RQ2):** SUPPORTED on ≤ 5% of ground-truth-not-necessary instances.
- **EH3 (RQ2):** IG top-1 selections of ground-truth-not-necessary heads are CONTRADICTED on ≥ 90%.
- **EH4 (RQ4):** the count-controlled head configuration (ALTERNATIVE null) CONTRADICTS ≥ 50% of the ground-truth-necessary head instances. This is an over-restrictive null, predicted from development.
- **EH5 (RQ5):** the zero-ablation configuration (STRESS_TEST, recorded threshold) SUPPORTS ground-truth-not-necessary instances at a rate at least 5 percentage points above PRIMARY.
- **EH6:** attribution-only audits: 100% of IG selection claims are UNSUPPORTED (structural).
- **RQ3 (descriptive):** the rate of reversal findings on clear vs ambiguous instances, and on reliable vs unreliable cases.

**Concepts, E2** (held-out case 39):
- **KE1:** no known-negative concept (K−, K0) is VALIDATED under any null, and no concept standing is SUPPORTED for them under either cap. A failure means the policy is too permissive.
- **KE2 (prediction from development):** the Tracr known-positive K+ is *not* VALIDATED under the PRIMARY covariance null.
- **KE3 (descriptive):** the SIIT-trained K+ outcome under both nulls.

**Central, C1** (held-out samples, and model D):
- **CH7:** among IG_necessary samples whose PRIMARY standing is SUPPORTED, ≥ 20% carry `alternative_reverses`, at ≥ 3 of the 5 held-out A/B/C sites (sites with ≥ 5 PRIMARY-supported samples).
- **CH8:** R_necessary PRIMARY SUPPORTED on ≤ 5% of samples at every held-out site and on D.
- **CH9:** at every held-out site and on D, IG_necessary has at least as many PRIMARY-SUPPORTED samples as R_necessary.

**NLP** (N2 held-out task; D and E held-out models):
- **N1 (prediction):** e-SNLI IG top-k_h necessary is PRIMARY-SUPPORTED on at least as many samples as the human selection. The point estimate is compared, and the paired difference is reported with its interval.
- **N2:** IG token F1 with annotator 1 exceeds the random selection's. The Bonferroni interval of the paired difference excludes 0.
- **N3:** the zero-embedding OOD percentile exceeds [MASK]'s on D and on E (two tests; Bonferroni intervals of the paired difference exclude 0).
- **N4:** the SNLI model's empty-premise accuracy on 500 e-SNLI test rows exceeds chance. The Bonferroni Wilson lower bound is above 1/3.

## 10. Failure criteria for the gate

- **NO-GO for the framework paper** if, on held-out reliable E1 cases:
  - EH2 fails with a PRIMARY false-positive rate > 10%, or EH1 falls below 80%: the audit does not separate correct from incorrect mechanisms;
  - or any attribution-only claim is SUPPORTED;
  - or KE1 fails under the PRIMARY null: the policy validates known negatives.
- **REQUIRES SCIENTIFIC CHANGES** if the save → restart → load → audit → WHY loop fails, or an unfixed BLOCKING API defect remains, or an important mutation survives unfixed.
- **Otherwise:** READY FOR PHASE 8, WITH EXPLICIT LIMITATIONS if predictions fail or scope limits remain material. Failed predictions are reported, and the policy is not changed afterwards.

## Deviations

(None at the freeze.)

### DV-1 (2026-09-26, after the held-out E1 run; seen: that case 124 errored, not any outcome of it)

- **Original rule (§7, E1):** the first 40 inputs of `get_clean_data(max_samples=400, seed=42, unique_data=True)` whose trained-model prediction equals the Tracr model's.
- **Problem:** circuits-benchmark's `unique_data` compares `str(input)` against a set of tuples, so it never removes duplicates. Case 124's first 40 agreeing inputs contained duplicates. The audit plan refused them (`duplicate plan samples`), and case 124 produced no result.
- **Why only case 124:** every other case's plan was accepted, and a duplicate would have been refused, so their samples were already distinct and this rule leaves them unchanged.
- **Replacement rule:** the first 40 *distinct* inputs (by token sequence), in the same order, whose predictions agree.
- **Status:** case 124 is re-run alone with this rule. Its result is reported as held-out, flagged "re-run after DV-1". The other 17 held-out cases stand as run.
