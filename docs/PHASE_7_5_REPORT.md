# Phase 7.5 Report: External Validation, Audit Refinement, API Freeze

- **Date:** 2026-09-27.
- **Local git only:** nothing was pushed or published.
- **Sources:** every number comes from a committed file in `experiments/phase7_5/results/`, computed by a committed script (`evaluate75.py`, `figure_data.py`).
- **Intervals:** 95% Wilson or percentile bootstrap over the stated unit, descriptive and unadjusted, except the four confirmatory NLP intervals (Bonferroni 98.75%, policy §5).

## 1. Starting commit

`0601cde` (Phase 7 gate: READY FOR PHASE 7.5 WITH EXPLICIT LIMITATIONS).

## 2. Ending commit

⟨D-FINAL: filled at the end⟩. See §52.

## 3. Baseline

At `0601cde`, before any change:
- 983 passed + 1 skipped without Captum, and 1004 passed with Captum, on Python 3.10 / 3.12 / 3.14;
- ruff, mypy --strict (with and without Captum), build, clean-wheel smokes (phases 1–7) and the 7 README examples were all green.

## 4. Literature

`docs/research/PHASE_7_5_LITERATURE.md` (`b1d6345`), written before the policy was frozen and chosen per weakness:

| topic | source(s) | decision |
|---|---|---|
| proportions | Brown, Cai & DasGupta 2001 | Wilson intervals |
| means and paired differences | Efron & Tibshirani; Koehn 2004; Dror et al. 2018 | percentile and paired bootstrap, resampling over samples |
| multiplicity | Holm 1979 | Bonferroni over the 4 confirmatory NLP intervals |
| sensitivity without a score | Steegen et al. 2016 multiverse; Simonsohn et al. 2020 specification curves | all configurations and roles reported; no summary inference |
| known mechanisms | InterpBench / SIIT, Tracr | used |
| | MIB, RAVEL, CausalGym | rejected, with reasons |
| rationales | e-SNLI; Jacovi & Goldberg; DeYoung et al. (ERASER) | plausibility kept separate from faithfulness |
| token OOD | Hooker et al. (ROAR); Hase et al.; Kim et al. | nativeness labels, percentile OOD scores |
| leakage and shortcuts | Gururangan et al. 2018; Poliak et al. 2018 | hypothesis-only probe |

## 5. Frozen policy commit

`ee3f91e`: `docs/PHASE_7_5_FROZEN_POLICY.md`, committed with the plan, the experiment code and the development evidence, before any held-out evaluation.

The consistency test `tests/test_phase75_analysis.py::test_development_cases_match_the_frozen_policy` fails if a held-out case becomes a development case (mutation-tested, §43).

## 6. Deviations

One deviation: **DV-1**, appended to the policy with the original rule, the problem, the reason, the replacement and its status.
- **Problem:** circuits-benchmark's `unique_data` does not remove duplicates. Case 124's plan was therefore refused (`duplicate plan samples`).
- **Replacement rule:** case 124 alone was re-run on the first 40 *distinct* inputs.
- **Visibility:** only the error was seen before the change; no outcome of case 124 was known.

**Not deviations**, recorded as findings:
1. The partial circularity of the E1 ground truth, identified at analysis time; no rule was changed (§17, §43).
2. Model D's run time (about 10–19 min per BERT-base sample). The frozen N = 40 was kept, by the owner's decision.

## 7. Audit refinement

These address Phase-7 limitations 1, 2 and 9:
- **ADR-048:** declared configuration roles (PRIMARY / ALTERNATIVE / STRESS_TEST, per axis; per sample where needed), with standings from PRIMARY only:
  - `alternative_reverses` (QUALIFYING), `stress_test_reverses` (INFORMATIONAL);
  - `undeclared_configuration`, `primary_untested`;
  - a `SensitivityProfile` per group;
  - `configuration_level_disagreement` findings;
  - `audit_plan` v2 with a v1 migration; report format v2; named replacements.
- **ADR-049:** uncertainty.
- **ADR-050:** `concepts.load_validation` and protocol versions.
- **ADR-051:** `metrics.margin`.
- **ADR-052:** required replacements.

## 8. Audit-profile design

A `SensitivityProfile` holds **raw structure only**:
- the tested configurations, with role and outcome;
- the outcome counts by role;
- the supporting / contradicting / inconclusive configuration ids;
- the sensitive axes (a single-axis change reverses the outcome) and stable axes (≥ 2 values, no reversal);
- the minimal reversals (the pairs with the fewest differing axes; at most 50 listed, all counted).

`describe()` reads, for example, "1 of 2 tested configurations SUPPORT (undeclared 1 of 2)".

**Rejected:** any aggregate number, and automatic downgrading of a PRIMARY standing by the alternatives (API review §3).

## 9. Configuration-level disagreement

Per sample, the finding is emitted when two configurations that share every other axis disagree on the pairing axes (replacement, k, method, dataset), independently of the standing.

**Held-out IG_necessary samples with such a finding:**

| site | samples |
|---|---|
| A/input | 48/51 (94%) |
| A/net.1 | 41/51 (80%) |
| B/pixels | 54/60 (90%) |
| B/relu2 | 34/60 (57%) |
| C/tokens | 37/40 (93%) |
| D | ⟨D⟩ |

On e-SNLI, random_necessary carries it on 9/40 samples.

## 10. Uncertainty methodology

`audits.wilson(k, n, level)`, `audits.bootstrap(values, seed, draws=10_000, level)`, `audits.paired_bootstrap(a, b, …)`:
- `Interval` records the quantity, estimate, low, high, level, method, unit, n, k, draws and seed;
- Wilson is exact 0 / 1 at k = 0 / n;
- per-sample claims carry a Wilson interval per standing;
- the resampling unit is always the input sample, never configurations or records.

**Confirmatory tests:** N2, N3-D, N3-E and N4 use 98.75% (Bonferroni over 4). Everything else is descriptive.

**Fixed in this phase:** `describe()` rounded 0.9875 to "99%". It now prints "98.75%", with a regression test; the stored levels were always correct.

## 11. Uncertainty results

Every hypothesis number below carries its interval (`results/hypotheses75.json`).

Intervals are wide at n = 40–60. For example, IG_necessary PRIMARY SUPPORTED on C/tokens is 14/40 = 0.35 [0.22, 0.50], and on B/relu2 1/60 = 0.017 [0.003, 0.089]. Differences between sites of less than about 15 percentage points are therefore not distinguishable.

## 12. External benchmark selection

- **E1:** InterpBench (SIIT-trained models, HF `a1242a84`; tasks and correspondences from circuits-benchmark `220791e4`).
  - 20 eligible cases (categorical; ≥ 1 head and ≥ 1 MLP in the circuit; ≤ 4 layers; d_model ≤ 64).
  - Development: 7 and 13. Held-out: 18 cases.
- **E2:** Tracr-compiled and SIIT-trained models of case 3 (dev) and case 39 (held-out; a weak held-out, from the same program family).
- **Not used:** MIB, RAVEL, CausalGym (`PHASE_7_5_LITERATURE.md` §3).

Details: `docs/PHASE_7_5_EXTERNAL_VALIDATION.md`.

## 13. Ground-truth definition

Per (node, sample), with the next sample as the interchange source:
- **Necessary:** the Tracr (HL) interchange of the corresponding HL node changes the HL argmax.
- **Not necessary:** a non-circuit node (the SIIT premise).
- **Ambiguous:** an independent TransformerLens resample of the trained (LL) model disagrees with that label. Ambiguous instances are never counted in TP/FP/TN/FN.

**Reliability:** a case is reliable if the clean LL-vs-HL agreement is ≥ 0.90. All 18 held-out cases are reliable (≥ 0.995); dev case 13 is not (0.19).

## 14. Known-correct results

EH1: PRIMARY SUPPORTED on 942/942 necessary node instances, 1.000 [0.9959, 1]. Largely by construction (§17).

## 15. Known-wrong results

- **EH2:** PRIMARY SUPPORTED on 0/8,840 not-necessary instances [0, 0.0004]. By construction (§17).
- **EH3:** IG top-1 selections of wrong heads were CONTRADICTED on 1,571/1,571 [0.9976, 1]. Attribution chose a wrong head in 80% of (layer, sample) selections.
- **EH6:** attribution-only IG selection claims were UNSUPPORTED on 1,960/1,960.
- **E2 known negatives:** KE1 held. No K− / K0 was validated or SUPPORTED, under either null or either cap.

## 16. False positives

**PRIMARY:** 0 on clear instances.

**Every single configuration treated as if it were primary** (fd1):

| configuration | FP |
|---|---|
| zero (recorded threshold) | 1,150/8,840 = 13.0% [12.3, 13.7] (EH5 held) |
| zero ×0.5 | 20.4% |
| mean (recorded) | 1.3% |
| resample ×0.5 | 3.0% |

**Qualitative look at every FP cluster:**
- **Zero ablation:** its FPs are whole non-circuit nodes on all or nearly all samples (case 14: L1.H3, L1 MLP; case 25: L0.H1 / L0.H2; case 103: L0 MLP; case 63: L1.H0 / L1.H1). These are nodes whose zero output is far from any natural activation.
- **Mean replacement:** its FPs are sample-specific, at most 8/40 per node.
- **The 18 ambiguous instances** are all non-circuit nodes that the trained model does use (case 21: L0 heads and MLP, 17; case 58: L1.H3, 1). PRIMARY reports them SUPPORTED, which is correct for the LL model. Under an HL-only ground truth they would be FPs (0.2%).

## 17. False negatives

**PRIMARY:** 0.

**Non-PRIMARY configurations:**
- the mean replacement misses 31.5% of necessary nodes (TP 645/942);
- zero misses 30.0%;
- the count-controlled head configuration misses 98.6% (EH4 held: 425/431 CONTRADICTED; over-restrictive, as predicted from development).

**INCONCLUSIVE:** 9 instances on case 110 (L0.H1, L0.H2), all not-necessary. On these samples the resample source equals the clean activation, so the intervention is a no-op; the audit correctly returns INCONCLUSIVE rather than a verdict.

**Circularity (found at analysis time):**
- The PRIMARY configuration is the benchmark's own semantics: a resample of the same LL node from the same source, with `min_drop` = clean margin, i.e. "the argmax changes".
- On *clear* instances the LL resample agrees with the label by the definition of "clear".
- EH1 and EH2 therefore verify that BeyondNN reproduces an independent TransformerLens computation (9,782/9,782 agreement) and passes it through without spurious findings or exclusions. **They do not show that the audit discovers correct mechanisms.**

This is the most important limitation of the phase (§42, §46).

## 18. Moderate realistic model

**Model D:**
- `textattack/bert-base-uncased-SST-2` (110M parameters, pinned revision), a held-out model;
- 40 SST-2 validation samples of 8–24 WordPieces, correctly classified, seeded order;
- label mapping fixed on 200 calibration rows only.

**Results:** ⟨D: IG / R PRIMARY SUPPORTED, alternative reversals, disagreement, CH8 / CH9 including D⟩.

**Cost:** about 10–19 min of evidence generation per sample on this CPU (§35).

## 19. NLP task 1: sentiment (SST-2; C held-out, D held-out)

**Central claims on C/tokens (held-out, 40 samples):**
- IG_necessary PRIMARY SUPPORTED on 14/40 = 0.35 [0.22, 0.50]; R_necessary on 0/40.
- 14/14 of the PRIMARY-supported IG samples carry `alternative_reverses`.
- Configuration-level disagreement on 37/40.

**Leakage:** 0 of the 40 C samples and 0 of the 40 D samples appear verbatim (case-insensitive) in the 66,978 SST-2 training sentences.

## 20. NLP task 2: NLI with human rationales (e-SNLI; model E)

**Setup:**
- **Model E:** `textattack/bert-base-uncased-snli` (109M; label mapping calibrated on 500 dev rows: 0.906).
- **Samples:** 40 e-SNLI test rows (14 contradiction, 14 neutral, 12 entailment), correctly classified, ≤ 64 WordPieces, annotator-1 highlight present.
- **Site:** the word embeddings.
- **Approaches:** human (annotator 1's highlighted hypothesis tokens), IG top-k_h (the same k as the human highlight), and a random span of k_h.
- **Roles:** replacement [MASK] PRIMARY, [PAD] ALTERNATIVE, zero STRESS_TEST.

**PRIMARY standings:**

| approach | necessary SUPPORTED | sufficient SUPPORTED |
|---|---|---|
| human | 10/40 = 0.25 [0.14, 0.40] | 5/40 |
| IG top-k_h | 7/40 = 0.175 [0.09, 0.32] | 6/40 |
| random | 3/40 | 1/40 |

**N1 (prediction: IG ≥ human): FAILED.**
- 7 < 10; paired difference −0.075 [−0.225, 0.075] (95%).
- Per-sample cross-tab (IG, human): both supported 3, IG only 4, human only 7, neither 26. The two select different evidence, and neither is necessary on most samples.

## 21. Human-rationale comparison: plausibility and faithfulness, kept separate

**Plausibility (agreement with annotator 1; never ground truth):**
- IG top-k_h token F1: 0.34 [0.26, 0.42].
- **N2 held:** IG minus random F1 = 0.149, Bonferroni 98.75% [0.031, 0.270].
- Inter-annotator IoU is 0.56 on average, so the human reference is itself noisy.

**Faithfulness (audited PRIMARY necessity, §20):** the human highlight is necessary on 10/40.

**Descriptive:** IG's F1 with annotator 1 is 0.50 on samples where IG is necessary and 0.31 where it is contradicted. This is not a test and not evidence that plausibility implies faithfulness.

**What is claimed and what is not:**
- Claimed: "IG agrees with human highlights more than random spans do; the human highlight is causally necessary for the prediction margin on a quarter of samples."
- Not claimed: human highlights as causal ground truth (`analysis75.rationale_summary` enforces `human_is_ground_truth: False`; mutation-tested).

## 22. Token OOD

**Probe:** `nlp_probes.py` (direct HF forward passes, outside BeyondNN's protocols). On models C (development), D and E (held-out), 40 samples each, every interior token is perturbed by zero / [MASK] / [PAD] / [UNK] / deletion. Measured:
- the |Δmargin|, the flip rate, the prediction KL, and the representation cosine;
- an **OOD percentile**: the rank of the perturbed final-layer [CLS] vector's nearest-neighbour distance to 200 reference inputs from the same source, among the references' leave-one-out distances.

**N3 (prediction: zero more OOD than [MASK], on D and on E):**

| model | zero minus [MASK], Bonferroni 98.75% | result |
|---|---|---|
| D | −5.53 [−7.56, −3.58] | **FAILED** (reversed: [MASK] is *more* OOD than zero on BERT-base SST-2) |
| E | +1.52 [0.21, 2.94] | held |
| C (descriptive) | −0.41 [−0.85, −0.004] (95%) | |

On D, [PAD] (−4.47) and deletion (−5.06) are also less OOD than [MASK].

**Reading:** which replacement is "in distribution" is model-specific. The fine-tuned SST-2 model reacts to [MASK] more than to a zero vector. This supports the policy of declaring replacements and reporting the dependence (ADR-052). It argues against any fixed claim that [MASK] is the safe choice.

## 23. Special tokens

Paired |Δmargin| of perturbing [CLS] / [SEP] minus the mean interior token (95%):
- **[CLS]:** always less influential than interior tokens (C: −0.17 / −0.22; D: −0.48 / −0.57; E: −0.68 / −0.85, for [MASK] / zero).
- **[SEP]:** on D (BERT-base SST-2), perturbing [SEP] shifts the margin **more** than an interior token: +2.12 [1.47, 2.82] with [MASK], +1.84 [1.20, 2.51] with zero. On C and E it shifts it less.

**Consequence:** on D, a removal test whose selection included [SEP] would measure the special-token effect. Phase 7 found that the IG top-k never selected [CLS] / [SEP] on C; D's selections are ⟨D⟩.

## 24. Padding and leakage

**Padding** (right-padding to 64 tokens, per sample; `nlp_probes.py`):

| model | with the attention mask | without the attention mask |
|---|---|---|
| C, D, E | max \|Δp\| < 1e-7 (padding is inert, as it should be) | |
| C | | 0/40 flips, mean \|Δmargin\| 0.60 |
| D | | **13/40 flips**, mean \|Δmargin\| 5.04 |
| E | | 0/40 flips, mean \|Δmargin\| 0.97 |

**Consequences for the recorded evidence:**
- **None for this phase's runs:** every BeyondNN run uses batch 1 without padding, and the evidence records the attention mask as a model keyword.
- **For batched use:** a wrong mask on D would change conclusions.
- **Not evaluated:** padded batches inside BeyondNN protocols.

**Word order:** shuffling the words of a sample (5 seeded shuffles) keeps the prediction on 94% (C) and 82.5% (D) of shuffles. Both SST-2 models behave largely as bag-of-words. For NLI (E) the figure is 44%.

**Leakage:**
- **SST-2:** no verbatim train/validation overlap in the samples (§19).
- **NLI (N4 held):** the SNLI model's accuracy with an **empty premise** is 228/500 = 0.456, Bonferroni [0.401, 0.512], above chance (1/3). With a [MASK]ed premise it is 0.462; with the full input 0.892.
- **Reading:** a large part of the model's NLI behaviour is available from the hypothesis alone (annotation artefacts; Gururangan et al.). A faithfulness claim about hypothesis tokens on this model may concern that shortcut, not inference.

## 25. Blind held-out evaluation

**Order:** the policy was frozen (`ee3f91e`) with the development evidence only. Held-out runs followed in the plan's order.

**Blindness controls:**
- **Selection by seeded order:** samples were chosen before any audit.
- **Quiet runs:** `--quiet` smoke runs print no outcomes.
- **Outcome-free deviation:** DV-1 was made on an error message.

| hypothesis | held? |
|---|---|
| EH1–EH6 | all held (EH1 / EH2 largely by construction) |
| KE1, KE2 | held |
| KE3 (descriptive) | SIIT K+ is not validated under either null (use test fails) |
| CH7 | held (3/3 qualifying sites) |
| CH8 | ⟨D⟩ (held on A/B/C) |
| CH9 | ⟨D⟩ (held on A/B/C) |
| N1 | **failed** |
| N2 | held |
| N3 | **failed on D**, held on E |
| N4 | held |

## 26. External-researcher workflow

`researcher_workflow.py`, run in a clean virtual environment holding only the built wheel and numpy, with the public API only, on a model new to BeyondNN.
- **First run:** failed (F-1, then F-2).
- **After the fixes:** `run ok` and `reload ok … same as before: True`, re-verified from the final wheel in the regression matrix (`results/validation_matrix.txt`).

## 27. API problems found

The 14 findings are in `docs/PHASE_7_5_API_REVIEW.md`:
- 3 BLOCKING API DEFECTS, all fixed (F-1 sample identity, F-2 evidence persistence, F-6 WHY after restart);
- 5.5-F-22 (implicit zero replacement), fixed by ADR-052;
- ergonomic, documentation and scientific-ambiguity items: fixed or bounded;
- 2 expected limitations.

## 28. Save / restart result

Save → fresh process → load → audit → `verify_report` → `load_validation` → WHY: **passes**, with an identical audit distribution, concept standing, validation status and WHY sections. It is tested in `tests/test_audit_75.py` and in the clean-wheel workflow.

## 29. Concept-policy validation

**Frozen POLICY_V1**, unchanged; 0.10 / 0.95 not tuned:
- no known negative is validated (KE1);
- **the known positive is not validated either**, on Tracr and SIIT, on dev and held-out. The use effect is −0.06, below `min_change` = 0.1.

So the concept policy has **no demonstrated external true positive** (G-N2).

## 30. Null validation

- **Faithfulness nulls:**
  - the count-controlled null is over-restrictive on 4-head InterpBench models (EH4: 98.6% of known-necessary heads rejected);
  - it was declared ALTERNATIVE, so it shows as a qualifying reversal, not as a rejection;
  - the audit cannot itself detect an over-restrictive null (F-11).
- **Concept nulls:** on dev case 3, the covariance null (PRIMARY) rejected encoding of Tracr K+ where the isotropic null accepted it (KE2 was predicted from this). On held-out case 39 both nulls accept encoding.

## 31. Central result under the frozen policy

**Held-out PRIMARY standings** (policy §C; 60 / 60 / 40 samples, A: 51 available):

| site | IG_necessary SUPPORTED | R_necessary SUPPORTED | PRIMARY-supported IG with `alternative_reverses` |
|---|---|---|---|
| A/input | 8/51 = 0.16 [0.08, 0.28] | 1/51 | 8/8 |
| A/net.1 | 4/51 = 0.08 [0.03, 0.18] | 0/51 | 2/4 |
| B/pixels | 14/60 = 0.23 [0.14, 0.35] | 1/60 | 13/14 |
| B/relu2 | 1/60 = 0.02 [0.003, 0.09] | 0/60 | 1/1 |
| C/tokens | 14/40 = 0.35 [0.22, 0.50] | 0/40 | 14/14 |
| D/tokens | ⟨D⟩ | ⟨D⟩ | ⟨D⟩ |

**Development** (IG_necessary SUPPORTED): A/input 5/60, A/net.1 1/60, B/pixels 20/60, B/relu2 4/60, C 8/40.

**Hypotheses:**
- CH7 held (A/input, B/pixels, C/tokens: ≥ 20%; in fact 93–100%).
- CH8 (R ≤ 5% everywhere) and CH9 (IG ≥ R everywhere) held on A/B/C; D is ⟨D⟩.

**Also recorded:**
- No results were excluded; `verify_report` passed on every report.
- The Phase-7 conclusion is reproduced with roles: IG top-k is more often necessary than random selections, and almost every such support depends on the choice of a reasonable alternative.

## 32. Strongest supporting result

**Attribution-selected wrong heads:** on 18 held-out InterpBench models, the audit contradicted every one of 1,571 IG selections of a known-unnecessary head, and it never let attribution-only evidence support a causal claim (1,960/1,960 UNSUPPORTED).

Both are independent of the circularity in §17: which head IG selects is not decided by the ground-truth procedure.

## 33. Strongest limiting result

**The external validation of RQ1/RQ2 is partly circular:** PRIMARY equals the ground-truth-defining intervention on clear instances (§17). So there is **no evidence yet that the audit discovers which mechanism is correct** beyond reproducing the benchmark's own interchange semantics.

A close second: **the concept policy rejects the known-used Tracr variable** (§29).

## 34. Strongest contradicting result

- **N3 on D:** the frozen expectation "zero embeddings are more OOD than [MASK]" is reversed on BERT-base SST-2 (−5.5 percentile points, Bonferroni interval excluding 0).
- **N1:** IG's selection is necessary on fewer e-SNLI samples than the human highlight (7 vs 10; the interval includes 0).

Neither contradicts an earlier BeyondNN conclusion. Both contradict Phase-7.5 predictions and are reported as failed.

## 35. API changes

`audits.role`; `roles=` on `claim` / `concept`; `SensitivityProfile`; `Interval`; `wilson` / `bootstrap` / `paired_bootstrap`; `sample_id` / `traces_of` / `save_evidence` / `load_evidence`; `concepts.load_validation`; `protocols.PROTOCOL_VERSIONS`; `metrics.margin`; `faithfulness.replacement(..., name=)`; required `replacement` (ADR-052, breaking); `audit_plan` v2; report v2; `Interval.describe` level formatting.

## 36. Rejected API changes

See `PHASE_7_5_API_REVIEW.md` §3:
- a profile score;
- automatic downgrading by alternatives;
- default roles;
- concept-threshold tuning;
- `selection(source=…)`;
- a compressed evidence format.

## 37. API freeze status

**FROZEN** (`docs/API_FREEZE.md`): names, signatures, record kinds and versions, report and persistence formats, standings, findings and roles.

**Not frozen:** `_testing`, `faithfulness.stats`, internals, and message wording.

## 38. Performance

Single runs, one torch thread, a machine shared with other experiments (`results/performance75.json`):

| setting | results / records | evidence generation | save / disk | load | audit build (integrity + re-derivation) | aggregation + profiles | verify | report | audit peak (tracemalloc) / process RSS |
|---|---|---|---|---|---|---|---|---|---|
| small: A/input, 3 samples | 216 / 38,263 | 7.3 s | 2.3 s / 100 MB | 7.0 s | 11.5 s | 1.1 s | 9.5 s | 1.25 MB | 8.2 MB / 838 MB |
| external: InterpBench case 7, 5 samples | 230 / 3,946 | 6.8 s | 0.4 s / 16 MB | 1.1 s | 2.0 s | 0.4 s | 1.6 s | 1.49 MB | 3.3 MB / 854 MB |
| moderate: D, BERT-base, 2 samples | 144 / 27,175 | **2,313 s** | 4.0 s / **449 MB** | 8.4 s | 12.5 s | 0.9 s | 11.4 s | 0.81 MB | 6.1 MB / 1,167 MB |

**Other timings:**
- Uncertainty for Wilson plus two 10k-draw bootstraps: about 0.2 s.
- Full held-out central audits: 34–58 s for 2,760–4,320 results.

**Reading:**
- The audit is cheap relative to evidence generation.
- Evidence generation and evidence size dominate: about 225 MB and about 19 min per BERT-base sample under contention (F-10).

## 39. Reproducibility

- **Seeds:** fixed (samples, controls, bootstraps 7501 / 7502 / 7504).
- **Pinned sources:** InterpBench, circuits-benchmark, e-SNLI and HF model revisions; `requirements_interpbench.txt`.
- **Checks:** `verify_report` passed on every report.
- **Re-runs at HEAD** (`results/reruns/README.md`):
  - Phase-7 scenarios 14/14 + tamper match (plus the new additive finding code);
  - Phase-6 ground truth: all outcomes identical;
  - Phase-5.5 A (3 samples): 840/840 rows identical except result ids.

## 40. Mutations

24/24 killed (`results/mutations75.json`). They cover:
- **Roles:** configuration counts, profile contents, role order, the PRIMARY-only standing, alternative reversals, sample-specific roles.
- **Disagreement and requirements:** configuration-level disagreement, the wrong requirement.
- **Uncertainty:** unpaired bootstrap, ignored seed, ignored Wilson level, dropped intervals.
- **Persistence and integrity:** profile serialisation, saved-evidence round trip, the wrong checkpoint, an unsupported protocol version.
- **Analysis:** a held-out case turned into a dev case, mislabelled ground truth, FP/FN counts, benchmark scope, ambiguous-as-clear, token-perturbation labels, human rationale as ground truth.

One pattern did not apply in the first run (a formatter line split). It was fixed and re-run, and killed; no code change was needed.

## 41. Test counts

- At the start: 983 + 1 skipped / 1004.
- In the matrix (`76c68eb`): 1014 + 1 skipped / 1035, on 3.10 / 3.12 / 3.14.
- At the end: one more test (`test_interval_description_keeps_a_non_integer_level`): 1015 + 1 / 1036 ⟨confirmed in the final check⟩.

## 42. Lint / type / build

ruff check and format: clean; mypy --strict with and without Captum: clean (112 files); build: sdist and wheel.

**Clean-wheel smokes:**
- phases 1–4, 6, 7 and 7.5 pass, as do the 7 README examples;
- the phase-5 and 5.5 scratch smokes needed the ADR-052 explicit replacement and then passed with identical outputs (`results/validation_matrix.txt`).

## 43. Scientific limitations

1. E1 RQ1/RQ2 are partly circular (§17).
2. The concept policy has no external true positive (§29).
3. E2's held-out case is weak (same program family).
4. n = 40–60 per site; single seeds, checkpoints and datasets per model.
5. Replacement OOD is model-specific (N3-D); no replacement is in-distribution for token removal.
6. The NLI model uses hypothesis-only shortcuts (N4).
7. Padding is not evaluated inside BeyondNN protocols. The direct probe shows D is mask-sensitive (13/40 flips without the mask).
8. The count null is over-restrictive on small multi-head models.
9. Human rationales are one annotator (inter-annotator IoU 0.56).

## 44. Engineering limitations

- Evidence size (F-10).
- BERT-base evidence generation takes tens of minutes per sample on CPU.
- `run_dataset` gaps (5.5-F-7 / F-8).
- One declared approach per audit (F-7).
- Two `sample_id` entry points (F-9).
- RB-1 to RB-3 (contacts, CI never run) remain.

## 45. NeurIPS readiness

See `research/PHASE_7_5_PAPER_READINESS.md` §2:
- **Must-fix:** G-N1 (non-circular external validation), and G-N2 if concepts are a headline claim.
- **Desirable:** comparison with multiverse / ROAR / CIF; seed variation; wider model families.

## 46. ACL readiness

See `research/PHASE_7_5_PAPER_READINESS.md` §3:
- **Must-fix:** G-A5 (= G-N1).
- **Desirable:** a third task type; control of hypothesis-only shortcuts.
- **Acceptable:** the rationale separation (as designed) and the token OOD (stated).

## 47. Must-fix vs acceptable

**Must-fix before the paper makes the corresponding claim:**
1. External ground truth independent of the PRIMARY intervention, or else narrow the claim to "reproduces known interchange results and exposes configuration dependence".
2. A concept known-positive validated on a new held-out case without tuning, or else no headline concept-validation claim.

**Acceptable when stated:** everything else in §43–§44.

## 48. Phase-8 requirements

Phase 8 must:
- keep the API freeze (changes only via ADR);
- address or explicitly narrow must-fix 1 and 2 before any paper text claims them;
- keep Trace 3B separate;
- resolve RB-1 to RB-3 before any public release.

No Phase-8 work has been started.

## 49. Gate

⟨D-FINAL⟩

## 50. Commits

⟨D-FINAL⟩

## 51. Git status

⟨D-FINAL⟩

## 52. Ending commit and uncommitted work

⟨D-FINAL⟩

## 53. Uncommitted work

⟨D-FINAL⟩
