# Phase 7.75 Report: Scientific Problem Resolution and Final Evidence Freeze

- **Dates:** 2026-09-29 to 2026-09-30.
- **Local git only:** nothing was pushed or published. No paper text was written, and Trace 3B was not started.
- **Sources:** every number comes from a committed file in `experiments/phase7_75/results/` (Phase-7.5 numbers from `experiments/phase7_5/results/`), computed by `evaluate775.py` into `hypotheses775.json`. Tables are in `results/figure_data/`.
- **Intervals:** 95% Wilson over the stated unit, descriptive.

## 1. Starting HEAD

`7ec2493` (Phase 7.5 gate: READY FOR PHASE 8 WITH EXPLICIT LIMITATIONS).

## 2. Ending HEAD

The commit that adds this final section text ("Phase 7.75 report: final commits and status"), directly after `8e93f2a`.

## 3. Baseline tests

At `7ec2493`:
- 1015 passed + 1 skipped without Captum, and 1036 with Captum, on Python 3.10 / 3.12 / 3.14;
- ruff and mypy --strict clean.

The first matrix run overlapped with the first code edits from Python 3.14 onwards. Python 3.14, ruff and mypy were therefore re-run in a clean worktree at `7ec2493` and matched.

## 4. Problems inherited from Phase 7.5

1. **Circular external validation:** InterpBench PRIMARY shared the resample operation with the definition of "clear" instances.
2. **Concept false negative:** the known-used Tracr variable was not validated (effect 0.06 < 0.1).
3. **Special tokens:** on BERT-base SST-2 (D), 10 of 11 IG supports selected [SEP].
4. **Replacement OOD is model-dependent:** the prediction "zero more OOD than [MASK]" failed on D.
5. **Zero-ablation false positives:** 13% on InterpBench.
6. **Count-matched null:** rejected 98.6% of known-necessary heads.
7. **Rationale, plausibility and NLI-shortcut results:** to preserve, not repair.

## 5. Independent-ground-truth literature review

`docs/research/PHASE_7_75_INDEPENDENT_GROUND_TRUTH.md`, written before the freeze.

- **MIB (both tracks), RAVEL and CausalGym** define success through interchange interventions, the same family of operation the audit uses. InterpBench's "not necessary" premise is itself an interchange-invariance training objective.
- **Only compiled Tracr programs** offer truth that is analytic: the program text plus the weights.

## 6. Chosen benchmark

**TD:** Tracr programs with decoys, written for this phase (`td_programs.py`).
- **Node claims:** 2 development and 8 held-out categorical programs.
- **Concept claims:** 1 development and 4 held-out numeric programs.
- **Trained-model evidence:** the compiled models are not trained; InterpBench remains the trained-model evidence.

## 7. Why its truth is independent

- **Labels from the source:**
  - a variable is **used** if the output `SequenceMap`'s function depends on it injectively;
  - it is a **decoy** if that function reads it and ignores it (`lambda x, y: x`), including decoys that are exact copies of the used variable;
  - numeric decoys are read with coefficient exactly 0.
- **Components mapped structurally:** each head / MLP is mapped to the variables it writes, through the non-zero rows of its output weights into labelled residual dimensions. Each held-out component writes exactly one variable.
- **Compiled model checked:** it reproduces the RASP interpreter on 100% of the selected samples in all 8 programs.
- **No intervention defines any label.** `analysis775.truth_is_independent` returns True for TD and False for InterpBench (tested and mutation-tested).

## 8. Preregistration commit

`b355129` (`docs/PHASE_7_75_PLAN.md`), committed after the framework fixes (`920f1cc`) and the development evidence, before any held-out run.

## 9. Deviations

- **DV-1 (2026-09-30):** the machine crashed about 2.5 h into the six model-D shard processes. Nothing had been written and no outcome had been seen.
  - D1 and D2 were re-partitioned from 3 into 10 resumable shards each.
  - This does not affect results: per-sample standings depend only on each sample's evidence.
- **Not a deviation (code bug fix):** the case-39 script looked up the `is_x` residual label by exact name and failed before computing anything. It was fixed to the prefix match Phase 7.5 used.
- **Operational:**
  - a queued A/B/C lane was stopped once by the system for low memory and re-run later;
  - the session scratch folder (old smoke scripts) was lost in the crash.

## 10. Independent mechanism results (TD held-out; PRIMARY standings)

8 programs × 40 samples; 0 results excluded; `verify_report` passed.

| truth | SUPPORTED | CONTRADICTED | UNSUPPORTED | INCONCLUSIVE | other |
|---|---|---|---|---|---|
| known true (19 components, 760 instances) | **591** | 169 | 0 | 0 | 0 |
| known false: decoys (10 components, 400) | **0** | 400 | 0 | 0 | 0 |
| empty (3 padding heads, 120) | 0 | 0 | 0 | 120 | 0 |

**IG top-1 head selections** (240; layers with ≥ 2 heads):

| selection | SUPPORTED | CONTRADICTED |
|---|---|---|
| used head (183) | 146 | 37 |
| decoy head (57) | 0 | 57 |

| hypothesis | result | held? |
|---|---|---|
| TD1: every known-true component SUPPORTED on ≥ 1 sample | minimum 28 of 40 | yes |
| TD2: decoys SUPPORTED ≤ 5% | 0/400 [0, 0.0095] | yes |
| TD3: known-true instances SUPPORTED ≥ 60% | 591/760 = 77.8% [74.7, 80.6] | yes |
| TD4: correlated decoys SUPPORTED ≤ 5% | 0/160 [0, 0.023] | yes |
| TD5: zero ablation FP ≥ PRIMARY + 5 pp | 400/400 [0.990, 1] vs 0 | yes |
| TD6: attribution-only UNSUPPORTED; empty never SUPPORTED | 240/240 UNSUPPORTED; 0 supports on empty under any configuration | yes |
| TD7: agreement with TransformerLens | 1,160/1,160 decisive instances | yes |
| TD8: IG-selected decoys SUPPORTED ≤ 5% | 0/57 [0, 0.063] | yes (the upper bound exceeds 5% at n = 57) |

## 11. False positives

- **PRIMARY:** none, in 400 decoy instances and 57 IG-selected decoys.
- **Zero ablation (STRESS_TEST):** supports **all 400** decoy instances. In Tracr's one-hot lookups a zeroed decoy is not a valid value, the lookup fails, and the output changes. Zero ablation cannot distinguish a used component from a decoy here.
- **Mean replacement (ALTERNATIVE):** 0/400.
- **Threshold ×0.5 and ×1.5 under resample:** 0/400.

## 12. False negatives

- **The 169 CONTRADICTED known-true instances** (22.2%) are single-counterfactual non-changes:
  - the resample source (the next sample) carries the same value of the used variable at the position that matters, so the output does not change;
  - the independent TransformerLens interchange agrees on every one of them (TD7).
- **Evidential adequacy (question B): correct.** That counterfactual does not show necessity.
- **Mechanism identification (question C): a false negative per instance.** This is the measured cost of testing necessity with one counterfactual per sample. At the component level no known-true component is missed (TD1).
- **Mean replacement** supports 586/760 known-true instances; zero supports 760/760 (while supporting every decoy).

## 13. Inconclusive cases

- **Empty padding heads:** 120/120 INCONCLUSIVE. Their output is zero under every replacement, so the intervention is a no-op. They are never SUPPORTED and never CONTRADICTED.
- **Count-null head tests** (560 instances in layers with 2–3 heads): 479 carry `control_criterion_unattainable`. The count null supports 0/280 known-true head instances. It is ALTERNATIVE, and no standing depends on it.

## 14. Implementation-consistency results (question A)

- **TD:** BeyondNN's PRIMARY outcome equals the independent TransformerLens interchange on 1,160/1,160 decisive instances.
- **InterpBench held-out** (re-audited under ADR-054; `interpbench_heldout.json`):
  - PRIMARY confusion identical to Phase 7.5 (TP 942, FP 0, TN 8,831, 9 INCONCLUSIVE, 18 ambiguous);
  - 0 standings changed;
  - it remains an implementation-consistency check. `truth_is_independent` is False for it.

## 15. Mechanism-identification results (question C)

Under program-defined truth, the audit's PRIMARY standing:
- never supported a decoy (0/400), including decoys perfectly correlated with the used variable (0/160) and decoys that attribution selected (0/57);
- supported every used component on most samples (28–38 of 40).

**Scope:** compiled Tracr programs (1–2 layers, ≤ 3 heads, one-hot or numeric variables). This is not evidence about trained models' messy representations.

## 16. Concept false-negative diagnosis

- Case 39's target is the numeric output "fraction of x so far" at the last position, whose values are small.
- Removing the used direction can lower it by at most about its own value (about 0.06 on the evaluated subset).
- An absolute `min_change` of 0.1 was therefore barely attainable on that target's scale.
- Phase 6 had used 0.25 and Phase 7.5 had used 0.1, with no scale argument for either.

## 17. Concept policy alternatives considered

| alternative | decision |
|---|---|
| lower the absolute threshold (e.g. 0.05) | rejected: post-hoc tuning, and still scale-dependent |
| effect relative to the clean margin | rejected: undefined for numeric targets |
| effect relative to the random-direction distribution | already required (`min_fraction_beyond_controls` ≥ 0.95); it measures specificity, not magnitude |
| a paired effect interval excluding a null region | considered; needs a null region in target units, the same scale problem |
| **effect relative to the target's natural variability (SD over the train split)** | **adopted** |

## 18. Chosen policy and justification

**ADR-055:** `min_change` = 0.2 × SD(clean target, train split), declared before the use test and recorded with the result.
- **Why a standardised effect:** the use claim is about a change in the model's output that is meaningful relative to how much that output varies across inputs.
- **Why 0.2:** Cohen's conventional "small" effect. The claim is use, not strong use; specificity is carried by the control criterion.
- **No API change.**

## 19. Calibration evidence

`results/concept_calibration.json`: constructed models y = s·(w·h0 + h2), with use weights w ∈ {0, 0.1, 0.25, 0.5, 1, 4} and output scales s ∈ {0.01, …, 100}.

- **Absolute 0.1 rule: not scale-invariant.** A used concept with w = 4 is not validated at s = 0.01; w = 0.1 is validated at s ≥ 10.
- **Relative rule: identical verdicts across all 5 scales.** It validates w ≥ 0.5 and does not validate w ≤ 0.25 (its operational meaning of "used").
- **Neither rule ever validated** the decodable-but-unused or the negative concept (30 cells each).

## 20. Held-out concept result

**TD numeric programs (blind; 4 programs):**

| concept | encoding (cov / iso) | use effect vs REL `min_change` | VALIDATED (REL, cov) | audit (strict and lenient) |
|---|---|---|---|---|
| K+ (used), 4 programs | supports / supports | −0.157 to −0.304 vs 0.012–0.021 | **4/4** | SUPPORTED 4/4 |
| K− (decodable, unused), 4 | supports / supports | 0.0 exactly | 0/4 | UNSUPPORTED 4/4 |
| K0 (permuted labels), 4 | contradicts / contradicts | (use passes: same direction) | 0/4 | UNSUPPORTED 4/4 |

- **KC1 held; KC2 held.**
- **KC3 (descriptive):** the absolute 0.1 rule also validated all 4 held-out K+ (their effects are 0.16–0.30). These programs therefore do **not** discriminate between the two rules. The case for ADR-055 rests on the calibration grid and the scale argument.

**Case 39 (NOT blind):** under REL (`min_change` about 0.005), the Tracr and the SIIT K+ are validated (effects −0.062 and −0.064); K0 is not. Under ABS they remain unvalidated, as in Phase 7.5.

## 21. Special-token policy

**ADR-053:**
- eligibility is declared by the caller (a mask and a name) and recorded in the selection, the test spec and the audit plan;
- rankings and controls use eligible units only;
- the audit matches claims by eligibility and flags near-misses (`eligibility_mismatch`);
- nothing is filtered silently, and an all-token claim still includes [CLS] / [SEP].

## 22. All-token D result (D1)

`central775_D_heldout_all.json`: 40 samples, 2,736 results, 0 excluded, `verify_report` passed in all 10 shards.

**IG_necessary:**
- PRIMARY SUPPORTED on **11/40 = 0.275 [0.16, 0.43]**, the same 11 samples as Phase 7.5;
- `alternative_reverses` on **11/11**;
- configuration-level disagreement on 33/40;
- **6 samples** moved from CONTRADICTED to INCONCLUSIVE (`control_criterion_unattainable`);
- `effect_without_competitive_advantage` on 6.

**Other claims:** R_necessary 1/40; G_necessary 5/40.

**Special tokens:** [SEP] (or [CLS]) is in the PRIMARY IG selection on **30/40** samples, and on **10 of the 11** supported ones.

## 23. Content-token D result (D2)

`central775_D_heldout_content_tokens.json`: 40 samples, 0 excluded, `verify_report` passed in all 10 shards.

**Eligibility:** eligible positions exclude [CLS] and [SEP]. There is no padding, since every run uses batch 1.

**IG_necessary:**
- PRIMARY SUPPORTED on **6/40 = 0.15 [0.07, 0.29]**;
- `alternative_reverses` on **6/6**;
- configuration-level disagreement on 20/40;
- 10 samples INCONCLUSIVE: with k = 1 content token, the control criterion is often unattainable.

**Other claims:** R_necessary 2/40 [0.014, 0.165]; G_necessary 3/40.

**Hypotheses:** SX1 held (6 ≤ 11); SX2 held (6/6 ≥ 20%); SX3 held (IG 6 ≥ R 2).

**Selected units:** sentiment-bearing words ("loved", "compelling", "painful", "superior", "well", "hole", "somber", …).

## 24. How many original D supports survive

**3 of the 11** D1 IG supports survive as content-token supports (samples 330, 363, 373).

| fate of the other 8 D1 supports under D2 | samples |
|---|---|
| CONTRADICTED (e.g. "unable" + "get" in place of "unable" + [SEP]) | 5 |
| INCONCLUSIVE (a single content token, unattainable control) | 3 |

D2 also has **3 new** supports (382, 659, 681), with different k and controls.

**Reading:**
- Most of the Phase-7.5 BERT-base "necessary top tokens" result depended on [SEP].
- The model's causal dependence under this test is concentrated in model-control tokens more than in lexical evidence.
- For content tokens alone, the support rate is 0.15, and every support is reversed by a reasonable alternative.

**BERT-tiny (C) for comparison:**
- C1 (all tokens): IG_necessary SUPPORTED 14/40; no special token in any PRIMARY selection.
- C2 (content tokens): 11/40.
- The 11 C2 supports are a subset of the 14. Three are lost because k and the control population are computed over content tokens only.

## 25. Replacement-OOD findings

- **Preserved (Phase 7.5):** zero minus [MASK] OOD percentile −5.5 on BERT-base SST-2 (prediction failed) and +1.5 on BERT-base SNLI. There is no universal replacement ordering.
- **No framework change:** replacement identity and role are recorded, and reversals are findings. Nativeness and OOD scores stay in experiment tables (`PHASE_7_75_SCIENTIFIC_FIXES.md`).

## 26. Zero-ablation findings

| setting | zero-ablation FP | zero-ablation TP |
|---|---|---|
| InterpBench (trained SIIT models; Phase 7.5) | 13.0% [12.3, 13.7] of 8,840 | 70.0% |
| TD (compiled; program-defined truth) | **100%** of 400 decoy instances, across 8 programs, heads and MLPs | 100% |

Zero is not invalid globally. It is an aggressive stress test whose false-positive rate depends on how far zero is from the activation distribution: total for one-hot lookups, partial for trained residual features. The audit shows it as STRESS_TEST reversals (1,400 `stress_test_reverses` findings on TD) without letting it decide a standing.

## 27. Count-null diagnosis

- **Mechanism:** random control sets are drawn from every (eligible) unit, including the selected set, and a tie never counts as "below". With k = 1 of 4 heads, about 25% of draws are the selection itself, so the 0.95 criterion is unattainable for any model.
- **Hypotheses examined:**
  - **confirmed:** the null population is mis-specified for small unit counts, and the statistic is not calibrated to the estimand;
  - **not needed to explain the result:** circuit redundancy; "many causally effective heads".
- **InterpBench re-audit:** of 431 count-null tests on known-necessary heads, 425 are now INCONCLUSIVE (unattainable) and 6 SUPPORT. None CONTRADICTS.
- **Central sites:**
  - B/relu2: 2 IG samples CONTRADICTED → INCONCLUSIVE;
  - C/tokens: 5 IG samples, 4 R samples and 5 G samples (short sentences, k = 1);
  - no SUPPORTED standing changed.

## 28. Control semantics

**ADR-054; no new ontology:**
- an unattainable control criterion yields INCONCLUSIVE plus `control_criterion_unattainable`;
- a PRIMARY contradiction that met its effect threshold and failed only the control yields `effect_without_competitive_advantage`. The standing stays CONTRADICTED for the claim as declared (beyond controls), and the report says the effect itself was present.
- Held-out IG_necessary samples with that finding: A/input 1, B/pixels 9, B/relu2 2, C/tokens 3 (D1 6, D2 2).

## 29. e-SNLI rationale result (preserved)

Human highlight necessary 10/40; IG 7/40; random 3/40. N1 failed. Not repaired: the framework separates human plausibility from model faithfulness, and this is what that separation shows.

## 30. Plausibility vs faithfulness (preserved)

IG minus random token F1 with annotator 1: +0.149, Bonferroni [0.031, 0.270] (N2 held). An explanation can be more human-aligned without being more causally necessary. The two are never merged.

## 31. NLI shortcut stratification (analysis only)

On 17 of the 40 e-SNLI samples the model predicts the same label with an **empty premise**.

| selection | necessary SUPPORTED: empty-premise-same (17) | needs premise (23) |
|---|---|---|
| human | 5 | 5 |
| IG | 5 | 2 |
| random | 1 | 2 |

IG's supports concentrate in the shortcut stratum (5 of its 7). These are descriptive counts at small n. No sample was removed.

## 32. Central A–D rerun differences

Phase-7.5 design, new audit; all units.

| site | IG_necessary SUPPORTED (7.5 → 7.75) | with `alternative_reverses` (7.5 → 7.75) | standings changed |
|---|---|---|---|
| A/input | 8 → 8 | 8/8 → 7/8 | 0 |
| A/net.1 | 4 → 4 | 2/4 → 2/4 | 0 |
| B/pixels | 14 → 14 | 13/14 → 13/14 | 0 |
| B/relu2 | 1 → 1 | 1/1 → 1/1 | 2 (CONTRADICTED → INCONCLUSIVE) |
| C/tokens | 14 → 14 | 14/14 → 13/14 | 5 (CONTRADICTED → INCONCLUSIVE) |
| D/tokens (D1) | 11 → 11 | 11/11 → 11/11 | 6 (CONTRADICTED → INCONCLUSIVE) | | |

One alternative reversal on A/input and one on C/tokens disappeared, because the reversing configuration's control criterion was unattainable.

## 33. Which Phase-7.5 results became invalid

- **EH4's reading** ("the count null rejects 98.6% of necessary heads") is invalid as a statement about evidence: those tests could not have supported any claim. Correct statement: the count null was unattainable on 4-head layers.
- **"E1 shows the audit separates correct from incorrect mechanisms"** was already disallowed. It is now replaced by the TD result.
- **The concept false negative (case 39)** was an artefact of an absolute threshold on a small-scale target.
- **D's headline "IG necessary on 11/40"** cannot be read as a content-token result. It is an all-token result in which [SEP] participates.

## 34. Which remained valid

- Every PRIMARY SUPPORTED standing of Phase 7.5 (InterpBench and central A–C; D: identical 11 supported samples in D1).
- The configuration-sensitivity story (§32).
- Attribution-only → UNSUPPORTED.
- IG-selected wrong heads → CONTRADICTED.
- Zero-ablation FP.
- The rationale, OOD, shortcut and padding results.

## 35. API changes

Additive, under the freeze:
- `eligible=` / `eligibility=` on `faithfulness.ranking` / `top_k` / `units`;
- `audits.selection(..., eligibility=)`;
- finding codes `eligibility_mismatch`, `control_criterion_unattainable`, `effect_without_competitive_advantage`.

No signature was removed or changed.

## 36. Schema migrations

- `evidence_selection` v2 → v3 (adds `eligible`, `eligibility` = None).
- `audit_plan` v2 → v3 (adds `selection.eligibility` = None).
- Both are tested and mutation-tested. Selection ids change with the version, as in ADR-034.

## 37. Save / reload verification

- **Unit eligibility round trip** (`test_eligibility_survives_save_and_reload`): plan JSON and saved evidence reloaded give an identical report.
- **Clean-wheel researcher workflow with eligibility** (`researcher_workflow775.py`; public API only): trace → attribution → intervention → faithfulness → concept → audit → WHY → save → restart → load → verify → WHY: `same as before: True`.
- **The Phase-7.5 workflow** on the same wheel is unchanged.

## 38. Mutations

19/19 killed (`results/mutations775.json`).
- **First run:** 16 killed. 3 survived:
  - the concept-threshold constant in the experiment (a test matched a docstring);
  - a dropped replacement name (not checked at the audit axis);
  - the audit-plan v2 → v3 migration of selection claims (the v1 fixture had none).
- **Fix:** three tests were added. No framework code changed. All 3 were killed on re-run.

## 39. Performance

- **Audit:** the new checks add one pass over recorded control units.
- **Full held-out central audits:** A/input 52 s → 118.6 s under heavy machine contention (not comparable). A clean benchmark (`benchmarks/bench_audit.py`, baseline `7ec2493` vs HEAD, idle machine; `results/performance/`) gives audit time within about 1% (e.g. 186.0 vs 188.2 ms for 48 results) and identical report bytes for the same 3,672 results (see `central775_*` `audit_seconds`).
- **Report and evidence sizes:** unchanged in kind. Evidence specs grow by the eligible-unit list only when eligibility is declared.
- **No large performance experiment was re-run.**

## 40. Regression matrix

`results/validation_matrix775.txt`:
- 1038 passed + 1 skipped without Captum, and 1059 with Captum, on 3.10 / 3.12 / 3.14;
- ruff clean; mypy --strict clean (after typing fixes in the two new test files);
- build OK;
- clean wheel: 7 README examples, 3 `examples/` scripts, and both researcher workflows pass.
- **Not reproduced identically:** the per-phase smoke scripts of earlier phases were session scratch files lost in the crash. The repo's own examples and workflows were used instead.

## 41. Strongest supporting result

**Program-defined truth:** 0 of 400 decoy instances supported, including 0 of 160 decoys that are exact copies of the used variable and 0 of 57 decoys selected by IG. Every used component was supported on 28–38 of 40 samples. Agreement with an independent implementation was 100%.

## 42. Strongest negative result

**Zero ablation supports every decoy** (400/400) under program-defined truth, and 13% of unnecessary nodes in trained InterpBench models. A necessity claim resting on zero ablation alone is unreliable in exactly the settings where ground truth is available.

## 43. Strongest limitation

**Independent truth is available only on compiled toy programs.** On trained models, the evidence is the partly circular InterpBench check plus configuration sensitivity without ground truth.

Also: a single resample counterfactual misses 22% of known-true instances (§12).

## 44. Strongest counterexample

**BERT-base SST-2:** 8 of the 11 Phase-7.5 samples where IG's top tokens were "necessary" lose that support once the claim is about content tokens. The Phase-7.5 headline for the moderate model was mostly a statement about [SEP].

## 45. Final paper evidence ledger

`docs/research/PAPER_EVIDENCE_LEDGER.md`, section "Final paper-claim freeze (Phase 7.75)": claims A–G with labels, allowed and disallowed wording, sources, counterevidence, scope and uncertainty.

## 46. NeurIPS remaining gaps

See `docs/research/PHASE_7_75_PAPER_READINESS.md` §2.

## 47. ACL remaining gaps

See `docs/research/PHASE_7_75_PAPER_READINESS.md` §3.

## 48. Paper claims now allowed

Exact wording is in the ledger.

| claim | label |
|---|---|
| A: configuration choices change conclusions | SUPPORTED FOR PAPER |
| B: attribution alone is insufficient | SUPPORTED FOR PAPER |
| C: claim / evidence typing catches unsupported inferences | SUPPORTED WITH REQUIRED QUALIFIER |
| D: save / restart / load reproducibility | SUPPORTED FOR PAPER |
| E: sensitivity surfaced without a score | SUPPORTED FOR PAPER |
| F: distinguishes correct and incorrect mechanisms | SUPPORTED WITH REQUIRED QUALIFIER ("on compiled Tracr programs with program-defined truth") |
| G: concept validation separates decodability from use | SUPPORTED WITH REQUIRED QUALIFIER ("on programs with known use; scale-relative criterion") |

**ACL-facing claims:**
- special tokens: qualified;
- plausibility vs faithfulness: qualified;
- replacement OOD model-dependence: qualified;
- NLI shortcut stratification: exploratory only.

## 49. Paper claims still disallowed

- "BeyondNN identifies correct mechanisms in trained models".
- Any use of InterpBench 942/942 or 0/8,840 as independent validation.
- "Zero ablation is invalid".
- "The count null shows heads are not better than random".
- "A universal safe replacement exists".
- "The case-39 concept is validated" as confirmatory evidence.
- "BERT-base's necessary tokens are sentiment words", without the [SEP] qualification.
- Any score, confidence or accuracy number for the audit.

## 50. Phase 8 requirements

Phase 8 is packaging, CI, documentation polish, security / contributing / code-of-conduct files, release metadata, TestPyPI / PyPI, `uv` lock, reproducibility instructions and benchmark scripts. It must not change scientific methodology.

**Carried into Phase 8:**
- RB-1 to RB-3;
- the per-phase clean-wheel smoke scripts must be re-created *in the repository*;
- evidence size (F-10).

## 51. Gate

**SCIENTIFICALLY READY FOR PHASE 8 WITH PAPER LIMITATIONS.**

**Applying the frozen criteria (plan §9):**
- **NO-GO: not triggered.**
  - TD decoy FP 0% (≤ 10%);
  - KC2 held;
  - no attribution-only claim was SUPPORTED.
- **REQUIRES FURTHER SCIENTIFIC VALIDATION: not triggered.**
  - Claims A–E rest on non-circular evidence: configuration reversals, attribution-only refusals, scenario suites, save / reload.
  - Save / reload passes; 19/19 mutations are killed.
- **Why "with paper limitations":**
  - Claims F and G are supported only with qualifiers: compiled programs with program-defined truth; programs with known use, with held-out programs that did not discriminate between threshold rules.
  - Claim F on trained models remains unestablished.

No policy was changed after seeing held-out results. DV-1 was procedural and outcome-blind.

## 52. Commits

18 local commits after `7ec2493`, the last being the report-completion commit. No Claude attribution trailers (owner's rule).

- `920f1cc` Declared unit eligibility (ADR-053); unattainable control criteria are inconclusive and competitive-only failures are distinguished (ADR-054); concept min_change in target scale (ADR-055)
- `b355129` Phase 7.75 FROZEN plan, independent-ground-truth review, experiment code (TD programs, concepts, central re-runs, InterpBench re-audit), development evidence; Phase-7.5 results pinned (before any held-out Phase-7.75 run)
- `fbb2cf0` Phase 7.75 adversarial tests (eligibility, control attainability, competitive failure, persistence, known concepts, independence), analysis helpers, mutation script, NLI shortcut stratification
- `a5fd344` Phase 7.75 held-out TD (independent program-defined truth): TD1-TD8 hold; evaluation script
- `3bd906f` Phase 7.75 external-researcher workflow with declared unit eligibility: passes from a clean wheel, identical after restart
- `83db388` Docs: unit eligibility, unattainable controls, competitive-only failure, report claim order; API-freeze additions; changelog
- `a62f301` Phase 7.75 scientific-fixes record (changes, justifications, affected evidence, re-runs)
- `b5478b4` Tests closing three surviving Phase-7.75 mutations (experiment threshold constant, replacement name in the audit axis, audit-plan v2 selection migration)
- `9f08859` Phase 7.75 mutations: 19/19 killed (3 after added tests)
- `6560d86` Phase 7.75 held-out concepts (KC1, KC2 hold), case-39 re-run (not blind; label lookup fixed as in Phase 7.5), InterpBench re-audit under ADR-054 (PRIMARY unchanged; count null unattainable)
- `7002ec3` Lint: td_concepts label lookup
- `a156d7e` Phase 7.75 deviation DV-1 (crash: D shards re-partitioned into 10 resumable shards; no outcome seen); central A held-out re-run under ADR-054
- `8d59f4b` Phase 7.75 figure-data script and tables (TD, concepts, calibration, central, NLI strata)
- `bfdba15` Phase 7.75: central B, C1, C2 held-out re-runs; regression matrix (1038+1 / 1059 on 3.10/3.12/3.14); typing fixes in the new tests
- `b407070` Phase 7.75 report draft (model-D sections, gate and claims pending)
- `bb303eb` Phase 7.75 model D held-out: D1 (all tokens) and D2 (content tokens), 10+10 shards merged; audit benchmark baseline vs HEAD
- `8e93f2a` Phase 7.75 docs: report sections for D1/D2, claims and gate; final paper-claim freeze in the ledger; paper readiness; figure candidates; roadmap; experiment log; docs index

## 53. Git status

Clean after the final commit. Git-ignored `experiments/phase7_75/artifacts/` holds logs, reports, queue scripts and the wheel-check environment.

## 54. Pushed / published status

Not pushed (the repository has no remote). Not published. No paper text written. Phase 8 not started. Trace 3B not started.
