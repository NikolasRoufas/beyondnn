# Phase 7.75 Plan: Scientific Problem Resolution and Final Evidence Freeze (FROZEN)

- **Frozen:** this plan is committed before any Phase-7.75 held-out run.
- **Deviations:** appended at the end as original rule / problem / reason / replacement / status.
- **Start:** `7ec2493` (Phase 7.5 gate: READY FOR PHASE 8 WITH EXPLICIT LIMITATIONS).
- **Baseline at `7ec2493`:**
  - 1015 passed + 1 skipped without Captum, 1036 with Captum, on 3.10 / 3.12 / 3.14;
  - ruff and mypy --strict clean.
  - Python 3.14, ruff and mypy were re-checked in a clean worktree, because the first matrix overlapped with code edits.
- **Phase-7.5 evidence is immutable.** Phase-7.75 outputs go to `experiments/phase7_75/results/` only, and a test pins the Phase-7.5 result files.

**Development evidence seen before this freeze** (allowed):
- InterpBench dev case 7 under ADR-054;
- TD dev programs `dev_copy_1_2` and `dev_map_decoy`;
- the TD numeric concept dev program `cdev_a_b`;
- the concept calibration grid;
- a one-sample C-dev content-token smoke run (not kept).

**Known before the freeze, and therefore not blind:**
- the Phase-7.5 case-39 use effect (about 0.06);
- the Phase-7.5 D result (IG selects [SEP]).

## 1. Problems and their treatment

| # | problem (Phase 7.5) | treatment | ADR |
|---|---|---|---|
| A | circular external validation | new confirmatory benchmark TD with program-defined truth (§3) | none (experiment) |
| B | concept false negative | scale-relative `min_change` (0.2 × train SD of the clean target), calibrated on constructed models; confirmed on new held-out TD concept programs (§4); case 39 re-run and labelled non-blind | ADR-055 |
| C | special tokens in D's IG support | declared unit eligibility; D re-run as D1 (all tokens) and D2 (content tokens), also C1 / C2 (§5) | ADR-053 |
| — | count-matched null rejects 98.6% of necessary heads | diagnosed as an unattainable criterion; the audit treats such tests as inconclusive; InterpBench and central audits re-run (§6) | ADR-054 |
| — | failure against competitive controls equated with no effect | `effect_without_competitive_advantage` finding | ADR-054 |
| — | replacement OOD (N3-D failed); zero FP 13% | preserved. Replacement identity is already recorded, and reversals already surface. TD adds zero-ablation evidence under independent truth. No framework change (§7) | none |
| — | NLI shortcut | stratification analysis of the e-SNLI results by empty-premise predictability (§8) | none |

## 2. Questions kept separate

- **A. Implementation correctness:** BeyondNN's PRIMARY intervention outcome vs an independent TransformerLens interchange of the same node (TD, InterpBench).
- **B. Evidential adequacy:** given the evidence, the audit status is what the evidence licenses:
  - attribution-only → UNSUPPORTED;
  - a no-op → INCONCLUSIVE, never SUPPORTED;
  - an unattainable control → not CONTRADICTED;
  - eligibility mismatch → not used.
- **C. Mechanism identification:** the audit standing vs program-defined truth (TD).

## 3. TD: independent known-mechanism benchmark (confirmatory)

**Programs and truth:**
- **Programs:** `experiments/phase7_75/td_programs.py`.
  - Development: `dev_copy_1_2`, `dev_map_decoy`.
  - **Held-out:** `ho_copy_2_1`, `ho_swap_1_3`, `ho_both_1_2`, `ho_map_decoy_2`, `ho_deep`, `ho_corr_1`, `ho_corr_2`, `ho_copy_1_3`.
- **Truth:** from the program source plus weights (`td_structure.py`).
  - Components: 19 known-true, 10 known-false (decoys; 4 of them in the correlated-decoy programs, perfectly correlated with the used variable), 3 empty (padding heads).
  - Structurally mixed components would be excluded; none exist.

**Samples and interventions:**
- **Samples:** `get_clean_data(max_samples=400, seed=42, unique_data=True)`, deduplicated by token sequence. Only samples whose compiled prediction equals the RASP interpreter's are kept; the first 40.
- **Evidence:** exactly the Phase-7.5 E1 design.
  - comprehensiveness, `min_drop` = clean margin (`metrics.margin`: the argmax changes);
  - replacements: resample (PRIMARY; source = next sample), mean (ALTERNATIVE), zero (STRESS_TEST);
  - null: none (PRIMARY); count controls (20, 0.95; ALTERNATIVE) for heads in layers with ≥ 2 heads;
  - thresholds ×0.5 (ALTERNATIVE), ×1.5 (STRESS_TEST);
  - IG top-1 per layer with ≥ 2 heads (32 steps, zero baseline).
- **Question A:** a TransformerLens interchange of each node from the same source.

**Reporting:**
- Per instance, the full truth × standing table: known-true / known-false / empty × SUPPORTED / CONTRADICTED / UNSUPPORTED / INCONCLUSIVE / other.
- Per component, the fraction of samples SUPPORTED.
- Every FP and FN is examined.
- Intervals: 95% Wilson over instances.

**Hypotheses (held-out):**
- **TD1 (C, component level):** every known-true component is PRIMARY-SUPPORTED on ≥ 1 of its 40 samples (necessity for the task is existential).
- **TD2 (C):** known-false instances are PRIMARY SUPPORTED on ≤ 5%. **Failure criterion:** > 10% means Claim F is not supported.
- **TD3 (C, instance level; prediction from development):** known-true instances are PRIMARY SUPPORTED on ≥ 60%. The remainder are expected to be single-counterfactual non-changes (CONTRADICTED) or no-ops (INCONCLUSIVE).
- **TD4 (C, plausible but wrong):** in `ho_corr_*`, decoy components are PRIMARY SUPPORTED on ≤ 5% of instances.
- **TD5 (RQ5):** zero ablation (STRESS_TEST, recorded threshold) supports known-false instances at a rate ≥ 5 percentage points above PRIMARY.
- **TD6 (B):**
  - attribution-only audits leave 100% of IG selection claims UNSUPPORTED;
  - empty components are never SUPPORTED under any configuration.
- **TD7 (A):** on decisive instances, BeyondNN's PRIMARY outcome agrees with the TransformerLens interchange (SUPPORTS ⇔ argmax changed) on ≥ 99%.
- **TD8 (C, attribution):** IG top-1 selections of known-false heads are PRIMARY SUPPORTED on ≤ 5%.
- **TD9 (descriptive):** the share of count-null head tests whose criterion is unattainable (ADR-054).

**Claim F decision:**
- **Supported for the paper, with the qualifier "on compiled programs with program-defined truth":** only if TD1, TD2, TD4 and TD8 hold.
- **Narrowed:** if TD2 or TD4 fails at ≤ 10%.
- **Rejected:** if TD2 fails at > 10%.

## 4. Concepts (ADR-055)

**Calibration** (done before the freeze; `results/concept_calibration.json`):
- the absolute 0.1 rule is scale-dependent;
- the relative rule is scale-invariant;
- no decodable-but-unused or negative concept is ever validated under either rule.

**TD numeric concept programs** (`td_concepts.py`):
- **Structure:** `is_u` (used) and `is_d` (decoy) numeric indicators, combined by `numerical(LinearSequenceMap(is_u, is_d, a, 0))` (the decoy coefficient is exactly 0), then prefix-averaged into the last-position output.
- **Development:** `cdev_a_b`.
- **Held-out:** `cho_c_d`, `cho_x_a`, `cho_half_e_b` (a = 0.5), `cho_b_c`.
- **Concepts:** K+ = the `is_u` residual direction at `blocks.0.hook_resid_post` (used); K− = the `is_d` direction (decodable, unused); K0 = K+ with permuted labels (seed 4242).
- **Label:** the fraction of the token > its train-split median.
- **Splits and tests:** as Phase-7.5 E2.
  - encoding: covariance (PRIMARY) and isotropic (ALTERNATIVE) nulls, 200 random directions + 200 label permutations, 0.95;
  - use: `remove(zero)` (numeric absence), 50 covariance random directions, 0.95;
  - policy: POLICY_V1;
  - audits: strict (0.10 / 0.10) and lenient (0.30 / 0.30) caps, with the Phase-7.5 concept roles.
- **Use rules:** REL (PRIMARY): `min_change` = 0.2 × SD(clean target, train split). ABS (the Phase-7.5 0.1): reported for comparison.

**Hypotheses (held-out):**
- **KC1:** every held-out K+ is VALIDATED under the PRIMARY covariance null with the REL rule.
- **KC2:** no K− and no K0 is VALIDATED, and none is SUPPORTED by an audit, under any null, rule or cap. **A failure means the policy is too permissive.**
- **KC3 (descriptive):** the ABS-rule outcomes on the same programs.

**Case 39 re-run** (`td_concepts.py case39`): the HL and SIIT-LL K+ under REL. Reported as **not blind**; it cannot confirm ADR-055.

**Claim G decision:**
- **Supported with the qualifier "on programs with known use":** only if KC1 and KC2 hold.
- **Narrowed to concept *testing*:** if KC1 fails.
- **Rejected:** if KC2 fails.

## 5. Special tokens (ADR-053): D1 / D2, C1 / C2

**Runs** (`central775.py`):
- the Phase-7.5 central design, unchanged (samples, checkpoint, targets, replacements and roles, p = 5 / 10 / 20% with PRIMARY 10%, 50 controls, thresholds);
- under the Phase-7.75 audit;
- **D1 / C1:** `--eligibility all` (every position, special tokens included; the Phase-7.5 claim).
- **D2 / C2:** `--eligibility content_tokens`.
  - Eligible positions are those whose input id is not a special id of the model's tokenizer.
  - k = ⌈p × number of eligible positions⌉.
  - IG / gradient rank and the random selection are drawn among eligible positions; controls are drawn from eligible positions.
- **Sharding:** D1 and D2 run in 3 shards each (samples i::3). Per-sample standings depend only on a sample's own evidence, so shards are audited separately and merged per sample.

**Reported:**
- IG / G / R PRIMARY SUPPORTED;
- `alternative_reverses` among the PRIMARY-supported samples;
- configuration-level disagreement;
- selected units (special tokens in D1 selections);
- how many D1 supports remain under D2.
- Phase-7.5 D (`central75_D_heldout.json`) stays as recorded.

**Hypotheses (predictions, descriptive):**
- **SX1:** D2 has no more IG PRIMARY supports than D1.
- **SX2:** ≥ 20% of D2's PRIMARY-supported IG samples carry `alternative_reverses` (if ≥ 5 are supported).
- **SX3:** under D2, IG_necessary ≥ R_necessary PRIMARY supports.

## 6. ADR-054 re-audits

- **InterpBench held-out** (18 cases): re-run with the Phase-7.5 code, unchanged, under the new audit (`rerun_interpbench.py`).
  - **Prediction:** PRIMARY standings are identical (the count null is ALTERNATIVE); count-null configurations on necessary heads become INCONCLUSIVE (unattainable).
- **Central A, B, C held-out:** re-run (`central775.py ... --eligibility all`).
  - **Report:** changes in PRIMARY standings, in `alternative_reverses` and in CH7–CH9, and the number of unattainable-control findings.

## 7. Replacements

No framework change. Replacement identity and role are recorded and reversals are findings (ADR-048). Nativeness and OOD measurements remain experiment-level (Phase-7.5 probes).

TD adds zero-ablation behaviour under independent truth (TD5).

## 8. NLI shortcut stratification (analysis only)

For the 40 e-SNLI samples, record whether the SNLI model predicts the same label with an **empty premise**, using the Phase-7.5 model, tokenizer and mapping. Human / IG / random PRIMARY necessity supports are then reported by stratum. No sample is removed.

## 9. Gate criteria

- **NO-GO FOR FRAMEWORK PAPER:** TD2 fails with > 10% FP, or KC2 fails, or any attribution-only claim is SUPPORTED.
- **REQUIRES FURTHER SCIENTIFIC VALIDATION:** a central paper claim (A–E) rests only on circular or invalid evidence after this phase, or save/reload fails, or an important mutation survives.
- **SCIENTIFICALLY READY FOR PHASE 8 WITH PAPER LIMITATIONS:** A–E hold, and F and/or G are supported only with qualifiers or narrowed.
- **SCIENTIFICALLY READY FOR PHASE 8:** A–G all hold as intended.

## Deviations

(None at the freeze.)
