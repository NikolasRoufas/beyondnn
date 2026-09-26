# Paper evidence ledger (Phase 7)

- **Status:** a ledger of what the recorded evidence supports, with allowed and disallowed wording. It is not the paper and not a draft of one.
- **Sources:** every number comes from a file in `experiments/phase7/results/` (see `RESULT_MANIFEST.json`) or from `docs/PHASE_7_REPORT.md`.
- **Data shared by every entry:**
  - Models A = MLP / breast_cancer (sites `input`, `net.1`), B = CNN / digits (`pixels`, `relu2`), C = BERT-tiny / SST-2 (`tokens`).
  - Held-out samples from Phase 5.5: 60 / 60 / 40.
  - Single deterministic runs, one torch thread, fixed seeds.
  - **No confidence intervals anywhere:** every quantity is an exact count over the declared samples.

---

### E1. The audit reproduces every pre-registered falsification scenario

- **Why it matters:** correctness of the audit's rules on evidence with known ground truth.
- **Experiment and artifact:** §29.3 scenarios A–N plus a tampered A; `results/scenarios.json`; `tests/test_audit.py::test_scenario_matches_preregistration`.
- **Result:** 14/14 scenarios gave exactly the pre-registered standing and finding codes, and the tampered A gave INTEGRITY_FAILURE with 1 result excluded.
- **Model/data and n:** hand-built fixed-weight models (`_testing`); 14 + 1.
- **Uncertainty:** deterministic.
- **Counterevidence:** the first version of scenario G was built wrong (D7), and the audit correctly refused it.
- **Limitations:** the scenarios are tests of the audit on toy models, not findings about models.
- **Reproduced?** Yes: the same builders run in the test suite on Python 3.10 / 3.12 / 3.14.
- **Allowed wording:** "On 14 pre-registered scenarios with known ground truth, the audit produced the pre-registered classification in every case."
- **Disallowed wording:** "The audit is correct", "the audit is validated".

### E2. Structural overclaim: attribution-only evidence never supports a causal claim

- **Why it matters:** operationalises the Jacovi & Goldberg / Joshi et al. principle.
- **Experiment and artifact:** §29.1 attribution-only audit; `results/central_{A,B,C}.json` → `attribution_only`.
- **Result:** G and IG `*_necessary` / `*_sufficient` are UNSUPPORTED (`attribution_is_not_intervention`) on 280/280 samples each, and R is NOT_EVALUATED on 280/280.
- **Model/data and n:** A, B, C; 280 samples × 4 claims.
- **Uncertainty:** deterministic.
- **Counterevidence:** none. The result holds **by construction** (CH4 is structural).
- **Limitations:** this is type checking. Its value is enforcement, not discovery.
- **Reproduced?** Yes: scenario B and a unit test.
- **Allowed wording:** "BeyondNN refuses to classify a causal claim as supported from attribution evidence alone."
- **Disallowed wording:** "Attributions are unfaithful" (not tested here).

### E3. Single-configuration support rarely survives declared invariance

- **Why it matters:** the central claim, that the audit separates what single-configuration reporting conflates.
- **Experiment and artifact:** §29.1; `central_{A,B,C}.json` (`naive_single_configuration_supports`, `audit.claims.ig_necessary.distribution`).
- **Result:**
  - The naive IG comprehensiveness test (r1, k at p = 10%, count controls at 0.95, threshold 0.5 · margin) SUPPORTS on 5 / 19 / 35 / 16 / 9 samples (A/input, A/net.1, B/pixels, B/relu2, C/tokens) = **84 / 280**.
  - The audited per-sample standing SUPPORTED, invariant over 3 replacements, 2–3 k values, 3 null values and thresholds ×0.5 / ×1.5, holds on **1 / 280** (C/tokens, sample 709).
- **n:** 280.
- **Uncertainty:** exact counts; no interval.
- **Counterevidence:** "SUPPORTED" requires agreement across every declared axis. Declaring more axes makes SUPPORTED rarer by construction. The ×1.5 threshold alternative alone can prevent SUPPORTED on samples whose recorded tests all support (B/pixels has 1 such sample).
- **Limitations:**
  - the invariance axes were chosen by the author;
  - the Phase 5.5 SUPPORTS shares were known;
  - D9 (control criterion 0.95) was set knowing the Phase-5.5 superiority medians.
- **Reproduced?** The evidence generation reproduces Phase 5.5 exactly (560/560 rows re-run, 0 differences).
- **Allowed wording:** "Under the declared invariances, the IG necessity claim was supported on 1 of 280 samples; a single-configuration test supported it on 84."
- **Disallowed wording:** "IG explanations are unreliable", "IG fails", "84 → 1 shows attribution is wrong".

### E4. The audit separates IG from random selections, by counts of non-contradicted samples

- **Experiment and artifact:** §29.1 CH3; `central_*.json`.
- **Result:** ASSUMPTION_SENSITIVE or SUPPORTED samples, IG vs R:

  | | A/input | A/net.1 | B/pixels | B/relu2 | C/tokens |
  |---|---|---|---|---|---|
  | IG | 55 | 60 | 54 | 59 | 34 |
  | R | 10 | 17 | 37 | 31 | 7 |

- **n:** 60 / 60 / 60 / 60 / 40.
- **Counterevidence:** the random selection is ASSUMPTION_SENSITIVE on 37/60 B/pixels samples. The standing does not distinguish "supported in most configurations" from "supported in one". The descriptive supporting-share distribution does: R has a share of 0 on 35–56 samples per site, whereas IG's shares spread across the bins (`results/analysis.json`).
- **Limitations:** ASSUMPTION_SENSITIVE is coarse (see REVIEWER_RISK C1).
- **Allowed wording:** "The audit classified random selections as CONTRADICTED on most samples (23–50 of 60; 33 of 40), and IG selections rarely (0–6)."
- **Disallowed wording:** "The audit ranks methods" (it does not).

### E5. Assumption sensitivity is pervasive and attributable to named axes

- **Artifact:** `central_*.json` → `samples_sensitive_by_axis`.
- **Result:** IG_necessary samples with an ASSUMPTION_SENSITIVE finding, per axis:

  | axis | A/input | A/net.1 | B/pixels | B/relu2 | C/tokens |
  |---|---|---|---|---|---|
  | k | 55 | 60 | 53 | 57 | 30 |
  | replacement | 51 | 57 | 50 | 47 | 11 |
  | null | 24 | 16 | 39 | 50 | 26 |
  | threshold | 54 | 58 | 42 | 54 | 24 |

- **Counterevidence:** k differs by design; the threshold alternatives are hypothetical re-evaluations.
- **Allowed wording:** "Sensitivity to the replacement was reported on 11–57 samples per site."
- **Disallowed wording:** "Replacement choice invalidates faithfulness metrics."

### E6. Necessity and sufficiency disagree at the configuration level; the standing-level check does not see it

- **Artifact:** `results/analysis.json` (descriptive; not pre-registered).
- **Result:**
  - IG samples with ≥ 1 (replacement, k) configuration where comprehensiveness SUPPORTS and sufficiency CONTRADICTS (or the reverse): 52 / 48 / 41 / 25 of 60, and 22 of 40.
  - The pre-registered cross-claim PROTOCOL_DISAGREEMENT check (on standings) fired on 0 samples.
- **Limitations:** a limitation of the audit design, recorded for Phase 7.5.
- **Allowed wording:** "Configuration-level necessity/sufficiency disagreements were frequent but were not surfaced by the standing-level check."

### E7. Absolute pass without beating matched controls

- **Artifact:** `analysis.json` → `samples_absolute_pass_but_controls_reject`; MISSING_CONTROL findings in `central_*.json`.
- **Result:** samples where an uncontrolled comprehensiveness test SUPPORTS and the count-controlled one CONTRADICTS (same k, r1):

  | | A/input | A/net.1 | B/pixels | B/relu2 | C/tokens |
  |---|---|---|---|---|---|
  | IG | 2 | 1 | 17 | 18 | 11 |
  | R | 2 | 1 | 21 | 9 | 3 |

- **Allowed wording:** "On the CNN and BERT sites, uncontrolled removal tests supported necessity on 9–21 samples where matched random controls did as well."

### E8. The concept audit reproduces the Phase-6 hand analysis; declared caps decide the one validated concept

- **Artifact:** `results/concepts_{A,B,C}.json`.
- **Result:**
  - KH1–KH4 held (A): only K4-direction validates; K1-direction encoding is null-sensitive and its use claim replacement-sensitive.
  - K4-direction is **UNSUPPORTED** under the strict caps (FP 0.277 > 0.10) and **SUPPORTED** under the lenient caps (≤ 0.30).
  - B: K5 (both kinds) and K6-neuron are ASSUMPTION_SENSITIVE (null); K6-direction is UNSUPPORTED.
  - C: see `concepts_C.json` and the report §20.
- **Counterevidence:** the caps were declared knowing the Phase-6 rates. Both rules are therefore reported, and the result depends on the rule.
- **Allowed wording:** "Whether the one validated concept counts as supported depends on the declared counterexample cap."
- **Disallowed wording:** "The model uses concept K4."

### E9. Persistence, re-derivation and verification

- **Artifacts:** `central_*.json` (`verify_report`), `concepts_*.json` (`reload_equal`), `performance.json` (`reload_group_mismatches`).
- **Result:**
  - `verify_report` passed on all 5 central reports and 6 concept reports;
  - 0 integrity failures in 20,112 re-derived central results;
  - concept A: an audit from 76 reloaded traces was byte-identical;
  - central reload subsets: see `performance.json`.
- **Allowed wording:** "Audits were re-derived from saved traces with identical output."

### E10. Mutation testing

- **Artifact:** `results/mutations.json`.
- **Result:** 25 code mutations of the audit. 19 were killed by the initial tests; 6 were not (5 survivors plus 1 unapplied pattern). After 5 added tests, all 25 were killed.
- **Allowed wording:** "All 25 mutations of the audit logic were detected by the final test suite."

### E11. NLP case study (BERT-tiny / SST-2; descriptive)

- **Artifact:** `results/nlp.json`.
- **Result:**
  - The IG top-k (p = 10%) never included [CLS] / [SEP].
  - The IG top-1 token was always interior and mostly sentiment-bearing ("fascinating", "irritating", "shame", "not", …).
  - Token-removal (replacement: zero / [MASK] / [PAD]) sensitivity appeared on 11 / 40 IG-necessary samples.
  - Length and negation cross-tabs show no pattern large enough to state at n = 40.
- **Limitations:** padding not evaluated (batch 1); label leakage not verifiable.
- **Allowed wording:** descriptive counts only.

### E12. Cost

- **Artifact:** `results/performance.json`; `central_*.json` (`audit_seconds`).
- **Result:** full central audits took 36–68 s for 2,832–4,320 results (460k–877k records), about 10–16 ms per result; `verify_report` costs the same again.
- **Allowed wording:** the measured times, with the machine and the single-run caveat.
