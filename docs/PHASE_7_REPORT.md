# Phase 7 Report: Scientific Audit Framework

- **Gate:** see §45.
- **Pre-registration:** `docs/PHASE_7_PLAN.md`, deviations D1–D13 and execution note E1.
- **Numbers:** every number below comes from `experiments/phase7/results/` (manifest: `RESULT_MANIFEST.json`) or the test suite.

## 1. Starting commit

`25fe462` (Phase 6 gate), with a clean tree.

## 2. Ending commit

The commit that adds this report. See §46 (`git log 25fe462..HEAD`).

## 3. Baseline

At `25fe462`:
- 934 passed + 1 skipped without Captum, and 955 with Captum, on Python 3.10, 3.12 and 3.14;
- ruff, format and mypy --strict clean.

These match the expected baseline.

## 4. Literature

`docs/research/PHASE_7_LITERATURE.md` (verified sources marked [V]).
- **Overlapping conceptual work:**
  - Joshi et al. 2026 (causality and interpretability claims);
  - Jacovi & Goldberg 2020;
  - Doshi-Velez & Kim 2017;
  - causal scrubbing and causal abstraction;
  - AAR 2026 (claim-level auditability of text research agents).
- **Metric and benchmark suites:** Quantus, OpenXAI, ERASER / ROAR / ROAD, MIB, RAVEL, CausalGym, Tracr, InterpBench.
- **Documentation frameworks:** Model Cards, Fact Sheets, Saliency Cards.
- **Statistical-validity work:** Méloux et al., CIF.
- **Conclusion:** no reviewed system implements a record-level audit with typed evidence status, provenance refusal and re-derivation.

## 5. Differentiation

`docs/research/PHASE_7_DIFFERENTIATION.md`.
- **Not novel:** the conceptual premise (match claims to evidence type).
- **What is differentiated:** an implemented, verifiable audit layer over typed, provenance-bound, re-derivable records, with no score.
- **Weak spots, stated:** no uncertainty quantification, small models, no DAS, no external ground truth, and the audit only sees recorded evidence.

## 6. Audit definition

`audit(evidence, plan) → AuditReport`:
- deterministic and model-free;
- re-derives every result;
- excludes evidence by integrity, provenance and scope;
- classifies each plan claim and concept by fixed rules;
- lists contradictions (never resolves them);
- missing evidence is NOT_EVALUATED;
- no score and no "trustworthy" (ADR-044).

## 7. AuditPlan

Record kind `audit_plan` v1 (ADR-045) declares:
- the checkpoint, the declared model, the samples and the concept datasets;
- the claims: subject or selection; target or **per-sample targets** (added by D13); scope; requirement; invariances;
- the requirements: policy, controls, alternative criteria;
- the concepts with their policy and invariances;
- the counterexample caps and `naive_auroc`.

There are no invisible defaults. The plan is serialisable with the schema codec and embedded in every report.

## 8. AuditReport

Contains:
- the plan;
- an evidence summary (digest; traces, records and results; included / excluded);
- the claim inventory;
- the claim audits (per-sample groups with tests, axes, findings; distributions; counterexamples; limitations);
- the concept audits (validations, tests, findings, FP/FN identities);
- report-level findings, diagnostics, coverage and limitations.

Available as properties: `supported`, `contradicted`, `mixed`, `assumption_sensitive`, `inconclusive`, `unsupported`, `not_evaluated`, `missing_evidence`, `missing_controls`, `counterexamples`, `protocol_disagreements`, `overclaims`, `sensitivity(axis)`, `concept_findings`, `faithfulness_findings` and `provenance_findings`.

The report also has deterministic JSON, `render()`, `save()`, `load_report()` and `verify_report()`.

## 9. Taxonomy

`docs/audit/taxonomy.md`.
- **Standings (7):** SUPPORTED, CONTRADICTED, MIXED, ASSUMPTION_SENSITIVE, INCONCLUSIVE, UNSUPPORTED, NOT_EVALUATED.
- **Finding kinds (13):** the 12 planned kinds plus LIMITATION (D2), each with closed-vocabulary codes.
- **Severities (3):** BLOCKING, QUALIFYING, INFORMATIONAL. They are categorical and never aggregated.
- **Precedence:** plan §22 with D1 (exclusion findings do not by themselves set UNSUPPORTED).

## 10. Claim inventory

Every `Claim` record in the evidence is inventoried with:
- relation, subject, target, scope and sample;
- its results (protocol, outcome, inclusion status);
- its recorded assessments;
- the plan claims it matched by formal structure.

The central reports inventory every faithfulness claim (4,320 results per MLP/CNN site).

## 11. Unsupported claims

The structural overclaim codes are exercised by scenarios B, H, M and G, and by tests:
- `attribution_is_not_intervention`;
- `decodability_is_not_use`;
- `generated_label_is_not_validation`;
- `narrower_estimand`;
- `invariance_untested`;
- `missing_required_controls`;
- the counterexample caps.

On realistic data:
- the attribution-only audit is UNSUPPORTED on 280/280 samples for the G and IG claims;
- MISSING_CONTROL (the only SUPPORTS came from uncontrolled tests) occurs on 1–13 samples per claim and site.

## 12. Contradictions

- **Rule.** A SUPPORTS/CONTRADICTS pair is explained by the axis it differs in. An unexplained pair gives a CONTRADICTION and the MIXED standing; nothing is resolved.
- **In the central evidence:** no unexplained contradiction and no MIXED standing. Every disagreement differs in at least one recorded axis.
- **PROTOCOL_DISAGREEMENT:**
  - attribution vs intervention (scenario C);
  - necessary vs sufficient (scenario D). On realistic data it fired on **0** samples: see §32 and §40.

## 13. Missing evidence

The following behave as specified (scenarios J, K, N; tests):
- NOT_EVALUATED (`no_evidence`, `no_validation`, `concept_not_supplied`);
- `required_protocol_missing` (only for protocols never recorded: D8);
- `invariance_untested`;
- `selection_source_not_supplied`;
- `feature_derivation_not_rederived`.

Missing evidence is never counted as positive or negative. Code mutations "missing → SUPPORTED" and "missing → CONTRADICTED" are killed.

## 14. Assumption sensitivity

The axes are protocol, threshold, replacement, k, null, method and dataset. Central IG_necessary samples with a finding on each axis:

| axis | A/input | A/net.1 | B/pixels | B/relu2 | C/tokens |
|---|---|---|---|---|---|
| k | 55 | 60 | 53 | 57 | 30 |
| replacement | 51 | 57 | 50 | 47 | 11 |
| null | 24 | 16 | 39 | 50 | 26 |
| threshold | 54 | 58 | 42 | 54 | 24 |

## 15. Replacement sensitivity

- Replacement findings are listed in §14 (IG).
- R (random): 6 / 7 / 14 / 12 / 1.
- Concepts: K1-direction, K7-direction and K9-direction use claims are replacement-sensitive (zero vs train-mean). This reproduces Phase 6.

## 16. Threshold sensitivity

- Declared alternatives: ×0.5 and ×1.5 of the recorded `min_drop` / `max_drop`. They are re-evaluated from the recorded effects and labelled; they never enter a verdict.
- Threshold findings are in §14.
- One consequence: a sample whose recorded tests all SUPPORT is ASSUMPTION_SENSITIVE if the ×1.5 threshold fails (B/pixels: 1 sample).
- Alternatives for `concept_encoding` are refused (D3).

## 17. Null sensitivity

- **Central:** null values are `count@0.95`, `magnitude@0.95` and `none` (the uncontrolled r1 test, D9). Null findings for IG are in §14.
- **Absolute pass but controls reject** (`analysis.json`):
  - IG: 2 / 1 / 17 / 18 / 11 samples;
  - R: 2 / 1 / 21 / 9 / 3.
- **Concepts:** encoding outcomes differ between isotropic and covariance nulls for K1-direction, K5 (both kinds), K6-neuron, K7 (both), and K8-direction.
- **Phase 6's count** was 7 of 18 (concept × feature × null) settings.
- **Phase 7's audit** flags `null_sensitive` on 7 of 18 concept audits (the same 7).

## 18. Counterexamples

- **Per-sample CONTRADICTED samples** are listed by id, with caps (declared None in the central plans): for example R_necessary has 50 counterexample samples on A/input.
- **Concept FP/FN identities** are kept, and the caps are applied:
  - under the strict rule (0.10) every concept exceeds a cap;
  - under the lenient rule (0.30) K1 (both kinds), K2-neuron, K4 (both), K5 (both), K7 (both) and K9-neuron are within the caps.
- **Scenario I:** validated concept, FP 0.256 > 0.2 → UNSUPPORTED, with 23 FP identities kept.

## 19. Faithfulness audit

Selection claims (method-selected units) are matched through the recorded `EvidenceSelection` and its verified source attribution.
- **k:** k differs per sample; favourable-k-only evidence is UNSUPPORTED (test).
- **Absolute vs controls:** see §17.
- **Comprehensiveness vs sufficiency:** the standings-level check never fired, but configuration-level disagreements were common (§32).

## 20. Concept audit

| model | concept (kind) | strict | lenient | key findings |
|---|---|---|---|---|
| A | K1 direction | ASSUMPTION_SENSITIVE | ASSUMPTION_SENSITIVE | null_sensitive, replacement_sensitive, controls_defeat_encoding |
| A | K1 neuron, K2 (both), K4 neuron | UNSUPPORTED | UNSUPPORTED | decodable_not_used |
| A | K3 (both) | UNSUPPORTED | UNSUPPORTED | controls_defeat_encoding, counterexample_heavy |
| A | **K4 direction** | **UNSUPPORTED** (FP 0.277 > 0.10) | **SUPPORTED** | the only validated concept |
| B | K5 (both), K6 neuron | ASSUMPTION_SENSITIVE | ASSUMPTION_SENSITIVE | null_sensitive, decodable_not_used |
| B | K6 direction | UNSUPPORTED | UNSUPPORTED | controls_defeat_encoding |
| C | K7 direction | ASSUMPTION_SENSITIVE | ASSUMPTION_SENSITIVE | null + replacement sensitive |
| C | K7 neuron, K8 direction | ASSUMPTION_SENSITIVE | ASSUMPTION_SENSITIVE | null_sensitive, decodable_not_used |
| C | K8 neuron, K9 neuron | UNSUPPORTED | UNSUPPORTED | decodable_not_used |
| C | K9 direction | ASSUMPTION_SENSITIVE | ASSUMPTION_SENSITIVE | replacement_sensitive |

- **Hypotheses:** KH1–KH5 and NH1–NH2 held. They are consistency checks against Phase 6, except KH5, which was a prediction.
- **Fitted features** were re-derived from the encoding traces; there was no `feature_derivation_not_rederived` finding.

## 21. Generated-label audit

- **Scenario H:** a GENERATED label asserted as validated without a validation → UNSUPPORTED (`generated_label_is_not_validation`).
- **A validated concept with a GENERATED label** gets an INFORMATIONAL `generated_label_unverified`: the validation tests the dataset concept, not the text.
- **No realistic concept in Phase 7 used a GENERATED label.** The Phase-6 SAE case (with its polarity-reversed label) was not re-run.

## 22. Dataset-level audit

- **Per-sample claims** give distributions, never claim-level truth values (for example IG_necessary on A/input: 55 ASSUMPTION_SENSITIVE / 5 CONTRADICTED of 60).
- **Finite-sample and population claims built from instance evidence** are UNSUPPORTED (`narrower_estimand`), with the instance distribution in the detail (test).

## 23. Provenance audit

- **Exclusions:**
  - other checkpoint → excluded, PROVENANCE_MISMATCH (scenario J);
  - another model declaration → excluded;
  - mixed checkpoints inside one result → excluded.
- **Refusal:** `bnn.audit(..., model=other)` refuses. `compose(audit=...)` refuses another checkpoint or an undeclared sample.
- **Central evidence:** 0 exclusions.

## 24. Coverage

Reported per report:
- claims by standing, and per-sample evidence counts;
- concepts evaluated;
- evidence statuses and protocols run;
- required protocols never run;
- untested invariances;
- caps declared or "not declared".

It is coverage, not confidence.

## 25. Persistence and reload

- **Traces → fresh process → load → audit:**
  - byte-identical for concept A (76 traces);
  - 54/54 per-sample groups equal on the central subsets (D10: 3 samples per site; disk);
  - subprocess test in the suite.
- **`verify_report`** passed on all 11 realistic reports; edited reports raise `AuditMismatchError` (test).
- **P6-4** is fixed for auditing (ADR-046) and still open for `compose`.

## 26. WHY integration

- `bnn.compose(trace, audit=report)` adds an AUDIT section with per-sample standings and findings, the distribution, and the context claims and concepts (ADR-047).
- `to_dict()` gains `audit` only when composed.
- Mismatched checkpoints and samples are refused.

## 27. MLP result (A)

| | naive single-config SUPPORTS | IG_necessary | G_necessary | R_necessary |
|---|---|---|---|---|
| input | 5/60 | AS 55, C 5 | AS 54, C 6 | AS 10, C 50 |
| net.1 | 19/60 | AS 60 | AS 38, C 22 | AS 17, C 43 |

## 28. CNN result (B)

| | naive | IG_necessary | G_necessary | R_necessary |
|---|---|---|---|---|
| pixels | 35/60 | AS 54, C 6 | AS 56, C 4 | AS 37, C 23 |
| relu2 | 16/60 | AS 59, C 1 | AS 43, C 17 | AS 31, C 29 |

## 29. Transformer / NLP result (C)

- **Tokens:** naive 9/40.
  - IG_necessary: SUPPORTED 1 (sample 709), AS 33, C 6.
  - G_necessary: AS 30, C 10.
  - R_necessary: AS 7, C 33.
- **NLP** (`nlp.json`, descriptive):
  - the IG top-k never included [CLS] / [SEP];
  - the top-1 tokens are interior sentiment words;
  - token-removal (replacement) sensitivity on 11/40;
  - no length or negation pattern worth stating at n = 40;
  - padding is not evaluated (batch 1); label leakage is not verifiable.
- **Concepts:** see §20.
- **Duplicates:** 48 duplicate test records on 2 short sentences (where p = 5% and 10% give the same k) were deduplicated as identical records. 2,832 distinct results were audited.

## 30. Central case study

- **Hypotheses (5 model/sites):**
  - **CH1 held on 5/5:** IG SUPPORTED ≤ ½ of the naive count. It held essentially by construction: SUPPORTED is 1/280 overall, against 84/280 naive.
  - **CH2 held:** R SUPPORTED on 0/280.
  - **CH3 held on 5/5:** IG {S, AS} > R {S, AS}: 55 > 10, 60 > 17, 54 > 37, 59 > 31, 34 > 7.
  - **CH4 held:** structural.
  - **CH5 held on 5/5:** AS is the most frequent IG standing.
  - **CH6 held:** 0 integrity failures in 20,112 results; `verify_report` passed; reload subsets equal.
- **What separates the approaches** is the count of CONTRADICTED samples, not SUPPORTED ones.

## 31. Strongest supporting result

- Declaring the invariances that a necessity claim implicitly makes turns 84/280 single-configuration successes into 1/280 SUPPORTED.
- The audit names which assumption each disagreement rests on (§14).
- Random selections are CONTRADICTED on 23–50 of 60 (33/40) samples, IG selections on 0–6.

## 32. Strongest limiting result

- **ASSUMPTION_SENSITIVE saturates.** It is the standing of 33–60 IG samples per site, and it does not distinguish "supported in most configurations" from "in one".
- **The pre-registered cross-claim necessity/sufficiency check fired on 0 samples,** while configuration-level disagreements existed on 22–52 IG samples per site (`analysis.json`).
- **At realistic scale, the audit's per-claim output is coarse.** Its discriminating information sits in the findings and the descriptive layer, not in the standing.

## 33. Tool comparison

| Capability | Quantus / OpenXAI | Captum / TransformerLens / NNsight | Causal scrubbing | BeyondNN audit |
|---|---|---|---|---|
| Output | scalar metrics / leaderboards | mechanics | scalar (loss recovered) | standings + findings, no score |
| Claims as records | no | no | hypothesis (not a record) | yes |
| Evidence-type enforcement | no | no | interventional only | yes |
| Refuses cross-checkpoint evidence | no | no | no | yes |
| Re-derives stored results | no | no | no | yes |
| Reload-and-verify an audit | no | no | no | yes |
| Uncertainty intervals | some metrics | no | no | **no** |
| Scale | large | large | medium | **small models only** |

This is a factual comparison, not a ranking. No quantitative head-to-head was run.

## 34. Mutations

- **Code** (`mutations.json`):
  - 25 mutations of the audit logic;
  - 19 killed initially; 6 not (5 survivors, plus 1 pattern not applied after formatting);
  - 5 tests added → **25/25 killed**.
- **Evidence and report mutations** (tests), each detected:
  - forged outcome (INTEGRITY_FAILURE);
  - INCONCLUSIVE relabelled as SUPPORTS;
  - dropped contradiction (→ UNSUPPORTED, never SUPPORTED);
  - changed target or subject; changed dataset scope; checkpoint; sample;
  - removed controls; removed counterexamples; forged validation;
  - favourable k only; promoted generated label;
  - edited standing or findings in a saved report.

## 35. Performance

**Full central audits** (one thread, no tracemalloc):

| site | results | audit | verify_report |
|---|---|---|---|
| A input | 4,320 | 52 s | 49 s |
| A net.1 | 4,320 | 68 s | 67 s |
| B pixels | 4,320 | 51 s | 48 s |
| B relu2 | 4,320 | 65 s | 61 s |
| C tokens | 2,832 | 36 s | 34 s |

That is about 12–16 ms per result.

**Reload subsets** (216 results each, under tracemalloc, which slows execution):
- load 6.4–7.8 s; integrity 1.1–1.3 s; re-derivation 9.9–12.1 s; aggregation 0.76–0.78 s; serialisation 0.017 s;
- audit peak tracemalloc 6.8–7.5 MB (the audit's own allocations; the traces themselves are excluded);
- trace size on disk 98–177 MB per 3 samples (≈0.45 / 0.8 MB per test trace); report ≈0.58 MB.

**Evidence generation:** 195–209 s per MLP/CNN site; 1,413 s for BERT.

**Benchmark:** `bench_audit.py` is roughly linear: 1 / 4 / 16 samples → 26 / 121 / 410 ms.

## 36. Reproducibility

- **Scripts:** `experiments/phase7/*.py`.
- **Pins:** `experiments/phase5_5/requirements.txt`.
- **Code identity:** no `beyondnn/` change between the first valid run and the last (manifest).
- **Integrity:** result files are listed with SHA-256; full reports are git-ignored, with their digests listed.
- **Phase-5.5 re-run:** 2 MLP samples reproduce 560/560 rows.
- **Phase-6 ground-truth re-run:** reproduces every outcome and status. One committed claim id differs (pre-existing: provenance includes the environment; identical at `25fe462` and HEAD).
- **Discarded runs** are kept in `artifacts/aborted_run{1,2}` (D13, E1).

## 37. Test matrix

- **pytest** on Python 3.10, 3.12 and 3.14: **983 passed + 1 skipped** without Captum and **1004 passed** with Captum, `-W error`. That is 49 new tests: 48 in `test_audit.py` and 1 benchmark smoke.
- **Clean-wheel smokes** for Phases 1–7 pass with and without Captum.
- **README:** 7 examples execute.

## 38. Lint / type / build

- ruff check and format: clean.
- mypy --strict: clean on 3.12 and 3.14, with and without Captum. Two annotation errors on 3.12 were fixed in `34e432d`.
- `uv build` produces the sdist and wheel.

## 39. Backwards compatibility

- **Unchanged:** existing record kinds, ids and the golden claim id (all Phase 1–6 tests pass unchanged, apart from guard lists).
- **New:** record kind `audit_plan`; public names `bnn.audit` and `bnn.audits` (reviewed-surface tests updated).
- **API additions:**
  - `compose` / `from_evidence` gain an optional `audit`;
  - `concepts.verify` gains trace-level functions (the object-level API is unchanged);
  - `ExplainResponse.to_dict()` is unchanged when no audit is composed.

## 40. Scientific weaknesses

- ASSUMPTION_SENSITIVE is coarse, and "84 → 1" is partly determined by how many axes the plan declares.
- The cross-claim disagreement check operates on standings and missed configuration-level disagreements.
- No uncertainty statements; single deterministic runs.
- Not blind: the Phase-5.5/6 results were known when choosing the invariances, D9 (control criterion 0.95) and the concept caps. The concept caps decide K4's standing.
- Small models only; no external ground truth; no DAS.
- `decodable_not_used` fires when encoding holds under **either** null, even if the other null rejects it.
- The audit sees only recorded evidence.

## 41. Engineering limitations

- The audit holds every record in memory: a central site peaked around 2 GB RSS.
- Evidence volume is ≈0.5–0.9 MB per test trace, so the full central evidence (≈11 GB) could not be saved. Reload was checked on a subset (D10).
- `compose` still needs live Phase-6 objects (P6-4).
- Re-derivation re-computes every id (integrity dominates audit time).
- Concurrent realistic runs thrashed the 16 GB machine (E1).
- A design defect (one target per claim) surfaced only in the first realistic run (D13).

## 42. NeurIPS gaps

- Uncertainty quantification (confidence sequences or intervals) on audited quantities.
- External validation on known-mechanism suites (Tracr, InterpBench, MIB).
- Larger models; trained featurisers (DAS).
- A finer standing (conditional or per-axis) so the realistic output is not dominated by ASSUMPTION_SENSITIVE.
- A blind (held-out) replication of the central experiment with invariances fixed before seeing any Phase-5.5 data.
- A clear framing as an operationalisation of Joshi et al. 2026.

## 43. ACL gaps

- The NLP evidence is one small model (BERT-tiny) and one dataset (SST-2), with n = 40.
- No token-level OOD measurement: replacement sensitivity is only a proxy.
- Padding, tokenisation variants and label leakage are untested.
- No lexical-shortcut intervention study beyond the K7/K9 concept audits.
- No comparison with rationale benchmarks (ERASER) or with human rationales.
- Multilingual and generation settings are absent.

## 44. Phase 7.5 requirements

1. Conditional standings (for example "SUPPORTED for replacement = r1 across k and null"), and configuration-level protocol-disagreement findings.
2. Uncertainty statements for finite-sample claims.
3. Compact or streaming evidence storage and streaming ingestion.
4. `compose` from saved traces (P6-4).
5. `decodable_not_used` gated on null-agreement.
6. A blind replication plan.
7. External ground-truth evaluation.

Release blockers RB-1 to RB-3 remain.

## 45. Gate decision

**READY FOR PHASE 7.5 WITH EXPLICIT LIMITATIONS.**
- **Every engineering criterion of plan §34 is met:**
  - matrix green; lint, type and build clean;
  - scenarios 14/14 plus tamper;
  - every evidence mutation detected; 25/25 code mutations killed after fixes;
  - CH6 held; experiments run and reported in full.
- **Why "with explicit limitations":** the realistic separation rests on CONTRADICTED counts and descriptive layers, because ASSUMPTION_SENSITIVE saturates, and the cross-claim check did not fire (§32, §40).
- **No NO-GO criterion was triggered:** no score was needed; no scenario was special-cased; no SUPPORTED was produced without deciding evidence; no competing system was found.

## 46. Commits

`git log 25fe462..HEAD`: from `6faca63` (literature) and `4786efc` (plan) through the report commit. The full list is printed in the final summary.

## 47. Git status

Clean after the report commit, apart from git-ignored artifacts. Nothing was pushed.

## 48. Uncommitted work

None. Git-ignored and not committed:
- `experiments/phase7/artifacts/`: the full reports, the 3-sample trace subsets, the logs and the discarded runs.
