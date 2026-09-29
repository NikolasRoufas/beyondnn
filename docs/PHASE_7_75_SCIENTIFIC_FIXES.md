# Phase 7.75 Scientific Fixes

- **Every change below** has an ADR, a stated justification that does not depend on the result it affects, the list of affected evidence, and the rerun that replaces it.
- **Phase-7.5 results** stay on disk unchanged (`tests/test_phase75_immutable.py`). Phase-7.75 results are in `experiments/phase7_75/results/`.
- **Numbers** below come from `results/hypotheses75.json` (Phase 7.5) and `results/hypotheses775.json` (Phase 7.75), computed by `evaluate775.py`.

## Fix 1: declared unit eligibility (ADR-053)

- **Problem:** "the IG top-k tokens are necessary" did not say whether [CLS] / [SEP] were part of the claim. On BERT-base SST-2, 10 of the 11 PRIMARY-supported IG samples selected [SEP] (Phase 7.5).
- **Justification (independent of outcome):** a content-token claim and an all-token claim are different estimands. Controls for a content-token claim must be drawn from content tokens, or the null compares against [SEP].
- **Change:** `eligible=` / `eligibility=` on selections; controls drawn from eligible units; the audit matches claims by eligibility (`eligibility_mismatch`); schema v3 with migrations. Nothing is filtered silently.

| old artifact | reason | new experiment | new artifact |
|---|---|---|---|
| `phase7_5/results/central75_D_heldout.json` (all tokens) | not invalid for the all-token claim, but re-audited under ADR-054 (Fix 2) | D1: `central775.py D heldout --eligibility all` (3 shards) | `phase7_75/results/central775_D_heldout_all.json` |
| (none) | content-token claim not previously expressible | D2: `--eligibility content_tokens` | `central775_D_heldout_content_tokens.json` |
| `central75_C_heldout.json` | as D | C1 / C2 | `central775_C_heldout_{all,content_tokens}.json` |

## Fix 2: unattainable control criteria and competitive-only failure (ADR-054)

- **Problem:** Phase 7.5 EH4: the count-matched null "rejected" 98.6% of known-necessary InterpBench heads.
- **Justification (combinatorial, independent of outcome):**
  - random control sets are drawn from all units, including the selected set;
  - identical sets tie with the selection, and ties never count as "below";
  - with one head among 4, about 25% of 20 draws are the head itself, so at most about 0.75 of controls can be beaten, less than the declared 0.95.
  - The test could not support the claim for any model.
- **Change:**
  - the audit computes the attainable fraction from the recorded control units; an unattainable CONTRADICTS becomes INCONCLUSIVE (`control_criterion_unattainable`);
  - a PRIMARY contradiction with the absolute effect met and only the control failed gets `effect_without_competitive_advantage`.
  - Evidence and protocol outcomes are unchanged.

| old artifact | reason | new experiment | new artifact |
|---|---|---|---|
| `external_interpbench_heldout.json` (+ case 124) | count-null CONTRADICTS were uninformative | re-run with the Phase-7.5 code, unchanged (`rerun_interpbench.py heldout`) | `interpbench_heldout.json` |
| `central75_{A,B,C,D}_heldout.json` | small-k token / feature configurations may have unattainable controls | `central775.py {A,B,C,D} heldout --eligibility all` | `central775_*_all.json` |

## Fix 3: concept use criterion in the target's scale (ADR-055)

- **Problem:** the known-used Tracr variable (case 39) was rejected: effect about 0.06 < `min_change` 0.1.
- **Justification (independent of that value):**
  - `min_change` is in target units; the same absolute number means different effects on different outputs. Phase 6 (0.25) and Phase 7.5 (0.1) used unrelated values.
  - On constructed models, rescaling the output by 0.01–100 flips the absolute rule's verdict for the same concept, both ways (`results/concept_calibration.json`). The relative rule does not.
- **Change (policy; no API change):** `min_change` = 0.2 × SD(clean target over the train split). 0.2 is Cohen's "small" effect, chosen from the convention and the calibration, never from a Tracr result.
- **Confirmation:** new held-out TD numeric programs. Case 39 is re-run and labelled **not blind**.

| old artifact | reason | new experiment | new artifact |
|---|---|---|---|
| `external_tracr_concepts_{dev,heldout}.json` | the absolute rule is not scale-invariant | TD concept programs (held-out, blind); case 39 (not blind) | `td_concepts_heldout.json`, `td_concepts_case39.json` |

## Fix 4: independent known-mechanism validation (no framework change)

- **Problem:** Phase-7.5 E1's clear instances were defined by agreement with a resample of the audited model, and the PRIMARY configuration was that resample.
- **Change:** a new benchmark, TD, whose truth comes from program source and weight structure (`research/PHASE_7_75_INDEPENDENT_GROUND_TRUTH.md`).
- InterpBench is kept as an implementation-consistency check.

## Considered and not changed

- **Replacement metadata (model-native status, OOD score):**
  - replacement identity and role are already recorded, and reversals are findings;
  - nativeness and OOD are properties of a model / replacement pair, measured by the Phase-7.5 probes and TD;
  - a field on `Replacement` would be a caller-asserted label with no verification. Kept in experiment tables.
- **Control-role ontology (negative / competitive / stress):**
  - every control BeyondNN draws is competitive (matched random unit sets);
  - the failure Phase 7.5 exposed was unattainability, which ADR-054 addresses;
  - `effect_without_competitive_advantage` states what a competitive failure means. A new ontology was not justified by the evidence.
- **Automatic special-token handling:** rejected (request §16).
- **Lowering 0.1 to 0.05, or any other tuning to the case-39 value:** rejected (request §10–§12).

## API finding found while writing the Phase-7.75 workflow

- **F-15 (DOCUMENTATION ISSUE):** `AuditReport.claims` is ordered by claim name, not by plan order.
- The first version of the Phase-7.75 researcher workflow read `claims[0]` / `claims[1]` and swapped two claims.
- Documented in `docs/audit/plans.md` ("Reading reports"). The API (`report.claim(name)`) is unchanged.
