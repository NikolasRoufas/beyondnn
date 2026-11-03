# Remaining limitations after the revision

Ordered by how much they limit a TNNLS submission. Each item states what is known, what is missing, and what would close it.

| # | limitation | what is known | what would close it |
|---|---|---|---|
| 1 | **Added value over a careful conventional workflow is not demonstrated** | On identical measurements with seven designed faults, a careful checker (54 lines) matched every detection. The audit differs in default availability, attainability semantics and tamper refusal. By lines of code it is not cheaper. | Re-audit *published* claims with their released artifacts, or run a user study with researchers who do not know the fault list |
| 2 | **No independent mechanism truth for trained central models** | Program labels exist only for 8 tiny compiled programs. Specification labels for 18 SIIT-trained transformers are independent of the audited evaluation, but SIIT training uses interchange; the re-analysis was post hoc. | A trained network whose mechanism is fixed by construction outside the interchange family; none was found that avoids triviality (INDEPENDENT_VALIDATION_PROTOCOL §1) |
| 3 | **TNNLS scope** | No learning method; the propositions hold by construction | Author decision (TNNLS_SUITABILITY.md); TAI is the lower-risk venue |
| 4 | **Statistics** | All intervals are unclustered and descriptive. Five checkpoints each for the MLP and CNN give ranges only. | Cluster bootstrap over programs and checkpoints; more checkpoints |
| 5 | **BERT models: one checkpoint each** | — | Independent fine-tunes (GPU-scale) |
| 6 | **Blinding** | 47/52 is a re-audit of seen samples. A fresh-sample confirmation exists only for the CNN and BERT-tiny (P3); the MLP pool is exhausted, and BERT-base is too costly on CPU. | Fresh held-out BERT-base samples (about 23 CPU-hours) |
| 7 | **Cost** | About 23× per-test overhead on a 30-feature MLP; audit + verify ≈ 33–39% of total on the small sites; 33–39 MB evidence per MLP sample | Leaner record retention (summary-only controls) and a measured scaling study |
| 8 | **Reader-side re-audit** | Re-execution is exact on the authors' side. No compact evidence bundle is published. | Publish a small evidence bundle (plan + traces + report) |
| 9 | **Single-counterfactual necessity** | 169 of 760 tests of structurally used Tracr components were non-changes | A multi-counterfactual estimand, which would change the API |
| 10 | **Scale** | The largest model is BERT-base (110M) on CPU; GPU execution is not validated | A GPU-validated run on a modern decoder (not justified for the current claims) |
| 11 | **Manuscript source** | The revised source is a reconstruction; the original LaTeX was not available | Authors to merge it into their own source if it exists |
| 12 | **AI-disclosure text and two reference details** | Drafted or flagged for author review | Author check |
