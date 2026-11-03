# Adversarial TNNLS review (Stage 6)

- **Reviewed:** the **revised** manuscript `paper_tnnls/main.pdf` (reconstructed IEEEtran source), not the original.
- **Stance:** an independent, skeptical reviewer.
- **Rounds:** Round 1 found the issues listed in §1. Round 2 (§2) re-reviews after corrections. The final recommendation is in §3.

**Severity:**
- **CRITICAL:** likely to cause rejection.
- **MAJOR:** must be addressed.
- **MINOR:** should be fixed.

## 1. Round 1

### R1-C1 (CRITICAL): TNNLS scope and type of contribution

- **Location:** §I, contributions; §III-F; §IV.
- **Problem:**
  - There is no learning method, architecture or learning-theoretic result.
  - The contribution is an audit semantics plus software.
  - The six propositions hold "by construction"; they restate rules rather than prove non-obvious properties.
- **Why vulnerable:** TNNLS asks for "novel contributions of archival nature in developing theories and/or innovative applications of neural networks and learning systems", not of incremental nature. An editor can desk-reject a tooling paper.
- **Evidence required:** none that experiments can supply. This is a framing and venue decision.
- **Correction:**
  - State the propositions as *properties of the semantics* (done).
  - Show that they are checked: tests, the τ × 0.5 consistency check, and role invariance on 47 samples (done).
  - Make the neural-network specificity explicit: unit eligibility, sites, replacements in activation space, checkpoints (done in §III).
  - **The venue decision remains with the authors** (TNNLS_SUITABILITY.md).
- **Status after revision:** **unresolved risk** (venue).

### R1-C2 (CRITICAL → MAJOR after revision): added value over a careful conventional workflow

- **Location:** §VI-C, Table IV.
- **Problem:**
  - The comparison shows *no* detection advantage over a careful checker.
  - The audit's advantage is that its checks are built in and reusable, but its plan has *more* lines of code than the checker.
  - A reviewer will ask why a community should adopt a new representation for checks it can write itself.
- **Why vulnerable:** the paper's central motivation ("conclusions move", "make it checkable") needs evidence that the representation changes outcomes in practice.
- **Evidence required:** real-world usage, either (a) a re-audit of *published* interpretability claims with their released artifacts, counting the claims whose stated conclusion the declared evidence does not support, or (b) a user study.
- **Correction made:**
  - The paper now reports the negative result plainly.
  - It identifies the two concrete semantic differences: attainability, and the refusal of tampered evidence.
  - It states the burden honestly.
  - No new experiment can be run in this revision without external artifacts or participants.
- **Status:** **MAJOR, partially addressed;** remains the strongest reviewer objection.

### R1-M1 (MAJOR): the 47/52 headline was a re-audit of already-seen samples

- **Location:** abstract; §VI-A; §V.
- **Problem:** the samples had been audited before the current rules were adopted (49/52), so the headline is not a fresh held-out estimate.
- **Correction:**
  - Disclosed in §V and §VII.
  - **New confirmatory run on fresh, never-inspected samples** (protocol P3; CNN and BERT-tiny) with pre-stated predictions F-i–F-v. Results in §2.
- **Status:** see §2.

### R1-M2 (MAJOR): no independent mechanism truth for trained central models

- **Location:** §VI-D; §VII.
- **Problem:**
  - The trained-network reference labels come from SIIT-trained InterpBench models: post-hoc and interchange-trained.
  - The MLP, CNN and BERT models have no reference mechanism.
- **Correction:** stated explicitly, and the independence argument is scoped precisely (label independent of the audited *evaluation*, not of the *intervention family* used in training).
- **Status:** **unresolved,** honestly bounded.

### R1-M3 (MAJOR): fault set and checker written by the authors

- **Location:** §VI-C.
- **Problem:** detection rates over a designed fault set are not error rates. The checker was written knowing the faults, which favours the checker, not the audit.
- **Correction:** stated in §VI-C and §VII. A blind fault set from a third party would strengthen it.
- **Status:** **disclosed, unresolved.**

### R1-M4 (MAJOR): statistical treatment

- **Location:** §V; all tables.
- **Problem:**
  - Intervals are unclustered.
  - Pooled counts (41/52, 0/400, 591/760) ignore nesting.
  - There is no inference beyond descriptive counts.
- **Correction:**
  - Per-site and per-checkpoint reporting.
  - Between-checkpoint ranges (Table V).
  - The text never presents intervals as population statements.
- **Status:** **acceptable for a descriptive audit paper;** a reviewer may still ask for cluster-robust intervals.

### R1-M5 (MAJOR): cost of the evidence layer

- **Location:** §VI-G; Table VI.
- **Problem:**
  - About 23× per-test overhead on a 30-feature MLP.
  - Audit + verify account for about 37% of the total on small models.
  - 33–39 MB of evidence per MLP sample.
- **Correction:** all reported, without minimising them.
- **Status:** **addressed (disclosed).**

### R1-m1 (MINOR): AI-disclosure placeholder

- **Location:** Acknowledgment.
- **Problem:** the bracketed note "for author review" must not be submitted.
- **Status:** **author action required.**

### R1-m2 (MINOR): two references with unconfirmed proceedings details

- **Location:** [Joshi 2026], [Méloux 2026].
- **Problem:** ICML 2026 acceptance is confirmed, but PMLR volume and pages are not. The bib now omits the unconfirmed volume and pages.
- **Status:** **author check** before submission.

### R1-m3 (MINOR): Table I judgments

- **Location:** Table I.
- **Problem:** cells such as "partial" need sources.
- **Correction:** BASELINE_CAPABILITY_MATRIX.md lists the sources. A supplementary version can be included.
- **Status:** **addressed in the repository.**

### R1-m4 (MINOR): the BERT-base decomposition depended on a long-running job

- **Location:** Table III.
- **Problem:** at the time of round 1 the BERT-base re-run was incomplete, so totals covered 41 of 52 supports.
- **Status:** see §2.

## 2. Round 2

Filled in after the BERT-base re-run and the fresh-sample run complete; see below.

## 3. Final recommendation

Given after round 2.
