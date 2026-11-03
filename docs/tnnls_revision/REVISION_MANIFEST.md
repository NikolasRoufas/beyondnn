# TNNLS revision manifest

- **Purpose:** the running research log of the TNNLS revision. There is one entry per stage.
- **Rules:**
  - Frozen experiments, `paper/` (when it exists) and the `v0.1.0` release are never modified.
  - No push without explicit authorisation.
  - No AI co-author trailers in commits.

## Stage 1: repository and scientific evidence audit

| field | value |
|---|---|
| date | 2026-11-03 |
| branch | `tnnls-revision` (from `main` @ `c9e43ad`) |
| commit | recorded in the git log (the Stage 1 commit adds this file) |
| scripts | `experiments/tnnls_revision/stage1/recalculate_stage1.py` (artifact-only, standard library) |
| data source | frozen results in `experiments/phase7/results`, `phase7_5/results` and `phase7_75/results`; SHA-256 recorded in `experiments/tnnls_revision/stage1/results/recalculation.json` |
| experiment ID | S1-RECALC (artifact recalculation); S1-REPRO-A (CPU re-run of `central775.py A heldout --eligibility all` in an isolated `git archive` copy outside the repository) |
| server location | none used for computation. Read-only diagnostics on `interstellar`; see `SERVER_READINESS.md` |
| model / checkpoint | S1-RECALC: none. S1-REPRO-A: model A, retrained deterministically (seed 0, CPU) in the copy |
| seed | as recorded in the frozen scripts (training seed 0; selection and control seeds in the plans) |
| compute cost | recalculation under 5 s; test suite about 2 min; S1-REPRO-A: 10 min 42 s wall clock on a local Mac CPU; exact reproduction (0 per-sample differences; identical checkpoint digest) |
| results | `MANUSCRIPT_RESULT_VERIFICATION.csv`: 36 claims in 37 rows (R08 split into a and b): 24 VERIFIED_FROM_ARTIFACTS, 10 SUPPORTED_BY_SUMMARY_ONLY, 2 UNVERIFIED, 1 CONTRADICTED_BY_ARTIFACTS, 0 REQUIRES_MODEL_RERUN |
| limitations | the manuscript was not available, so the manuscript-to-repository consistency check is blocked; saved traces are not available, so research-scale re-audit was not repeated |
| manuscript use | none yet; the paper has not been rewritten |

### Deliverables

- `docs/tnnls_revision/EVIDENCE_INVENTORY.md`
- `docs/tnnls_revision/MANUSCRIPT_RESULT_VERIFICATION.csv`
- `docs/tnnls_revision/REPRODUCIBILITY_AUDIT.md`
- `docs/tnnls_revision/SCIENTIFIC_WEAKNESSES.md`
- `docs/tnnls_revision/REVISION_EXPERIMENT_PLAN.md`
- `docs/tnnls_revision/SERVER_READINESS.md`

### Gate

STOP before Stage 2, pending Prof. Kanavos's review.

## AI-assistance record (for IEEE disclosure)

| date | stage | assistance | human review status |
|---|---|---|---|
| 2026-11-03 | 1 | An AI coding assistant (Claude Code, Anthropic) did several things: inspected the repository; wrote the recalculation script; ran the software checks and the CPU reproduction; searched literature with web search; drafted every Stage 1 document. | pending author review |
