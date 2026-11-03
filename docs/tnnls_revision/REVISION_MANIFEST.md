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
| results | `MANUSCRIPT_RESULT_VERIFICATION.csv`: 66 rows (R01–R36 evidence claims; M01–M29 manuscript-specific): 42 VERIFIED_FROM_ARTIFACTS, 12 SUPPORTED_BY_SUMMARY_ONLY, 2 UNVERIFIED, 10 CONTRADICTED_BY_ARTIFACTS, 0 REQUIRES_MODEL_RERUN. Manuscript PDF SHA-256 `0471417fcfae81692d523e00873b9c118bfaaef95b5614946e0011e072a1fa60` (16 pp.; not committed). |
| limitations | the manuscript was checked from the compiled PDF only (no LaTeX, no supplementary appendices); saved traces are not available, so research-scale re-audit was not repeated |
| manuscript use | none yet; the paper has not been rewritten |

### Deliverables

- `docs/tnnls_revision/EVIDENCE_INVENTORY.md`
- `docs/tnnls_revision/MANUSCRIPT_RESULT_VERIFICATION.csv`
- `docs/tnnls_revision/REPRODUCIBILITY_AUDIT.md`
- `docs/tnnls_revision/SCIENTIFIC_WEAKNESSES.md`
- `docs/tnnls_revision/REVISION_EXPERIMENT_PLAN.md`
- `docs/tnnls_revision/SERVER_READINESS.md`
- `docs/tnnls_revision/MANUSCRIPT_CONSISTENCY.md`

### Gate

STOP before Stage 2, pending Prof. Kanavos's review.

## Stages 2–6 (2026-11-03, same day; authorised by the owner to continue without an intermediate review)

### Experiments

| ID | stage | protocol (commit before run) | model / data | checkpoint / seed | result | source | compute | status |
|---|---|---|---|---|---|---|---|---|
| S2-CFG | 2 | P1 (`140bb2e`) | A, B, C, D held-out (302 samples) | frozen (A/B rebuilt on macOS, digests verified); C, D pinned HF | exact re-execution (0 diffs); 47/52 = 23 fixed-selection + 24 k-only; τ × 0.5 reverses 0 | `results/runs/S2_*`, `results/aggregate.json` | about 19 CPU-h (D: 17.7) | completed |
| S4-SEED | 4 | P1 | A, B held-out | seeds 0–4, retrained on Linux | hypotheses (i)–(ii) hold on all 24 checkpoint-site pairs; CNN-pixel fixed share 60–83% | `results/runs/S4_*_seed*` | about 2 CPU-h | completed |
| S4-FRESH | 4 | P3 (`d023b8c`) | B, C, never-inspected samples (60 + 40) | frozen / pinned | predictions F-i–F-v hold | `results/runs/S4_FRESH_*` | about 1 CPU-h | completed |
| S4-COST | 4 | P1 | A–D | frozen | per-phase timings; reload / re-audit identical | `aggregate.json` → `timing`; `results/bench_audit_interstellar.json` | included above | completed |
| S4-XPLAT | 4 | P1 | A–D | frozen vs Linux retrain | audit outputs identical; retraining digests differ by platform | `aggregate.json` | — | completed |
| S3-CMP v1 | 3 | P2 (`84e91df`) | A input, 20 held-out samples | frozen + seed 1 | baseline bug (signed ranking) → 4/20 selections differ | `runs/S3_CMP` (server only) | < 5 min | **failed run, superseded** |
| S3-CMP v2 | 3 | P2 (fix `601d640`) | same | same | equivalence 20/20; all 7 faults detected by B1 (with checks) and by BeyondNN; B0 none | `results/S3_CMP_v2`, `stage3/results/comparative_tally.json` | < 5 min | completed |
| S3-IBSPEC | 3 | INDEPENDENT_VALIDATION_PROTOCOL.md (artifact-only; post hoc) | 18 InterpBench SIIT models | committed Phase-7.5 rows | 942/942; 18/8858 (all are nodes the trained model uses) | `stage3/results/interpbench_specification_labels.json` | seconds | completed (not blind) |
| planted-surgery validation | 3 | INDEPENDENT_VALIDATION_PROTOCOL.md §1 | — | — | not run: trivial or duplicates existing evidence | — | 0 | **NO-GO (justified)** |
| fresh samples A, D | 4 | P3 | — | — | not run: A pool exhausted; D about 23 CPU-h | — | 0 | **blocked (cost / pool)** |
| BERT checkpoint replication | 4 | — | — | — | not run: GPU-scale fine-tuning | — | 0 | **not justified now** |

### Failures and corrections

- **Smoke-test launch:** the first launch was run twice; the second exited with "directory exists". The first launch's output is used.
- **tmux:** the first tmux launch reported no running server. The cause was not established; the run directory already existed from an earlier, interrupted launch. As a precaution `TMUX_TMPDIR` was moved to `/data`.
- **S3-CMP v1 baseline ranking bug:** fixed and re-run as v2; v1 is kept.
- **Archive transfer:** a tar stream was corrupted by a stray `du` line; it was re-sent cleanly.
- **Shell safety:** one shell command was blocked by a safety check (`rm` of a variable path). Nothing in it ran; it was re-issued without the deletion.

### Server

- **Where:** `interstellar`, `/data/roufas/research/beyondnn_tnnls_revision/`.
- **CPU and GPU:** 22.7 timed CPU-hours; 0 GPU-hours.
- **Storage:** 6.5 GB, including 3.7 GB of saved MLP evidence and the full per-run outputs.
- **Root filesystem:** 100% full before and after; never written.
- **Jobs:** no other user's jobs were touched.

### Manuscript

- `paper_tnnls/` is a reconstructed IEEEtran source (no original LaTeX was available).
- 11 pages; 6 figures; 6 tables.
- Change log: `MANUSCRIPT_CHANGELOG.md`; traceability: `RESULT_SOURCE_TRACEABILITY.csv`.

### Review

`ADVERSARIAL_REVIEW.md`, two rounds. Final recommendation: **REQUIRES SUBSTANTIAL ADDITIONAL RESEARCH** (for TNNLS).

## AI-assistance record (for IEEE disclosure)

| date | stage | assistance | human review status |
|---|---|---|---|
| 2026-11-03 | 2–6 | Claude Code wrote protocols P1–P3 and the run, analysis, figure and table scripts; executed the server experiments; drafted all reports, the reconstructed manuscript (all sections) and the adversarial review | pending author review |
| 2026-11-03 | 1 | An AI coding assistant (Claude Code, Anthropic) did several things: inspected the repository; wrote the recalculation script; ran the software checks and the CPU reproduction; searched literature with web search; drafted every Stage 1 document. | pending author review |
