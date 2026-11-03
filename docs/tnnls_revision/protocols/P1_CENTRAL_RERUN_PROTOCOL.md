# Protocol P1: central re-run with per-configuration export, checkpoint replication and cost breakdown

- **Status:** written and committed **before** any of the runs below were executed (2026-11-03).
- **Covers:** experiments S2-CFG (Stage 2), S4-SEED, S4-COST and S4-XPLAT (Stage 4).
- **Script:** `experiments/tnnls_revision/run_central_configs.py`. It wraps the *unchanged* frozen functions of `experiments/phase7_5/central75.py` and `experiments/phase7_75/central775.py`, and adds nothing to the evidence or the audit.
- **Execution:** on `interstellar`, CPU only, in an isolated clone under `/data/roufas/research/beyondnn_tnnls_revision/`. Frozen result files are never written.

## Shared design (identical to the frozen Phase 7.75 central design)

| field | value |
|---|---|
| models | A MLP / breast_cancer (sites `input`, `net.1`); B CNN / digits (`pixels`, `relu2`); C BERT-tiny SST-2 (`tokens`, HF `M-FAC/bert-tiny-finetuned-sst2` @ `41ad670`); D BERT-base SST-2 (`tokens`, HF `textattack/bert-base-uncased-SST-2` @ `95f0f6f`) |
| samples | the frozen held-out rule: correctly classified test inputs in seeded order (`SELECTION_SEED` 1234), positions n…2n−1 (A/B 60, C 40); D: 40 eligible SST-2 validation sentences |
| evidence | G / IG / R top-k at p ∈ {5, 10, 20}%; comprehensiveness and sufficiency with 50 count-matched controls at q = 0.95 under 3 replacements; under the PRIMARY replacement also magnitude-matched and no controls; threshold 0.5 × clean margin |
| roles | PRIMARY: mean / [MASK] replacement (A/input: zero = train mean), k at 10%, count null, recorded threshold. ALTERNATIVE: resample / [PAD], k 5% and 20%, magnitude null, threshold × 0.5. STRESS_TEST: zero (A/input: train min), no null, threshold × 1.5 |
| audit | `bnn.audit` + `verify_report`, BeyondNN at this branch (semantics identical to v0.1.0) |

## S2-CFG: fixed-selection decomposition of the 47/52 result

- **Research question:** among IG necessity claims PRIMARY-SUPPORTED on held-out samples, which declared changes reverse the outcome when every other axis is held at its PRIMARY value?
- **Why the existing artifacts are insufficient:**
  - The committed files store only role × outcome counts per sample.
  - Their `reversal_axes` field merges ALTERNATIVE and STRESS_TEST findings.
  - The per-configuration axis values exist only in audit reports, which were not committed.
- **Unit:** one (site, held-out sample) with IG necessity PRIMARY-SUPPORTED.
- **Configuration families:** every configuration recorded in the audit report's sensitivity profile, with its axis values (replacement, k, null, threshold) and declared role.
- **Single-axis (fixed-subject) comparisons:** a configuration that differs from the PRIMARY configuration in exactly one axis, with all other axes at their PRIMARY values.
  - Replacement-only, null-only and threshold-only changes keep the selected units fixed: IG top-k at the same k is the same set, because the attribution does not depend on the replacement.
  - k-only changes alter the selected units, so they are reported as **subject changes**, separately.
- **Multi-axis:** a configuration differing in ≥ 2 axes. It is reported as an interaction only if no single-axis configuration on that sample already reverses.
- **Reversal:** the configuration outcome is `contradicts` while PRIMARY `supports`. `inconclusive` is a separate third category, never counted as a reversal or as agreement.
- **Roles:** ALTERNATIVE and STRESS_TEST are tabulated separately. The headline replication uses ALTERNATIVE only.
- **Statistics:**
  - Exact counts per site and per axis, with paired, per-sample comparisons.
  - Wilson intervals are reported only per site and labelled unclustered.
  - No pooled interval across checkpoints.
  - No hypothesis test; this is a descriptive decomposition.
- **Replication check before analysis:**
  - Per-sample standings and codes must equal the frozen `central775_*_heldout_all.json`.
  - Any difference is reported. Linux x86-64 can flip exact float ties (docs/REPRODUCIBILITY.md).
  - The decomposition then uses the re-run records, with the differences listed.

## S4-SEED: independent-checkpoint replication (A, B)

- **Question:** do the qualitative central findings hold across independently trained checkpoints?
  - (i) IG necessity support exceeds random-selection support.
  - (ii) Most IG supports have at least one alternative reversal.
  - (iii) The fixed-selection reversal pattern from S2-CFG holds.
- **Why needed:** every finding currently rests on one checkpoint per model.
- **Checkpoints:** A and B retrained with seeds 1, 2, 3 and 4. The training code is identical to `common.train` except for the seed (`torch.manual_seed` and the batch-order generator). Seed 0 must reproduce the frozen checkpoint digest; this is the validation gate.
- **Samples:** the same held-out *rule* per checkpoint. The held-out sets therefore differ by checkpoint (correctness-conditional).
- **Unit:** sample nested in checkpoint.
- **Reporting:**
  - Per-checkpoint counts and between-checkpoint spread (min–max).
  - Samples are never pooled across checkpoints without stating the nesting.
  - Intervention-control seeds are unchanged and are *not* model replication.
- **Hypotheses**, stated before running:
  - (i) IG support > R support on every checkpoint and site;
  - (ii) on every checkpoint with ≥ 3 IG supports, at least half have an ALTERNATIVE reversal.
- **Failure handling:** if a checkpoint violates a hypothesis, it is reported as such; nothing is re-run.

## S4-COST and S4-XPLAT

- **Cost:** for every run, separately record:
  - evidence-generation wall time (model execution plus BeyondNN record construction);
  - `bnn.audit` time;
  - `verify_report` time;
  - report serialization time and size;
  - peak RSS.
- **Model A, seed 0 only:** additionally `save_evidence` time and bytes, `load_evidence` + re-audit time, and whether the re-audit from disk equals the in-memory report.
- **Cross-platform:** the S2-CFG seed-0 runs on Linux x86-64 are compared with the frozen macOS arm64 results (S4-XPLAT).

## Compute budget and stopping rule

- **Hardware:** CPU only, one torch thread per process, at most 40 concurrent processes. GPU not used.
- **Budget:** at most 6 h wall clock and 200 CPU-hours in total; at most 60 GB on `/data`.
- **D:** run in 20 shards of 2 samples. If a D shard exceeds 3 h, D is stopped and reported as incomplete.
- **Stopping rule:** if the seed-0 A or B checkpoint digest does not match the frozen digest, S4-SEED stops before any audit result is read.

## Expected outputs

`/data/roufas/research/beyondnn_tnnls_revision/runs/<tag>/`:
- `summary.json` (central775 format);
- `configs.csv` (per sample × configuration for `*_necessary` claims);
- `timing.json`;
- `log.txt`.

Compact copies go to `experiments/tnnls_revision/results/`.
