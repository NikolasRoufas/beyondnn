# Computational evaluation (Stage 4, S4-COST)

## Environment

- **Platform:** `interstellar`, Linux x86-64, CPU only, one torch thread per process.
- **Software:** Python 3.12.3, torch 2.14.0 (CPU wheel), BeyondNN 0.1.0 at branch `tnnls-revision`.
- **Concurrency:** up to 40 concurrent single-thread processes on 48 cores. Wall times can include some contention, which inflates absolute times uniformly.
- **Runs:** single timed runs per configuration; not repeated.

**Sources:**
- `experiments/tnnls_revision/results/aggregate.json` → `runs.S2_*_frozen.timing`, produced by `run_central_configs.py`, which times each phase separately;
- `results/S3_CMP_v2/summary.json` → `cost`;
- `results/bench_audit_interstellar.json`, produced by `benchmarks/bench_audit.py`.

## 1. Per-phase cost of the full central design, per held-out sample

| site | samples | results/sample | evidence generation (s) | audit (s) | verify_report (s) | audit + verify share | report (MB / site) | peak RSS (MB / process) |
|---|---|---|---|---|---|---|---|---|
| MLP input | 51 | 72 | 3.66 | 0.97 | 1.15 | 36.7% | 20.3 | 5906* |
| MLP hidden | 51 | 72 | 4.41 | 1.03 | 1.21 | 33.7% | 20.0 | 6866* |
| CNN pixels | 60 | 72 | 3.65 | 1.13 | 1.18 | 38.8% | 23.9 | 3364 |
| CNN relu2 | 60 | 72 | 4.28 | 1.10 | 1.03 | 33.2% | 23.6 | 3836 |
| BERT-tiny | 40 | 69 | 70.4 | 1.21 | 1.11 | 3.2% | 15.4 | 1427 |
| BERT-base | 40 | 68 | 1589.3 | 0.91 | 0.91 | 0.11% | 15.7 | 2140 |

\* The MLP run also saved, reloaded and re-audited the evidence, so its peak memory includes the reloaded traces.

**What each phase contains:**
- **Evidence generation:** model execution *and* BeyondNN record construction (hashing, provenance, controls). It is not attributable to BeyondNN alone.
- **Audit and `verify_report`:** model-free. Only these are pure audit-layer cost.

## 2. The evidence layer versus plain PyTorch at the level of one test

From S3-CMP v2: one PRIMARY IG necessity test (Captum IG, 32 steps; top-3; 50 count-matched controls) on the MLP (30 input features), 20 samples, median of 3 repeats.

| path | median per test |
|---|---|
| BeyondNN (`A.attribute` + `F.top_k` + `F.run`) | **55.3 ms** |
| plain PyTorch + Captum, identical computation | **2.38 ms** |

- The two paths are measurement-equivalent: identical selections and outcomes on 20/20 samples, and a maximum drop difference of 2.9 × 10⁻⁶.
- The ratio is about **23×**. For a model this small, record construction and hashing dominate.
- The ratio was **not measured for BERT models.** Model execution dominates there: BERT-base evidence generation is about 26 min per sample, against a model-free audit of 1.8 s.

## 3. Storage

| artifact | size |
|---|---|
| saved evidence, MLP input site (51 samples) | 1.69 GB (33 MB per sample) |
| saved evidence, MLP hidden site (51 samples) | 1.97 GB (39 MB per sample) |
| audit report per site | 15–24 MB |

Evidence is large because every control effect (50 per test) is a retained, content-addressed record.

## 4. Reload and re-audit (MLP)

| step | input | hidden |
|---|---|---|
| `save_evidence` | 136 s | 120 s |
| `load_evidence` + full re-validation + re-audit | 235 s | 259 s |
| `verify_report` from disk | 243 s | 276 s |
| identical to the in-memory report | yes | yes |

## 5. Scaling of the audit (model-free benchmark)

`benchmarks/bench_audit.py`: comprehensiveness tests with 20 controls under 3 replacements on a 32-unit model; median of 10 iterations.

| samples | records | results | audit (ms) | verify_report (ms) | report (bytes) |
|---|---|---|---|---|---|
| 1 | 539 | 3 | 9.7 | 10.2 | 10,351 |
| 4 | 1,706 | 12 | 45.8 | 46.5 | 29,845 |
| 16 | 6,374 | 48 | 188.7 | 193.0 | 116,809 |

- **Records:** audit time grows about linearly with the number of records, roughly 0.03 ms per record in this range.
- **Full central runs:** the audit cost about 13–18 ms per result across all four models (2,736–4,320 results per site), independent of model size.
- **No extrapolation** is made beyond these sizes.
- **Not measured:** scaling with the number of claims or of alternative configurations separately. Those vary together in the central design.

## 6. Conclusions for the manuscript

- **Audit overhead is constant per result.** It is negligible relative to evidence generation for BERT-scale models (0.1–3%), but it is a third of total time on small MLP/CNN sites.
- **The recording layer costs about 23× a plain implementation per test on a tiny MLP.** It is not free.
- **Storage of full evidence is substantial** (tens of MB per sample).
- All of this is reported in §VI-G and Table VI of the revised manuscript.
