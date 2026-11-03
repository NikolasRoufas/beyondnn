# Protocol P3: confirmatory run on fresh, never-inspected samples (S4-FRESH)

- **Status:** written and committed before execution (2026-11-03).
- **Script:** `experiments/tnnls_revision/run_central_configs.py <model> fresh`.

## Why

The headline 47/52 and its decomposition are a re-audit of held-out samples whose earlier audit (49/52) had been seen before the current audit rules were adopted (ADR-053/054). Those rules are now frozen, at audit semantics 3. A run on samples that no earlier phase has touched tests whether the pattern holds without that exposure.

## Design

| field | value |
|---|---|
| models | CNN (B), frozen checkpoint `d799d1e8…`; BERT-tiny (C), pinned HF revision |
| not included | MLP (A): its pool of correctly classified test inputs (111) is exhausted by the dev and held-out sets. BERT-base (D): its CPU cost (~35 min per sample) is too high for this budget |
| samples | the frozen selection rule, positions 2n…3n−1 of the seeded order (B: 60, C: 40). Never used by any earlier phase |
| design, roles, audit | identical to the frozen central design (P1) |
| unit | held-out sample |

## Predictions (stated before running)

| id | prediction |
|---|---|
| F-i | IG necessity support > random-selection support at every site |
| F-ii | at sites with ≥ 3 IG supports, ≥ 50% of them carry an ALTERNATIVE reversal |
| F-iii | CNN pixels: ≥ 50% of the IG supports are reversed by an ALTERNATIVE that keeps the selected units fixed |
| F-iv | BERT-tiny: ≥ 50% of the ALTERNATIVE reversals arise only from configurations that change k |
| F-v | no PRIMARY-supported claim is reversed by the τ × 0.5 alternative (Proposition 4; a consistency check) |

**Reporting:** every prediction is reported as holding or failing; nothing is re-run.

## Compute

- CPU only; B in 1 process, C in 4 shards.
- At most 1.5 h wall clock.
- Output: `/data/roufas/research/beyondnn_tnnls_revision/runs/S4_FRESH_*`.
