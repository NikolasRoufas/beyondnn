# TNNLS revision experiments

This folder is kept separate from the frozen phase folders. Nothing in it modifies `experiments/phase*/`.

| stage | folder | contents | status |
|---|---|---|---|
| 1 | `stage1/` | `recalculate_stage1.py`: artifact-only recalculation of headline numbers (standard library; no model). `results/recalculation.json`: its output, including source-file SHA-256. | complete |
| 2–4 | `stage2/` … | not started; each requires a committed protocol and the authors' approval | — |

Run Stage 1 from the repository root:

```bash
python3 experiments/tnnls_revision/stage1/recalculate_stage1.py
```
