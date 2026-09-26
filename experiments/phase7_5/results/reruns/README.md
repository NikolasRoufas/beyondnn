# Re-runs of representative earlier experiments at the Phase-7.5 HEAD

| re-run | command | comparison with the committed result | outcome |
|---|---|---|---|
| `phase7_scenarios.json` | `python experiments/phase7/scenarios.py` | `experiments/phase7/results/scenarios.json` | 14/14 scenarios + tamper match the pre-registration; the only difference is the Phase-7.5 finding code `configuration_level_disagreement` added to the necessary-not-sufficient scenario (ADR-048, additive) and timings |
| `phase6_ground_truth.json` | `python experiments/phase6/ground_truth.py` | `experiments/phase6/results/ground_truth.json` | all outcomes identical; one use-claim id differs. The Phase-7 HEAD (`0601cde`) produces the same id as Phase 7.5, so the change predates Phase 7.5 (the committed Phase-6 file was not regenerated after Phase 7) |
| `phase5_5_faithfulness_A_limit3.json.gz` | `python experiments/phase5_5/run_faithfulness.py A --limit 3` | first 3 samples of `experiments/phase5_5/results/faithfulness_A.json.gz` | 840/840 rows identical in every field except `result_id`; result ids differ (not investigated per record; record identities include record versions changed since Phase 5.5) |
