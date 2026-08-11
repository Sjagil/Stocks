# Phase 6.1 Robustness & Failure Attribution Status

Status:

```text
PHASE6_1_ROBUSTNESS_AND_FAILURE_ATTRIBUTION_GO
decision_status                         INSUFFICIENT_SAMPLE
FINANCIAL_FINALIST_GO                   false
PAPER_STRATEGY_AUTHORITY                blocked
```

Walk-forward diagnostics:

```text
fold_count                              5
positive_folds                          3
worst_period_pf                         0.0
worst_pf_zero_reason                    NO_TRADES
benchmark_comparisons                   25
strategy_beats_benchmark_count          12
```

Fold attribution:

```text
fold 1  2021  net_pnl  14.450  period_pf 1.4562  episode_pf  5.1145  episodes 19  LOW_CONFIDENCE
fold 2  2022  net_pnl  -9.999  period_pf 0.7648  episode_pf  0.3947  episodes 14  LOW_CONFIDENCE
fold 3  2023  net_pnl  -0.025  period_pf 0.0000  episode_pf  null    episodes  0  INSUFFICIENT_SAMPLE
fold 4  2024  net_pnl   2.615  period_pf 1.2079  episode_pf  1.8888  episodes 14  LOW_CONFIDENCE
fold 5  2025  net_pnl  15.666  period_pf 1.7893  episode_pf 17.2836  episodes 10  LOW_CONFIDENCE
```

Robustness:

```text
single_asset_gt_40pct                   false
single_region_gt_60pct                  false
single_sleeve_gt_70pct                  false
single_year_gt_50pct                    false
leave_one_out_tests                     28
break_even_cost_bps                     252.0994
PBO                                     0.2
deflated_sharpe_probability             0.1336
bootstrap_probability_PF_gt_1           0.9933
```

Cost stress:

```text
5 bps   net_return 1.067  period_pf 1.1411
10 bps  net_return 1.032  period_pf 1.1379
20 bps  net_return 0.963  period_pf 1.1316
30 bps  net_return 0.897  period_pf 1.1252
50 bps  net_return 0.772  period_pf 1.1125
```

Interpretation:

```text
PF=0 in fold 3 is not evidence of all losing trades.
It is explained as NO_TRADES: no closed position episodes were generated in that test fold.

The strategy remains technically diagnosable, but the financial evidence is not yet sufficient
for finalist, shadow, paper, or live authority.
```

Artifacts:

```text
output/research/phase6_1/fold-diagnostics.json
output/research/phase6_1/benchmark-comparison.json
output/research/phase6_1/concentration.json
output/research/phase6_1/leave-one-out.json
output/research/phase6_1/regime-analysis.json
output/research/phase6_1/cost-stress.json
output/research/phase6_1/parameter-plateau.json
output/research/phase6_1/multiple-testing.json
output/research/phase6_1/bootstrap-monte-carlo.json
output/research/phase6_1/sample-size-gate.json
output/research/phase6_1/phase6_1-status.json
```

Phase 6.1 remains offline research only. It does not grant broker authority.
