# Phase 6 Baselines & Strategy Evidence Status

Status:

```text
PHASE6_BASELINES_AND_STRATEGY_EVIDENCE_GO
FINANCIAL_FINALIST_GO                   NOG NIET
SHADOW_TRADING_GO                       NOG NIET
PAPER_EXECUTION_GO                      NOG NIET
LIMITED_LIVE_CANARY_GO                  NOG NIET
```

Dataset audit:

```text
instrument_count                         17
common_history_start                     2017-08-04
common_history_end                       2026-07-20
common_history_sessions                  2250
point_in_time_universe_min               14
point_in_time_universe_max               17
financial_calls                          0
```

Evidence:

```text
benchmark_families                       5
benchmark_results                        21
strategy_configurations                  108
positive_full_history_configs            108
parameter_plateau                        true
walk_forward_folds                       5
positive_walk_forward_test_folds         3
median_walk_forward_test_PF              1.207941964518837
worst_walk_forward_test_PF               0.0
financial_finalist_status                NO_FINANCIAL_FINALIST_YET
```

Baseline sample:

```text
equal_weight_monthly                     net_return 1.2656  period_PF 1.1531  max_drawdown -0.2150
inverse_volatility_monthly               net_return 0.9678  period_PF 1.1509  max_drawdown -0.1814
trend_200_cash_filter                    net_return 0.8756  period_PF 1.1250  max_drawdown -0.2496
momentum_12_1_top4_trend_filter          net_return 0.7868  period_PF 1.0954  max_drawdown -0.2655
```

Verification:

```text
pytest                                   397 passed
ruff                                     GO
compileall                               GO
phase1_static_audit                      GO
total_return_cache_validation            GO
order_calls                              0
```

Phase 6 remains offline research only. It does not grant broker authority or paper/live execution authority.
