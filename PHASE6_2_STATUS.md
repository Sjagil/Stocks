# Phase 6.2 Sample Sufficiency & Forward OOS Status

Status:

```text
PHASE6_2_SAMPLE_SUFFICIENCY_AND_FORWARD_OOS_GO
decision_status                         NO_FINANCIAL_EDGE
FINANCIAL_FINALIST_GO                   false
PAPER_STRATEGY_AUTHORITY                blocked
FORWARD_RESEARCH_SHADOW                 available
```

Episode accounting:

```text
carry_policy                            CONTINUOUS_TIMELINE_EPISODES_NOT_CLOSED_AT_FOLD_BOUNDARY
annual_fold_count                       5
artificial_fold_closure_detected         false
NO_TRADES_REASON                         none on continuous annual accounting
```

Forward OOS evidence:

```text
annual_windows                          5
semiannual_windows                      10
total_testwindows                       15
aggregate_closed_oos_episodes           369
effective_sample_size                   2494.0
evaluable_testwindows                   12
positive_evaluable_window_ratio         0.6667
unevaluable_no_trade_windows            0
aggregate_oos_episode_pf                1.8191
median_oos_period_pf                    1.0634
20bps_stress_pf                         1.1316
benchmark_win_rate                      0.48
PBO                                     0.20
DSR_probability                         0.1336
history_policy                          POINT_IN_TIME_ONLY_NO_RETROACTIVE_UNIVERSE_EXTENSION
P5_break_even_cost_bps_bootstrap         251.2678
```

Gate interpretation:

```text
sample_sufficiency                      GO
raw_oos_episode_pf                      GO
20bps_cost_stress                       GO
benchmark_comparisons_won_gt_60pct      false
DSR_probability_candidate               false
financial_edge_decision                 NO_FINANCIAL_EDGE
```

Cost reliability:

```text
break_even_cost_bps                     reported with turnover, closed episodes, gross profit/loss and bootstrap P5
conservative_cost_artifact              output/research/phase6_2/cost-reliability.json
```

Forward shadow:

```text
authority                               FORWARD_RESEARCH_SHADOW
closed_bars_only                        true
orders_enabled                          false
account_positions_used                  false
parameter_changes_allowed_during_run    false
shadow_records                          12
latest_decision                         2026-07-20T23:59:59+00:00
```

The strategy now has enough OOS sample to reject the previous `INSUFFICIENT_SAMPLE` blocker, but it still does not prove incremental financial edge versus simple benchmarks or a sufficiently deflated Sharpe signal. No shadow-to-paper or paper execution authority is granted.

Verification:

```text
pytest                                  406 passed
ruff                                    GO
compileall                              GO
phase1_static_audit                     GO
total_return_cache_validation           GO
financial_calls                         0
```

Artifacts:

```text
output/research/phase6_2/episode-accounting.json
output/research/phase6_2/semiannual-oos.json
output/research/phase6_2/aggregate-oos.json
output/research/phase6_2/point-in-time-history.json
output/research/phase6_2/cost-reliability.json
output/research/phase6_2/forward-shadow.json
output/research/phase6_2/phase6_2-status.json
```
