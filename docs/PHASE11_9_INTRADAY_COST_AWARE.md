# Phase 11.9 Intraday Cost-Aware Research

## Scope

The final campaign evaluates 21 base strategies and 9 ensembles on:

```text
1h, 4h, 1d, 1w, 1mo
```

The 1h source is native EODHD history. The 4h bars are deterministic closed
aggregations of the selected native 1h provider partition. Providers are never
silently blended.

Every strategy/timeframe combination uses nested validation selection,
next-bar-open execution, whole shares, EUR normalization, global security
netting, gross exposure no greater than 100%, and per-side cost stress at 5,
10, 20, 30, and 50 bps.

## Runs

The final artifact contains:

```text
strategy/timeframe combinations  150
global hypotheses                450
OOS portfolio executions       14,400
```

The original fast intraday profiles are preserved under:

```text
output/research/phase11_9/generation1_fast_profiles/
```

The second generation uses slower, predeclared 1h and 4h swing windows to
reduce turnover. Its direct comparison is:

```text
output/research/phase11_9/intraday-cost-aware-comparison.json
```

## Result

The slower windows improved 1h and 4h cost resilience. No 1h strategy passes
the full portfolio gate. One 4h strategy, volatility contraction breakout,
passes the basic cost-adjusted gate only marginally:

```text
median OOS portfolio PF     1.1139
median OOS CAGR             1.98%
median OOS Sharpe           0.5450
50 bps median PF            1.0009
worst OOS drawdown         -49.79%
positive fold ratio         55.56%
worst fold PF               0.3650
```

It does not pass the stricter shortlist because fold consistency and worst-fold
evidence are insufficient.

The full cross-timeframe campaign still has ten portfolio-gate passes, eight
independent economic outcomes, and seven benchmark-incremental passes. These
are backtest-positive research results, not an independent financial finalist.

```text
FINANCIAL_FINALIST_GO       false
STRATEGY_AUTHORITY          NONE
EXECUTION_AUTHORITY         NONE
```

No broker or order calls occur in this research path.
