# Phase 11.10 Status

```text
PHASE11_10_CAUSAL_MULTITIMEFRAME_RESEARCH_GO
FINANCIAL_FINALIST_GO        false
STRATEGY_AUTHORITY           NONE
EXECUTION_AUTHORITY          NONE
BROKER_CALLS                 0
ORDER_CALLS                  0
```

## Campaign

```text
architectures                10
parameter profiles           3
global hypotheses            30
nested selections            188
OOS portfolio runs           940
cost stress                  5–50 bps per side
```

One architecture passed every predeclared research-shortlist gate:

```text
architecture                 weekly_daily_pullback
higher timeframe             1w
lower timeframe              1d
entry                        EMA pullback
folds                        20
positive folds               80%
median OOS portfolio PF      1.1452
worst OOS portfolio PF       0.8254
median OOS CAGR              17.60%
median OOS Sharpe            0.8142
worst OOS drawdown          -28.93%
50 bps median PF             1.0806
parameter plateau folds      80%
median excess CAGR           6.66%
positive excess folds        85%
```

This remains historical discovery. The already consumed history is not an
independent forward holdout, so the result cannot grant paper or live
authority.

## Runtime

The canonical Windows task observes the Phase 11.10 watchlist in
`SIGNALS_ONLY`. Its first integrated cycle completed with:

```text
runtime                      GO
multitimeframe watchlist     GO
orders generated             0
broker writes                0
execution authority          NONE
```

## Evidence Hashes

```text
phase11_10.py
E61BEA1BCC0D2D3CB57DDB08EE9C2AF50572E3F2D041BD94000EFC73D4BF0118

status.json
81F1F3D0F96236440D7757D22A96BB8EDE08A81ACDF9B8C8AFEDCF5B8A9B333F

nested-results.parquet
40FCC17AB30409B08E003E8C5F0F907D99D3F537EAA2E8867749A5BF342B944F
```
