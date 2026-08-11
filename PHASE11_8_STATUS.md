# Phase 11.8 Realistic Multi-Strategy Forward Campaign

## Decision

```text
PHASE11_8_REALISTIC_MULTISTRATEGY_CAMPAIGN_GO
PHASE11_8_FORWARD_OBSERVATION_CANDIDATE_FROZEN_GO
FINANCIAL_FINALIST_GO              false
FORWARD_RESEARCH_SHADOW            OBSERVATION_ONLY
STRATEGY_AUTHORITY                 NONE
EXECUTION_AUTHORITY                NONE
PAPER_STRATEGY_AUTHORITY           NONE
LIVE_STRATEGY_AUTHORITY            NONE
BROKER_CALLS                       0
ORDER_CALLS                        0
```

Phase 11.8 is technically complete. It does not grant financial-finalist,
paper, or live authority.

## Data Coverage

The campaign audited 32,774,820 point-in-time price rows across 13,824
identities. Native daily, weekly, and monthly histories were usable. Native
1-hour and 4-hour histories did not meet the registered minimum-history gate
and were blocked. No daily data was upsampled.

The following licensed or point-in-time datasets were not present locally:

```text
licensed PMI history
broad earnings revisions
broad fundamentals and valuation history
broad earnings-event history
```

Quality momentum remains blocked until those inputs are available with
license and point-in-time provenance.

## Portfolio Contract

```text
whole shares                         enforced
global gross exposure                <= 100%
security netting                     global
simultaneous selection               score desc, security ID asc
execution                            next bar open
base currency                        EUR
historical FX conversion             point-in-time EURUSD
explicit FX friction                 1 bp per side
cost stress                          5, 10, 20, 30, 50 bps per side
primary selection metric             portfolio period profit factor
```

The invariant audit is `GO`.

## Walk-Forward Result

The bounded campaign used 12 stock identities, 15 ETFs and 5 commodity ETFs.
It produced 2,025 fold/cost results for 21 financially testable
strategy/timeframe pairs.

The robust selection policy rejected the largest raw peak and prioritized
parameter plateau, worst-fold behavior, cost stress, drawdown and then median
portfolio PF.

```text
candidate                   asymmetric_ma
universe                    stocks
timeframe                   1w
folds                       20
positive-fold ratio         0.65
median OOS portfolio PF     1.1347
worst OOS portfolio PF      0.7582
median OOS CAGR             3.88%
worst OOS drawdown          -45.35%
50 bps median PF            1.1215
plateau-fold ratio          0.55
```

This is a frozen forward-observation candidate, not a financial finalist.

## Forward Holdout

The baseline date is 2026-07-27. Independent future observations are zero.
The minimum observation window is three months. Historical 2019-2026 evidence
is consumed and cannot be reused as independent confirmation.

The private append-only registry is:

```text
data/research/phase11_8/private/forward-holdout.sqlite3
```

The Phase 9 route is blocked. The existing Phase 8.2 infrastructure may only
observe strategy-agnostic signals with authority `NONE`.
