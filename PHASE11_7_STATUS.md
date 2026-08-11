# Phase 11.7 Financial Finalist Campaign

## Decision

```text
PHASE11_7_TECHNICAL_CAMPAIGN_GO
BEST_AVAILABLE_RESEARCH_DECISION   NO_FINANCIAL_FINALIST
FINANCIAL_FINALIST_GO              false
FORWARD_SHADOW_GO                  false
STRATEGY_AUTHORITY                 NONE
EXECUTION_AUTHORITY                NONE
BROKER_CALLS                       0
```

The best development-selected candidate was:

```text
ROT_63_100_Q__TOP_2__GROSS_0.50
```

This is a quarterly, risk-adjusted 63-bar momentum rotation with a 100-day
trend gate, two positions and 50% maximum gross exposure.

## Historical Confirmation

The 2019-2025 period was already consumed by earlier research and is not a new
untouched holdout.

```text
CAGR                         10.84%
Sharpe                       0.545
period profit factor         1.111
closed-episode profit factor 1.184
maximum drawdown            -56.46%
positive years               5 / 7
closed episodes              47
```

At 20 bps per side, closed-episode profit factor remained 1.166. At 50 bps it
remained 1.115.

## Passed Gates

```text
positive confirmation years  GO
aggregate episode PF          GO
20 bps cost stress PF         GO
sample size                   GO
single-security concentration GO
PBO                           GO
accounting                    GO
data quality                  GO
stale positions               GO
```

## Blocking Gates

```text
fold-level episode sample     BLOCKED
benchmark excess CAGR         BLOCKED
drawdown budget               BLOCKED
single-year concentration     BLOCKED
Deflated Sharpe Ratio         BLOCKED
future forward holdout        UNAVAILABLE
historical Shariah history    UNAVAILABLE
delisting settlement model    UNAVAILABLE
```

The candidate underperformed the no-cost dynamic equal-weight benchmark by
2.16 percentage points of CAGR. Its DSR probability after 72 trials was 0.0.
The 5,000-path block bootstrap estimated an 85.66% probability of positive
total return, but its 95% CAGR interval still crossed zero.

## Data Corrections

Phase 11.7 quarantined 1,652 identities with split/ticker discontinuities,
used point-in-time 20-session dollar volume, used causal strategy scores,
normalized USD values to EUR, and used official XNYS sessions. The frozen
Phase 11.6 outputs were not overwritten.

## Next Safe Work

Do not optimize the consumed 2019-2025 period again. The next valid evidence
step is a frozen, strategy-agnostic forward observer with authority `NONE`,
after implementing historical Shariah reconstruction and explicit delisting
cash-settlement accounting.
