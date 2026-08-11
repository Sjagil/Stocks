# Phase 11.9 Accelerated Multi-Timeframe Discovery

## Cost-Aware EODHD Rerun

The current canonical run contains 14,400 OOS portfolio executions across 30
systems/ensembles and five timeframes. Native EODHD 1h history and deterministic
closed 4h aggregation replaced the earlier short intraday sample.

Ten candidates pass the portfolio gate, representing eight independent
economic outcomes; seven pass the benchmark-incremental gate. The best
available research candidate is `triple_ma_trend` on `1d`.

The lower-timeframe cost-aware rerun produced one marginal basic 4h pass
(`volatility_contraction_breakout`) and no full 1h pass. It is not a strict
shortlist candidate.

```text
FINANCIAL_FINALIST_GO       false
FORWARD_RESEARCH_SHADOW     BLOCKED_NEW_DISCOVERY
PAPER_STRATEGY_AUTHORITY    NONE
LIVE_STRATEGY_AUTHORITY     NONE
EXECUTION_AUTHORITY         NONE
BROKER_CALLS                0
ORDER_CALLS                 0
```

## Decision

```text
PHASE11_9_ACCELERATED_MULTITIMEFRAME_DISCOVERY_GO
PROMISING_ACCELERATED_RESEARCH_CANDIDATE
FINANCIAL_FINALIST_GO       false
STRATEGY_AUTHORITY          NONE
EXECUTION_AUTHORITY         NONE
BROKER_CALLS                0
ORDER_CALLS                 0
```

## Data

Twenty liquid stocks and ETFs were collected from yfinance with 3,473 native
one-hour bars per instrument. Closed four-hour bars were built only from
native one-hour bars. Long adjusted daily histories were aggregated into
closed weekly and monthly bars.

```text
1h     20 instruments, 3,473 bars each
4h     20 instruments, 993 bars each
1d     19 instruments, 3,565-6,679 bars
1w     19 instruments, 741-1,386 bars
1mo    19 instruments, 171-319 bars
```

No daily bars were upsampled into intraday history.

## Research Scope

Twelve base strategies and five fixed ensembles were evaluated on five
timeframes. Two parameter profiles were selected inside nested validation
folds. The campaign produced:

```text
strategy/timeframe pairs   85
global hypotheses          170
nested selections          1,088
OOS cost executions        5,440
cost stress                5-50 bps per side
```

Every run used whole shares, one globally netted EUR ledger, maximum 100%
gross exposure, lagged daily EURUSD conversion and next-bar execution.

## Best Available Formula

```text
strategy                    20-week Donchian breakout
entry                       weekly close > prior 20-week high
exit                        weekly close < current 20-week mean
ranking                     descending 20-week momentum
maximum positions           4
gross exposure              <= 100%
execution                   next weekly bar open
dominant profile            13 of 20 folds
```

OOS portfolio evidence at 10 bps per side:

```text
folds                       20
positive folds              70%
median portfolio PF         1.3273
worst fold PF               0.7320
median CAGR                 12.72%
median Sharpe               0.729
worst drawdown              -33.36%
PF at 50 bps                1.1412
parameter plateau folds     60%
```

Against the fixed SPY/IWM/EEM/TLT whole-share benchmark:

```text
median excess CAGR          +5.64%
positive excess folds       65%
median excess Sharpe        +0.256
benchmark incremental gate  GO
```

The 10,000-run fold bootstrap estimated a 98.87% probability that median PF
is above one. Its 5th percentile median PF was 1.048.

## Intraday Result

No one-hour strategy survived the complete 50 bps cost gate. Four-hour
volume breakout, EMA pullback and breakout consensus remain low-confidence
watchlist items because only two OOS folds are available. They are not
promoted above the weekly candidate.

## Limitation

The historical campaign evaluated 170 global hypotheses and has no new
independent holdout. The Donchian formula is the best available accelerated
research candidate, not a financial finalist or execution-authorized
strategy.
