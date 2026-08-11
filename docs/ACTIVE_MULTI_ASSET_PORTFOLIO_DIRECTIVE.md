# Active Multi-Asset Swing-Trading And Portfolio-Management Directive

The system is developed as one coordinated, active swing-trading portfolio
manager for stocks, ETFs, commodity securities and, only when the required
contract and roll controls exist, commodity futures through Interactive
Brokers.

It is not merely a signal generator. The complete lifecycle is:

1. Build and validate the investable universe.
2. Scan 1-hour, 4-hour, daily and weekly data.
3. Evaluate technical, fundamental, news, macro, volatility and cross-asset
   context.
4. Rank opportunities across all eligible strategy families.
5. Select a diversified portfolio rather than trading every valid signal.
6. Size positions using equity, stop distance, volatility, liquidity,
   correlation, concentration, portfolio heat and drawdown.
7. Define entry, stop, exit and maximum holding rules.
8. Reassess current positions and issue `HOLD`, `ADD`, `REDUCE`, `EXIT`,
   `REPLACE`, `TAKE_PARTIAL_PROFIT`, `TIGHTEN_STOP`,
   `UPDATE_TRAILING_STOP` or `BLOCK_NEW_ENTRY`.
9. Reconcile positions, orders, fills and cash with IBKR.
10. Maintain complete decision, risk, signal, order and execution audit trails.

## Timeframe Responsibilities

```text
1 week  structural context
1 day   regime, ranking and portfolio selection
4 hour  setup confirmation
1 hour  entry refinement and position management
```

The 5-minute and 15-minute timeframes are research and execution diagnostics,
not primary swing-strategy authorities.

## Portfolio Sleeves

The central allocator coordinates:

- individual stocks;
- regional, country, sector, factor and thematic ETFs;
- bond and defensive ETFs;
- commodity ETFs and ETCs;
- futures only after expiry, multiplier, margin, notice date, roll, liquidity
  migration and executable-contract controls are complete.

No sleeve allocates capital independently from total portfolio constraints.

## Strategy Breadth

Research must continue across economically distinct families, including trend,
time-series and cross-sectional momentum, dual momentum, pullbacks, relative
strength, volatility compression, Donchian and range breakouts, Bollinger
setups, VWAP and anchored-VWAP reclaims, quality momentum, earnings revisions,
post-earnings drift, gap continuation, failed breakouts and breakdowns,
regime-filtered mean reversion, ETF rotation, commodity trend/carry and
macro-sensitive allocation.

Simple strategies remain valid research candidates. Complexity is not a
selection criterion. Frozen parameters, realistic costs, next-bar execution,
walk-forward evidence and portfolio-level results remain mandatory.

The research generator and fine-tuning system may continuously propose and
test new DNA, but may not mutate an active strategy or grant itself execution
authority.

## Portfolio Decision Contract

Every opportunity receives comparable technical, family-breadth, timeframe,
fundamental, liquidity, regime, relative-strength and setup scores, with
explicit penalties for event risk, weak data, correlation, concentration and
unresolved contracts.

The allocator enforces:

- whole-share execution accounting;
- gross exposure at or below 100%;
- global security netting;
- position, sleeve, sector, region and correlation-cluster caps;
- portfolio heat and drawdown limits;
- EUR costs and FX;
- liquidity and implementation-shortfall controls;
- a configurable replacement threshold above switching costs.

Cash is a valid allocation.

## Rebalancing And Reporting

```text
Daily    signals, ranking, risk, position actions and exits
Weekly   sleeve weights, clusters and deeper portfolio rebalance
Monthly  universe, macro, execution-cost and strategy-health review
```

Urgent risk-reducing actions may occur sooner. Normal rotation must avoid
unnecessary turnover.

Daily Telegram reporting includes ranked and trending signals, portfolio
actions, data and regime status, important news and upcoming macro events.
Notifications never change authority or submit orders.

Private portfolio planning uses the latest complete read-only broker snapshot
to calculate equity/cash-aware whole-share shadow targets, current and target
exposure, portfolio heat and netted quantity deltas. Financial values and
position quantities remain under `data/portfolio/private`; public artifacts
contain only ratios, hashes, status, counts and advisory actions.

Each decision cycle and position action is written idempotently to an
append-only private ledger. Replaying the same broker snapshot, signals and
policy must not create a second decision or action event.

## Authority

Normal operation uses `main.py` and the native IBKR API. No strategy may bypass
the central allocator, risk engine, order state machine or reconciliation
layer.

Progression remains:

```text
research
-> frozen shadow
-> IBKR paper
-> small supervised live
-> evidence-based capital scaling
-> controlled portfolio management
```

Positive research does not equal proven live profitability. Live order
submission remains fail-closed until all execution, reconciliation, financial,
account-fingerprint and operator-approval gates are satisfied.
