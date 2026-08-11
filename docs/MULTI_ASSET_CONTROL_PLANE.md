# Multi-Asset Control Plane

This project is not a single trading bot for stocks, ETFs and commodities. It is a multi-asset control plane with specialized sleeves and one central allocator.

## Sleeves

Initial sleeves:

```text
equity_momentum
etf_core_rotation
bond_duration
commodity_trend
commodity_carry
defensive_cash
optional_mean_reversion
```

The allocator decides:

```text
which markets are allowed
which sleeves are eligible in the current regime
how much risk each sleeve receives
which positions represent the same underlying risk
whether expected edge exceeds costs and uncertainty
```

Dynamic allocation does not mean continuous churn. It means controlled risk-budget shifts only when economic regime, volatility, correlations and strategy evidence change enough to justify turnover.

## Control Flow

```text
DATA GATE
    -> MARKET REGIME
    -> SLEEVE ELIGIBILITY
    -> SIGNAL ENSEMBLE
    -> CONFIDENCE
    -> RISK BUDGET PER SLEEVE
    -> ASSET RANKING
    -> CORRELATION CLUSTERING
    -> PORTFOLIO OPTIMIZATION
    -> TURNOVER/CAPACITY FILTER
    -> ORDER INTENTS
    -> IBKR EXECUTION GATEWAY
```

Phase 1 currently ends at the read-only IBKR connection service. There is no order gateway authority.

## First Serious Baseline

The first financial baseline should be intentionally plain:

```text
ETF trend and rotation
liquid stock momentum
commodity trend
volatility targeting
correlation caps
defensive allocation
```

More complex ML or AI layers must later prove incremental net value over this baseline after transaction costs, slippage, data confidence and out-of-sample reliability.

## Offline Prequalification Backtest

The current implementation includes an offline multi-asset rotation prequalification command. It only reads the local historical bar cache and never calls IBKR, EODHD or order methods.

```powershell
python .\main.py strategy multi-asset schema
python .\main.py strategy multi-asset status
python .\main.py strategy multi-asset backtest `
  --interval 1d `
  --data-type TRADES `
  --source LOCAL `
  --lookback-bars 3 `
  --top-n-per-sleeve 2 `
  --cost-bps 10
```

The backtest uses completed historical bars only:

```text
lookback momentum
price above lookback average trend gate
volatility-adjusted score
sleeve risk budgets
defensive cash residual
turnover cost
period profit factor
drawdown
expectancy
annualized volatility
Sharpe
Sortino
Calmar
cash and sleeve exposure
region exposure
```

If the local bar cache is empty, the command returns `NO_DATA`. A positive profit factor is only evidence for the local dataset and configuration; it is not live-trading proof and must not be optimized by leaking future data or mining parameters.

## Explicit Non-Goals

Do not build:

```text
one RSI rule for every asset
one AI model that predicts every market
one fixed stop for stocks and futures
daily full-portfolio churn
margin-as-risk budgeting
continuous futures as executable contracts
backtests using current index constituents projected backward
order authority before Phase 12+
```

## Current Gate

Phase 1 is still `PARTIAL_GO` until the live TWS paper forced-disconnect drill is proven.

Phase 2 work may define offline types, schemas and tests. It must not send live IBKR contract requests until Phase 1 is frozen.
