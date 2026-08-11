# Roadmap

The build order is fixed by safety and evidence, not by strategy excitement.

```text
Phase 0   Official ibapi and read-only TWS paper connectivity
Phase 1   Central read-only connection service through main.py
Phase 2   Contract resolver for STK, ETF and FUT
Phase 3   Trading calendars and sessions
Phase 4   Historical IBKR/EODHD collector and cache
Phase 5   Corporate actions and point-in-time universes
Phase 6   Futures chain, expiry and roll engine
Phase 7   Feature registry and formula validation
Phase 8   Vectorized prequalification backtests
Phase 9   Event-driven portfolio backtester
Phase 10  Risk and allocation engine
Phase 11  Shadow multi-asset runner
Phase 12  IBKR paper order state machine
Phase 13  Reconciliation and restart recovery
Phase 14  Small supervised paper/canary
Phase 15  Limited autonomous paper authority
```

## Current State

```text
Phase 0  GO
Phase 1  PARTIAL_GO
Phase 2  gated CLI, offline schema, futures metadata and local Parquet cache preparation only
Phase 3  local-cache-only market session command surface prepared, overnight window handling and calendar cross-check covered, full calendars still pending
Phase 4  offline historical bar schema, status, manifest, request-policy, partition, validation, Phase 2 identity-link and gap-detection contracts only
Research universe  small unvalidated STK ETF/commodity manifest prepared
Phase 7  offline formula contracts prepared
Phase 8  offline multi-asset prequalification backtest prepared, local bar cache only
```

The open Phase 1 gate is:

```text
forced TWS disconnect correctly detected and bounded reconnect proven live
```

Use:

```powershell
python .\main.py ibkr disconnect-drill-preflight
python .\main.py ibkr disconnect-drill --seconds 180 --poll-seconds 2
```

The preflight must return `GO` before the operator drill starts. Close or restart TWS paper manually during the drill. Phase 1 is frozen only after the drill command returns `GO`.

Current Phase 2 command surface is intentionally fail-closed:

```powershell
python .\main.py ibkr contract status
python .\main.py ibkr contract schema
python .\main.py ibkr contract init-cache
python .\main.py ibkr contract validate-cache
python .\main.py ibkr contract resolve-stock --symbol SPY --asset-class etf --currency USD --exchange SMART --primary-exchange ARCA
python .\main.py ibkr contract resolve-future --symbol GC --exchange COMEX --currency USD --expiry 202612
```

Resolve commands return `PHASE1_NOT_FROZEN` until the freeze report exists.

Offline Phase 3 preparation can inspect cached contract hours without broker access:

```powershell
python .\main.py market status --con-id 265598
python .\main.py market next-open --con-id 265598
python .\main.py market sessions --date 2026-07-20
```

These commands only read local contract cache rows. They do not request contract details, market data or historical data.
`market sessions` lists all cached contracts for the date unless `--con-id` is supplied as a filter.
Reports include known cached hours coverage, explicit `CLOSED` day flags and `NO_KNOWN_NEXT_OPEN` when the local cache has no later trading window.
For supported exchanges, `market sessions` also includes a local `exchange-calendars` cross-check that reports whether the cached IBKR trading day matches the reference calendar session day.

Offline Phase 4 preparation can inspect the bar-cache contract without provider access:

```powershell
python .\main.py data bars schema
python .\main.py data bars status
python .\main.py data bars init-cache
python .\main.py data bars validate-cache
python .\main.py data bars request-policy
```

These commands do not request IBKR historical data and do not call EODHD. They only report the supported intervals/data types, manifest, request queue policy, partition layout, local cache validation, Phase 2 contract-identity links and fail-closed data authority.

Offline research universe preparation:

```powershell
python .\main.py research universe schema
python .\main.py research universe init-manifest
python .\main.py research universe validate-manifest
python .\main.py research universe status
```

This writes `data/instruments/research_universe.yaml` with a small `UNVALIDATED` STK universe. The symbols and exchanges are candidates only; Phase 2 must resolve canonical IBKR `conId` values before any bar request is allowed.

Offline Phase 8 preparation can run a local-only multi-asset sleeve rotation backtest when bar cache files already exist:

```powershell
python .\main.py strategy multi-asset schema
python .\main.py strategy multi-asset status
python .\main.py strategy multi-asset backtest --interval 1d --data-type TRADES --source LOCAL
```

This command reports instrument count, date range, bar count, rebalance count, trade count, gross/net return, CAGR, annualized volatility, Sharpe, Sortino, maximum drawdown, Calmar, profit factor, expectancy, win rate, average win/loss, turnover, total costs, cash exposure, region exposure and sleeve exposure. It does not fetch data, does not request IBKR market data and does not produce order intents.
