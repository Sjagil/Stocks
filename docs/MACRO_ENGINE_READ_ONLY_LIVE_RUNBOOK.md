# Macro Engine Read-Only Live Runbook

## Authority

The macro engine is a bounded research observer:

```text
macro_analysis_authority = RESEARCH_ONLY
strategy_authority       = NONE
execution_authority      = NONE
paper_strategy_authority = NONE
live_strategy_authority  = NONE
```

It never calls IBKR, creates an order intent, changes a portfolio ledger, or
promotes a strategy. A macro score is context, not an entry signal.

## Canonical update

Run from the repository root:

```powershell
.\.venv-ibkr\Scripts\python.exe .\main.py macro update
.\.venv-ibkr\Scripts\python.exe .\main.py macro readiness
```

`macro update` is one-shot and protected by
`data/macro/private/macro-update.lock`. A concurrent invocation returns
`UPDATE_ALREADY_RUNNING_BLOCKED`; it does not wait indefinitely.

For Windows Task Scheduler use:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File C:\Users\alhar\Documents\Stocks\scripts\run_macro_update.ps1
```

The scheduler remains operator-controlled. The repository does not install or
modify an operating-system schedule automatically.

## Source policy

```text
FRED/ALFRED     official US observations and historical vintages
ECB             official policy, balance-sheet and money data
Eurostat        official latest EU data; historical vintages unavailable
Yahoo Finance   raw, unadjusted cross-asset closes
Local cache     transparent market breadth
Datascraper     limited five-symbol PIT filing feasibility aggregate
```

Actual PMI histories are licensed and unavailable. OECD business-confidence
series are labelled as proxies and are never called PMI. Earnings and valuation
scores based on the five-symbol feasibility ledger receive a 0.25 confidence
multiplier and are not broad-market evidence.

## Required gates

The read-only update layer is technically ready when:

```text
PIT validation                 GO
provider conflicts             0
all configured providers       healthy
historical release calendar    available
future ECB calendar            available
macro paired validation        GO
broker calls                   0
order calls                    0
```

`READ_ONLY_LIVE_READY_DEGRADED_DATA` is an acceptable infrastructure state.
It means the updater can run safely while some licensed or broad-universe
series remain unavailable. It does not grant financial or trading authority.

## Failure handling

Provider failures, stale inputs, insufficient history and schedule-fetch
failures remain visible in public artifacts. The engine does not forward-fill
an unavailable provider with invented values and does not silently replace a
named official series with a proxy.

Important artifacts:

```text
output/macro/collection.json
output/macro/score.json
output/macro/events.json
output/macro/provider-conflict-resolution.json
output/macro/live-readiness.json
output/research/macro_pairs/status.json
output/research/macro_pairs/decision.json
```
