# IBKR Phase 1 Connection Service Frozen

Status:

```text
IBKR_PHASE1_CONNECTION_SERVICE_FROZEN_GO
```

Phase 1 is frozen on the read-only TWS paper connection service through `main.py`.

The canonical proof artifact is:

```text
output/ibkr/phase1-disconnect-drill-20260720-133116.json
```

The application gate accepts:

```text
PHASE1_FREEZE_REPORT.md
```

Only after this gate passes may Phase 2 use the connection service for read-only contract identity requests.

## Safety Invariants

```text
IBKR_HOST                       127.0.0.1
IBKR_PORT                       7497
IBKR_ORDER_AUTHORITY            NONE
IBKR_READ_ONLY                  true
IBKR_LIVE_TRADING_ENABLED       false
IBKR_ALLOW_ORDER_TRANSMISSION   false
```

Forbidden in Phase 1 and Phase 2:

```text
placeOrder
cancelOrder
reqGlobalCancel
reqMktData
reqHistoricalData
reqRealTimeBars
```

Allowed in Phase 2 only:

```text
reqContractDetails
reqMatchingSymbols
reqMarketRule
```

Current implementation uses `reqContractDetails` for contract identity validation. It does not request market data or historical bars.

## Freeze Evidence

```text
status                          GO
initial_health                  HEALTHY
disconnect_detected             true
disconnected_or_stale_seen      true
bounded_reconnect_attempted     true
reconnect_attempts              4
reconnect_recovered             true
final_health                    HEALTHY
thread_leak                     false
financial_calls                 0
```

The freeze gate rechecks the drill artifact hash and the current hashes of the frozen Phase 0/1 files before it reports `PHASE1_FROZEN`.
