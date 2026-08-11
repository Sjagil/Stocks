# Phase 2 Status

Status:

```text
IBKR_PHASE2_CONTRACT_RESOLVER_READ_ONLY_GO
```

Phase 2 is a read-only IBKR contract identity resolver. It supports only:

```text
STK
FUT
```

It does not support options, CFDs, combos, paperorders, liveorders, historical bars or realtime market data.

## Live Read-Only Evidence

```text
AAPL USD SMART NASDAQ          RESOLVED, conId 265598
ASML EUR SMART AEB             RESOLVED, conId 117589399
SPY USD SMART ARCA             RESOLVED as STK/ETF, conId 756733
GC USD COMEX chain             AMBIGUOUS_BLOCKED, 34 candidates, no selected conId
```

All live validation calls used:

```text
reqContractDetails             allowed read-only
reqMktData                     0
reqHistoricalData              0
reqRealTimeBars                0
financial_calls                0
```

## Cache

```text
cache_dir                      output/ibkr/contracts
stocks_parquet                 row_count 3
futures_parquet                row_count 0
request_audit_rows             3
error_audit_rows               3
cache_validation               GO
```

The GC chain result is intentionally not cached as a resolved future contract because the request returned multiple valid candidates.

## Implemented Invariants

```text
0 valid matches                NOT_FOUND
1 valid match                  RESOLVED
>1 valid matches               AMBIGUOUS_BLOCKED
fresh exact cache hit          no broker request
stale cache                    refresh required
duplicate conId                blocked
deterministic request_hash     GO
deterministic contract_hash    GO
callback timeout               CALLBACK_TIMEOUT
```

`conId` is the primary technical identity. Hashes are SHA-256 over canonical sorted payloads.
