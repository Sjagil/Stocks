# Phase 4 Historical Bars Status

Status:

```text
IBKR_PHASE4_HISTORICAL_BARS_READ_ONLY_GO
```

Scope:

```text
Security type          STK
Interval               1d
Data type              TRADES
useRTH                 true
Source                 IBKR historical read-only
Realtime data          disabled
Broker writes          0
```

Collected contracts:

```text
AAPL conId 265598          rows 1644    2020-01-02 to 2026-07-20
ASML conId 117589399       rows 1677    2020-01-02 to 2026-07-20
SPY  conId 756733          rows 2902    2015-01-02 to 2026-07-20
```

Cache validation:

```text
instrument_count       19
file_count             19
row_count              51457
duplicate_rows         0
invalid_ohlc_rows      0
missing_sessions       0
unexpected_sessions    0
timezone_errors        0
currency_errors        0
contract_mismatches    0
content_hash           026862A3CA644214DF5A605465F6BBBBED59129ED66210008E558E59B12DC78A
```

Artifacts:

```text
data/bars/security_type=STK/con_id=265598/interval=1d/data_type=TRADES/bars.parquet
data/bars/security_type=STK/con_id=117589399/interval=1d/data_type=TRADES/bars.parquet
data/bars/security_type=STK/con_id=756733/interval=1d/data_type=TRADES/bars.parquet
output/ibkr/bars/requests.jsonl
output/ibkr/bars/errors.jsonl
output/ibkr/bars/collection-manifest.json
output/ibkr/bars/gap-report.json
output/ibkr/bars/cache-validation.json
output/ibkr/phase4-historical-bars-status.json
```

Verification:

```text
Phase 1 immutable service hashes       unchanged
Phase 2 contract cache                 GO
Phase 3 session cache                  GO
pytest                                 381 passed
ruff                                   GO
compileall                             GO
static audit                           GO
account leaks                          0
order calls                            0
market-data streaming calls            0
historical data calls                  collector only
```

Parallel collector policy:

```text
historical client id                   IBKR_CLIENT_ID + 1000
same-client parallel collectors         fail closed
collision status                        CLIENT_ID_COLLISION
data committed on collision             false
```

Non-canonical error classification:

```text
error_class                             CLIENT_ID_COLLISION
canonical_run                           false
retryable                               true
data_committed                          false
phase4_blocking                         false
artifact                                output/ibkr/bars/error-classification.json
```

Current expanded cache includes Phase 4.1 research universe bars plus the initial AAPL and ASML validation files.

Phase 4 is read-only and does not grant order authority.
