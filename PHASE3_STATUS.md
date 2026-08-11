# Phase 3 Market Sessions Status

Status:

```text
IBKR_PHASE3_MARKET_SESSIONS_READ_ONLY_GO
IBKR_PHASE3_1_SESSION_CONFLICT_POLICY_GO
```

Scope:

```text
Phase 1 connection service      FROZEN_GO
Phase 2 contract resolver       READ_ONLY_GO
Phase 3 market sessions         READ_ONLY_GO
Historical bars                 Phase 4 daily STK enabled
Realtime market data            not enabled
Broker writes                   0
```

Implemented:

```text
Canonical session schema        GO
IBKR tradingHours parsing       GO
IBKR liquidHours parsing        GO
Explicit endpoint dates         GO
Overnight session support       GO
Exchange-calendar validation    GO
Calendar conflicts explicit     GO
Effective collection boundary   GO
Conflict classification         GO
Session cache                   GO
Session hash determinism        GO
```

Validated local contracts:

```text
AAPL conId 265598       SESSION_READY
ASML conId 117589399    SESSION_DEGRADED, explicit XAMS liquid-close conflict
SPY  conId 756733       SESSION_READY
```

Conflict policy:

```text
exact match                          SESSION_READY
deviation within tolerance           SESSION_DEGRADED
deviation outside tolerance          SESSION_BLOCKED
missing or unparseable hours         SESSION_BLOCKED
collection boundary source           IBKR contract hours
calendar role                        independent validator
execution authority                  none
```

Session cache:

```text
data/sessions/sessions.parquet          rows 15
data/sessions/session_manifest.json     GO
data/sessions/session_conflicts.jsonl   rows 5
data/sessions/session_errors.jsonl      rows 0
```

Cache validation:

```text
instrument_count       3
file_count             4
row_count              15
duplicate_rows         0
timezone_errors        0
contract_mismatches    0
content_hash           93AFC2C99963B2BFBC45921EADE101F2B9AD37B78A1CEC5CBD77A28C06AD66ED
```

Verification:

```text
pytest                 380 passed
ruff                   GO
compileall             GO
static audit           GO
financial calls        0
market data calls      0
historical data calls  0
```

Phase 4 now imports Phase 1/2/3 outputs and does not copy connection logic.
