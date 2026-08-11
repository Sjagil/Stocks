# Phase 3 Market Sessions

Phase 3 builds an offline, read-only market-session layer from cached IBKR contract identities.

Primary source:

```text
output/ibkr/contracts/stocks.parquet
output/ibkr/contracts/futures.parquet
```

Fields used:

```text
conId
symbol
secType
exchange
primaryExchange
currency
timeZoneId
tradingHours
liquidHours
contract_hash
```

Exchange calendars are validation inputs only. When IBKR cached hours and the local exchange calendar disagree, the engine writes an explicit conflict instead of silently choosing one source.

CLI:

```powershell
python .\main.py market sessions schema
python .\main.py market sessions resolve --con-id 265598 --date 2026-07-20
python .\main.py market sessions resolve --con-id 117589399 --date 2026-07-20
python .\main.py market sessions status --con-id 756733
python .\main.py market sessions next-open --con-id 265598
python .\main.py market sessions validate-cache
python .\main.py market sessions range --con-id 265598 --start 2026-07-20 --end 2026-07-22
```

Artifacts:

```text
data/sessions/sessions.parquet
data/sessions/session_manifest.json
data/sessions/session_conflicts.jsonl
data/sessions/session_errors.jsonl
```

The canonical primary key is:

```text
con_id
session_date
session_type
```

The deterministic `session_hash` is SHA256 over:

```text
con_id
session_date
session_open_utc
session_close_utc
liquid_open_utc
liquid_close_utc
timezone_id
```

Readiness:

```text
SESSION_READY      deterministic, no blocking conflict
SESSION_DEGRADED   deterministic with explicit non-fatal calendar conflict
SESSION_BLOCKED    stale, unresolved, invalid timezone, or malformed hours
```

Phase 3 does not request historical bars, realtime market data, account data, or orders.
