# Phase 4 Historical Bars

Phase 4 is not enabled yet. This document captures the offline bar-cache contract that can be validated before any IBKR historical-data request or EODHD HTTP request exists.

Current commands:

```powershell
python .\main.py data bars schema
python .\main.py data bars status
python .\main.py data bars init-cache
python .\main.py data bars validate-cache
python .\main.py data bars request-policy
```

These commands are local only. They do not call IBKR, EODHD or any other data provider.

## Supported First Pass

```text
intervals       1d, 1h, 15m
data types      TRADES, MIDPOINT, BID, ASK, ADJUSTED_LAST
sources         IBKR, EODHD
sec types       STK, FUT
```

`EODHD` is only planned for daily `STK` bars at this stage. Intraday EODHD planning and EODHD futures planning are blocked. Futures bars remain IBKR-only until futures-chain, roll and provider-symbol mapping rules are defined. `ADJUSTED_LAST` is blocked for futures.

## Cache Layout

```text
data/bars/
└── sec_type=STK/
    └── con_id=265598/
        └── interval=1d/
            └── data_type=TRADES/
                └── source=EODHD/
                    └── bars.parquet
```

Required fields:

```text
con_id
sec_type
interval
data_type
source
timestamp_utc
open
high
low
close
volume
available_at
```

Target collection fields for the real collector:

```text
timestamp_utc
session_date
con_id
symbol
security_type
currency
exchange
interval
data_type
open
high
low
close
volume
wap
bar_count
source
downloaded_at
```

## Fail-Closed Rules

Historical bar requests are disabled until Phase 4 explicitly grants data authority. The request object validates the schema only after `data_phase_enabled=True`.

Current guards:

```text
con_id positive
request and cache layout con_id positive integer
partition con_id positive integer
record con_id positive integer
categorical cache fields strictly one of the supported enum values
runtime request, bar and cache layout enum values strictly validated
runtime request and bar datetime values must be timezone-aware datetime objects
request policy numeric fields strictly typed as integers
runtime OHLC values must be Decimal instances
runtime volume must be null or a non-negative integer
timezone-aware start/end
parseable ISO-8601 timestamp fields without booleans or surrounding whitespace
end after start
EODHD daily-only planning
EODHD STK-only planning
EODHD TRADES/ADJUSTED_LAST only
ADJUSTED_LAST blocked for FUT
OHLC coherence
finite and parseable OHLC decimals
OHLC decimals without booleans or surrounding whitespace
non-negative volume
volume null or integer
UTC bar timestamps
available_at on or after timestamp_utc for daily bars
available_at on or after timestamp_utc + interval for intraday bars
intraday duplicate timestamp detection
timestamp_utc ascending order per cache file
intraday exact-step gap detection
cache writer refuses out-of-order timestamps, duplicate timestamps and exact intraday gaps
daily gap detection deferred to market calendar logic
financial calls always 0
```

## Request Policy

The queue planner validates request shape and deduplicates identical requests, but does not execute them.

```text
max_concurrent_ibkr_requests    3
max_concurrent_eodhd_requests   2
request_timeout_seconds         60
max_retries                     3
retry_backoff_seconds           2, 5, 15
IBKR hard historical cap         50
execution_enabled               false until Phase 4
```

Policy fields are fail-closed: concurrency, timeout, retry and backoff values must be real integers, not booleans, floats, strings or list-shaped substitutes.

The request planner can also receive local Phase 2 contract cache rows. When those rows are supplied, each planned request must match an existing `(sec_type, con_id)` identity. Missing identities and security-type mismatches are rejected before a request can enter the queue.

`write_bar_cache_file` refuses to create shards with out-of-order timestamps, duplicate timestamps or exact intraday gaps. `validate-cache` reads existing local `bars.parquet` files under the canonical partition layout. An empty cache is valid before collection starts when no bar files exist, but its `research_readiness_status` is `NO_DATA`. An existing `bars.parquet` file must contain at least one row. Existing files are checked for exact record schema, supported enum values for `sec_type`, `interval`, `data_type` and `source` without booleans or surrounding whitespace, positive-integer `con_id` values in request planning, cache path construction, partition paths and records, parseable ISO-8601 timestamp fields without booleans or surrounding whitespace, partition/record consistency, source/security-type/data-type compatibility, finite parseable OHLC decimals without booleans or surrounding whitespace, null-or-integer non-negative volume, OHLC coherence, point-in-time `available_at` ordering, ascending `timestamp_utc` order, duplicate timestamps and intraday exact-step gaps. Runtime request/bar datetime fields must be timezone-aware `datetime` objects, runtime OHLC values must be `Decimal` instances and runtime volume must be null or a non-negative integer. Rejected request reports remain JSON-safe for invalid datetime inputs. Runtime request, bar and cache-layout enum fields are also validated before queueing, key generation or path construction. Missing required fields and unexpected extra fields are both rejected.

`validate-cache` now reports cache-quality fields needed before real research:

```text
instrument_count
file_count
row_count
duplicate_rows
invalid_ohlc_rows
missing_sessions
unexpected_sessions
stale_instruments
timezone_errors
currency_errors
contract_mismatches
first_timestamp
last_timestamp
content_hash
```

The hard research-readiness gate is stricter than empty-cache validity:

```text
research_readiness_status GO
file_count                > 0
row_count                 > 0
duplicate_rows            0
invalid_ohlc_rows         0
timezone_errors           0
contract_mismatches       0
financial_calls           0
```

For `15m` and `1h` bars, the local contract treats `timestamp_utc` as the bar start and requires `available_at` to be at or after `timestamp_utc + interval`. This keeps intraday research from consuming a bar before that bar has completed. Daily bar close-time validation is deferred until market-calendar-specific collection rules are enabled.

When bar files exist, `validate-cache` also checks that each referenced `(sec_type, con_id)` exists in the local Phase 2 contract cache. Missing contract identities are `NO_GO`; this prevents research data from becoming detached from canonical IBKR contract identity. When no bar files exist yet, contract-cache validation is skipped and the empty bar cache remains `GO`.

`init-cache` writes `data/bars/bar_manifest.json`. The manifest records the bar schema, supported intervals/data types/sources, request policy, partition layout and zero financial-call counters. `validate-cache` rejects manifest drift, including changed request policy, changed schema fields or nonzero financial-call counters.

EODHD configuration remains secret-safe. Status reports only show booleans such as `api_key_configured` and `requested_enabled`; API keys are not printed.
