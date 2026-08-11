# Phase 2 Contract Resolver

Phase 2 is enabled after `PHASE1_FREEZE_REPORT.md` proved the real TWS paper disconnect/reconnect drill with a verified canonical artifact path/name, parseable artifact timestamp, matching SHA256, ordered valid drill JSON and matching frozen Phase 0/1 file hashes, including the phase-gate implementation itself.

The command surface is already present through `main.py`:

```powershell
python .\main.py ibkr contract status
python .\main.py ibkr contract schema
python .\main.py ibkr contract init-cache
python .\main.py ibkr contract validate-cache
python .\main.py ibkr contract export-identity --con-id 265598
python .\main.py ibkr contract resolve-stock --symbol SPY --asset-class etf --currency USD --exchange SMART --primary-exchange ARCA
python .\main.py ibkr contract resolve-future --symbol GC --exchange COMEX --currency USD --expiry 202612
```

Until Phase 1 is frozen, resolve commands return `PHASE1_NOT_FROZEN` and make no IBKR socket calls.
They still include a prepared `ibkr_contract_spec_v1` when the input is sufficient, so the native API field mapping is testable before live qualification.
The same spec can now build an official `ibapi.contract.Contract` object locally. This is object construction only; it does not create an `EClient`, socket connection or contract-details request.
The resolver maps official `ibapi.contract.ContractDetails` objects into canonical `ResolvedContract` values through the frozen Phase 1 callback path. Live callback wiring is limited to read-only contract identity requests.
ContractDetails batches are evaluated through the same ambiguity policy as cache/candidate lists; invalid payloads count as rejected and are never silently selected.
Evaluated payloads can now produce deterministic resolution artifacts: an audit record for every outcome and a cache row only when the status is exactly `RESOLVED`.
Resolved artifacts also expose an `ibkr_contract_identity_v1` document with canonical IBKR field names, `contract_hash`, `resolved_at`, `server_version`, cache TTL and zero financial-call counters.
Cached identities can be exported by `conId` through `main.py` without broker access. The export command validates the local cache first, returns `NOT_FOUND` for absent `conId` values, and never requests contract details.
Resolution artifacts can be persisted locally: `RESOLVED` records go to request audit plus cache upsert, while `NOT_FOUND` and `AMBIGUOUS_BLOCKED` go to error audit without cache writes.

Prepared native API fields:

```text
STK  symbol, secType, exchange, currency, primaryExchange when supplied
FUT  symbol, secType, exchange, currency, lastTradeDateOrContractMonth
```

FUT specs require an explicit expiry/contract month. A continuous futures chain is a later research feature, not an exact tradable contract identity.
Request validation is fail-closed before the Phase 1 gate: asset class must match the IBKR `secType`, symbol, exchange and primary exchange must already be uppercase canonical IBKR codes without surrounding whitespace, STK requests may not carry expiry, FUT requests must carry expiry, and futures specs only accept IBKR `YYYYMM` or `YYYYMMDD` contract month/date strings.

Ambiguity policy:

```text
0 matches   NOT_FOUND
1 match     RESOLVED
>1 matches  AMBIGUOUS_BLOCKED
```

The resolver must never select the first returned contract automatically.

Candidate filters before any contract can be selected:

```text
symbol
security type
currency
exchange
primary exchange when requested
expiry when requested
resolved contract storage validation
```

Wrong currency, wrong exchange and invalid futures details are treated as non-matches.
`ResolvedContract.validate_phase2_required_fields()` enforces the required STK/FUT field sets before a contract can be persisted as canonical.

Canonical storage fields for resolved contracts:

```text
conId
symbol
localSymbol
secType
exchange
primaryExchange
currency
tradingClass
multiplier
minTick
expiry
firstNoticeDay
lastTradeDay
deliveryType
settlementType
contractSizeUnit
rollGroup
validExchanges
marketRuleIds
longName
industry
category
subcategory
lastTradeDateOrContractMonth
realExpirationDate
lastTradeTime
underConId
timeZoneId
liquidHours
tradingHours
```

Cache layout:

```text
output/ibkr/contracts/
├── stocks.parquet
├── futures.parquet
├── contract_requests.jsonl
├── contract_errors.jsonl
└── contract_manifest.json
```

DuckDB schema inspection:

```powershell
python .\main.py ibkr contract schema
```

The schema command prints `ibkr_contracts` DDL with `con_id` as primary key and `contract_hash` as a required column.

The cache can be initialized without broker access:

```powershell
python .\main.py ibkr contract init-cache
```

The local Parquet cache can be validated without broker access:

```powershell
python .\main.py ibkr contract validate-cache
python .\main.py ibkr contract export-identity --con-id 265598
```

An empty cache is valid before live Phase 2 resolution starts. Existing Parquet rows are checked for required STK/FUT fields, uppercase 3-letter currency codes, positive `minTick`, parseable IBKR `timeZoneId`, `tradingHours`, `liquidHours`, `marketRuleIds`, STK `validExchanges` and FUT lifecycle fields, expected security type, valid `con_id` values and duplicate `con_id` values across both cache files.
Validation also reconstructs each canonical contract and verifies `contract_hash`; a hash mismatch marks the cache `NO_GO`.
Optional futures reference metadata is modeled for later data phases: `firstNoticeDay`, `lastTradeDay`, `deliveryType`, `settlementType`, `contractSizeUnit` and `rollGroup`. The non-date reference fields are all-or-none for FUT and are rejected on STK contracts.
Cache rows and audit rows may have `server_version = null` for offline artifacts, but any present `server_version` must be a positive integer.
Cache writes and JSONL audit validation require timezone-aware `resolved_at` timestamps, so naive timestamps are blocked before they become canonical evidence.
JSONL audit files are validated too: request audit rows must be `RESOLVED` with `selected_conId` and `contract_hash`; error audit rows must be non-resolved with no selected contract. Audit validation also enforces canonical request-code fields and status semantics: `RESOLVED` must have exactly one returned match, `NOT_FOUND` must have zero, and `AMBIGUOUS_BLOCKED` must have more than one.
When present, `contract_manifest.json` is validated against the expected file layout, required fields, optional futures reference fields, cache policy and zero financial-call counters.

Resolution audit fields:

```text
requested_symbol
requested_security_type
requested_currency
requested_exchange
requested_primary_exchange
returned_match_count
selected_conId
resolution_status
resolved_at
server_version
contract_hash
```

Resolved and failed resolution attempts are written as JSONL audit records:

```text
contract_requests.jsonl  resolved/cache-hit style records
contract_errors.jsonl    not found, ambiguous or validation failure records
```

Cache policy:

```text
STK  7 days
FUT  24 hours
```

Offline cache invariants:

```text
fresh exact cache hit       allowed
stale cache hit             miss, later refresh required
duplicate conId             blocked
ambiguous request match     blocked
request/error JSONL audit   required fields and status/match-count semantics
missing required fields      blocked
malformed session fields     blocked
unknown timeZoneId           blocked
malformed marketRuleIds      blocked
malformed validExchanges     blocked
malformed FUT lifecycle      blocked
incomplete FUT metadata      blocked
FUT metadata on STK          blocked
malformed currency code      blocked
nonpositive minTick          blocked
nonpositive FUT underConId   blocked
invalid server_version       blocked when present
naive resolved_at            blocked
STK/FUT Parquet write        required field validation first
Parquet readback             contract_hash, resolved_at and server_version preserved
typed cache readback         ResolvedContract and ContractCacheRow restored
hash mismatch on readback    blocked
validate-cache command       local integrity check, no broker access
resolution artifacts         audit always, cache row only for RESOLVED
contract identity export     canonical IBKR fields plus contract_hash
future reference metadata    optional identity/cache roundtrip
identity export command      local cache only, conId lookup
artifact persistence         resolved upsert, non-resolved audit-only
request validation           pre-gate canonical codes, asset/secType consistency, STK no-expiry, FUT required YYYYMM/YYYYMMDD
contract_hash validation     cache row hash mismatch blocks
audit JSONL validation       request/error routing, canonical request fields, required fields and match counts
manifest validation          layout, schema and financial counters
manifest schema drift        blocked
```

EODHD is treated as a research data source for later data phases. The current code only reports safe booleans such as `api_key_configured` and `requested_enabled`; it never prints the key. Even when `EODHD_ENABLED=true` is configured, actual `enabled` remains false until a later data phase explicitly passes read-only data authority.
