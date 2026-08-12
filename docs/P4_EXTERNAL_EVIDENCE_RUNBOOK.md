# P4 external evidence runbook

## Current boundary

The internal P4 and RL foundation is operational. Production readiness remains
blocked until independently sourced data and natural forward observations pass
the frozen gates. Historical backfills never count as forward evidence. A
current constituent list or current Shariah verdict never replaces history.

## Recommended acquisition order

1. Obtain a small US daily-equity package with delisted securities, historical
   index membership, corporate actions, stable security identifiers and an
   explicit research license. Norgate documents Windows/Python access to
   survivorship-bias-free US data and historical constituents; it does not
   provide live or intraday data. WRDS CRSP plus Compustat Point-in-Time is the
   broader institutional alternative. Validate a sample before purchase.
2. Obtain historical Shariah classifications with effective dates,
   publication/availability dates, methodology versions and stable identifiers.
   Muslim Xchange publicly states that historical compliance data and API access
   are available. A marketing claim is not evidence: require a sample export,
   license terms and explicit point-in-time semantics before attesting it.
3. Enable the exchange-specific IBKR Level-1 subscriptions needed for the
   bounded universe. The runtime requires genuine live bid/ask callbacks and
   rejects delayed data. Network A, B and C cover the main US listing venues;
   choose the exact subscriptions using IBKR's Market Data Assistant.

No subscription or purchase is made by this repository.

## Required normalized datasets

| Dataset | Required proof |
|---|---|
| `security_master` | Stable ID, symbol/exchange/currency, effective interval, availability time and listing state |
| `universe_membership` | Universe and stable ID membership intervals plus availability time |
| `delistings` | Delisting date, availability time, return and reason; missing return remains missing |
| `corporate_actions` | Action date, availability time, type and value |
| `fundamentals` | Period end, availability time, metric, value and revision ID |
| `shariah_classification` | Effective interval, availability time, verdict and methodology version |
| `daily_prices` | Stable ID, session/availability time, OHLCV and adjustment version |

The exact machine-readable schemas are defined in
`src/stocks/p4/data.py`. CSV, TSV and Parquet input are supported.

## Ingest procedure

Stage the vendor export outside the public output tree. Inspect it for secrets
and contractual restrictions. Then ingest one immutable snapshot at a time:

```powershell
$env:PYTHONPATH = "src"
.\.venv-ibkr\Scripts\python.exe -m stocks.p4 ingest daily_prices `
  C:\path\from\vendor\daily_prices.parquet `
  --provider "PROVIDER_LEGAL_NAME" `
  --source-version "VENDOR_RELEASE_ID" `
  --license-id "RESEARCH_LICENSE_ID" `
  --obtained-at "2026-08-11T00:00:00Z" `
  --operator "OPERATOR_NAME" `
  --licensed-for-research `
  --complete-history-attested `
  --point-in-time-semantics-attested
```

Only set an attestation flag when the contract and supplied data explicitly
support it. Repeat for all seven datasets, then run:

```powershell
.\.venv-ibkr\Scripts\python.exe .\main.py p4 audit-data
.\.venv-ibkr\Scripts\python.exe .\main.py p4 publish
```

For a complete delivery, prefer the fail-closed bundle command. Copy
`config/p4_pit_bundle_manifest.example.json`, fill the seven source paths and
their individual attestations, and run:

```powershell
$env:PYTHONPATH = "src"
.\.venv-ibkr\Scripts\python.exe -m stocks.p4 ingest-bundle `
  C:\path\to\p4-pit-bundle.json
```

The command validates every file and the complete cross-dataset bundle before
it registers any snapshot. A rejected preflight writes
`output/p4/bundle-ingest-status.json` and leaves the catalog unchanged. A
restart after an interrupted successful registration is idempotent because
snapshot identities are content-addressed.

Expected production data gates are `PIT_DATA_GO`, `SURVIVORSHIP_GO` and
`SHARIAH_PIT_GO`. A partial package remains `EXTERNAL_DATA_REQUIRED`.

Attestation flags alone are not sufficient. The catalog also verifies the
normalized bundle as one bounded universe:

- every referenced `security_id` must exist in the security master;
- the configured `target_universe` must be present;
- every positive member must have daily prices, fundamentals and a historical
  Shariah classification;
- the security master must contain delisted or inactive securities and every
  such security must have a delisting record;
- membership history must contain an actual exit or ended interval;
- the newest production snapshot is authoritative; a corrupt or incomplete
  replacement cannot silently fall back to an older snapshot.

The detailed result is published under `bundle_coherence` in
`output/p4/data-catalog-status.json`. Local current-universe, reconstructed
Shariah or partial provider data may be ingested for research diagnostics, but
must not be attested as complete production evidence.

## Quote proof

Keep the official native TWS API, paper/read-only settings and zero order
authority. During a natural qualified setup, the existing quote capture must
prove contract routing, primary exchange, live market-data callback type,
positive bid/ask and sizes, acceptable spread and age. Snapshot, delayed or
synthetic data may not satisfy `LIVE_QUOTE_GO`.

## Promotion sequence

After the data gates pass, rebuild the same frozen trials, continue the nine
preregistered candidates, retrain PPO on corrected data and compare it against
the deterministic engine. Promotion still requires sufficient trades and
episodes, regime robustness, cost stress, bootstrap confidence and separate
operator approval. Until then the only valid state is cash plus `SHADOW_ONLY`.
