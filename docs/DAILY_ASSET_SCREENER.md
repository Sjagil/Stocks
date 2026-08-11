# Daily Asset Screener V1

## Purpose

The screener is a research-only, long-only daily process for stocks, ordinary ETFs,
and approved commodity ETFs or ETCs. It runs after a completed market session,
creates no order intent, and always reports:

```text
execution_authority = NONE
strategy_authority  = NONE
broker_calls        = 0
order_calls         = 0
```

## Point-in-time sources

The source adapter uses qualified local data without silently blending providers:

1. Yahoo Finance auto-adjusted daily OHLCV cache.
2. EODHD daily OHLCV from the Phase 11.3 causal store.
3. SEC Company Facts with filing acceptance timestamps.
4. Phase 11.3 mover events for top-winner and top-loser context.
5. Phase 11.3 historical Shariah screens.
6. Expiring manual Shariah attestations with explicit observation timestamps.
7. Phase 11.4 security master metadata.
8. The read-only IBKR contract cache for `conId` identity only.

Yahoo and EODHD are compared on overlapping dates. The source with the latest
eligible bar is selected as a complete series. Provider histories are never
spliced together. A material close-price conflict blocks the asset.

## Hard filters

An asset is rejected when any hard filter fails:

- incomplete, invalid, future, or stale price data;
- less than 253 daily observations;
- price below USD 5;
- insufficient 20-day volume or dollar volume;
- inactive or delisted identity;
- leveraged or inverse product;
- unavailable, incomplete, stale, or non-compliant Shariah evidence;
- missing critical point-in-time fundamentals;
- micro-cap, extreme volatility, or extreme overextension;
- missing benchmark data or a material provider conflict.

Missing bid-ask data is disclosed as a warning. An available spread above the
configured limit is a hard rejection.

## Score formula

All components are bounded to `[0, 100]`:

```text
total_score =
    0.40 * fundamental_score
  + 0.40 * technical_score
  + 0.10 * liquidity_score
  + 0.10 * risk_score
```

The weights and all thresholds are in
`config/screener/daily_screener_v1.json`. No parameter optimization is run.

The fundamental score covers positive earnings or FCF, earnings yield, FCF
yield, profitability, debt, accruals, dilution, and dividend quality. Missing
non-critical components remain missing and reduce reported coverage; they are
not replaced by invented values.

The technical score covers EMA 50/200 trend, 3/6/12-month momentum, six-month
relative strength, ATR-normalized trend strength, volatility, EMA distance, and
relative volume.

## Classifications

```text
HIGH_POTENTIAL  total >= 75, fundamental >= 60, technical >= 65,
                all hard filters pass, positive long trend and all momentum gates
WATCHLIST       total >= 60 and all hard filters pass
NEUTRAL         total >= 45 and all hard filters pass
REJECTED        hard-filter failure or total < 45
```

## CLI

```powershell
python .\main.py screener run
python .\main.py screener status
python .\main.py screener report
python .\main.py screener history --symbol ASML
python .\main.py screener top --limit 20
```

`run --as-of YYYY-MM-DD` and `report --as-of YYYY-MM-DD` are available for
causal historical verification. A second run for the same screening date returns
`ALREADY_REGISTERED`; it never overwrites that day's observations.

## Storage

The source of truth is the append-only private database:

```text
data/screener/private/daily_screener.sqlite3
```

Public artifacts are written under:

```text
output/screener/<screening-date>/
```

Each daily directory contains JSON, Parquet, a summary, and a Markdown report.
`output/screener/candidate-history.parquet` is a derived historical registry.

## Current missing fields

- complete current Shariah data for most instruments;
- ETF holdings fundamentals and ETF-level Shariah certificates;
- bid-ask spreads;
- free cash flow, accruals, and dividend facts for many SEC identities;
- non-US point-in-time fundamentals;
- current EODHD bars beyond 2026-07-21;
- a provider-native point-in-time sector benchmark map.

Every missing critical field excludes the asset. These gaps must be filled
before the screener can safely produce broad finalist candidates.
