# Multi-Timeframe Market Data

This research-only layer keeps provider observations separate and supports the
canonical intervals `5m`, `15m`, `1h`, `4h`, `1d`, `1w`, and `1mo`. The CLI
also accepts `1m` as an explicit alias for one month. One minute is not part of
this contract and would be spelled `1min` in a future revision.

## Sources

- yfinance: native 5m, 15m, 1h, and 1d observations.
- EODHD: native 5m, 1h, and 1d observations when the configured plan permits.
- frozen local IBKR Phase 4 daily bars.
- local Stocks Yahoo and PIT EODHD caches.
- validated datascraper EODHD exports, Yahoo cache, EODHD historical lake, and
  silver research files.

Website and RSS scrapers are context sources, not OHLCV providers. The
datascraper `HistoricalCandleServiceV1` generates deterministic fixture data and
is blocked from this cache.

## Derivation

- 15m may be derived from 5m.
- 4h is derived from 1h and anchored to each exchange session's first bar.
- 1w and 1mo are derived from daily bars.
- Lower-frequency data is never upsampled into a smaller interval.

The final US regular-hours 4h bucket is normally partial because a regular
session is 6.5 hours. It remains marked `is_partial=true`; no synthetic bar is
inserted.

## Commands

```powershell
python .\main.py data multitimeframe schema
python .\main.py data multitimeframe inventory
python .\main.py data multitimeframe collect --symbols SPY,ON --intervals 5m,15m,1h,4h,1d,1w,1m --providers all --lookback-days 30
python .\main.py data multitimeframe validate-cache
python .\main.py data multitimeframe audit
python .\main.py data multitimeframe status
```

All commands have strategy authority `NONE`, execution authority `NONE`, and no
broker write capability. Provider partitions are never silently blended.
