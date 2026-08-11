# Phase 11.10 Causal Multi-Timeframe Swing Research

Phase 11.10 tests higher-timeframe trend eligibility with lower-timeframe
entries. It is an offline research phase and grants no trading authority.

## Architectures

The fixed research set contains pullback and volatility-contraction breakout
variants for:

- monthly to weekly;
- weekly to daily;
- weekly to four-hour;
- daily to four-hour;
- daily to one-hour.

The higher timeframe is resampled from the same lower-timeframe source. Its
gate is shifted by one complete higher bar before it is forward-filled onto
the lower index. This conservative availability rule prevents an unfinished
higher bar from influencing a lower-timeframe decision.

## Portfolio And Evidence

Every architecture uses:

- next-lower-bar-open execution;
- whole shares;
- EUR-normalized prices;
- global security netting;
- maximum 100% gross exposure;
- nested validation selection;
- 5, 10, 20, 30, and 50 bps cost stress;
- a four-asset benchmark comparison.

Historical discovery cannot create an independent future holdout. Therefore
`FINANCIAL_FINALIST_GO`, strategy authority, paper authority, and live
authority remain false regardless of the raw result.

## Commands

```powershell
.\.venv-ibkr\Scripts\python.exe .\main.py research phase11-10 schema
.\.venv-ibkr\Scripts\python.exe .\main.py research phase11-10 run
.\.venv-ibkr\Scripts\python.exe .\main.py research phase11-10 watchlist
.\.venv-ibkr\Scripts\python.exe .\main.py research phase11-10 status
```

The watchlist is observation-only and cannot route to Phase 9.
