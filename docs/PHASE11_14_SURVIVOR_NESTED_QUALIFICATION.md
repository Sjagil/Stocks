# Phase 11.14 Survivor Nested Qualification

Phase 11.14 converts broad Phase 11.12 discovery results into a bounded,
diversified survivor cohort. It is a research and observation layer. It does
not grant strategy or execution authority.

## Evidence contract

- Candidate selection starts from 10 and 50 bps Phase 11.12 survivors.
- Economic duplicates are removed before cohort construction.
- Selection is balanced across `1h`, `4h`, `1d`, `1w`, stocks, ETFs and
  commodity proxies where qualified data exists.
- Each candidate is tested over six nested outer folds.
- The responsive, balanced or conservative profile is selected only from the
  preceding validation segment.
- Portfolio simulation uses whole shares, EUR accounting, next-bar execution,
  gross exposure at or below 100%, an exposure-matched benchmark and
  10/20/50/100 bps cost stress.
- The qualification boundary is frozen before forward observation.
- Historical results are explicitly
  `SELECTION_CONDITIONED_REUSED_HISTORY`, not independent confirmation.

## Current qualification

```text
evaluated candidates              16
research passes                   16
robust observer candidates         9
1h robust candidates               0
4h robust candidates               2
1d robust candidates               4
1w robust candidates               3
financial finalist                 false
execution authority                NONE
broker/order calls                 0
```

The robust formulas are:

```text
nr7_breakout                 1w  STOCK
ppo_trend                    1d  STOCK
choppiness_breakout          4h  STOCK
nr7_breakout                 1w  COMMODITY_PROXY
choppiness_breakout          1d  STOCK
trend_quality_52w            4h  COMMODITY_PROXY
trend_pullback_consensus     1w  ETF
rsi14_trend_pullback         1d  COMMODITY_PROXY
ma_crossover                 1d  ETF
```

No 1h candidate passed the frozen robustness and 50 bps cost gates. The
observer must not weaken those gates or substitute a lower timeframe result.

## Forward observer

`phase11-14 observe` evaluates only frozen robust candidates against the latest
closed bars. Raw signals and currently attested targets remain separate.
Only current Shariah-attested symbols may become research target weights.

The operations service runs this observer at most once per hour, before the
central portfolio plan. The portfolio manager may rank and size the resulting
research signals, but marks every one as:

```text
portfolio eligible source           true
research allocation eligible        true
deployment eligible                 false
execution authority                 NONE
```

Every successful survivor observation also refreshes the existing Telegram
shadow digest. The digest keeps broad Phase 11.12 signals, frozen Phase 11.14
targets, central security-netted portfolio ranking, current news and macro
event context in separate sections. A qualification-hash mismatch, non-frozen
boundary or missing current attestation suppresses the survivor section.

The independent forward audit counts only bars that both:

1. closed strictly after the frozen qualification data end; and
2. were first observed after qualification.

Backfilled bars and the qualification boundary bar never count as independent
forward evidence.

## Commands

```powershell
.\.venv-ibkr\Scripts\python.exe .\main.py research phase11-14 schema
.\.venv-ibkr\Scripts\python.exe .\main.py research phase11-14 status
.\.venv-ibkr\Scripts\python.exe .\main.py research phase11-14 observe
.\.venv-ibkr\Scripts\python.exe .\main.py portfolio plan
```

Re-running qualification requires a new explicit research phase. The existing
frozen boundary is never silently replaced by the observer or operations
runtime.
