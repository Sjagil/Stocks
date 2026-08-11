# Phase 11.8 Realistic Multi-Strategy Forward Research

Phase 11.8 adds a bounded research campaign above the existing frozen
research and execution-control layers. It does not activate a strategy.

## Commands

```powershell
python .\main.py research phase11-8 schema
python .\main.py research phase11-8 data-coverage
python .\main.py research phase11-8 portfolio-audit
python .\main.py research phase11-8 run --max-stock-identities 12
python .\main.py research phase11-8 finalize
python .\main.py research phase11-8 status
```

## Strategy Scope

The campaign supports MA crossover, asymmetric MA, MA channel, Bollinger
breakout, volatility-contraction breakout, ETF trend and commodity-ETF trend.
Quality momentum is registered but blocked when broad point-in-time
fundamentals and earnings revisions are unavailable.

Only native timeframes with sufficient history are admitted. Daily bars are
aggregated causally into closed weekly and monthly bars. Intraday data is
never synthesized from daily bars.

## Portfolio Semantics

Signals are selected causally using the prior closed bar. Ties are resolved
by security ID, so input ordering cannot decide which simultaneous signal is
accepted. Trades execute at the next bar open.

All instruments share one EUR cash ledger. Positions use whole shares, one
security can occupy only one global slot, gross exposure cannot exceed 100%,
and cash cannot become negative. Historical prices use point-in-time EURUSD
conversion. Each fill records transaction costs and a separate 1 bp FX
friction per side.

Nested walk-forward selection uses validation portfolio PF and CAGR. Final
research-candidate selection requires a plateau, positive median CAGR,
50-bps survival, bounded drawdown and acceptable worst-fold PF. It does not
select decimal parameter peaks.

## Governance

Macro data remains context-only until a paired corrected OOS ablation proves
positive incremental CAGR and Sharpe. A selected candidate is written to an
immutable public artifact and an append-only private holdout registry.

```text
FINANCIAL_FINALIST_GO       false
STRATEGY_AUTHORITY          NONE
EXECUTION_AUTHORITY         NONE
PAPER_STRATEGY_AUTHORITY    NONE
LIVE_STRATEGY_AUTHORITY     NONE
```

No Phase 11.8 command makes broker or order calls.
