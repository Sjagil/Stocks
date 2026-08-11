# Phase 11.6 Multi-Timeframe Walk-Forward

Phase 11.6 is an offline research layer above the frozen Phase 11.5 portfolio ledger.
It audits provider-specific bars, preserves per-bar provenance, runs nested walk-forward
selection, evaluates fixed parameters across cohorts, and executes voting, sleeve, and
hierarchical combinations through the global whole-share v2 ledger.

## Safety contract

- `FINANCIAL_FINALIST_GO=false`
- `FORWARD_SHADOW_GO=false`
- `PAPER_STRATEGY_AUTHORITY=NONE`
- `LIVE_STRATEGY_AUTHORITY=NONE`
- `EXECUTION_AUTHORITY=NONE`
- `BROKER_CALLS=0`

The historical 2019-2026 confirmation period is consumed. Future data is unavailable
and no historical result is represented as a sealed future holdout.

## Canonical commands

```powershell
python .\main.py data multitimeframe coverage
python .\main.py research phase11-6 data-audit
python .\main.py research phase11-6 walk-forward --max-walk-forward-identities 500
python .\main.py research phase11-6 cohorts --max-walk-forward-identities 500
python .\main.py research phase11-6 combine --max-combination-identities 500
python .\main.py research phase11-6 audit
python .\main.py research phase11-6 status
```

Intraday histories are fail-closed when their real bar count is below the registered
timeframe minimum. No daily data is upsampled to create synthetic intraday history.

