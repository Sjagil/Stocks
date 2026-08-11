# Daily Asset Screener Status

```text
DAILY_ASSET_SCREENER_V1          GO
EXECUTION_AUTHORITY              NONE
STRATEGY_AUTHORITY               NONE
BROKER_CALLS                     0
ORDER_CALLS                      0
FINANCIAL_FINALIST_GO            false
```

The canonical 2026-07-24 run screened 221 identities using Yahoo Finance,
EODHD, SEC Company Facts, mover events, historical Shariah screens, the
security master, and the read-only IBKR contract cache.

```text
screened identities              221
HIGH_POTENTIAL                   0
WATCHLIST                        0
REJECTED                         221
data quality                     DEGRADED
```

The dominant blocker is explicit:

```text
SHARIAH_DATA_INCOMPLETE          193
SHARIAH_DATA_UNAVAILABLE          28
```

AAPL and NVDA have dual-source external attestations observed on 2026-07-26.
They are deliberately not applied retroactively to the 2026-07-24 run. The
attestations expire on 2026-08-25 and must then be refreshed or become blocked.

The latest report is:

```text
output/screener/2026-07-24/daily-report.md
```

The result is technically valid but not a financial finalist. Turning the
finalist flag on without current complete eligibility evidence would invalidate
the research contract.
