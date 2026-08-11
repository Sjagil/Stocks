# Macro Engine and Analyst V1

## Scope

The Macro Engine supplies deterministic, point-in-time research context to the
existing screener, research autopilot, strategy registry and portfolio layer.
It cannot generate orders, promote strategies or grant execution authority.

```text
macro_analysis_authority = RESEARCH_ONLY
strategy_authority       = NONE
execution_authority      = NONE
paper_strategy_authority = NONE
live_strategy_authority  = NONE
```

The implementation lives in `src/stocks/macro`. Configuration is versioned in
`config/macro/macro_v1.json`. Private append-only observations are stored in
`data/macro/private/macro.sqlite3`; public aggregate artifacts are written to
`output/macro`.

## Point-In-Time Contract

Each observation records its economic period, publication timestamp,
`available_at`, revision status, source, provider, value, frequency, region,
vintage and quality status. A value is selected only when `available_at` is not
later than the requested decision time. Multiple vintages are retained.

For FRED monthly and quarterly series without a historical release calendar,
V1 uses the conservative rule:

```text
estimated availability = economic period end + configured release lag
```

Legacy period-start estimates remain append-only but are quarantined from
feature calculations. Where actual historical vintages are unavailable, the
engine publishes `VINTAGE_HISTORY_UNAVAILABLE`. This prevents a current revised
series from being presented as genuine historical release evidence.

Provider payload conflicts are never overwritten. Collection quarantines and
counts them by series.

## Scores

The registry contains 44 series and 15 score families:

```text
growth, inflation, labor, liquidity, monetary, credit,
financial_stress, housing, consumer, currency, commodity,
breadth, valuation, earnings_cycle, risk_appetite
```

Transforms include year-over-year change, period change, rate of change and
transparent level anchors. Features use robust median/MAD normalization,
direction adjustment and clipping to `[-100, 100]`. Every score publishes
coverage, confidence, missing and stale inputs, and positive and negative
contributions. Insufficient critical coverage produces `UNKNOWN` or
`DATA_INCOMPLETE`, never an invented neutral value.

## Regimes

The engine classifies growth, inflation, liquidity, monetary, credit, market,
currency and commodity axes. The overall state uses configured thresholds,
hysteresis and minimum confirmations. During a pending transition the public
state is `TRANSITION`. Regime output is descriptive and is not a trading
signal.

Historical reconstruction uses only information available at each as-of
timestamp. Future SPY returns over 5, 10, 21, 63 and 126 trading days are joined
after classification and published as descriptive outcome analysis. They do
not alter regimes, selection or authority.

## Macro Analyst

The analyst is a fixed Dutch-language template. It reports regime, directions,
drivers, conflicts, vulnerable and supported areas, missing data and stale
inputs. Daily, weekly and monthly JSON and Markdown reports are content
addressed and immutable. No LLM, ML model or opaque classifier is used.

## Integrations

The screener may allocate at most 10% weight to available macro context. When
macro context is unavailable, its weight is removed and the original weights
are renormalized. Existing Shariah, fundamental and technical hard gates remain
unchanged.

The research autopilot supports at most two macro filters per strategy. A macro
filter cannot be the sole asset signal. Missing point-in-time macro history
blocks the variant. Any macro variant requires a matched no-macro baseline.

The portfolio integration only applies configured bounded exposure multipliers
from 0.5 to 1.1. It remains long-only and cannot create leverage, shorts or
orders.

## CLI

```powershell
python .\main.py macro collect --start 2000-01-01 --end 2026-07-27
python .\main.py macro update
python .\main.py macro validate
python .\main.py macro status
python .\main.py macro score --as-of 2026-07-27
python .\main.py macro regime --as-of 2026-07-27
python .\main.py macro history --rebuild
python .\main.py macro events
python .\main.py macro report --period daily
python .\main.py macro report --period weekly
python .\main.py macro report --period monthly
python .\main.py macro explain
python .\main.py macro compare --date-a 2025-01-31 --date-b 2026-01-31
python .\main.py macro sector-impact
python .\main.py macro strategy-impact --strategy-id <STRATEGY_ID>
python .\main.py macro audit
python .\main.py macro freeze
```

## Known Limitations

- Historical release vintages are not available for every configured series.
- Several official/manual series are currently unavailable or stale.
- Structured historical event schedules and consensus histories are absent.
- Local breadth depends on the available canonical market cache universe.
- Forward regime samples are overlapping and descriptive, not causal proof.
- Macro context is not financial-finalist evidence and cannot activate shadow,
  paper or live strategy authority.

