# Hierarchical Context And Entry Engine

## Purpose

The engine separates information by trading function instead of adding macro,
COT, GEX and order-flow fields to every strategy DNA.

```text
CONTEXT -> BIAS -> SETUP -> ENTRY CONFIRMATION -> RISK AND EXIT
```

It is observation-only:

```text
strategy_authority = NONE
execution_authority = NONE
automatic_orders = 0
```

## Functional leaderboards

Research results are published in separate roles:

```text
STRATEGIC_ALLOCATION  1w and 1mo
ACTIVE_SWING          1h, 2h, 4h and 1d
TACTICAL_EXECUTION    observed entry overlays
EXPLORATORY_FORWARD   incomplete or pending evidence
```

Cross-role ranking is forbidden because frequency, turnover, capacity and
holding period are not comparable.

```powershell
python .\main.py research registry roles
```

Artifacts are written below `output/research/<role>/` and summarized in
`output/research/role_leaderboards/status.json`.

## Asset-specific context

`config/context/asset_transmission_v1.json` maps symbols to economic
transmission groups. Macro families are weighted differently for technology,
financials, defensive equities, commodities and duration assets.

COT is collected from the official CFTC public reporting API. The collector
uses the conservative Tuesday plus 93-hour publication boundary, stores an
append-only private history and never treats weekly positioning as an entry
trigger.

```powershell
python .\main.py market context cot-update --start 2018-01-01
python .\main.py market context cot-status
python .\main.py market context transmission
```

GEX is accepted only as estimated context for options-liquid instruments. It
may affect bias and target interpretation, but it has no standalone trading
authority. Missing context lowers confidence instead of becoming neutral
evidence.

## Entry observer

The observer shortlists current 1h, 2h, 4h and 1d setups. It requires observed
tape data before confirmation and reserves depth checks for the highest-ranked
symbols.

The optimized decision contract applies the layers in order:

```text
hard data/instrument/event/economics vetoes
-> regime family router
-> daily directional bias
-> primary 4h setup
-> optional 2h refinement
-> 1h confirmation
-> optional 15m price improvement
```

A daily signal is directional context only. A 4h signal remains a setup, and a
1h signal cannot become an execution candidate without current daily direction
and a current 4h setup. A 15m observation can improve a price but can never
create the swing thesis.

Stale signals that were originally actionable are retained in the funnel as
explicit vetoed observations. They are not silently omitted and cannot be
rescued by a high confidence score. Setup scores are only comparable within a
strategy family; no universal macro-plus-indicator score is published.

```powershell
python .\main.py market context observe --max-symbols 20 --depth-symbols 5
python .\main.py market context observer-status
```

Observed private inputs are expected at:

```text
data/market_context/private/equity-trades.parquet
data/market_context/private/equity-orderbook.parquet
```

Bar-derived flow and CVD are explicitly classified as proxies and can never
confirm an entry. Without observed tape, a valid setup remains
`ENTRY_DATA_PENDING_OBSERVED_TAPE`.

Each observation is appended to
`data/market_context/private/entry-episodes.jsonl`. An episode records the data
cutoff, context, setup, entry evidence and pending forward outcome. Public
shortlist artifacts contain no broker authority.

## Asset-type specialization

The existing observer applies separate ranking contracts for stocks, ETFs and
commodity proxies. A stock profile emphasizes company/event context and
observed equity flow. An ETF profile reserves weight for breadth, underlying
assets, futures confirmation and fair value. A commodity-proxy profile reserves
weight for curve, physical/fundamental context, futures flow and proxy quality.

Unavailable specialized inputs remain `null` and are listed under
`missing_components`. They are never replaced by bar-flow proxies or invented
neutral observations. The coverage-adjusted score is ranking context only and
cannot independently create an entry.

## Forward episodes and ML

Each signal thesis has one stable append-only episode identity based on signal,
strategy, symbol, timeframe and data cutoff. The first decision snapshot freezes
the deterministic gates, context, setup, observed tape/depth and hypothetical
order assumptions. Outcome labels remain pending until later market observations
can establish fills, MFE, MAE, exits, costs and net R.

The runtime publishes
`NOT_TRAINED_INSUFFICIENT_FORWARD_LABELS` until those labels exist. No ML model
may change direction, authority or broker state from this observer contract.

## Qualification boundary

Phase 11.10 evidence is frozen independently from changing runtime bars. Its
qualification manifest records the research source, strategy catalog, selected
parameters, universe, historical cutoff, cutoff-scoped data hash, cost model,
result hash and qualified strategy identifiers. Previous evidence is archived
by content hash. Multiprocessing workers receive the same cutoff as the parent,
and publication fails if any worker coverage exceeds it.

```powershell
python .\main.py research phase11-10 qualification-audit
python .\main.py research phase11-10 run --historical-cutoff 2026-07-24
python .\main.py research phase11-10 qualification-freeze
```

## Terminal episode outcomes

Feature snapshots and outcomes use separate append-only stores. The outcome
engine writes at most one terminal result per current-schema episode. Missed
limits carry no PnL, and MFE/MAE begin after the hypothetical fill. A candle
that cannot establish stop/target ordering becomes
`DATA_FAILURE / INTRABAR_PATH_AMBIGUOUS`; the engine does not invent a path.
Legacy episodes stay preserved but are excluded from the current completeness
denominator.

```powershell
python .\main.py market context settle-episodes
python .\main.py market context episode-status
```

## Operational boundary

The hierarchy improves selection and timing research. It does not establish a
financial finalist and does not connect signals to Phase 9. Promotion requires
closed forward episodes showing incremental performance after costs versus the
same setup without the entry overlay.
