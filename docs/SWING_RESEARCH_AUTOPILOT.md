# Swing Research Autopilot

## Scope

The autopilot extends the existing screener, provider caches, Phase 11 research,
shadow controls and Phase 9 safety boundaries. It does not create execution
authority. Every research and forward artifact reports:

```text
strategy_authority = NONE
execution_authority = NONE
broker_calls = 0
order_calls = 0
```

Supported research timeframes are `1h`, `2h`, `4h`, `6h`, `12h`, `1d`, `1w`
and `1mo`. `1min`, `5m`, `15m`, `30m` and tick paths fail closed. Existing
legacy 5m/15m cache files are preserved but quarantined from research.

## Architecture

```text
daily point-in-time screener
  -> exact-date eligibility gate
  -> deterministic component-based strategy generator
  -> append-only strategy and parameter registry
  -> staged smoke/research/cost-stress engine
  -> append-only trial and decision ledger
  -> research watchlist
  -> frozen forward observer (authority NONE)
  -> existing human-approved Phase 9 paper canary only
```

The same backtest core builds signals, weights and accounting for every family.
Signals use closed candles and weights are shifted one bar before returns are
credited. Portfolios are long-only, capped at 100% gross exposure and retain
unused capital as cash. The risk layer enforces position, sector, region,
currency, turnover, liquidity, order-notional, per-position loss and portfolio
drawdown limits. Supported construction models include equal, inverse-volatility,
score, rank, capped risk-adjusted, sector-first and regional-sleeve allocation.

Every completed actual-data trial is compared, with the same next-bar and cost
rules, to equal weight, inverse volatility, 200-day trend, monthly top-three
momentum rotation and a world-index buy-and-hold benchmark. The fixed primary
benchmark is `ACWI` or `SPY` when available, otherwise equal weight. An
unavailable world benchmark is never represented as a zero-return winner.

## Initial Families

1. `quality_momentum`: quality, six/twelve-month momentum and EMA-200 trend.
2. `trend_pullback`: EMA-20/50 pullback within daily/weekly positive trend.
3. `etf_rotation`: three/six/twelve-month relative-strength rotation.
4. `volatility_contraction_breakout`: contraction, breakout and volume gate.
5. `commodity_etf_trend`: approved product trend/momentum with defensive cash.

Strategy IDs and hashes are deterministic. Exact duplicate payloads are
idempotent. Parameter sets, trials, decisions and forward observations are
append-only in:

```text
data/research/autopilot/private/research_autopilot.sqlite3
```

Public artifacts are written below:

```text
output/research/autopilot/
output/research/forward/
```

## Commands

```powershell
python .\main.py research components
python .\main.py research generate --budget 100
python .\main.py research smoke
python .\main.py research campaign --family etf_rotation
python .\main.py research daily
python .\main.py research weekly
python .\main.py research monthly
python .\main.py research candidates
python .\main.py research strategy --id <STRATEGY_ID>
python .\main.py research compare --ids <ID1> <ID2>
python .\main.py research leaderboard
python .\main.py research rejected
python .\main.py research audit
python .\main.py research autopilot-status
python .\main.py research freeze

python .\main.py portfolio backtest --strategy-id <STRATEGY_ID>
python .\main.py portfolio stress --strategy-id <STRATEGY_ID>

python .\main.py forward register --strategy-id <STRATEGY_ID>
python .\main.py forward run
python .\main.py forward status
```

The bounded operator runner can be invoked by Windows Task Scheduler:

```powershell
.\scripts\run_swing_research_autopilot.ps1 -Cadence auto
```

It uses a single-flight file lock, exits after the selected cadence, and never
invokes IBKR or Phase 9 commands.

## Evidence Boundary

Synthetic fixtures prove engine correctness only. They never count as financial
evidence. Historical promotion requires an eligible screener observation on the
same decision date. Current Shariah attestations are not back-projected, so
historical campaigns correctly report `PIT_ELIGIBILITY_UNAVAILABLE` until
sufficient forward point-in-time history exists.

The current total-return contract also reports
`delisting_settlement = false`. Both that contract and historical exact-date
eligibility must be complete before any actual-market campaign can be financial
evidence. Monthly reports and leaderboards keep fixture and actual-market
results in separate fields.

`FINANCIAL_FINALIST_GO`, strategy authority and execution authority therefore
remain false/none. A forward registration is rejected unless an append-only
decision explicitly grants `FORWARD_OBSERVER_CANDIDATE`; even then the observer
has authority `NONE`.
