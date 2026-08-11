# Phase 11.11 Status

```text
PHASE11_11_HMM_REGIME_RESEARCH_GO
```

## Evidence

```text
strategy/timeframe pairs expected    150
strategy/timeframe pairs evaluated   150
paired fold/cost rows                 4,440
train-fold HMM models                74

HMM overlays promoted                0
profitable baselines retained        29
not promoted                         121
```

The 29 retained research candidates consist of:

```text
1D     2
4H     1
1W    26
```

All are registered as `FROZEN_SHADOW` with authority `NONE`. No monthly
candidate is promoted because only five evaluable monthly folds are
available. No HMM overlay is promoted because none passed every preregistered
incremental-performance and model-stability gate.

Across the retained candidates:

```text
median baseline PF       1.4095
median HMM PF            1.3342
median baseline CAGR     13.87%
median HMM CAGR           6.43%
median baseline Sharpe    0.9260
median HMM Sharpe         0.7253
```

The HMM frequently reduced drawdown, but generally reduced return and Sharpe
too much. It remains an observational risk overlay.

## Current state

At the latest production fit:

```text
as of                    2026-07-24
RISK_ON_TREND            60.54%
NEUTRAL_CHOPPY           39.36%
STRESS_HIGH_VOL           0.09%
risk multiplier           0.8418
```

This is a filtered model estimate, not a prediction or trading instruction.

## Governance

```text
FINANCIAL_FINALIST_GO       false
STRATEGY_AUTHORITY          NONE
EXECUTION_AUTHORITY         NONE
PAPER_STRATEGY_AUTHORITY    NONE
LIVE_STRATEGY_AUTHORITY     NONE
BROKER_CALLS                0
ORDER_CALLS                 0
```

The historical sample has already been used for discovery. Independent
forward evidence remains unavailable, so the 29 candidates may be observed
but not automatically traded.

## Artifacts

Canonical evidence is under:

```text
output/research/phase11_11/
```

Important files:

```text
paired-results.parquet
paired-summary.csv
model-fold-audit.csv
promotion-registry.json
frozen-shadow-registry.json
current.json
audit.json
status.json
manifest.json
```

