# Phase 6.3 Status

Status: `PHASE6_3_BENCHMARK_CHAMPION_AND_INCREMENTAL_ALPHA_GO`

Financial decision: `NO_EXISTING_FINANCIAL_EDGE`

Strategy status: `REJECTED_NO_INCREMENTAL_FINANCIAL_EDGE`

## Summary

Phase 6.3 evaluated the existing frozen EUR total-return research dataset and existing Phase 6/6.1/6.2 OOS windows only. No broker writes, account reads, streaming market data calls, historical data requests, or provider downloads were made.

Benchmark candidate count: `21`

OOS windows: `15`

Champion benchmark: `BUY_AND_HOLD_GLD`

Champion family: `BUY_AND_HOLD`

Decision reason: the simple champion has positive OOS performance but is dominated by one instrument, one region, and one sleeve. The DSR probability is also below the finalist threshold. The previously complex Phase 6.2 strategy has negative incremental alpha versus the champion.

## Key Evidence

Champion OOS CAGR: `0.16203865277608642`

Champion Sharpe: `0.9991329044270596`

Champion max drawdown: `-0.2365914935742759`

Champion period PF: `1.1931306865760682`

Champion 20 bps stress PF: `1.1910011595226462`

Champion positive window ratio: `0.8666666666666667`

Champion DSR probability: `0.023831005717873`

Single asset contribution max: `1.0`

Single region contribution max: `1.0`

Single sleeve contribution max: `1.0`

Incremental alpha status: `NEGATIVE_INCREMENTAL_ALPHA`

Strategy annualized active return versus champion: `-0.1424554899313492`

Strategy information ratio versus champion: `-0.8917062238223891`

## Counters

```json
{
  "financial_calls": 0,
  "order_calls": 0,
  "market_data_calls": 0,
  "historical_data_calls": 0
}
```

## Artifacts

- `output/research/phase6_3/benchmark-ranking.json`
- `output/research/phase6_3/benchmark-results.parquet`
- `output/research/phase6_3/paired-comparisons.parquet`
- `output/research/phase6_3/champion-analysis.json`
- `output/research/phase6_3/incremental-alpha.json`
- `output/research/phase6_3/leave-one-out.parquet`
- `output/research/phase6_3/parameter-plateau.json`
- `output/research/phase6_3/statistical-validation.json`
- `output/research/phase6_3/decision.json`
- `output/research/phase6_3/manifest.json`
- `output/research/phase6_3/freeze-status.json`

No `forward-shadow-spec.json` was published because the decision is below `PROMISING_SIMPLE_CANDIDATE`.
