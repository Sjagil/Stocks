# Phase 11.9 Expanded Research Report

Technical status: `GO`.

## Scope

- 21 base strategies;
- 9 voting ensembles;
- 5 timeframes: 1h, 4h, 1d, 1w, 1mo;
- 3 preregistered parameter profiles per strategy/timeframe;
- 5 cost levels from 5 through 50 bps;
- whole-share, EUR, globally netted portfolio simulation;
- validation selection followed by untouched outer-fold testing;
- 9,600 OOS cost runs and 1,920 fold selections;
- 450 global hypotheses.

## Results

- 10 published shortlist candidates passed the fixed portfolio gates.
- 8 economically independent outcomes remain after duplicate-DNA audit.
- 7 published candidates passed the incremental benchmark gate.
- `triple_ma_trend` on 1d is the current best available accelerated
  research candidate.
- Weekly ADX, Donchian/channel, ATR/Keltner, and robust trend consensus
  families remain promising research candidates.

`atr_breakout` and `keltner_breakout` produced identical economic outcomes.
`donchian_breakout` and `ma_channel` also produced identical outcomes. These
pairs must not be counted as independent alpha sources.

## Authority

```text
FINANCIAL_FINALIST_GO=false
FORWARD_RESEARCH_SHADOW=BLOCKED_NEW_DISCOVERY
STRATEGY_AUTHORITY=NONE
EXECUTION_AUTHORITY=NONE
BROKER_CALLS=0
ORDER_CALLS=0
```

The historical sample has now been used for discovery. It cannot be relabeled
as independent confirmation. The candidates may enter bounded forward
observation, but not automatic paper or live execution under the current
authority contract.
