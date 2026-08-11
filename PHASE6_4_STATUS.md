# Phase 6.4 Status

Technical marker: `PHASE6_4_PREREGISTERED_MECHANISM_RESEARCH_AND_FORWARD_SELECTION_GO`

Financial decision: `NO_NEW_FINANCIAL_CANDIDATE`

Preregistration hash: `6ED4A089C1F8AEBC75734862781980880916682C93EA6F983EF6D2FE72B9D46A`

Highest permitted promotion in this phase: `FORWARD_RESEARCH_SHADOW_ELIGIBLE`

Actual promotion: none.

## Scope

Phase 6.4 used only frozen EUR total-return research inputs and existing Phase 6 windows/artifacts. It did not create a new historical holdout and all evidence remains `RESEARCH_EVIDENCE`.

The previously complex strategy remains `REJECTED_NO_INCREMENTAL_FINANCIAL_EDGE`.

`BUY_AND_HOLD_GLD` remains a required reference benchmark but is not eligible as diversified benchmark champion because of concentration.

## Hypotheses

```text
HYP_A_DIVERSIFIED_DUAL_MOMENTUM
HYP_B_TREND_BREADTH_RISK_ALLOCATION
HYP_C_DIVERSIFIED_TREND_RISK_PARITY
HYP_D_CRISIS_RESILIENT_SLEEVE_ROTATION
```

Best ranked hypothesis: `HYP_B_TREND_BREADTH_RISK_ALLOCATION`

Window count: `46`

Ablation count: `4`

## Decision

The best hypothesis has positive aggregate CAGR and PF, but Phase 6.4 blocks promotion because the evidence fails required gates:

```text
positive_testwindow_ratio_ge_60       false
no_blocking_concentration             false
component_not_harmful                 false
DSR_gt_phase6_2                       false
bootstrap_beat_diversified_champion   false
hypothesis_mechanism_positive         false
```

No forward-shadow protocol was published.

## Counters

```json
{
  "financial_calls": 0,
  "order_calls": 0,
  "market_data_calls": 0,
  "historical_data_calls": 0,
  "account_calls": 0
}
```

## Artifacts

- `output/research/phase6_4/preregistration.json`
- `output/research/phase6_4/hypothesis-results.parquet`
- `output/research/phase6_4/window-results.parquet`
- `output/research/phase6_4/decision-log.parquet`
- `output/research/phase6_4/incremental-alpha.parquet`
- `output/research/phase6_4/ablation-results.parquet`
- `output/research/phase6_4/leave-one-out.parquet`
- `output/research/phase6_4/statistical-validation.json`
- `output/research/phase6_4/candidate-ranking.json`
- `output/research/phase6_4/decision.json`
- `output/research/phase6_4/manifest.json`
- `output/research/phase6_4/freeze-status.json`

`output/research/phase6_4/forward-shadow-spec.json` is intentionally absent.
