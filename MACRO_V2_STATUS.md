# Macro V2 Status

```text
MACRO_DATA_PIPELINE                    GO
POINT_IN_TIME_VALIDATION               GO
PROVIDER_CONFLICT_RESOLUTION           GO
READ_ONLY_LIVE_READINESS               DEGRADED_GO
MACRO_PAIR_VALIDATION                  GO
FINANCIAL_FINALIST_GO                  false
STRATEGY_AUTHORITY                     NONE
EXECUTION_AUTHORITY                    NONE
```

## Data

```text
registered series                     47
stored observations                   281,574
historical ALFRED releases            11,119
future ECB statistical releases       26
provider conflicts                    0
legacy records overwritten            0
```

Official ECB and Eurostat series are active. Thirteen revision-sensitive
monthly or quarterly FRED series use historical ALFRED vintages. Actual PMI
history remains licensed; OECD business-confidence proxies are labelled as
proxies. The earnings and valuation aggregates use a five-symbol PIT
feasibility ledger and therefore receive a 0.25 confidence multiplier.

## Conflict resolution

The original 2,414 quarantined conflicts included 2,408 DBC
`GLOBAL_COMMODITY_INDEX` rows. Yahoo adjusted closes changed retroactively
after distributions. Macro market levels now use versioned raw closes,
provisional unavailable bars are excluded, and the corrected rows coexist with
legacy rows append-only. A repeated full collection produced zero conflicts.

## Paired evidence

```text
ledger strategy pairs                 38
financial strategy/timeframe pairs    12
identities                            500
outer results                         180
cost stress                           5/10/20/30/50 bps
PBO                                   0.3333
retained macro variants               0
```

Parameters were selected only from the no-macro inner-validation sample.
Baseline and macro variants then used identical parameters, universe and costs
in the outer tests. No macro variant met the predefined OOS uplift and DSR
requirements.

The correct financial decision is:

```text
NO_MACRO_VARIANT_DEMONSTRATED_OOS_VALUE_ADD
FINANCIAL_FINALIST_GO = false
```
