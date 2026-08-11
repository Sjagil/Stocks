# Phase 4.1 Universe Expansion Status

Status:

```text
IBKR_PHASE4_1_UNIVERSE_EXPANSION_BARS_GO
```

Research universe:

```text
manifest                         data/instruments/research_universe.yaml
resolved_instruments             17
blocked_instruments              0
regions                          7
sleeves                          4
research_daily_bar_rows          48136
total_bar_cache_rows             51457
```

Sleeves:

```text
cash
commodity
defensive
equity
```

Important correction:

```text
INDA primary_exchange             BATS
reason                            exact IBKR match after ARCA returned NOT_FOUND
silent fallback                   no
```

Cache validation:

```text
contract_cache                    GO
session_cache                     GO
bar_cache                         GO
duplicate_rows                    0
invalid_ohlc_rows                 0
timezone_errors                   0
contract_mismatches               0
financial_calls                   0
```

Verification:

```text
pytest                            381 passed
ruff                              GO
compileall                        GO
static audit                      GO
order calls                       0
market-data streaming calls       0
historical data calls             collector only
```

Phase 5 dependencies:

```text
corporate_action_status           GO
fx_normalization_status           GO
eur_total_return_status           GO
```

Therefore:

```text
MULTI_ASSET_RESEARCH_DATASET_GO    GO
reason                            17 causal EUR total-return series available
```

Phase 4.1 remains read-only and does not grant order authority.
