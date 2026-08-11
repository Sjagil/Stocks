# Canonical Strategy Taxonomy Integration

The supplied taxonomy is the repository's coverage contract, not an instruction
to activate every named indicator. Proprietary aliases and mathematically
equivalent variants are not duplicated.

Machine-readable evidence:

```text
output/research/autopilot/component-taxonomy-coverage.json
```

The contract covers 24 sections and requires every active component to publish:

```text
name
version
category
formula
required_fields
supported_assets
supported_timeframes
lookback
warmup
available_at_rules
missing_data_policy
causality_status
unit
output_range
test_status
```

The repository currently registers 116 transparent components. Supported swing
timeframes begin at `1h`; `1m`, `5m`, `15m`, `30m` and tick data fail closed.
No AI or machine learning component is part of this contract.

Every strategy specification must expose:

```text
universe filter
fundamental eligibility
ranking
entry
confirmation
regime filter
position sizing
exit
portfolio constraints
cost model
benchmark
```

Five priority families have executable deterministic templates. Remaining
taxonomy families are an explicit research backlog and receive no financial or
execution authority merely because they are listed.

All 38 strategies already present in the research ledger have a strict
no-macro/macro pair identity in:

```text
output/research/macro_pairs/registry-pair-inventory.json
```

Only strategies with a complete causal data mapping may contribute financial
evidence. The Phase 11.6 regression shortlist supplied 12 evaluable
strategy/timeframe pairs. Missing mappings are reported, not substituted with
fixtures.
