# Stocks Parameter Research V2

## Scope

The parameter campaign is offline research. It uses the causal v2 global
portfolio ledger and retains dynamic cash and NAV, whole-share orders, actual
order notional and per-order fees, proportional costs, slippage, FX conversion,
global caps, security netting and the causal investability gate.

It grants no strategy or execution authority.

## Registry

The source contract is
config/research_contracts/stocks_parameter_space_registry_v2.json.

Each campaign writes a fully resolved copy containing all 32 runtime strategies
and rotational momentum. Unknown strategies, parameters, values and timeframes
fail closed. Integer periods are never searched at decimal precision. Explicit
continuous thresholds may use deterministic, pre-registered midpoint
refinements.

## Timeframes

- 1d uses native Phase 11.4 PIT bars.
- 1w is resampled from fully closed source weeks; signals execute on the next
  weekly open.
- 1h and 15m are blocked until genuine local historical bars exist.

Annualization is inferred from the observed index instead of assuming 252 for
every timeframe.

## Reference Scripts

rsi2_adx5_vwap.py and strategy_research_hub_v3_1_0_definitive.py are formula
references only. Their standalone data and capital engines are not imported.

strategy1.py is never imported because it contains broker and external data
client dependencies. This prevents the offline campaign from acquiring broker
authority.

## Search

Small valid spaces are exhaustive. Larger spaces use deterministic Sobol,
Latin Hypercube or seeded random sampling. The baseline is always first.
Normal and double-cost variants run through the same ledger. Configurations
need at least 30 validation trades before development selection, regardless of
a lower technical smoke-test threshold.

Weighted global-netted sleeves are executed by the existing v2 combination
engine. Confirmation voting and hierarchical combinations remain blocked until
their distinct causal candidate semantics are implemented and tested.

## Commands

Plan and seal:

    .\.venv-ibkr\Scripts\python.exe -B .\strategy_combo_research_lab.py parameter-research --parameter-search --parameter-search-method sobol --timeframes 1d,1w,1h --max-coarse-trials-per-strategy 20 --plan-only

Bounded micro-run:

    .\.venv-ibkr\Scripts\python.exe -B .\strategy_combo_research_lab.py parameter-research --parameter-search --parameter-search-method sobol --include-strategies ma_crossover,rsi_adx --timeframes 1d,1w --max-coarse-trials-per-strategy 4 --max-symbols 20 --bootstrap-runs 20

The 2019-2026 interval is a consumed historical confirmation set. A genuine
sealed holdout remains unavailable until future data arrives.
