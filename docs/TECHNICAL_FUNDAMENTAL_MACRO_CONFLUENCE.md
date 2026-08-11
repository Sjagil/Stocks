# Technical, Fundamental and Macro Confluence

## Purpose

The confluence layer combines existing evidence. It does not create a new
strategy and cannot create an entry without a current technical setup.

```text
current causal technical setup
-> point-in-time fundamental quality
-> asset-specific macro transmission
-> confluence ranking and allocation gate
-> existing portfolio risk controls
```

## Layer roles

- Technical: timing, structure, trend, momentum and timeframe agreement.
- Fundamental: business quality and durability for operating-company stocks.
- Macro: asset-specific tailwind or headwind using the configured sensitivity
  map. Macro context is never a standalone trigger.

ETFs and commodity proxies do not receive fabricated company fundamentals.
Their fundamental layer is not required; their asset-specific context remains
the responsibility of the existing ETF, commodity, COT and macro layers.

## Combination method

The score uses a weighted geometric mean:

```text
technical weight   50%
fundamental weight 30%
macro weight       20%
```

The geometric mean penalizes a weak layer more strongly than an arithmetic
average. Thresholds and weights are versioned in
`config/portfolio/active_manager_v1.json`.

## Fail-closed behavior

- A stock without point-in-time fundamentals cannot pass allocation.
- Missing macro context is not silently converted into positive evidence.
- A generic regime fallback is explicitly labelled and receives low
  confidence.
- A severe, high-confidence macro headwind reduces ranking and risk.
- An adverse technical setup or adverse stock fundamental layer blocks new
  allocation.
- Stale market references invalidate the technical setup before confluence.

## Artifacts

```text
output/portfolio/confluence-audit.json
output/portfolio/opportunity_ranking.json
output/portfolio/active_portfolio_plan.json
```

The audit stores the base score, adjusted score, score delta, layer statuses,
macro source and blockers. This provides a deterministic ablation trail.

## Authority

```text
standalone_entry_allowed = false
strategy_authority       = NONE
execution_authority      = NONE
automatic_execution      = false
```
