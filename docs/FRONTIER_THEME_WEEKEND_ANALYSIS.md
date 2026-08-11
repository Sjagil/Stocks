# Frontier Theme Weekend Analysis

This research-only layer analyzes quantum computing and nuclear/uranium
instruments without changing strategy or execution authority.

Run both themes:

```powershell
.\.venv-ibkr\Scripts\python.exe .\scripts\analyze_frontier_themes.py
```

Refresh bounded official SEC fundamentals before the theme run:

```powershell
.\.venv-ibkr\Scripts\python.exe .\scripts\refresh_theme_fundamentals.py
```

Refresh theme-specific company news and official NRC/DOE feeds:

```powershell
.\.venv-ibkr\Scripts\python.exe .\scripts\refresh_theme_news.py
```

Resolve or refresh the configured theme identities through the dedicated,
strict read-only live TWS connection:

```powershell
.\.venv-ibkr\Scripts\python.exe .\scripts\refresh_theme_contracts.py
.\.venv-ibkr\Scripts\python.exe .\scripts\refresh_theme_shariah.py
```

This command permits only bounded `reqContractDetails` calls. It cannot request
quotes, reserve order IDs, or submit or cancel orders.

Run the complete bounded weekend cycle:

```powershell
.\.venv-ibkr\Scripts\python.exe .\scripts\run_frontier_weekend_research.py
```

Install the four-hour external scheduler once:

```powershell
.\scripts\install_frontier_weekend_task.ps1
```

The scheduled launcher starts every four hours in `--no-bars` mode. It refreshes
news, context, theme rankings, and evidence status while the canonical runtime
remains responsible for regular bar collection. Its internal calendar guard
makes zero provider calls on weekdays. A single-flight lock blocks overlapping
weekend runs. Run the complete command above when an explicit weekend bar
refresh is required. Use `--force` only for a deliberate read-only weekday
research run.

Run one theme:

```powershell
.\.venv-ibkr\Scripts\python.exe .\scripts\analyze_frontier_themes.py `
  --theme nuclear_uranium
```

Artifacts are written to:

```text
output/analysis/themes/frontier-technology-energy.json
output/analysis/themes/quantum-computing.json
output/analysis/themes/nuclear-uranium.json
output/analysis/themes/instrument-summary.parquet
output/analysis/themes/fundamental-coverage.json
output/analysis/themes/contract-coverage.json
output/analysis/themes/shariah-coverage.json
output/analysis/themes/theme-news.json
output/analysis/themes/event-risk-calendar.json
output/analysis/themes/opening-session-watchplan.json
output/analysis/themes/provisional-candidate-assessment.json
output/analysis/themes/weekend-run.json
```

The analysis uses only completed local bars. On weekends, the latest completed
Friday session remains the market cutoff. No weekend bars are synthesized.
News, SEC filing coverage, macro state, and official source links are context
only. They cannot create an entry. A technical theme score is not a current
strategy setup; the report separately publishes current frozen-strategy setup
status.

The event-risk calendar queries the current yfinance issuer calendar, probes
the licensed EODHD calendar capability, joins recent accepted SEC material
filings, and reuses the official macro schedule. Provider dates retain their
source and date-only precision. EODHD entitlement failures, missing dates and
provider conflicts are published rather than replaced. ETF and physical
vehicles are marked not applicable for corporate earnings instead of being
misclassified as missing company data.

The opening-session gate uses the following semantics:

```text
EVENT_RISK_IMMINENT   hard block for a normal non-event setup
EVENT_DATE_CONFLICT   hard block
EVENT_DATE_UNCERTAIN  hard block until verified
EVENT_RISK_NEAR       soft evidence/risk penalty
EVENT_RISK_POST_EVENT soft evidence/risk penalty
EVENT_CLEAR           no event penalty
```

Macro schedule proximity remains context and can only reduce hypothetical
research risk. Neither an earnings date, SEC filing nor macro event can create
an entry, grant strategy authority, or submit an order.

Current news is classified by event type and summarized with source
concentration, directional conflict and uncertainty shrinkage. A large number
of positive headlines from one aggregator cannot receive full catalyst weight.
The opening-session watchplan joins this context to existing forward setups and
theme leaders, then publishes every contract, freshness, Shariah, hierarchy and
theme-structure blocker. It creates no entry or order and remains authority
`NONE`.

The provisional-candidate assessment separates immutable hard gates from soft
evidence uncertainty. Missing current Shariah review, stale data, unresolved
contracts and failed setup gates remain blocking. Weak breadth, concentrated
news and incomplete statistical evidence reduce an informational risk
multiplier. The multiplier is never applied to execution: executable risk stays
zero, promotion is manual, and strategy and execution authority remain `NONE`.

The report keeps technical leadership separate from contextual conviction.
Contextual conviction combines available technical, fundamental, directly
relevant company-news, and macro context with explicit weights. Missing
components are not silently replaced with bullish values. Extreme fundamental
ratios are shrunk toward neutral. Both rankings remain research context and are
not buy or sell signals.

The report also joins the current hierarchical entry shortlist without changing
old episodes. It publishes contract coverage, hard-veto status, hierarchy
readiness, and missing tape/depth confirmation per observed setup. Historical
feature snapshots remain immutable after a contract is resolved later.

The `sector_structure` block preserves the economics of each theme. Nuclear
and uranium separately measure the physical proxy, uranium funds and the
fuel-cycle producer basket. Quantum separately measures speculative pure plays,
diversified platform enablers and quantum security. These confirmation states
are descriptive context only and cannot create a setup or order.

Current Shariah coverage reuses the repository's manually reviewed, expiring
attestation contract. Missing stock attestations and ETF/product-structure
reviews remain explicit and fail closed. The generated review queue prioritizes
assets with a current forward observation, but it never infers compliance or
changes strategy/execution authority.

Every instrument remains:

```text
standalone_entry_allowed = false
strategy_authority       = NONE
execution_authority      = NONE
```

Missing fundamentals, stale bars, incomplete macro inputs, missing realtime
tape/depth entitlements, and unknown current Shariah eligibility are reported
as explicit gaps.

SEC provider availability is not treated as decision-ready fundamental
coverage. The fundamental artifact separately reports core-metric completeness,
filing age, extreme denominator-sensitive ratios and decision usability.
Extreme growth or margin values remain visible in the raw context but are
excluded from the bounded quality score until reviewed. Sparse fundamentals are
shrunk more strongly toward neutral rather than receiving false precision.
