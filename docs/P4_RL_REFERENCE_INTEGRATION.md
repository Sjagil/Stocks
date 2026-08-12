# P4 and RL reference-repository integration

This matrix records the capabilities inspected in `reference_repos`. The
product implementation is native to this repository. No reference repository
is added to the runtime import path, no code is blindly copied, and no reference
component receives broker or money authority.

| Repository | Applied capability | Product use | Boundary |
|---|---|---|---|
| `finrl_x` | Portfolio DRL workflow and feature normalization patterns | Informed the bounded single-policy experiment and explicit feature scaler | Not imported; its broad portfolio agent is not used as execution logic |
| `trademaster` | Environment/trainer separation and cost-aware portfolio accounting | Informed separate Gymnasium environment, trainer and evaluator contracts | Legacy Gym environment not copied; product uses current Gymnasium API |
| `qlib` | Composable reward components and finite-environment lifecycle | Applied as decomposed reward logging and deterministic terminal handling | Qlib/Tianshou runtime not imported |
| `vectorbt` | Vectorized research and portfolio analysis patterns | Existing repo analytics remain the fast deterministic comparison layer | Not used for broker execution |
| `quantstats` | Standard return, drawdown and tail-risk reporting | Metrics include CAGR, Sharpe, Sortino, Calmar, drawdown and CVaR | Metrics are implemented locally against the same return series |
| `pybroker` | Bootstrap evaluation and chronological strategy testing | Applied bootstrap confidence and purged walk-forward evaluation | Commons-Clause dependency is not vendored or imported |
| `lean` | Portfolio/risk models outside alpha generation | Reinforces hard drawdown, exposure and risk gates outside PPO | LEAN execution and broker adapters are not used |
| `nautilus_trader` | Event identities and restart-safe state machines | Applied immutable decision/outcome identities and recovery events | Runtime engine is not imported |
| `lumibot` | Separation of backtest, paper and live modes | Applied explicit `SHADOW_ONLY` default and separate future paper stage | Lumibot broker layer is not used |
| `rd_agent` | Research/challenger lifecycle | Applied ACTIVE/CHALLENGER/REJECTED/ARCHIVED registry states | No autonomous code mutation or policy promotion |
| `fingpt` | Financial text context | Existing NLP output may be an observation feature with explicit missingness | LLM output cannot create entries or authority |
| `finrobot` | Multi-source research/risk context | Existing fundamental and risk outputs remain upstream context | Agent framework is not used as trader |
| `ib_async` | Async IBKR abstraction | Explicitly excluded | Product remains on the official native local `ibapi` |
| `lean_ibkr` | Alternative IBKR integration | Explicitly excluded | It cannot replace the canonical native IBKR writer |

The central question is not whether PPO can make a positive backtest. It is
whether the same-period PPO challenger adds statistically credible net value
over the existing deterministic engine after costs, regimes and forward shadow
evidence. A rejection is a valid result.
