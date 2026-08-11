# Phase 11.11 HMM Regime Research

## Scope

Phase 11.11 tests a Hidden Markov Model as a causal context and risk
overlay for every Phase 11.9 strategy/timeframe pair. It does not create
entry signals and has no broker or execution authority.

The evaluated matrix is:

```text
30 strategies x 5 timeframes = 150 paired hypotheses
timeframes: 1h, 4h, 1d, 1w, 1mo
variants: baseline and HMM risk overlay
costs: 10 bps and 50 bps
```

## Model

The observation equation is a Markov-switching regression:

```text
y_t = alpha[S_t] + beta' x_t + epsilon_t
epsilon_t ~ Normal(0, sigma[S_t]^2)
```

The hidden state has a fixed transition matrix after each train-only fit:

```text
P[i,j] = Pr(S_t = j | S_(t-1) = i)
```

Out-of-sample probabilities use only the recursive Hamilton filter:

```text
predicted_t = filtered_(t-1) x P
filtered_t  = normalize(predicted_t x emission_likelihood_t)
```

Kim smoothing and any other full-sample state inference are forbidden.

## Canonical states

Raw state numbers are not stable between fits. Every fitted model therefore
maps states to deterministic economic labels:

```text
highest conditional variance                  STRESS_HIGH_VOL
highest conditional return among remaining    RISK_ON_TREND
remaining state                               NEUTRAL_CHOPPY
```

The current configuration uses three states. A fourth
`INFLATION_RATE_SHOCK` state is supported only when an explicit inflation
signature is available and separately validated.

## Features

Daily, weekly and monthly models use cross-asset price features from SPY,
TLT and DBC plus optional point-in-time macro observations:

```text
world return
20/60 period realized volatility
downside volatility
equity/bond correlation
bond return
commodity return
USD return
credit-spread change
yield-curve velocity
financial-conditions change
VIX log change
```

The 1H and 4H models use closed-bar short-term market-state features:

```text
return
realized volatility
range expansion
downside range
volume surprise
overnight gap
```

Macro observations are joined on `available_at`, carried forward only after
release, and never linearly interpolated or backfilled. Optional macro
features are selected separately inside each training fold only when both
train and OOS availability meet the preregistered coverage rule. Missing
early macro history therefore cannot remove otherwise valid market history.

## Risk overlay

The filtered state probabilities produce:

```text
m_t = sum_s Pr(S_t=s | information_t) x state_multiplier[s]
```

The overlay is applied with one closed-bar lag:

```text
target exposure_t = baseline target_t x m_(t-1)
trade risk_t       = baseline trade risk_t x m_(t-1)
```

`m` is bounded to `[0, 1]`. The HMM can trim exposure or block a new entry;
it cannot increase exposure, activate a strategy, or generate an order.

## Stability gates

Each train-fold model is checked for:

```text
optimizer convergence
minimum train-state occupancy
minimum expected state duration
maximum one-bar chatter
probabilities summing to one
```

Promotion of an HMM overlay additionally requires:

```text
at least 10 folds
median HMM PF > 1
positive HMM CAGR in at least 60% of folds
50 bps stressed PF > 1
Sharpe at least 15% above baseline
drawdown at least 20% better
PF better in at least 60% of paired folds
stable HMM in at least 60% of folds
```

## Runtime integration

`regimes current` advances only the saved filtered state. The machine cycle
runs it before the dynamic portfolio preview. The public state contains only
probabilities, a bounded multiplier, timestamps and hashes.

The dynamic portfolio engine combines it multiplicatively with the existing
deterministic regime and drawdown throttles. Missing or invalid HMM state is
treated as an optional unavailable overlay with multiplier `1.0`; it never
raises the pre-existing risk budget.

Authority remains:

```text
strategy authority   NONE
execution authority  NONE
paper authority      NONE
live authority       NONE
broker calls         0
```

## Commands

```powershell
.\.venv-ibkr\Scripts\python.exe .\main.py regimes schema
.\.venv-ibkr\Scripts\python.exe .\main.py regimes fit
.\.venv-ibkr\Scripts\python.exe .\main.py regimes walk-forward
.\.venv-ibkr\Scripts\python.exe .\main.py regimes current
.\.venv-ibkr\Scripts\python.exe .\main.py regimes audit
.\.venv-ibkr\Scripts\python.exe .\main.py regimes status
```

The long walk-forward command is checkpointed per timeframe.

