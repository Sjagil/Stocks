# Phase 11.9 Accelerated Multi-Timeframe Discovery

## Commands

```powershell
python .\main.py data multitimeframe collect `
  --symbols "SPY,QQQ,IWM,EFA,EEM,TLT,GLD,SLV,DBC,AAPL,MSFT,NVDA,AMZN,GOOGL,META,JPM,XOM,ON,ASML,INTC" `
  --intervals "1h,4h" `
  --providers yfinance `
  --lookback-days 729

python .\main.py research phase11-9 schema
python .\main.py research phase11-9 run
python .\main.py research phase11-9 diagnose
python .\main.py research phase11-9 watchlist
python .\main.py research phase11-9 status
```

## Registered Families

The base registry contains MA crossover, asymmetric MA, Donchian breakout,
Bollinger breakout, volatility-contraction breakout, RSI(2)+ADX pullback,
RSI(14) trend pullback, MACD trend, Keltner breakout, volume breakout, ROC
trend and EMA pullback.

Fixed ensembles combine trend, breakout, trend-pullback, momentum and
diversified votes. Ensemble thresholds are specified before evaluation.

## Evaluation Contract

- Signals use only closed bars.
- Entries execute at the next bar open.
- Simultaneous candidates rank by score and then security ID.
- The portfolio uses whole shares and no leverage.
- A security occupies at most one global position.
- USD prices use lagged daily EURUSD conversion.
- Costs are stressed from 5 through 50 bps per side.
- Parameters are selected inside validation folds.
- OOS portfolio PF is primary; pooled theoretical trade PF is not used.
- Four-hour results are labeled low confidence.
- One-hour candidates must survive 50 bps before shortlisting.

The outputs are written to:

```text
output/research/phase11_9/
```

No command grants strategy, paper or live authority.
