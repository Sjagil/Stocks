# Dynamic Multi-Strategy Engine

The canonical entrypoint is `main.py dynamic`. The engine orchestrates the
existing survivor recovery, signal, portfolio, Telegram, forward-observation,
Phase 9 paper and live-canary readiness layers.

It does not optimize parameters at runtime. Strategy parameters are frozen.
Only regime compatibility, bounded scores, weights and portfolio exposure
change according to the versioned causal rules in
`src/stocks/dynamic/service.py`.

## Safety

- Signal and execution authority are separate.
- Default execution authority is `NONE`.
- Total proposed exposure is capped at 25%.
- A strategy is capped at 25%; a family is capped at 50%.
- The proposal uses whole shares and EUR notionals.
- Assets without a unique local IBKR contract identity are not proposed.
- When the screener config enables the PIT Shariah register, missing or expired
  attestations fail closed before Telegram and portfolio selection.
- No live scaling is automatic.
- Every selected strategy has an append-only forward registration and
  observation stream in the existing research-autopilot SQLite database.
  Forward observations are not used for parameter retuning.

## Commands

```powershell
python .\main.py dynamic status
python .\main.py dynamic regime
python .\main.py dynamic strategies
python .\main.py dynamic signals
python .\main.py dynamic portfolio
python .\main.py dynamic explain --symbol SPY
python .\main.py dynamic daily
python .\main.py dynamic paper-campaign
```

`dynamic daily` refreshes the canonical signal scan, nets strategy votes per
asset, writes `output/dynamic`, and invokes the existing deduplicating Telegram
delivery. It never submits an order.
