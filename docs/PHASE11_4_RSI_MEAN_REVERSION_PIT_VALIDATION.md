# Phase 11.4 RSI Mean Reversion PIT Validation

Phase 11.4 is an offline falsification pipeline for `RSI_MEAN_REVERSION_CAUSAL_V1`.
It never grants strategy or execution authority.

## Frozen rule

- Compute Wilder RSI(3) using information available at close `t`.
- Signal when RSI(3) is below 5.
- Enter at the next valid session open.
- Signal an exit on the first close above the previous close.
- Exit at the following valid session open.
- Charge 10 bps per side in the base run.
- Hold at most one position per symbol.

## Data gates

Signals are financially valid only when historical membership, listing and delisting dates,
split-consistent OHLC, and next-open execution prices are proven point in time. Current symbol
membership is never back-projected. Incomplete sources may produce provisional diagnostics, but
`valid_for_candidate_gate` remains false and the decision fails closed.

The Shariah cohort accepts only unexpired causal screens with complete business activity,
financial ratios, and non-permissible-income evidence. Current compliance is never back-projected.

## Commands

Run the commands under `python main.py research rsi-pit`, in the order shown by `main.py -h`.
The final `status` command records the financial decision; `freeze` hashes the evidence without
changing `strategy_authority=NONE` or `execution_authority=NONE`.

Private symbol-level evidence is append-only in
`data/research/phase11_4/private/rsi_mean_reversion_pit.sqlite3`. Aggregated artifacts are written
to `output/research/rsi_pit`.
