# Phase 8 IBKR Read-Only Reconciliation

Phase 8 connects the frozen Phase 1 read-only IBKR connection constraints with the frozen Phase 7 offline execution control plane.

Allowed IBKR methods:

```text
reqCurrentTime
reqAccountSummary
cancelAccountSummary
reqPositions
cancelPositions
reqOpenOrders
reqAllOpenOrders
reqExecutions
```

Forbidden methods include order placement, order cancellation, order-id reservation, auto-binding, market data streaming, and historical data collection outside the dedicated earlier phases.

Account IDs are converted immediately to:

```text
HMAC_SHA256(IBKR_ACCOUNT_FINGERPRINT_KEY, raw_account_id)
```

Public artifacts live under:

```text
output/ibkr/phase8/
```

Private broker snapshots live under:

```text
data/broker/phase8/private/broker_observation.sqlite3
```

Reconciliation is observation-only. Mismatches are classified and may recommend manual review or a reconciliation kill-switch state, but Phase 8 never mutates the Phase 7 ledger and never imports broker state.
