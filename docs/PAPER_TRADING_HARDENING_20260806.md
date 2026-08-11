# Paper trading hardening - 2026-08-06

## Safety contract

This hardening release does not grant strategy, paper, or live execution
authority. The runtime remains fail closed:

```text
LIVE_TRADING_ALLOWED=false
execution_authority=NONE
standalone_entry_allowed=false
IBKR_READ_ONLY=true
```

The live-writer manifest is an integrity proof only. A valid writer hash does
not activate the writer and does not satisfy Phase 9 broker-evidence gates.

## Order state machine

Canonical progression:

```text
NONE
-> PREPARED
-> APPROVED
-> ORDER_ID_ALLOCATED
-> SUBMIT_SENT
-> WORKING
-> PARTIALLY_FILLED | FILLED | CANCEL_REQUESTED | REJECTED
-> CANCELLED | FILLED
-> CLOSED
```

Important constraints:

- A fill requires a canonical IBKR execution identity.
- A cancel request is not a cancellation. `BROKER_ORDER_CANCELLED` requires a
  complete read-only snapshot and broker-side absence with complete execution
  scope.
- A position is closed only after a closing SELL fill. No local close is
  inferred from an order request.
- SELL quantity cannot exceed the reconciled long quantity. Negative
  positions are blocked.
- Duplicate execution identities are idempotent; conflicting duplicates are
  blocked.
- Every event records timestamp, cause, source, and correlation ID.
- Unknown or forbidden transitions remain fail closed and require review.

The machine-readable transition table is published in:

```text
output/ibkr/phase9/schema.json
output/ibkr/phase9/order-state-machine-audit.json
```

## Capital lifecycle

Opening BUY intents reserve capital after pre-trade risk succeeds. Reservation
events are append only:

```text
RESERVED -> DEPLOYED -> RELEASED
```

- Broker reject releases an unfilled reservation.
- A cancel call does not release capital.
- Broker-confirmed cancel releases an unfilled reservation.
- Partial BUY fills convert the reservation to deployed capital.
- A full canonical closing SELL releases deployed capital.
- Replays are idempotent and conflicting reservation amounts are blocked.

Private state is stored in:

```text
data/execution/phase9/private/paper_execution.sqlite3
```

## Writer integrity

The v2 manifest hashes normalized source contents, blocks symlinks, uses POSIX
relative paths, and records category hashes for configuration, strategy,
authority, adapter, risk, routing, and reconciliation code. Re-freeze requires
an explicit operator, reason, and confirmation.

```powershell
python .\main.py live writer-integrity inspect
python .\main.py live writer-integrity verify
python .\main.py live writer-integrity diff
```

Do not re-freeze a mismatch merely to clear a blocker. Review the diff and run
the full tests first.

## Data and research gates

- IBKR data capabilities are machine-readable and never silently downgrade
  realtime requirements to delayed or bar proxies.
- SEC data is delayed context with `RANKING_OVERLAY_ONLY`, bounded to +/-4
  points, and cannot create an entry.
- ML accepts only canonical Phase 9 broker-fill labels. Bar simulations and
  open episodes are not trainable rows.
- Strategy promotion is explicit and never grants broker authority.

Artifacts:

```text
output/ibkr/data-capabilities/capability-matrix.json
output/research/sec_intelligence/status.json
output/research/active_swing/selective_ml/status.json
output/ibkr/live/freeze-status.json
output/ibkr/live/writer-integrity-history.jsonl
```

## Current proof boundary

Offline lifecycle and safety tests are green. Phase 9 remains operationally
blocked until current paper broker evidence proves reconciliation, a complete
submit/cancel canary, and a canonical closing SELL. No live activation follows
automatically from those paper proofs.
