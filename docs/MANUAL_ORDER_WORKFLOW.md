# Manual Signal And Position Workflow

## Boundary

A model signal is not an order. A manually executed signal remains owned by
the operator until it is explicitly claimed. Registration, claim and unclaim
perform no broker writes and grant no execution authority.

Ownership classifications:

```text
MANUAL_TRACKED
BOT_MANAGED
```

Broker match classifications:

```text
UNVERIFIED
MATCHED
QUANTITY_MISMATCH
BROKER_POSITION_NOT_FOUND
AMBIGUOUS_BROKER_POSITION_BLOCKED
BROKER_SNAPSHOT_UNAVAILABLE
BROKER_SNAPSHOT_STALE
BROKER_POSITION_SNAPSHOT_INCOMPLETE
```

`BOT_MANAGED` does not by itself permit automatic orders. Until a separate
broker identity and quantity match is proven, the effective status is:

```text
BOT_MANAGED_PENDING_BROKER_MATCH
automatic_execution_eligible = false
```

## Register A Manual Fill

After executing an existing signal manually:

```powershell
.\.venv-ibkr\Scripts\python.exe .\main.py positions register-manual `
  --signal-id <SIGNAL_ID> `
  --quantity <QUANTITY> `
  --fill-price <ACTUAL_FILL_PRICE> `
  --environment paper
```

This writes the quantity, fill price, stop, targets and slippage only to:

```text
data/signals/private/signals.sqlite3
```

Public artifacts contain identities, classifications and counts, but no
quantity, fill price or PnL.

Repeating the exact registration is idempotent. A conflicting quantity or
fill price is blocked.

## Match The Broker Position

First produce a current complete read-only broker snapshot through the
existing reconciliation path. Then match the private registration to exactly
one broker position:

```powershell
.\.venv-ibkr\Scripts\python.exe .\main.py positions broker-match `
  --position-id <POSITION_ID> `
  --environment paper
```

For a future live registration use `--environment live`. The live command
reads only the dedicated live observation store and never falls back to a
Paper snapshot.

A match requires:

```text
same environment
resolved STK conId
complete snapshot
snapshot no older than five minutes
exactly one matching conId
exact position quantity
```

The private event log may retain contract and quantity evidence. The public
artifact contains only status, counts, snapshot age and hashes. Matching
never imports the broker position, changes the broker account, or grants
execution authority.

## Claim

Explicitly classify the tracked position as bot-managed:

```powershell
.\.venv-ibkr\Scripts\python.exe .\main.py positions claim `
  --position-id <POSITION_ID> `
  --mode bot-managed `
  --yes
```

This is an ownership classification, not broker adoption. Automatic exits or
adjustments remain blocked until the position is matched to the intended
broker account, contract identity and quantity and normal live authority is
active.

After a successful match, a claimed position reports:

```text
BOT_MANAGED_BROKER_MATCHED_AUTHORITY_REQUIRED
automatic_execution_eligible = false
execution_authority = NONE
```

## Unclaim

Return management to the operator:

```powershell
.\.venv-ibkr\Scripts\python.exe .\main.py positions unclaim `
  --position-id <POSITION_ID> `
  --yes
```

Unclaim never sends a closing order.

## Status

```powershell
.\.venv-ibkr\Scripts\python.exe .\main.py positions status
```

The public status reports:

- manual position count;
- ownership counts;
- unmatched position count;
- automatic-execution-eligible count;
- paper and live reconciliation status.

Financial values remain private.
