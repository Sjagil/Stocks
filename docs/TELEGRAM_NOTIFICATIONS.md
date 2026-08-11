# Telegram Notifications

The Telegram integration is an outbound-only adapter after the existing signal
and risk layers. It never changes signal, paper, live, or execution authority.

## Commands

```powershell
.\.venv-ibkr\Scripts\python.exe .\main.py telegram health
.\.venv-ibkr\Scripts\python.exe .\main.py telegram test
.\.venv-ibkr\Scripts\python.exe .\main.py telegram status
.\.venv-ibkr\Scripts\python.exe .\main.py telegram preview
.\.venv-ibkr\Scripts\python.exe .\main.py telegram send-latest-signals
.\.venv-ibkr\Scripts\python.exe .\main.py telegram retry-failed
```

`telegram test` performs no IBKR call and submits no order. `preview` writes
local text and HTML only. `send-latest-signals` consumes the existing public
signal artifact and never invokes the signal generator or broker.

The daily workflow can include Telegram or explicitly skip it:

```powershell
.\.venv-ibkr\Scripts\python.exe .\main.py daily
.\.venv-ibkr\Scripts\python.exe .\main.py daily --signals-only
.\.venv-ibkr\Scripts\python.exe .\main.py daily --research-only
.\.venv-ibkr\Scripts\python.exe .\main.py daily --no-telegram
```

Telegram failures are non-blocking. They cannot stop signal generation,
research, reconciliation, or alter order state.

## Privacy And Storage

Credentials are loaded from `.env` and never included in public output. The
chat is represented by a one-way SHA-256 prefix. Private queue state is stored
in:

```text
data/notifications/private/telegram.sqlite3
```

Public sanitized artifacts are stored in:

```text
output/notifications/
```

The queue is persistent, append-auditable, capped at 10,000 notifications,
rate-limited, retry-bounded, and duplicate-suppressed. A process crash with a
request in flight quarantines that notification because its delivery outcome
is unknown; it is not automatically resent.

## Authority

```text
Telegram authority       OUTBOUND_NOTIFICATION_ONLY
Execution authority      NONE
Automatic order creation false
Inbound commands         unsupported
```

Only `MANUAL_ACTIONABLE` BUY signals pass the actionable filter. Shadow output
can be sent as `WATCHLIST`, which always states that no order should be placed.
Futures BUY messages are blocked unless the exact contract identity, expiry,
multiplier, tick size, and tick value are all present.
