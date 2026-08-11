# Telegram Notification Status

Status: `ENABLED`

The Telegram Bot API healthcheck and one safe real delivery completed
successfully. Nine shadow watchlist messages were subsequently accepted within
the configured persistent rate limit. These messages are not order
instructions and cannot trigger execution.

```text
Telegram authority       OUTBOUND_NOTIFICATION_ONLY
Signal authority         SHADOW
Execution authority      NONE
Broker calls             0
Orders generated         0
```

Current operational state:

```text
sent                     10
system test               1
watchlists                9
pending                  41
retries                   0
final failures            0
queue in-flight           0
```

The remaining messages are persisted in the private SQLite queue and are
processed by later explicit or daily cycles subject to the configured
ten-messages-per-minute limit.

Privacy checks confirm that the Telegram token, full chat ID, and IBKR account
identifiers do not occur in public notification artifacts.

Verification:

```text
pytest                    874 passed
ruff main.py/src/tests    GO
compileall                GO
```

The repository-wide `ruff check .` also scans the virtual environment and
legacy/frozen top-level research scripts. It reports 86 pre-existing findings
in those excluded historical files; canonical application code and tests are
clean.
