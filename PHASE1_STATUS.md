# Phase 1 Status

Status:

```text
IBKR_PHASE1_CONNECTION_SERVICE_FROZEN_GO
```

Canonical freeze report:

```text
PHASE1_FREEZE_REPORT.md
```

Canonical disconnect drill evidence:

```text
output/ibkr/phase1-disconnect-drill-20260720-133116.json
```

## Proven Values

```text
initial_health                 HEALTHY
disconnect_detected            true
disconnected_or_stale_seen     true
bounded_reconnect_attempted    true
reconnect_attempts             4
reconnect_delays               [2.0, 5.0, 15.0, 30.0]
reconnect_recovered            true
final_health                   HEALTHY
thread_leak                    false
errors                         []
financial_calls                0
```

## Frozen Contract

```text
canonical_entrypoint           main.py
host                           127.0.0.1
paper_port                     7497
order_authority                NONE
read_only                      true
live_trading_enabled           false
allow_order_transmission       false
ibapi_version                  10.48.1
server_version                 225
```

The frozen Phase 1 gate validates `PHASE1_FREEZE_REPORT.md`, the drill artifact SHA-256, ordered disconnect/reconnect evidence, financial counters and hashes for the canonical Phase 0/1 files.

## Frozen Files

```text
main.py
requirements.lock.txt
src/stocks/application/config.py
src/stocks/application/context.py
src/stocks/application/phase_gates.py
src/stocks/application/lifecycle.py
src/stocks/ibkr/client.py
src/stocks/ibkr/callbacks.py
src/stocks/ibkr/connection.py
src/stocks/ibkr/errors.py
src/stocks/ibkr/health.py
```

## Current Gate

```text
phase1_freeze_status           PHASE1_FROZEN
phase2_contract_runtime        enabled, read-only only
brokerwrites                   0
market_data_calls              0
historical_data_calls          0
```

Phase 2 may import the frozen Phase 1 connection service. Socket, callback, heartbeat and reconnect behavior must not be copied into a second implementation.
