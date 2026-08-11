# Signals, Research Autopilot, and Live Canary

`main.py` remains the only operational entrypoint.

## Authority separation

Manual signals never submit broker orders. A recovered strategy must first be
classified as `FROZEN_SHADOW`, then explicitly promoted with:

```powershell
python .\main.py strategies promote-manual-signals `
  --strategy-id <ID> `
  --approval "<EXACTE_PHRASE>"
```

The phrase is supplied through `SIGNAL_MANUAL_APPROVAL_PHRASE` and is never
written to public artifacts.

## Daily operation

```powershell
python .\main.py daily
python .\main.py daily --signals-only
python .\main.py daily --research-only
python .\main.py daily --no-autopilot
```

The research scheduler is bounded and externally triggered. `autopilot start`
enables scheduling but does not create an unbounded hidden daemon.

Each due cycle advances two independent research tracks under one shared daily
trial budget:

- the legacy family campaign, conservatively reserving at most nine trial rows
  per selected strategy;
- the append-only Phase 11.12 two-block queue, defaulting to 25 previously
  unevaluated DNA records with both 10 bps and 50 bps cost stress.

The queue is resumable by `strategy_id + cost_bps`, so completed DNA records
are not selected again. Generation-budget exhaustion does not stop pending
queue work, while the total daily trial, runtime, disk, failure, and
single-instance limits remain fail-closed. Optional process-level limits:

```text
AUTOPILOT_REAL_BACKTESTS_PER_CYCLE=5
AUTOPILOT_PHASE11_12_DNA_PER_CYCLE=25
AUTOPILOT_MAX_TOTAL_BACKTESTS_PER_DAY=1000
AUTOPILOT_MAX_RUNTIME_MINUTES_PER_CYCLE=120
AUTOPILOT_MIN_CYCLE_INTERVAL_HOURS=6
AUTOPILOT_MAX_DISK_GB=25
```

Inspect the persisted counters and latest queue advance with:

```powershell
python .\main.py autopilot status
```

This research path always publishes `execution_authority=NONE`, zero broker
calls, and zero generated orders. A positive historical result never receives
automatic strategy, paper, or live authority.

## Live boundary

Use `.env.ibkr.live`, based on `.env.ibkr.live.example`. Live preflight sends no
order:

```powershell
python .\main.py live preflight
```

The current live writer is intentionally not frozen. Phase 9 fill/close,
live reconciliation, an eligible strategy, and a frozen writer remain mandatory.
No live order can be sent while any of those gates is open.
