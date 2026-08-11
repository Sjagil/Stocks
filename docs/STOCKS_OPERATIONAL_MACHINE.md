# Stocks Operational Machine

The operational supervisor is available through the canonical `main.py`
entrypoint. It composes the existing daily data and signal workflow, dynamic
portfolio engine, macro engine, Phase 9 reconciliation, research autopilot and
Telegram delivery. It does not duplicate their stores or broker clients.

## Commands

```powershell
python .\main.py execution status
python .\main.py execution preflight --environment paper
python .\main.py execution paper-fill-close-canary
python .\main.py execution activate-paper `
  --approval "ACTIVATE BOUNDED AUTOMATIC PAPER"
python .\main.py execution deactivate-paper

python .\main.py autopilot run `
  --mode SIGNALS_ONLY `
  --max-cycles 1 `
  --interval-seconds 300
```

`autopilot run` is bounded. A filesystem lock prevents concurrent cycles.
Private state, heartbeats and append-only cycle records are stored under
`data/operations/private`. Public status is written under `output/operations`.
The current step heartbeat is `output/operations/cycle-progress.json`.

Component scheduling is bounded independently:

```text
daily workflow       at most once per 20 hours
dynamic signals      at most once per hour
market data refresh  at most once per 6 hours
macro refresh        at most once per 6 hours
reconciliation       every cycle
Telegram retry       every cycle
research             only when its frozen runtime says it is due
```

Only a complete successful component advances its due timestamp. Timeouts,
errors, blocked results and unknown statuses become explicit cycle blockers.
On Windows, timeout cleanup terminates the complete subprocess tree so a
stale analysis cannot overlap the next cycle.

The Windows Task Scheduler helper defaults to signal-only operation:

```powershell
.\scripts\install_stocks_autopilot_task.ps1 `
  -Mode SIGNALS_ONLY `
  -RepetitionMinutes 10
```

The task is registered at normal scheduler priority (`4`). This prevents the
bounded supervisor from being starved by persistent background research
processes while retaining `IgnoreNew` and the two-hour hard execution limit.
Each scheduled invocation writes process output directly to:

```text
output/operations/scheduler-last.stdout.json
output/operations/scheduler-last.stderr.log
```

The task result remains the actual Python exit code.

## Authority

Paper activation requires a completed real Phase 9 fill-and-close canary,
empty or valid reconciliation, intact Phase 1 freeze evidence and the exact
operator activation phrase. Live activation additionally remains blocked by
the live preflight and frozen-writer requirements.

Research discovery can promote only to frozen observation. It cannot activate
paper or live authority.

The current research registry contains 62 backtest-positive
strategy/timeframe pairs across 28 named strategies. Ten pairs satisfy the
strict shortlist gate, but duplicate-DNA auditing reduces these to eight
economically distinct outcomes. These are research results, not permission to
trade; `FINANCIAL_FINALIST_GO` remains false.
