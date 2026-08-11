# Canonical Stocks Runtime

The only continuous application entrypoint is:

```powershell
.\.venv-ibkr\Scripts\python.exe .\main.py run
```

The default mode is `SIGNALS_ONLY`. It keeps research, data, reconciliation,
signals, reporting, and Telegram active while execution authority remains
`NONE`.

## Windows task

Install the current-user task:

```powershell
.\scripts\install_windows_service.ps1 -Mode SIGNALS_ONLY
```

Control it with:

```powershell
.\scripts\start_bot.ps1
.\scripts\status_bot.ps1
.\scripts\restart_bot.ps1
.\scripts\stop_bot.ps1
```

The task uses `main.py run`, normal process priority, a single-instance file
lock, bounded cycles, a 26-hour hard limit, and at most three task restarts.

Runtime health is written atomically to:

```text
runtime/heartbeat.json
```

The daily read-only cycle also publishes the current Phase 11.10 causal
multi-timeframe watchlist. This observer always has execution authority
`NONE` and cannot route candidates to Phase 9.

Logs are written to:

```text
output/operations/service-last.stdout.json
output/operations/service-last.stderr.log
```

## Authority

`launch preflight` reports all live blockers:

```powershell
.\.venv-ibkr\Scripts\python.exe .\main.py launch preflight
```

`launch live` is deliberately fail-closed while the live execution writer,
paper fill/close canary, dedicated live environment, reconciliation, and
financial finalist gates are not proven. Runtime installation never grants
paper or live authority.

## Manual Phase 9 Canary

The interactive wrapper keeps prepare, approval, submit, and reconciliation
separate while routing every broker action through `main.py`:

```powershell
.\scripts\run_phase9_manual_canary.ps1 `
  -Side BUY `
  -LimitPrice <CURRENT_MARKETABLE_PAPER_LIMIT>
```

It requires the generated Phase 9 approval challenge verbatim and then a
second exact submit phrase. It cannot grant strategy or live authority. A
closing SELL is a separate invocation after the BUY fill has been reconciled.
After submission it performs bounded read-only reconciliation polling for up
to 90 seconds. A timeout never triggers an automatic cancellation.

Verify the complete wrapper route without creating an intent or making a
broker write:

```powershell
.\scripts\run_phase9_manual_canary.ps1 `
  -Side BUY `
  -LimitPrice 90 `
  -PreflightOnly
```
