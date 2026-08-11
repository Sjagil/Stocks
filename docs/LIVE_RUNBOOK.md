# Controlled Live Runbook

## Scope

The live path is a fail-closed Level-1 deployment for the
`autonomous_multi_asset_v1` profile. It supports one explicit initial
operator activation followed by policy-controlled automatic orders.

Level-1 remains:

```text
max order value              EUR 10
max total live exposure      EUR 25
max open positions           1
max new orders per day       1
max daily live loss          EUR 5
whole shares                 required
margin                       disabled
leverage                     disabled
shorting                     disabled
options                      disabled
futures                      disabled
automatic capital promotion  disabled
```

Logging into Live TWS does not grant execution authority. A capability,
authority state and running service are separate controls.

## Before Live Login

Keep TWS in Paper Trading until all paper evidence is complete:

```powershell
.\.venv-ibkr\Scripts\python.exe .\main.py ibkr phase9 status
.\.venv-ibkr\Scripts\python.exe .\main.py ibkr phase9 reconcile
.\.venv-ibkr\Scripts\python.exe .\main.py execution paper-session-audit
```

Required evidence includes a filled and closed paper canary, commission
reconciliation, duplicate prevention, restart recovery and one complete
paper session.

## Read-Only Live Validation

After the paper gates pass, log into the intended Live TWS account. Confirm
that the configured live socket is enabled and that the dedicated
`.env.ibkr.live` file points to the live environment. Never print this file.

Run read-only reconciliation and preflight:

```powershell
.\.venv-ibkr\Scripts\python.exe .\main.py live reconcile
.\.venv-ibkr\Scripts\python.exe .\main.py live preflight
.\.venv-ibkr\Scripts\python.exe .\main.py live status
```

The live account must match the configured fingerprint. The first Level-1
baseline must be empty, buying power must be proven, the allowlist must be
non-empty, and both writer freezes must match current source hashes.

## Capability And Activation

Create one capability:

```powershell
.\.venv-ibkr\Scripts\python.exe .\main.py live capability-create `
  --profile autonomous_multi_asset_v1 `
  --yes

.\.venv-ibkr\Scripts\python.exe .\main.py live capability-status
```

The capability:

- expires after 15 minutes;
- is single-use;
- is bound to the profile, live account fingerprint, reconciliation state,
  frozen strategy allowlist and Level-1 configuration;
- grants no execution authority by itself.

Consume it using the exact private phrase configured in `.env.ibkr.live`:

```powershell
.\.venv-ibkr\Scripts\python.exe .\main.py live activate `
  --profile autonomous_multi_asset_v1 `
  --approval "<EXACT_PRIVATE_PHRASE>" `
  --yes
```

Do not place the phrase in documentation, shell history shared with others,
logs or public artifacts.

## Composite Launch

The one-command path repeats reconciliation and preflight, creates and
consumes a fresh capability, installs the canonical task in
`CONTROLLED_LIVE` mode and starts it:

```powershell
.\.venv-ibkr\Scripts\python.exe .\main.py live launch `
  --profile autonomous_multi_asset_v1 `
  --continuous `
  --resume `
  --approval "<EXACT_PRIVATE_PHRASE>" `
  --yes
```

Expected terminal state:

```text
launch_status        LIVE_ACTIVE
execution_authority  LIVE_LEVEL_ONE
runtime_started      true
real_live_order      false at activation
```

If scheduler installation or startup fails after activation, the launcher
pauses Level-1 authority and restores `SIGNALS_ONLY`.

## Monitoring

```powershell
.\.venv-ibkr\Scripts\python.exe .\main.py live status
.\.venv-ibkr\Scripts\python.exe .\main.py live automatic-cycle-status
.\scripts\status_bot.ps1
```

An automatic order is still blocked unless the frozen strategy, contract,
fresh closed-bar signal, session, notional, portfolio and broker
reconciliation checks all pass.

## Stop And Incident Controls

Pause new automatic live actions:

```powershell
.\.venv-ibkr\Scripts\python.exe .\main.py live pause `
  --reason "operator maintenance"
```

Stop the canonical runtime:

```powershell
.\scripts\stop_bot.ps1
```

The canonical stop path first pauses active Level-1 authority, verifies
`execution_authority = NONE`, and only then reports `STOPPED`. It returns
`STOP_INCOMPLETE` if authority cannot be cleared.

Activate the persistent kill switch:

```powershell
.\.venv-ibkr\Scripts\python.exe .\main.py live kill-switch activate `
  --reason "describe the incident"
```

The kill switch and automatic demotion may reduce authority immediately.
Capital promotion always requires a separate operator decision.

## Current External Gates

Software readiness does not prove that the external live environment is
ready. Typical blockers are:

```text
PHASE9_FILL_CLOSE_CANARY_REQUIRED
ONE_COMPLETE_PAPER_SESSION_REQUIRED
LIVE_TWS_SOCKET_UNREACHABLE
LIVE_ACCOUNT_FINGERPRINT_MISMATCH
LIVE_RECONCILIATION_EMPTY_NOT_PROVEN
LIVE_BUYING_POWER_NOT_PROVEN
LIVE_EXECUTION_WRITER_NOT_FROZEN
EXACT_OPERATOR_APPROVAL_REQUIRED
```

Do not remove or bypass these blockers. Resolve the underlying evidence and
rerun reconciliation, preflight, audits and freezes.
