param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("BUY", "SELL")]
    [string]$Side,

    [Parameter(Mandatory = $true)]
    [decimal]$LimitPrice,

    [int]$ConId = 8677881,

    [ValidateRange(1, 1)]
    [int]$Quantity = 1,

    [string]$Reason = "ON manual paper fill close canary",

    [ValidateRange(15, 300)]
    [int]$ObserveSeconds = 90,

    [ValidateRange(1, 10)]
    [int]$PollSeconds = 2,

    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv-ibkr\Scripts\python.exe"
$Main = Join-Path $ProjectRoot "main.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python runtime not found: $Python"
}

function Invoke-MainJson {
    param([string[]]$Arguments)

    $Output = & $Python $Main @Arguments | Out-String
    if ($LASTEXITCODE -ne 0) {
        throw "main.py failed: $($Arguments -join ' ')`n$Output"
    }
    return $Output | ConvertFrom-Json
}

$Preflight = Invoke-MainJson @("ibkr", "phase9", "preflight")
if ($Preflight.status -ne "GO") {
    throw "Phase 9 preflight is not GO."
}

$Reconciliation = Invoke-MainJson @("ibkr", "phase9", "reconcile")
if ($Reconciliation.reconciliation_status -ne "PAPER_RECONCILED_EMPTY" -and
    $Reconciliation.reconciliation_status -ne "PAPER_RECONCILED") {
    throw "Phase 9 reconciliation is not safe for a manual canary."
}

if ($PreflightOnly) {
    [pscustomobject]@{
        status = "GO"
        mode = "PREFLIGHT_ONLY"
        phase9_preflight = $Preflight.status
        reconciliation_status = $Reconciliation.reconciliation_status
        broker_position_count = $Reconciliation.broker_position_count
        broker_open_order_count = $Reconciliation.broker_open_order_count
        execution_authority = "NONE"
        broker_write_attempts = 0
        intent_created = $false
    } | ConvertTo-Json -Depth 4
    return
}

$Prepare = Invoke-MainJson @(
    "ibkr", "phase9", "prepare",
    "--con-id", "$ConId",
    "--side", $Side,
    "--quantity", "$Quantity",
    "--limit-price", "$LimitPrice",
    "--reason", $Reason
)
if ($Prepare.prepare_status -ne "AWAITING_MANUAL_APPROVAL") {
    $Prepare | ConvertTo-Json -Depth 8
    throw "Intent preparation did not reach AWAITING_MANUAL_APPROVAL."
}

Write-Host ""
Write-Host "Exact approval challenge:"
Write-Host $Prepare.approval_challenge -ForegroundColor Yellow
Write-Host ""
$Approval = Read-Host "Type the exact approval challenge"
if ($Approval -cne $Prepare.approval_challenge) {
    throw "Exact approval challenge mismatch. Nothing was approved or submitted."
}

$Approved = Invoke-MainJson @(
    "ibkr", "phase9", "approve",
    "--intent-id", $Prepare.intent_id,
    "--approval", $Approval
)
if ($Approved.status -ne "GO") {
    $Approved | ConvertTo-Json -Depth 8
    throw "Intent approval failed."
}

$SubmitChallenge = "SUBMIT PAPER INTENT $($Prepare.intent_id)"
$SubmitApproval = Read-Host "Type '$SubmitChallenge' to submit"
if ($SubmitApproval -cne $SubmitChallenge) {
    throw "Submit confirmation mismatch. The approved intent was not submitted."
}

$Submitted = Invoke-MainJson @(
    "ibkr", "phase9", "submit",
    "--intent-id", $Prepare.intent_id
)
$ObservationStarted = Get-Date
$ObservationDeadline = $ObservationStarted.AddSeconds($ObserveSeconds)
$ObservationCount = 0
$FillReconciled = $false
$After = $null
while ((Get-Date) -lt $ObservationDeadline) {
    $After = Invoke-MainJson @("ibkr", "phase9", "reconcile")
    $ObservationCount += 1
    $PositionCount = [int]$After.broker_position_count
    $OpenOrderCount = [int]$After.broker_open_order_count
    if ($Side -eq "BUY") {
        $FillReconciled = (
            $PositionCount -ge 1 -and $OpenOrderCount -eq 0
        )
    } else {
        $FillReconciled = (
            $PositionCount -eq 0 -and $OpenOrderCount -eq 0
        )
    }
    if ($FillReconciled) {
        break
    }
    Start-Sleep -Seconds $PollSeconds
}

if ($null -eq $After) {
    $After = Invoke-MainJson @("ibkr", "phase9", "reconcile")
    $ObservationCount += 1
}

[pscustomobject]@{
    status = if ($FillReconciled) {
        "FILL_RECONCILED"
    } else {
        "SUBMITTED_OBSERVATION_TIMEOUT"
    }
    intent_id = $Prepare.intent_id
    side = $Side
    quantity = $Quantity
    limit_price = $LimitPrice
    submit_status = $Submitted.submit_status
    reconciliation_status = $After.reconciliation_status
    broker_position_count = $After.broker_position_count
    broker_open_order_count = $After.broker_open_order_count
    broker_execution_count = $After.broker_execution_count
    broker_commission_count = $After.broker_commission_count
    fill_reconciled = $FillReconciled
    observation_count = $ObservationCount
    observation_seconds = [math]::Round(
        ((Get-Date) - $ObservationStarted).TotalSeconds,
        3
    )
    automatic_cancellation = $false
    execution_authority = "MANUAL_PAPER_CANARY"
    automatic_submission = $false
    live_authority = "NONE"
} | ConvertTo-Json -Depth 6
