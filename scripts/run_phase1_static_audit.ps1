param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$ErrorActionPreference = "Stop"

Set-Location $ProjectRoot

$python = Join-Path $ProjectRoot ".venv-ibkr\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Python venv niet gevonden: $python"
}

$outputDir = Join-Path $ProjectRoot "output\ibkr"
New-Item -ItemType Directory -Force $outputDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$artifact = Join-Path $outputDir "phase1-static-audit-$timestamp.json"

function Invoke-AuditCommand {
    param(
        [string]$Name,
        [scriptblock]$Command
    )

    $global:LASTEXITCODE = 0
    $output = & $Command 2>&1
    $exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
    return [ordered]@{
        name = $Name
        exit_code = $exitCode
        output = (($output | Out-String).Trim())
        passed = ($exitCode -eq 0)
    }
}

function New-AuditResult {
    param(
        [string]$Name,
        [int]$ExitCode,
        [string]$Output
    )

    return [ordered]@{
        name = $Name
        exit_code = $ExitCode
        output = $Output
        passed = ($ExitCode -eq 0)
    }
}

$checks = @()

$checks += Invoke-AuditCommand "doctor" {
    & $python ".\main.py" "doctor"
}
$checks += Invoke-AuditCommand "disconnect_drill_preflight_config" {
    & $python ".\main.py" "ibkr" "disconnect-drill-preflight" "--skip-socket-check"
}
$checks += Invoke-AuditCommand "env_example_config" {
    & $python "-c" "import sys; sys.path.insert(0, 'src'); from stocks.application.config import load_ibkr_settings; [load_ibkr_settings(path) for path in ('.env.ibkr.example', 'env.ibkr.example')]; print('IBKR env examples parse: GO')"
}
$checks += Invoke-AuditCommand "phase1_freeze_gate" {
    if (-not (Test-Path ".\PHASE1_FREEZE_REPORT.md")) {
        Write-Output "PHASE1_FREEZE_REPORT.md is missing; external gate remains open"
        $global:LASTEXITCODE = 0
        return
    }

    $statusJson = & $python ".\main.py" "ibkr" "contract" "status"
    if ($LASTEXITCODE -ne 0) {
        $statusExitCode = $LASTEXITCODE
        Write-Output $statusJson
        $global:LASTEXITCODE = $statusExitCode
        return
    }
    $status = $statusJson | ConvertFrom-Json
    if ($status.phase1.status -eq "PHASE1_FROZEN" -and $status.phase1.frozen -eq $true) {
        Write-Output "PHASE1_FREEZE_REPORT.md validated by application gate"
        $global:LASTEXITCODE = 0
        return
    }

    Write-Output "PHASE1_FREEZE_REPORT.md exists but application gate rejected it: $($status.phase1.reason)"
    $global:LASTEXITCODE = 2
    return
}
$checks += Invoke-AuditCommand "pytest" {
    & $python "-m" "pytest" "-q"
}
$checks += Invoke-AuditCommand "ruff" {
    & $python "-m" "ruff" "check" ".\main.py" ".\src" ".\tests"
}
$checks += Invoke-AuditCommand "compileall" {
    & $python "-m" "compileall" "-q" ".\main.py" ".\src" ".\tests"
}
$forbiddenOrderPattern = ("place" + "Order|cancel" + "Order|req" + "Global" + "Cancel|req" + "Ids")
$scanOutput = & rg -n $forbiddenOrderPattern -- ".\main.py" ".\src" ".\tests" 2>&1
$scanExit = $LASTEXITCODE
$orderViolations = @()
if ($scanExit -eq 0) {
    $orderViolations = @($scanOutput | Where-Object {
        $_ -notmatch 'src[\\/]stocks[\\/]ibkr[\\/]paper_execution[\\/]submission\.py:.*placeOrder' -and
        $_ -notmatch 'src[\\/]stocks[\\/]ibkr[\\/]paper_execution[\\/]cancellation\.py:.*cancelOrder' -and
        $_ -notmatch 'src[\\/]stocks[\\/]ibkr[\\/]paper_execution[\\/]order_ids\.py:.*reqIds' -and
        $_ -notmatch 'tests[\\/]test_phase9_paper_execution\.py:.*placeOrder' -and
        $_ -notmatch 'tests[\\/]test_phase9_paper_execution\.py:.*cancelOrder'
    })
}
$checks += New-AuditResult "forbidden_order_method_scan" $(if ($scanExit -eq 1 -or $orderViolations.Count -eq 0) { 0 } else { 1 }) $(if ($scanExit -eq 1 -or $orderViolations.Count -eq 0) { "Forbidden IBKR write-method scan: GO" } else { ($orderViolations | Out-String).Trim() })

$forbiddenDataPattern = ("req" + "Mkt" + "Data|req" + "Real" + "Time" + "Bars")
$dataScanOutput = & rg -n $forbiddenDataPattern -- ".\main.py" ".\src" ".\tests" ".\scripts" 2>&1
$dataScanExit = $LASTEXITCODE
$checks += New-AuditResult "live_ibkr_data_request_scan" $(if ($dataScanExit -eq 1) { 0 } else { 1 }) $(if ($dataScanExit -eq 1) { "Live IBKR data request scan: GO" } else { ($dataScanOutput | Out-String).Trim() })

$historicalDataPattern = ("req" + "Historical" + "Data")
$historicalScanOutput = & rg -n $historicalDataPattern -- ".\main.py" ".\src" ".\tests" ".\scripts" 2>&1
$historicalScanExit = $LASTEXITCODE
$historicalViolations = @()
if ($historicalScanExit -eq 0) {
    $historicalViolations = @($historicalScanOutput | Where-Object { $_ -notmatch 'src[\\/]stocks[\\/]data[\\/]ibkr_historical\.py' })
}
$checks += New-AuditResult "historical_data_request_allowlist_scan" $(if ($historicalScanExit -eq 1 -or $historicalViolations.Count -eq 0) { 0 } else { 1 }) $(if ($historicalScanExit -eq 1 -or $historicalViolations.Count -eq 0) { "Historical data request allowlist scan: GO" } else { ($historicalViolations | Out-String).Trim() })

$accountScanOutput = & rg -n "DU[0-9]{3,}|U[0-9]{3,}" -- ".\main.py" ".\src" ".\tests" ".\docs" ".\scripts" ".\README_PHASE_1_IBKR_SERVICE.md" ".\PHASE1_STATUS.md" 2>&1
$accountScanExit = $LASTEXITCODE
$checks += New-AuditResult "account_identifier_scan" $(if ($accountScanExit -eq 1) { 0 } else { $accountScanExit }) $(if ($accountScanExit -eq 1) { "Account identifier scan: GO" } else { ($accountScanOutput | Out-String).Trim() })

$processes = Get-CimInstance Win32_Process -Filter "name = 'python.exe'" |
    Where-Object { $_.CommandLine -like '*main.py ibkr*' } |
    Select-Object ProcessId,ParentProcessId,CommandLine
$processText = if ($processes) { ($processes | Format-List | Out-String).Trim() } else { "No active main.py ibkr processes" }
$checks += New-AuditResult "active_ibkr_process_scan" $(if ($processes) { 1 } else { 0 }) $processText

$allPassed = -not ($checks | Where-Object { -not $_.passed })
$phase1FreezeCheck = $checks | Where-Object { $_.name -eq "phase1_freeze_gate" } | Select-Object -First 1
$externalGateStatus = if ((Test-Path ".\PHASE1_FREEZE_REPORT.md") -and $phase1FreezeCheck -and $phase1FreezeCheck.passed) {
    "forced TWS disconnect/reconnect drill GO; Phase 1 freeze gate validated"
} else {
    "forced TWS disconnect/reconnect drill still requires operator action"
}
$report = [ordered]@{
    schema = "phase1_static_audit_v1"
    generated_at = (Get-Date).ToUniversalTime().ToString("o")
    status = $(if ($allPassed) { "GO" } else { "NO_GO" })
    external_gate = $externalGateStatus
    checks = $checks
}

$report | ConvertTo-Json -Depth 8 | Out-File -Encoding utf8 $artifact
$report | ConvertTo-Json -Depth 8

if ($allPassed) {
    exit 0
}

exit 2
