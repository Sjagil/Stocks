param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [double]$Seconds = 180,
    [double]$PollSeconds = 2,
    [switch]$RequireOperatorReady
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
$artifact = Join-Path $outputDir "phase1-disconnect-drill-$timestamp.json"
$verifier = Join-Path $ProjectRoot "scripts\verify_phase1_disconnect_drill_artifact.ps1"
$staticAudit = Join-Path $ProjectRoot "scripts\run_phase1_static_audit.ps1"

if (-not (Test-Path $verifier)) {
    throw "Verifier niet gevonden: $verifier"
}
if (-not (Test-Path $staticAudit)) {
    throw "Static audit niet gevonden: $staticAudit"
}

Write-Host ""
Write-Host "IBKR Phase 1 forced-disconnect drill" -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot"
Write-Host "Duration: $Seconds seconds"
Write-Host "Poll: $PollSeconds seconds"
Write-Host "Artifact: $artifact"
Write-Host ""

$preflightArguments = @(
    ".\main.py",
    "ibkr",
    "disconnect-drill-preflight"
)

$preflightOutput = & $python @preflightArguments 2>&1
$preflightExitCode = $LASTEXITCODE
$preflightText = ($preflightOutput | Out-String).Trim()

Write-Host "Preflight:"
Write-Host $preflightText
Write-Host ""

try {
    $preflight = $preflightText | ConvertFrom-Json
} catch {
    Write-Host "DRILL NO_GO: preflight output was not valid JSON." -ForegroundColor Red
    exit 2
}

if ($preflightExitCode -ne 0 -or $preflight.status -ne "GO") {
    Write-Host "DRILL NO_GO: preflight failed." -ForegroundColor Red
    Write-Host "Blocking checks: $($preflight.blocking_checks -join ', ')"
    Write-Host "Start TWS paper, enable API sockets, verify read-only port 7497, then rerun this helper."
    exit 2
}

Write-Host "Operator action required:" -ForegroundColor Yellow
Write-Host "  1. Laat TWS paper nu verbonden."
Write-Host "  2. Sluit of herstart TWS paper tijdens deze drill."
Write-Host "  3. Log opnieuw in op paper zodat bounded reconnect kan slagen."
Write-Host "  4. Een GO vereist disconnect_observed=true en reconnect_successful=true."
Write-Host ""

if ($RequireOperatorReady) {
    Write-Host "Operator confirmation required before the countdown starts." -ForegroundColor Yellow
    Write-Host "Type READY only when you can close and restart TWS paper during the drill."
    $ready = Read-Host "Operator ready"
    if ($ready -ne "READY") {
        Write-Host "DRILL NO_GO: operator did not confirm READY." -ForegroundColor Red
        exit 2
    }
    Write-Host ""
}

Write-Host "Disconnect window started: $(Get-Date -Format o)"
Write-Host ""

$arguments = @(
    ".\main.py",
    "ibkr",
    "disconnect-drill",
    "--seconds",
    "$Seconds",
    "--poll-seconds",
    "$PollSeconds"
)

$rawOutput = & $python @arguments 2>&1
$exitCode = $LASTEXITCODE
$rawText = ($rawOutput | Out-String).Trim()
$rawText | Out-File -Encoding utf8 $artifact

Write-Host $rawText
Write-Host ""
Write-Host "Raw command exit code: $exitCode"

try {
    $report = $rawText | ConvertFrom-Json
} catch {
    Write-Host "DRILL NO_GO: output was not valid JSON." -ForegroundColor Red
    exit 2
}

if ($report.status -eq "GO") {
    Write-Host "DRILL GO: forced disconnect observed and bounded reconnect succeeded." -ForegroundColor Green
    Write-Host "Verifying artifact and writing freeze report..."
    & powershell -NoProfile -ExecutionPolicy Bypass -File $verifier -ArtifactPath $artifact
    $verifierExitCode = $LASTEXITCODE
    if ($verifierExitCode -ne 0) {
        exit $verifierExitCode
    }

    Write-Host "Running post-freeze static audit..."
    & powershell -NoProfile -ExecutionPolicy Bypass -File $staticAudit
    $staticAuditExitCode = $LASTEXITCODE
    if ($staticAuditExitCode -ne 0) {
        Write-Host "POST-FREEZE STATIC AUDIT NO_GO" -ForegroundColor Red
        exit $staticAuditExitCode
    }

    Write-Host "POST-FREEZE STATIC AUDIT GO" -ForegroundColor Green
    exit 0
}

Write-Host "DRILL NO_GO: $($report.failure_reason)" -ForegroundColor Red
exit 2
