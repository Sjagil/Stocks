param(
    [Parameter(Mandatory = $true)]
    [string]$ArtifactPath,
    [string]$ProjectRoot = "",
    [switch]$NoWriteFreezeReport
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $ProjectRoot = (Resolve-Path (Join-Path $scriptDir "..")).Path
}

Set-Location $ProjectRoot

function Get-Sha256Hex {
    param([string]$Path)

    $stream = [System.IO.File]::OpenRead((Resolve-Path $Path).Path)
    try {
        $sha = [System.Security.Cryptography.SHA256]::Create()
        try {
            $hashBytes = $sha.ComputeHash($stream)
            return (($hashBytes | ForEach-Object { $_.ToString("X2") }) -join "")
        } finally {
            $sha.Dispose()
        }
    } finally {
        $stream.Dispose()
    }
}

function Test-JsonInteger {
    param([object]$Value)

    return (
        $null -ne $Value `
        -and -not ($Value -is [bool]) `
        -and (
            $Value -is [byte] `
            -or $Value -is [sbyte] `
            -or $Value -is [int16] `
            -or $Value -is [uint16] `
            -or $Value -is [int] `
            -or $Value -is [uint32] `
            -or $Value -is [long] `
            -or $Value -is [uint64]
        )
    )
}

function Test-JsonNumber {
    param([object]$Value)

    return (
        (Test-JsonInteger $Value) `
        -or (
            $null -ne $Value `
            -and -not ($Value -is [bool]) `
            -and (
                $Value -is [float] `
                -or $Value -is [double] `
                -or $Value -is [decimal]
            )
        )
    )
}

$resolvedArtifact = Resolve-Path $ArtifactPath
$raw = Get-Content -Raw $resolvedArtifact

try {
    $report = $raw | ConvertFrom-Json
} catch {
    throw "Artifact is not valid JSON: $resolvedArtifact"
}

$errors = New-Object System.Collections.Generic.List[string]

if ($report.schema -ne "ibkr_forced_disconnect_drill_v1") {
    $errors.Add("schema must be ibkr_forced_disconnect_drill_v1")
}
if ($report.status -ne "GO") {
    $errors.Add("status must be GO")
}
if ($report.host -ne "127.0.0.1") {
    $errors.Add("host must be 127.0.0.1 for the TWS paper Phase 1 freeze drill")
}
if (-not (Test-JsonInteger $report.port) -or [int64]$report.port -ne 7497) {
    $errors.Add("port must be 7497 for the TWS paper Phase 1 freeze drill")
}
if (-not (Test-JsonInteger $report.client_id) -or [int64]$report.client_id -le 0) {
    $errors.Add("client_id must be a positive integer")
}
if ($report.disconnect_observed -ne $true) {
    $errors.Add("disconnect_observed must be true")
}
if ($report.reconnect_successful -ne $true) {
    $errors.Add("reconnect_successful must be true")
}
if (-not (Test-JsonNumber $report.seconds) -or [double]$report.seconds -lt 180.0) {
    $errors.Add("seconds must be at least 180 for a Phase 1 freeze")
}
if (
    -not (Test-JsonNumber $report.poll_seconds) `
    -or (
        (Test-JsonNumber $report.seconds) `
        -and ([double]$report.poll_seconds -le 0.0 -or [double]$report.poll_seconds -gt [double]$report.seconds)
    )
) {
    $errors.Add("poll_seconds must be positive and no greater than seconds")
}
if ($null -ne $report.failure_reason -and -not [string]::IsNullOrWhiteSpace([string]$report.failure_reason)) {
    $errors.Add("failure_reason must be null or blank for a Phase 1 freeze")
}
if (-not (Test-JsonInteger $report.financial_calls.place_order) -or [int64]$report.financial_calls.place_order -ne 0) {
    $errors.Add("financial_calls.place_order must be 0")
}
if (-not (Test-JsonInteger $report.financial_calls.cancel_order) -or [int64]$report.financial_calls.cancel_order -ne 0) {
    $errors.Add("financial_calls.cancel_order must be 0")
}
if (-not (Test-JsonInteger $report.financial_calls.global_cancel) -or [int64]$report.financial_calls.global_cancel -ne 0) {
    $errors.Add("financial_calls.global_cancel must be 0")
}

$healthyStartIndex = -1
$disconnectIndex = -1
$reconnectIndex = -1
for ($index = 0; $index -lt @($report.observed_statuses).Count; $index++) {
    $item = $report.observed_statuses[$index]
    if ($healthyStartIndex -lt 0 -and $item.phase -eq "start" -and $item.status -eq "HEALTHY") {
        $healthyStartIndex = $index
    }
    if (
        $disconnectIndex -lt 0 `
        -and $healthyStartIndex -ge 0 `
        -and $index -gt $healthyStartIndex `
        -and ($item.status -eq "DISCONNECTED" -or $item.status -eq "STALE")
    ) {
        $disconnectIndex = $index
    }
    if (
        $reconnectIndex -lt 0 `
        -and $disconnectIndex -ge 0 `
        -and $index -gt $disconnectIndex `
        -and $item.phase -eq "reconnect" `
        -and ($item.status -eq "HEALTHY" -or $item.status -eq "DEGRADED")
    ) {
        $reconnectIndex = $index
    }
}

if ($healthyStartIndex -lt 0) {
    $errors.Add("observed_statuses must include a start phase with HEALTHY")
}
if ($disconnectIndex -lt 0) {
    $errors.Add("observed_statuses must include DISCONNECTED or STALE after start")
}
if ($reconnectIndex -lt 0) {
    $errors.Add("observed_statuses must include a reconnect phase with HEALTHY or DEGRADED after disconnect")
}

if (-not $NoWriteFreezeReport) {
    $outputRoot = (Resolve-Path (Join-Path $ProjectRoot "output\ibkr")).Path
    $artifactDir = Split-Path -Parent $resolvedArtifact.Path
    if (-not $artifactDir.StartsWith($outputRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        $errors.Add("freeze-writing verification requires an artifact under output\ibkr")
    }
    $artifactName = Split-Path -Leaf $resolvedArtifact.Path
    if ($artifactName -notmatch '^phase1-disconnect-drill-\d{8}-\d{6}\.json$') {
        $errors.Add("freeze-writing verification requires artifact name phase1-disconnect-drill-YYYYMMDD-HHMMSS.json")
    } else {
        $artifactTimestamp = $artifactName -replace '^phase1-disconnect-drill-(\d{8})-(\d{6})\.json$', '$1$2'
        $parsedTimestamp = [datetime]::MinValue
        if (-not [datetime]::TryParseExact(
            $artifactTimestamp,
            "yyyyMMddHHmmss",
            [System.Globalization.CultureInfo]::InvariantCulture,
            [System.Globalization.DateTimeStyles]::None,
            [ref]$parsedTimestamp
        )) {
            $errors.Add("freeze-writing verification requires parseable artifact timestamp YYYYMMDD-HHMMSS")
        }
    }
}

if ($errors.Count -gt 0) {
    $result = [ordered]@{
        schema = "phase1_disconnect_drill_artifact_verification_v1"
        generated_at = (Get-Date).ToUniversalTime().ToString("o")
        status = "NO_GO"
        artifact = $resolvedArtifact.Path
        errors = @($errors)
    }
    $result | ConvertTo-Json -Depth 8
    exit 2
}

$freezeReportPath = Join-Path $ProjectRoot "PHASE1_FREEZE_REPORT.md"
$artifactHash = Get-Sha256Hex $resolvedArtifact
$probeHash = Get-Sha256Hex (Join-Path $ProjectRoot "ibkr_tws_probe.py")
$lockHash = Get-Sha256Hex (Join-Path $ProjectRoot "requirements.lock.txt")
$immutablePhase1ServiceHashFiles = [ordered]@{
    "src/stocks/application/config.py" = Join-Path $ProjectRoot "src\stocks\application\config.py"
    "src/stocks/ibkr/connection.py" = Join-Path $ProjectRoot "src\stocks\ibkr\connection.py"
    "src/stocks/ibkr/client.py" = Join-Path $ProjectRoot "src\stocks\ibkr\client.py"
    "src/stocks/ibkr/callbacks.py" = Join-Path $ProjectRoot "src\stocks\ibkr\callbacks.py"
    "src/stocks/ibkr/errors.py" = Join-Path $ProjectRoot "src\stocks\ibkr\errors.py"
    "src/stocks/ibkr/health.py" = Join-Path $ProjectRoot "src\stocks\ibkr\health.py"
}
$immutablePhase1ServiceHashes = [ordered]@{}
foreach ($entry in $immutablePhase1ServiceHashFiles.GetEnumerator()) {
    $immutablePhase1ServiceHashes[$entry.Key] = Get-Sha256Hex $entry.Value
}
$immutablePhase1ServiceHashLines = ($immutablePhase1ServiceHashes.GetEnumerator() | ForEach-Object {
    "{0,-34} {1}" -f $_.Key, $_.Value
}) -join "`r`n"
$mutableApplicationEntrypointHashes = [ordered]@{
    "main.py" = Get-Sha256Hex (Join-Path $ProjectRoot "main.py")
}
$mutableApplicationEntrypointHashLines = ($mutableApplicationEntrypointHashes.GetEnumerator() | ForEach-Object {
    "{0,-34} {1}" -f $_.Key, $_.Value
}) -join "`r`n"
$Fence = '```'

$freezeReport = @"
# Phase 1 Freeze Report

Status:

${Fence}text
IBKR_PHASE_1_READ_ONLY_CONNECTION_SERVICE_GO
PHASE1_CONNECTION_SERVICE_FROZEN_GO
${Fence}

Verified artifact:

${Fence}text
$($resolvedArtifact.Path)
${Fence}

Artifact SHA256:

${Fence}text
$artifactHash
${Fence}

Evidence:

${Fence}text
schema                  $($report.schema)
status                  $($report.status)
host                    $($report.host)
port                    $($report.port)
client_id               $($report.client_id)
disconnect_observed     $($report.disconnect_observed)
reconnect_successful    $($report.reconnect_successful)
place_order             $($report.financial_calls.place_order)
cancel_order            $($report.financial_calls.cancel_order)
global_cancel           $($report.financial_calls.global_cancel)
${Fence}

Frozen Phase 0 hashes:

${Fence}text
ibkr_tws_probe.py       $probeHash
requirements.lock.txt   $lockHash
${Fence}

Immutable Phase 1 service hashes:

${Fence}text
$immutablePhase1ServiceHashLines
${Fence}

Mutable application entrypoint hash:

${Fence}text
$mutableApplicationEntrypointHashLines
${Fence}

Phase 1 remains read-only. This report does not grant order authority.
"@

if (-not $NoWriteFreezeReport) {
    $freezeReport | Out-File -Encoding utf8 $freezeReportPath
}

$result = [ordered]@{
    schema = "phase1_disconnect_drill_artifact_verification_v1"
    generated_at = (Get-Date).ToUniversalTime().ToString("o")
    status = "GO"
    artifact = $resolvedArtifact.Path
    artifact_sha256 = $artifactHash
    immutable_phase1_service_hashes = $immutablePhase1ServiceHashes
    mutable_application_entrypoint_hash = $mutableApplicationEntrypointHashes
    freeze_report = $(if ($NoWriteFreezeReport) { $null } else { $freezeReportPath })
}
$result | ConvertTo-Json -Depth 8
exit 0
