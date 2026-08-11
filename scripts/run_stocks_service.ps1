param(
    [ValidateSet(
        "SIGNALS_ONLY",
        "PAPER_AUTOMATIC",
        "LIVE_CANARY_AUTOMATIC",
        "CONTROLLED_LIVE"
    )]
    [string]$Mode = "SIGNALS_ONLY",
    [ValidateRange(1, 1440)]
    [int]$MaxCycles = 1440,
    [ValidateRange(60, 86400)]
    [int]$IntervalSeconds = 60
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv-ibkr\Scripts\python.exe"
$Main = Join-Path $ProjectRoot "main.py"
$OutputRoot = Join-Path $ProjectRoot "output\operations"
$Stdout = Join-Path $OutputRoot "service-last.stdout.json"
$Stderr = Join-Path $OutputRoot "service-last.stderr.log"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python runtime not found: $Python"
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$Arguments = @(
    $Main,
    "run",
    "--mode",
    $Mode,
    "--max-cycles",
    "$MaxCycles",
    "--interval-seconds",
    "$IntervalSeconds"
)
Push-Location $ProjectRoot
try {
    & $Python @Arguments 1> $Stdout 2> $Stderr
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
