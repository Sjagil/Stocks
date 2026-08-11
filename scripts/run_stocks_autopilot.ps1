param(
    [ValidateSet(
        "SIGNALS_ONLY",
        "PAPER_AUTOMATIC",
        "LIVE_CANARY_AUTOMATIC",
        "CONTROLLED_LIVE"
    )]
    [string]$Mode = "SIGNALS_ONLY"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv-ibkr\Scripts\python.exe"
$Main = Join-Path $ProjectRoot "main.py"
$OutputRoot = Join-Path $ProjectRoot "output\operations"
$Stdout = Join-Path $OutputRoot "scheduler-last.stdout.json"
$Stderr = Join-Path $OutputRoot "scheduler-last.stderr.log"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python runtime not found: $Python"
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$Arguments = @(
    $Main,
    "autopilot",
    "run",
    "--mode",
    $Mode,
    "--max-cycles",
    "1",
    "--interval-seconds",
    "300"
)
$Process = Start-Process `
    -FilePath $Python `
    -ArgumentList $Arguments `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $Stdout `
    -RedirectStandardError $Stderr `
    -Wait `
    -PassThru
exit $Process.ExitCode
