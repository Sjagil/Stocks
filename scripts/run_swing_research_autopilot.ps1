[CmdletBinding()]
param(
    [ValidateSet("daily", "weekly", "monthly", "auto")]
    [string]$Cadence = "auto",
    [int]$WeeklyMaxTrials = 40
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $ProjectRoot ".venv-ibkr\Scripts\python.exe"
$LockPath = Join-Path $ProjectRoot "data\research\autopilot\private\autopilot.lock"
$LogRoot = Join-Path $ProjectRoot "output\research\autopilot\operator-logs"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment not found: $Python"
}

New-Item -ItemType Directory -Force -Path (Split-Path $LockPath) | Out-Null
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

try {
    $Lock = [System.IO.File]::Open(
        $LockPath,
        [System.IO.FileMode]::OpenOrCreate,
        [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::None
    )
}
catch {
    throw "AUTOPILOT_SINGLE_FLIGHT_LOCKED"
}

try {
    Set-Location $ProjectRoot
    $Stamp = (Get-Date).ToString("yyyyMMdd-HHmmss")
    $Log = Join-Path $LogRoot "autopilot-$Stamp.log"

    & $Python .\main.py screener run *>> $Log
    & $Python .\main.py research daily *>> $Log

    $RunWeekly = $Cadence -eq "weekly" -or (
        $Cadence -eq "auto" -and (Get-Date).DayOfWeek -eq "Sunday"
    )
    $RunMonthly = $Cadence -eq "monthly" -or (
        $Cadence -eq "auto" -and (Get-Date).Day -le 3
    )
    if ($RunWeekly) {
        & $Python .\main.py research weekly --max-trials $WeeklyMaxTrials *>> $Log
    }
    if ($RunMonthly) {
        & $Python .\main.py research monthly *>> $Log
    }
}
finally {
    if ($null -ne $Lock) {
        $Lock.Dispose()
    }
}
