param(
    [ValidateSet(
        "SIGNALS_ONLY",
        "PAPER_AUTOMATIC",
        "LIVE_CANARY_AUTOMATIC",
        "CONTROLLED_LIVE"
    )]
    [string]$Mode = "SIGNALS_ONLY",
    [string]$TaskName = "Stocks Bounded Autopilot",
    [ValidateRange(5, 1440)]
    [int]$RepetitionMinutes = 10
)

$ErrorActionPreference = "Stop"
$Runner = Join-Path $PSScriptRoot "run_stocks_autopilot.ps1"
if (-not (Test-Path -LiteralPath $Runner)) {
    throw "Autopilot runner not found: $Runner"
}

$PowerShell = (Get-Command powershell.exe).Source
$Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$Runner`" -Mode $Mode"
$Action = New-ScheduledTaskAction `
    -Execute $PowerShell `
    -Argument $Arguments `
    -WorkingDirectory (Split-Path -Parent $PSScriptRoot)
$Trigger = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes $RepetitionMinutes)
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -MultipleInstances IgnoreNew `
    -Priority 4

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "Runs the bounded Stocks supervisor through main.py." `
    -Force
