param(
    [string]$TaskName = "Stocks Canonical Runtime",
    [ValidateSet(
        "SIGNALS_ONLY",
        "PAPER_AUTOMATIC",
        "LIVE_CANARY_AUTOMATIC",
        "CONTROLLED_LIVE"
    )]
    [string]$Mode = "SIGNALS_ONLY"
)

$ErrorActionPreference = "Stop"
$Runner = Join-Path $PSScriptRoot "run_stocks_service.ps1"
if (-not (Test-Path -LiteralPath $Runner)) {
    throw "Canonical runtime runner not found: $Runner"
}

$PowerShell = (Get-Command powershell.exe).Source
$Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$Runner`" -Mode $Mode"
$Action = New-ScheduledTaskAction `
    -Execute $PowerShell `
    -Argument $Arguments `
    -WorkingDirectory (Split-Path -Parent $PSScriptRoot)
$CurrentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$LogonTrigger = New-ScheduledTaskTrigger -AtLogOn -User $CurrentUser
# Keep one canonical runtime durable across a clean 24-hour bounded runner
# exit or an unexpected process failure. MultipleInstances=IgnoreNew prevents
# overlap while this five-minute recovery trigger closes the restart gap.
$RecoveryTrigger = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes 5)
$Triggers = @($LogonTrigger, $RecoveryTrigger)
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 26) `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 2) `
    -Priority 4

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Triggers `
    -Settings $Settings `
    -User $CurrentUser `
    -Description "Runs only the canonical Stocks main.py run entrypoint." `
    -Force
