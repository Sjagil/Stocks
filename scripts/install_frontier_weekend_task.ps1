$ErrorActionPreference = "Stop"

$taskName = "Stocks Frontier Weekend Research"
$runner = Join-Path $PSScriptRoot "run_frontier_weekend_research.ps1"
$powershell = (Get-Command powershell.exe).Source
$action = "`"$powershell`" -NoProfile -ExecutionPolicy Bypass -File `"$runner`""

& schtasks.exe /Create /TN $taskName /TR $action /SC HOURLY /MO 4 /F | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Unable to register scheduled task: $taskName"
}

Get-ScheduledTask -TaskName $taskName | Select-Object TaskName, State, TaskPath
