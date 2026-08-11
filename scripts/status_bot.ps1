param([string]$TaskName = "Stocks Canonical Runtime")

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv-ibkr\Scripts\python.exe"
$Main = Join-Path $ProjectRoot "main.py"

$Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
$TaskInfo = Get-ScheduledTaskInfo -TaskName $TaskName -ErrorAction SilentlyContinue
[pscustomobject]@{
    TaskName = $TaskName
    Installed = $null -ne $Task
    State = if ($Task) { $Task.State } else { "NOT_INSTALLED" }
    LastRunTime = if ($TaskInfo) { $TaskInfo.LastRunTime } else { $null }
    LastTaskResult = if ($TaskInfo) { $TaskInfo.LastTaskResult } else { $null }
}
& $Python $Main launch status
exit $LASTEXITCODE
