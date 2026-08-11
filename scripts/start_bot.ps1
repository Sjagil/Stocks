param([string]$TaskName = "Stocks Canonical Runtime")

$ErrorActionPreference = "Stop"
$Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
if ($Task.State -eq "Disabled") {
    Enable-ScheduledTask -TaskName $TaskName -ErrorAction Stop | Out-Null
    $Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
}
if ($Task.State -ne "Running") {
    Start-ScheduledTask -TaskName $TaskName
}
Get-ScheduledTaskInfo -TaskName $TaskName
