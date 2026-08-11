param([string]$TaskName = "Stocks Canonical Runtime")

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv-ibkr\Scripts\python.exe"
$Main = Join-Path $ProjectRoot "main.py"

& $Python $Main launch stop
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
Disable-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue |
    Out-Null

function Get-CanonicalRuntimeProcess {
    $ResolvedMain = [regex]::Escape((Resolve-Path $Main).Path)
    return @(
        Get-CimInstance Win32_Process |
            Where-Object {
                $_.Name -eq "python.exe" -and
                $_.CommandLine -match $ResolvedMain -and
                $_.CommandLine -match "\srun\s+--mode\s"
            }
    )
}

$Deadline = (Get-Date).AddSeconds(75)
$RuntimeProcesses = Get-CanonicalRuntimeProcess
while ($RuntimeProcesses.Count -gt 0 -and (Get-Date) -lt $Deadline) {
    Start-Sleep -Seconds 1
    $RuntimeProcesses = Get-CanonicalRuntimeProcess
}

# The scheduler owns only the PowerShell parent. Bound forced cleanup to the
# exact project main.py run command when graceful shutdown missed its window.
if ($RuntimeProcesses.Count -gt 0) {
    $RuntimeProcesses | ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 2
}

if ((Get-CanonicalRuntimeProcess).Count -gt 0) {
    throw "Canonical runtime process did not stop within the bounded window."
}

[pscustomobject]@{
    status = "GO"
    runtime_status = "STOPPED"
    task_name = $TaskName
    remaining_runtime_processes = 0
} | ConvertTo-Json
