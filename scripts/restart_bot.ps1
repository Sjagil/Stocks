param([string]$TaskName = "Stocks Canonical Runtime")

$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "stop_bot.ps1") -TaskName $TaskName
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
& (Join-Path $PSScriptRoot "start_bot.ps1") -TaskName $TaskName
