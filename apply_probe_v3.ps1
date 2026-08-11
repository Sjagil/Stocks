param(
    [string]$ProjectRoot = "C:\Users\alhar\Documents\Stocks"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location $ProjectRoot

$source = Join-Path $PSScriptRoot "ibkr_tws_probe.py"
$target = Join-Path $ProjectRoot "ibkr_tws_probe.py"
$backup = Join-Path $ProjectRoot (
    "ibkr_tws_probe_v2_backup_" +
    (Get-Date -Format "yyyyMMdd_HHmmss") +
    ".py"
)
$python = Join-Path $ProjectRoot ".venv-ibkr\Scripts\python.exe"

if (-not (Test-Path $source)) {
    throw "Nieuwe probe ontbreekt naast het patchscript: $source"
}
if (-not (Test-Path $target)) {
    throw "Bestaande probe ontbreekt: $target"
}
if (-not (Test-Path $python)) {
    throw "Virtuele omgeving ontbreekt: $python"
}

Copy-Item $target $backup
Copy-Item $source $target -Force

Write-Host "Backup: $backup"
Write-Host "Nieuwe probe: $target"

& $python -m py_compile $target
if ($LASTEXITCODE -ne 0) {
    Copy-Item $backup $target -Force
    throw "Compilecheck faalde. De oude probe is teruggezet."
}

$forbidden = Select-String `
    -Path $target `
    -Pattern "\.placeOrder\s*\(|\.cancelOrder\s*\(|\.reqGlobalCancel\s*\("

if ($forbidden) {
    Copy-Item $backup $target -Force
    throw "Verboden financiële ordermethodes gevonden. Patch teruggedraaid."
}

Write-Host "Probe v3 compile: GO" -ForegroundColor Green
Write-Host "Ordermethodes afwezig: GO" -ForegroundColor Green
Write-Host ""
Write-Host "Probe starten..."
& $python $target "--env-file" (Join-Path $ProjectRoot ".env.ibkr")
exit $LASTEXITCODE
