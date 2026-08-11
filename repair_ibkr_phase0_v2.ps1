param(
    [string]$ProjectRoot = (Get-Location).Path,
    [string]$ApiSource = "C:\TWS API\source\pythonclient"
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "IBKR Phase 0 Repair v2" -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot"
Write-Host "Official API source: $ApiSource"
Write-Host ""

Set-Location $ProjectRoot

$python = Join-Path $ProjectRoot ".venv-ibkr\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw ".venv-ibkr niet gevonden. Draai eerst .\install_ibkr_windows.ps1."
}

if (-not (Test-Path $ApiSource)) {
    throw "Officiele TWS API Pythonbron niet gevonden: $ApiSource"
}

$setupPy = Join-Path $ApiSource "setup.py"
$pyproject = Join-Path $ApiSource "pyproject.toml"
if ((-not (Test-Path $setupPy)) -and (-not (Test-Path $pyproject))) {
    throw "Geen setup.py of pyproject.toml gevonden in $ApiSource"
}

Write-Host "[1/7] pip tooling bijwerken..."
& $python -m pip install --upgrade pip setuptools wheel

Write-Host "[2/7] Third-party brokerwrappers en oude ibapi verwijderen..."
$packagesToRemove = @(
    "ib",
    "iba",
    "ibapi",
    "ib_async",
    "ib_insync",
    "aeventkit",
    "eventkit"
)
& $python -m pip uninstall -y @packagesToRemove

Write-Host "[3/7] Projectrequirements opnieuw normaliseren..."
$requirements = Join-Path $ProjectRoot "requirements.txt"
if (-not (Test-Path $requirements)) {
    throw "requirements.txt niet gevonden in $ProjectRoot"
}
& $python -m pip install -r $requirements

Write-Host "[4/7] tzdata herstellen naar actuele releasefamilie..."
& $python -m pip install --upgrade "tzdata>=2026.3"

Write-Host "[5/7] Officiele lokale IBKR Python API installeren..."
& $python -m pip install --upgrade --force-reinstall $ApiSource

Write-Host "[6/7] Importpad en dependencyconsistentie controleren..."
& $python -c "import ibapi; print('ibapi module path:', ibapi.__file__)"
& $python -m pip show ibapi
& $python -m pip check

Write-Host "[7/7] Lockbestand schrijven..."
& $python -m pip freeze | Out-File -Encoding utf8 (Join-Path $ProjectRoot "requirements.lock.txt")

Write-Host ""
Write-Host "IBKR PHASE 0 REPAIR V2 GO" -ForegroundColor Green
Write-Host "Start TWS in paper/read-only op poort 7497 en draai daarna:"
Write-Host "  python .\ibkr_tws_probe.py --env-file .env.ibkr"
