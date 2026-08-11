param(
    [string]$ProjectRoot = (Get-Location).Path,
    [string]$PythonVersion = "3.12"
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "IBKR Phase 0 installer" -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot"
Write-Host ""

Set-Location $ProjectRoot

$requirements = Join-Path $ProjectRoot "requirements.txt"
if (-not (Test-Path $requirements)) {
    throw "requirements.txt niet gevonden in $ProjectRoot"
}

$venv = Join-Path $ProjectRoot ".venv-ibkr"
$python = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Host "[1/6] Virtuele omgeving maken met Python $PythonVersion..."
    & py "-$PythonVersion" -m venv $venv
}

if (-not (Test-Path $python)) {
    throw "Virtuele omgeving kon niet worden gemaakt. Installeer Python $PythonVersion x64 en controleer 'py -0p'."
}

Write-Host "[2/6] pip, setuptools en wheel bijwerken..."
& $python -m pip install --upgrade pip setuptools wheel

Write-Host "[3/6] Algemene dependencies installeren..."
& $python -m pip install -r $requirements

Write-Host "[4/6] Officiële IBKR Python API zoeken..."
$candidates = @(
    "C:\TWS API\source\pythonclient",
    (Join-Path $env:USERPROFILE "TWS API\source\pythonclient"),
    (Join-Path $env:USERPROFILE "Downloads\TWS API\source\pythonclient"),
    (Join-Path $env:USERPROFILE "Downloads\twsapi\source\pythonclient")
)

$apiPath = $null
foreach ($candidate in $candidates) {
    if (Test-Path (Join-Path $candidate "setup.py")) {
        $apiPath = $candidate
        break
    }
    if (Test-Path (Join-Path $candidate "pyproject.toml")) {
        $apiPath = $candidate
        break
    }
}

if ($null -eq $apiPath) {
    Write-Host ""
    Write-Host "De algemene Python-packages zijn geïnstalleerd." -ForegroundColor Yellow
    Write-Host "De officiële IBKR TWS API-package is nog niet gevonden." -ForegroundColor Yellow
    Write-Host "Installeer eerst de officiële TWS API voor Windows."
    Write-Host "De verwachte Pythonbron staat daarna meestal in:"
    Write-Host "  C:\TWS API\source\pythonclient"
    Write-Host ""
    Write-Host "Voer dit installatiescript daarna opnieuw uit."
    exit 2
}

Write-Host "IBKR Python client gevonden: $apiPath"
& $python -m pip install --upgrade $apiPath

Write-Host "[5/6] Configuratiebestand voorbereiden..."
$envExample = Join-Path $ProjectRoot ".env.ibkr.example"
$envTarget = Join-Path $ProjectRoot ".env.ibkr"
if ((Test-Path $envExample) -and (-not (Test-Path $envTarget))) {
    Copy-Item $envExample $envTarget
    Write-Host ".env.ibkr aangemaakt uit het veilige voorbeeld."
} elseif (Test-Path $envTarget) {
    Write-Host ".env.ibkr bestaat al en is niet overschreven."
}

Write-Host "[6/6] Installatie verifiëren en lockbestand schrijven..."
& $python -c "import ibapi, numpy, pandas, scipy, pyarrow, duckdb, pydantic; print('IBKR en quant dependencies: IMPORT GO'); print('ibapi:', ibapi.__file__)"
& $python -m pip freeze | Out-File -Encoding utf8 (Join-Path $ProjectRoot "requirements.lock.txt")

Write-Host ""
Write-Host "INSTALLATIE GO" -ForegroundColor Green
Write-Host "Activeer later met:"
Write-Host "  .\.venv-ibkr\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Start TWS in PAPER, configureer poort 7497 en draai:"
Write-Host "  python .\ibkr_tws_probe.py --env-file .env.ibkr"
