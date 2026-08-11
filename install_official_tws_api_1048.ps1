param(
    [string]$ProjectRoot = "C:\Users\alhar\Documents\Stocks",
    [string]$ApiVersion = "1048.01",
    [string]$DownloadUri = "https://interactivebrokers.github.io/downloads/TWS%20API%20Install%201048.01.msi"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host $Message -ForegroundColor Cyan
}

Write-Host ""
Write-Host "Official IBKR TWS API installer" -ForegroundColor Green
Write-Host "Project: $ProjectRoot"
Write-Host "API package: $ApiVersion"
Write-Host ""

if (-not (Test-Path $ProjectRoot)) {
    throw "Projectmap bestaat niet: $ProjectRoot"
}

$uri = [Uri]$DownloadUri
if ($uri.Scheme -ne "https" -or $uri.Host -ne "interactivebrokers.github.io") {
    throw "Download geblokkeerd: alleen de officiële HTTPS-host interactivebrokers.github.io is toegestaan."
}

$repairScript = Join-Path $ProjectRoot "repair_ibkr_phase0_v2.ps1"
if (-not (Test-Path $repairScript)) {
    throw "Reparatiescript ontbreekt: $repairScript"
}

$tempDir = Join-Path $env:TEMP "IBKR_TWS_API_$ApiVersion"
$msiPath = Join-Path $tempDir "TWS_API_Install_$ApiVersion.msi"

New-Item -ItemType Directory -Force -Path $tempDir | Out-Null

Write-Step "[1/6] Officiële IBKR MSI downloaden..."
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Invoke-WebRequest -Uri $DownloadUri -OutFile $msiPath -UseBasicParsing

$file = Get-Item $msiPath
if ($file.Length -lt 1MB) {
    throw "Gedownloade MSI is verdacht klein: $($file.Length) bytes"
}
Write-Host "Download: $msiPath"
Write-Host "Grootte: $([math]::Round($file.Length / 1MB, 2)) MB"

Write-Step "[2/6] Digitale handtekening controleren..."
$signature = Get-AuthenticodeSignature -FilePath $msiPath
Write-Host "Signature status: $($signature.Status)"
if ($signature.SignerCertificate) {
    Write-Host "Signer: $($signature.SignerCertificate.Subject)"
    Write-Host "Thumbprint: $($signature.SignerCertificate.Thumbprint)"
}

if ($signature.Status -ne "Valid") {
    throw "MSI-installatie geblokkeerd: digitale handtekening is niet geldig ($($signature.Status))."
}

Write-Step "[3/6] Interactieve Windows-installer openen..."
Write-Host "Volg de wizard en behoud de standaardinstallatiemap op C:\TWS API."
$process = Start-Process `
    -FilePath "msiexec.exe" `
    -ArgumentList "/i `"$msiPath`"" `
    -Wait `
    -PassThru

if ($process.ExitCode -notin @(0, 3010)) {
    throw "MSI-installatie faalde met exitcode $($process.ExitCode)."
}

if ($process.ExitCode -eq 3010) {
    Write-Host "De installer vraagt een herstart. Rond eerst deze controle af en herstart daarna Windows." -ForegroundColor Yellow
}

Write-Step "[4/6] Officiële Pythonbron verifiëren..."
$pythonClient = "C:\TWS API\source\pythonclient"
$setupPy = Join-Path $pythonClient "setup.py"
$pyproject = Join-Path $pythonClient "pyproject.toml"

if (-not (Test-Path $pythonClient)) {
    throw "Installatiepad ontbreekt na MSI-installatie: $pythonClient"
}
if (-not (Test-Path $setupPy) -and -not (Test-Path $pyproject)) {
    throw "pythonclient bestaat, maar setup.py/pyproject.toml ontbreekt: $pythonClient"
}

Write-Host "Python API gevonden: $pythonClient" -ForegroundColor Green

$versionFile = "C:\TWS API\API_VersionNum.txt"
if (Test-Path $versionFile) {
    Write-Host "Geïnstalleerde API-versie:"
    Get-Content $versionFile
}

Write-Step "[5/6] Phase 0-omgeving repareren..."
$repairProcess = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "`"$repairScript`"",
        "-ProjectRoot", "`"$ProjectRoot`"",
        "-OfficialApiPath", "`"$pythonClient`""
    ) `
    -Wait `
    -PassThru

if ($repairProcess.ExitCode -ne 0) {
    throw "Phase 0-reparatiescript faalde met exitcode $($repairProcess.ExitCode)."
}

Write-Step "[6/6] Installatie en imports verifiëren..."
$python = Join-Path $ProjectRoot ".venv-ibkr\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Python uit .venv-ibkr ontbreekt: $python"
}

& $python -c @'
import ibapi
from importlib import metadata
from pathlib import Path

print("ibapi import: GO")
print("ibapi version:", metadata.version("ibapi"))
print("ibapi module:", Path(ibapi.__file__).resolve())
'@

if ($LASTEXITCODE -ne 0) {
    throw "De officiële ibapi kon na installatie niet worden geïmporteerd."
}

& $python -m pip check
if ($LASTEXITCODE -ne 0) {
    throw "pip check rapporteert dependencyproblemen."
}

Write-Host ""
Write-Host "TWS API INSTALLATION GO" -ForegroundColor Green
Write-Host ""

$portOpen = Test-NetConnection 127.0.0.1 -Port 7497 -InformationLevel Quiet -WarningAction SilentlyContinue
if ($portOpen) {
    Write-Host "TWS paperpoort 7497 is bereikbaar. Read-only probe starten..."
    Push-Location $ProjectRoot
    try {
        & $python ".\ibkr_tws_probe.py" "--env-file" ".env.ibkr"
    }
    finally {
        Pop-Location
    }
}
else {
    Write-Host "TWS paperpoort 7497 luistert nog niet." -ForegroundColor Yellow
    Write-Host "Start TWS met Paper Trading en controleer:"
    Write-Host "  Edit -> Global Configuration -> API -> Settings"
    Write-Host "  Enable ActiveX and Socket Clients = aan"
    Write-Host "  Read-Only API = aan"
    Write-Host "  Socket Port = 7497"
    Write-Host ""
    Write-Host "Draai daarna:"
    Write-Host "  cd $ProjectRoot"
    Write-Host "  .\.venv-ibkr\Scripts\Activate.ps1"
    Write-Host "  python .\ibkr_tws_probe.py --env-file .env.ibkr"
}
