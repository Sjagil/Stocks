$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv-ibkr\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "PYTHON_ENVIRONMENT_UNAVAILABLE"
}

Push-Location $ProjectRoot
try {
    & $Python ".\main.py" macro update
    if ($LASTEXITCODE -ne 0) {
        throw "MACRO_UPDATE_FAILED"
    }
    & $Python ".\main.py" macro readiness
    if ($LASTEXITCODE -ne 0) {
        throw "MACRO_READINESS_FAILED"
    }
}
finally {
    Pop-Location
}
