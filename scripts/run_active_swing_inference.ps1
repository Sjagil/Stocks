param()

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv-ibkr\Scripts\python.exe"
$Runner = Join-Path $ProjectRoot "src\stocks\ai\active_swing_fast_path.py"
$OutputRoot = Join-Path $ProjectRoot "output\ai\decision-intelligence"
$Stdout = Join-Path $OutputRoot "active-swing-fast-path.stdout.json"
$Stderr = Join-Path $OutputRoot "active-swing-fast-path.stderr.log"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python runtime not found: $Python"
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
Push-Location $ProjectRoot
try {
    & $Python $Runner --project-root $ProjectRoot 1> $Stdout 2> $Stderr
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}

