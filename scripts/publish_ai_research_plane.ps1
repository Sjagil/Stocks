$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$SourceRoot = Join-Path $ProjectRoot "src"
$Python = Join-Path $ProjectRoot ".venv-ibkr\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Canonical project Python is missing: $Python"
}

Push-Location $ProjectRoot
try {
    & $Python -c "import runpy,sys; sys.path.insert(0, r'$SourceRoot'); runpy.run_module('stocks.ai', run_name='__main__')"
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
