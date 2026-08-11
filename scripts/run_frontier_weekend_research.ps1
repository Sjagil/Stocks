$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv-ibkr\Scripts\python.exe"
$runner = Join-Path $projectRoot "scripts\run_frontier_weekend_research.py"
$logRoot = Join-Path $projectRoot "output\analysis\themes"

New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
& $python $runner --no-bars 1> (Join-Path $logRoot "weekend-task.stdout.log") 2> (Join-Path $logRoot "weekend-task.stderr.log")
exit $LASTEXITCODE
