$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$reidPython = Join-Path $projectRoot '.runtime/reid-env/Scripts/python.exe'
if (!(Test-Path -LiteralPath $reidPython)) { throw 'Run scripts/setup-reid.ps1 first.' }
$env:PYTHONPATH = Join-Path $projectRoot 'backend'
& $reidPython -m iris
if ($LASTEXITCODE) { throw 'ReID server stopped with an error.' }
