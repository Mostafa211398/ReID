$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location (Join-Path $projectRoot 'backend')
try {
    & (Join-Path $projectRoot '.runtime/reid-env/Scripts/python.exe') -m pytest -q
    if ($LASTEXITCODE) { throw 'Backend tests failed.' }
} finally { Pop-Location }
Push-Location (Join-Path $projectRoot 'frontend')
try {
    npm.cmd run build
    if ($LASTEXITCODE) { throw 'Frontend build failed.' }
    npm.cmd run test:e2e
    if ($LASTEXITCODE) { throw 'Browser tests failed.' }
} finally { Pop-Location }
