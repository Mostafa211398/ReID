param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$reidEnvironment = Join-Path $projectRoot '.runtime/reid-env'
& $Python -c "import sys; assert sys.version_info[:2] == (3, 10), 'Python 3.10 is required'"
if ($LASTEXITCODE) { throw 'Select Python 3.10 using -Python <path-to-python.exe>.' }
& $Python -m venv $reidEnvironment
if ($LASTEXITCODE) { throw 'Could not create the standalone environment.' }
$reidPython = Join-Path $reidEnvironment 'Scripts/python.exe'
& $reidPython -m pip install -r (Join-Path $projectRoot 'backend/requirements.txt')
if ($LASTEXITCODE) { throw 'Dependency installation failed.' }
& $reidPython -m pip install -e (Join-Path $projectRoot 'backend')
if ($LASTEXITCODE) { throw 'Backend installation failed.' }
Push-Location (Join-Path $projectRoot 'frontend')
try {
    npm.cmd ci
    if ($LASTEXITCODE) { throw 'Frontend installation failed.' }
    npm.cmd run build
    if ($LASTEXITCODE) { throw 'Frontend build failed.' }
} finally { Pop-Location }
Write-Host 'Ready. Run scripts/start.ps1. Use download-reid-models.ps1 only for missing models.'
