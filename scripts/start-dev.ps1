$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$reidPython = Join-Path $projectRoot '.runtime/reid-env/Scripts/python.exe'
if (!(Test-Path -LiteralPath $reidPython)) { throw 'Run scripts/setup-reid.ps1 first.' }
$env:PYTHONPATH = Join-Path $projectRoot 'backend'
$backendProcess = Start-Process -FilePath $reidPython -ArgumentList '-m', 'iris' -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru
Push-Location (Join-Path $projectRoot 'frontend')
try { npm.cmd run dev } finally {
    Pop-Location
    if (!$backendProcess.HasExited) { Stop-Process -Id $backendProcess.Id }
}
