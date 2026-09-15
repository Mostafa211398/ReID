$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'download-reid-models.ps1') -Encoders coca_visual
if ($LASTEXITCODE) { throw 'CoCa visual encoder extraction failed.' }
Write-Host "Standalone CoCa visual encoder created under $projectRoot/.data/models/reid/coca_visual"
