$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'download-reid-models.ps1') -Encoders coca_l14_visual
if ($LASTEXITCODE) { throw 'CoCa ViT-L/14 visual extraction failed.' }
Write-Host "Standalone CoCa ViT-L/14 visual encoder created under $projectRoot/.data/models/reid/coca_l14_visual"
