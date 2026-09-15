$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'download-reid-models.ps1') -Encoders siglip2_visual
if ($LASTEXITCODE) { throw 'SigLIP2 visual encoder extraction failed.' }
Write-Host "Standalone SigLIP2 visual encoder created under $projectRoot/.data/models/reid/siglip2_visual"
