param(
    [ValidateSet('siglip', 'dinov2', 'fastreid', 'openvino', 'transreid', 'coca', 'coca_visual', 'coca_l14', 'coca_l14_visual', 'siglip2', 'siglip2_visual')]
    [string[]]$Encoders = @('siglip', 'dinov2', 'fastreid', 'coca', 'coca_visual', 'coca_l14', 'coca_l14_visual', 'siglip2', 'siglip2_visual'),
    [string]$TransReIDCheckpoint
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $projectRoot 'backend'
$reidPython = Join-Path $projectRoot '.runtime/reid-env/Scripts/python.exe'
$modelRoot = & $reidPython -c "from iris.config import Settings; s=Settings(); print(s.reid_model_dir or s.data_dir / 'models/reid')"
if ($LASTEXITCODE) { throw 'Run scripts/setup-reid.ps1 first.' }
$reidArguments = @('-m', 'iris.reid_setup', '--models', $modelRoot, '--encoders') + $Encoders
if ($TransReIDCheckpoint) { $reidArguments += @('--transreid-checkpoint', (Resolve-Path -LiteralPath $TransReIDCheckpoint).Path) }
& $reidPython @reidArguments
if ($LASTEXITCODE) { throw 'Download failed. Rerun to resume.' }
