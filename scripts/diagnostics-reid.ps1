$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $projectRoot 'backend'
$reidPython = Join-Path $projectRoot '.runtime/reid-env/Scripts/python.exe'
& $reidPython -c "import importlib.util; assert all(importlib.util.find_spec(p) is None for p in ('ultralytics','mmdet','mmcv')), 'Unexpected detector package'; from iris.config import Settings; print(Settings().model_dump())"
if ($LASTEXITCODE) { throw 'Environment diagnostics failed.' }
$modelRoot = & $reidPython -c "from iris.config import Settings; s=Settings(); print(s.reid_model_dir or s.data_dir / 'models/reid')"
if ($LASTEXITCODE) { throw 'Could not resolve the model directory.' }
& $reidPython -m iris.reid_setup --models $modelRoot --verify --encoders siglip dinov2 fastreid openvino transreid coca coca_visual coca_l14 coca_l14_visual siglip2 siglip2_visual
if ($LASTEXITCODE) { throw 'Embedding verification failed.' }
