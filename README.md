# IRIS ReID

A standalone local workspace for vehicle embeddings and identity matching. Upload one already cropped vehicle per image, choose its camera and class, and compare embedding models. The application has no detection or video pipeline.

## Start on Windows

Requires Python 3.10 and Node.js/npm. From this folder:

```powershell
./scripts/setup-reid.ps1 -Python 'C:/path/to/python.exe'
./scripts/start.ps1
```

Open http://127.0.0.1:8003. Setup creates a separate environment, installs pinned PyTorch/torchvision and ReID dependencies, and builds the frontend. The CUDA 11.8 wheels also support CPU execution; encoder device selection is automatic. No YOLO installation is required.

Five existing embedding models have been copied into `.data/models/reid`. Full CoCa and its independently loadable stripped visual tower are the sixth and seventh encoders. To install all models:

```powershell
./scripts/download-reid-models.ps1 -Encoders siglip,dinov2,fastreid,openvino,transreid,coca,coca_visual
./scripts/diagnostics-reid.ps1
```

TransReID installation also accepts `-TransReIDCheckpoint <official-checkpoint-path>`. Model downloads are explicit; inference uses local files.

## Development and configuration

`./scripts/start-dev.ps1` starts the API and Vite frontend on http://127.0.0.1:5176; stopping it stops the API process it launched. Copy `.env.example` to `.env` to override ports and paths. Relative paths resolve from this project's root, including when launched from another directory. Use `IRIS_HOST=127.0.0.1` for the local development proxy.

The backend lives in `backend/iris`, the React application in `frontend`, model files in `.data/models/reid`, and experiment records in `.data/reid.db` and `.data/reid/`. The database starts empty. The original RTMDet-Tiny app and its data are preserved. Runtime files, models and frontend dependencies are ignored by Git; include the model folder separately when transferring an offline installation. Recreate the environment after moving the project.

The **Compare two images** tab offers full CoCa criteria/fusion and a visual-encoder-only raw-cosine mode. Full mode reports deterministic color, HOG shape, LBP texture, gradient detail, constrained cargo/brand/viewpoint attributes, an editable combined score, and a separate experimental criteria score. Run `./scripts/extract-coca-visual.ps1` to recreate the portable 512D visual checkpoint from the pinned full model. Completed experiments provide side-by-side query/gallery comparison views and portable result packages. The Research page calculates labeled-folder precision, recall, F1, accuracy, identification metrics, and confusion matrices for every selected encoder.

## Verification

```powershell
./scripts/verify.ps1
./scripts/diagnostics-reid.ps1
```

Browser tests use installed Chrome and a disposable test database on port 8013. They exercise pair comparison, ingestion, matching, review, and exports with deterministic embedding fixtures. Diagnostics load every installed model checkpoint and verify its checksum and output dimension.

To additionally run actual model pipeline and one-epoch training/import/evaluation checks:

```powershell
$env:IRIS_REID_REAL_SMOKE = '1'
./.runtime/reid-env/Scripts/python.exe -m pytest backend/tests/test_reid_real.py backend/tests/test_reid_vehicle_real.py -v
Remove-Item Env:IRIS_REID_REAL_SMOKE
```

These synthetic tests verify mechanics, not vehicle recognition accuracy. See `LICENSE`, `NOTICE`, and `THIRD_PARTY_NOTICES.md` for attribution.
