# Third-party notices

The application source is licensed under Apache-2.0. Important runtime dependencies include:

- OpenCV and Pydantic: Apache-2.0.
- PyTorch: BSD-family license.
- FastAPI, Uvicorn, HTTPX, PyYAML, NumPy and psutil: permissive MIT/BSD-compatible licenses.
- React, Vite, TypeScript and Lucide: MIT.
- Official model checkpoints and user datasets remain subject to their respective terms.

Package lockfiles are the authoritative dependency inventory for a built release.

This standalone extraction retains the original project's LICENSE and NOTICE.
Embedding model revisions and checksums are recorded in `.data/models/reid/manifest.json`:

- SigLIP: `google/siglip-base-patch16-224`; local vision-only checkpoint and preprocessing configuration.
- DINOv2: `facebook/dinov2-small`; local checkpoint and preprocessing configuration.
- FastReID: pinned JDAI-CV/fast-reid source and official VeRi SBS R50-IBN weights. Its source license is included in `.data/models/reid/fast-reid/LICENSE`.
- OpenVINO: Open Model Zoo vehicle-reid-0001 ONNX model.
- TransReID: pinned damo-cv/TransReID source and VeRi checkpoint. Its source license is included in `.data/models/reid/transreid/source/LICENSE`.
- CoCa ViT-B/32: `laion/CoCa-ViT-B-32-laion2B-s13B-b90k`, revision `47aff38863cd40aa76d915bb04e4a3d8edf5824c`, used through OpenCLIP. The model card and OpenCLIP package declare MIT licensing; the exact checkpoint checksum is recorded in the local manifest.
- CoCa visual encoder: locally derived from only the pinned CoCa checkpoint's `visual.*` tensors. Its Safetensors manifest records the parent revision and SHA-256, and redistribution remains subject to the same upstream terms.

Model weights and datasets retain their upstream terms. Encoder source files are copied with their existing copyright headers. Detection models and their runtime packages are not part of this standalone application.
