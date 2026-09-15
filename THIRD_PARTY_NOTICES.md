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
<<<<<<< HEAD
=======
- CoCa ViT-L/14: `laion/CoCa-ViT-L-14-laion2B-s13B-b90k`, revision `74207cb7fde8eafc9864451ebd332fa8e75b150f`, used through OpenCLIP. The pinned checkpoint SHA-256 is `73725652298ad76ed2162caffdae96d8653a05d7a29b6281103e4df81d0ff8ea` and its byte size is `2554109637`.
- CoCa ViT-L/14 visual encoder: locally derived from only the pinned L/14 checkpoint's `visual.*` tensors. The deterministic 1,226,933,392-byte Safetensors artifact has SHA-256 `51e977d10c02b43c4c1dcbce07d3699c14c77ad4903291aba117b6c3ea45c728`; its manifest also records extraction and parent provenance. Redistribution remains subject to the same upstream terms.
- SigLIP2 Base Patch16 384: `google/siglip2-base-patch16-384`, revision `f775b65a79762255128c981547af89addcfe0f88`, Apache-2.0, loaded through Hugging Face Transformers.
- SigLIP2 visual encoder: locally derived from only the pinned SigLIP2 checkpoint's `vision_model.*` tensors. Its Safetensors manifest records the parent revision and SHA-256, and redistribution remains subject to the same upstream terms.
>>>>>>> 370b3fd (Initial ReID application)

Model weights and datasets retain their upstream terms. Encoder source files are copied with their existing copyright headers. Detection models and their runtime packages are not part of this standalone application.
