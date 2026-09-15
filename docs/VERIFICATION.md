# Standalone verification

Verified on Windows with the isolated Python 3.10 environment in `.runtime/reid-env`.

- Dependency integrity: `pip check` passed.
- Backend regression suite: 90 passed; 25 opt-in real-model checks skipped by default.
- CoCa smoke checks: the pinned 1,014,488,932-byte checkpoint matched SHA-256 `df49af430529127b5f019825ddab06867429a63cec4006e34c3cb0bc4c8b4a04`; real inference produced finite normalized 512D image vectors, a normalized 24D semantic vector, and a complete `criteria-v1` report for two chosen crops. The stripped visual checkpoint matched full CoCa output within `1e-5` while running from an isolated directory without the parent model.
- CoCa ViT-L/14 smoke checks: the pinned 2,554,109,637-byte checkpoint matched SHA-256 `73725652298ad76ed2162caffdae96d8653a05d7a29b6281103e4df81d0ff8ea`; real CPU inference produced finite, repeatable, normalized 768D vectors, normalized semantic prompt features, and a complete `criteria-v1` report. The deterministic 1,226,933,392-byte standalone checkpoint matched SHA-256 `51e977d10c02b43c4c1dcbce07d3699c14c77ad4903291aba117b6c3ea45c728` and matched full-model image features within `1e-5` in an isolated directory.
- SigLIP2 integration checks pin the 1,501,968,264-byte `google/siglip2-base-patch16-384` checkpoint at revision `f775b65a79762255128c981547af89addcfe0f88` and SHA-256 `ed72c0ace85020ae610fc817c2538b9cae5a477b012a50859c60af5b3ad30857`. The extracted 372,729,048-byte visual Safetensors file has SHA-256 `0e1821175882644dbe8fc5d6d1c8c41588478b6fe8e3b1c8ed6416ca8f3925b7`. Real inference produced deterministic normalized 768D vectors, the isolated visual tower matched the full model within `1e-5`, and full prompt semantics produced normalized 24D vectors.
- Real-model suite covers absence of detector packages; encoder dimensions, finite normalized outputs and repeatability; actual crop uploads, matching and retries; and one-epoch DINOv2, SigLIP and FastReID training followed by checkpoint import and evaluation.
- Labeled-folder benchmark: the current `chosen crops` directory was verified as 10 identities, 46 unique images/viewpoints, 1,035 image pairs, and 45 eligible identification queries. The opt-in real benchmark is configured to load all eleven registered encoders.
- Model-grouped comparison export: passed background preparation, progress/status, cancellation and retry, cache reuse, missing-artifact regeneration, threshold invalidation, the 50,000-card limit, corruption-safe atomic output, per-model JPEG/CSV layout, and deletion cleanup without running an embedding stage.
- Chrome workflow: passed full CoCa B/32, CoCa L/14, and SigLIP2 structured criteria, all three visual-only raw-cosine modes, previews, vector JSON, scoring-only recalculation and deletion, plus crop experiments, review, comparison views, an eleven-model folder benchmark, model-grouped comparison-image download, ZIP download, research validation and deletion.
- Frontend production build and TypeScript checks: passed.
- Installed application: frontend and ReID status endpoint returned HTTP 200; default ports are 8003 and 5176.
- Model inventory: eleven embedding spaces are registered. The local manifest remains the source of truth for which checkpoint pairs have been downloaded and for every installed file checksum.
- All nine PowerShell scripts parsed successfully.
- All 37 copied original source files retained their pre-extraction SHA-256 hashes. No edits were made to the original application.

The real-model tests use synthetic images to verify execution and data integrity, not recognition accuracy. Upstream Starlette and PyTorch emitted deprecation warnings; tests passed.

See the README for commands to repeat these checks. Test artifacts are stored separately from the new application's fresh database.
