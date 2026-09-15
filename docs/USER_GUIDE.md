# ReID user guide

## Create an experiment

1. Start the application with `scripts/start.ps1` and open http://127.0.0.1:8003.
2. Create a site and enter camera IDs, one per line. Sites keep separate identity galleries.
3. Enter an experiment name and select embedding models. The site's active encoder is always included; DINOv2 is the default. SigLIP and FastReID are also selected initially.
4. Add JPEG, PNG, WebP, or BMP crops. Each image must contain one already cropped vehicle. Choose its camera and select **Car** or **Truck** for every file.
5. Optionally provide the capture time; leave it blank when unknown. Click **Upload & run experiment**.

Up to 32 images are accepted per experiment. The same file cannot appear twice in one experiment. Orientation metadata is applied and images are converted to RGB. Each crop becomes exactly one observation; the app neither locates vehicles nor infers vehicle classes.

## Matching and review

Experiments ingest crops, compute embeddings, compare saved identities, and generate labeled images plus CSV/JSON results. Eleven models are supported: SigLIP Base (768 dimensions), DINOv2 Small (384), FastReID VeRi (2048), OpenVINO vehicle-reid-0001 (512), TransReID VeRi (3840), full CoCa ViT-B/32 and its stripped visual encoder (512 each), full CoCa ViT-L/14 and its stripped visual encoder (768 each), and full SigLIP2 Base Patch16 384 and its stripped visual encoder (768 each). All CoCa and SigLIP2 variants are inference/evaluation only.

## Compare two images with CoCa B/32, CoCa L/14, or SigLIP2

Open **Compare two images**, select exactly two already cropped vehicle images, and choose **Embed and compare**. The saved job normalizes EXIF orientation and RGB color without creating observations, identities, gallery references, or datasets. Results show:

- the raw cosine between the two normalized image embeddings (512D for CoCa B/32; 768D for CoCa L/14 and SigLIP2);
- a 72D foreground-aware color similarity;
- an 8,100D HOG shape similarity from a 128×128 aspect-preserving grayscale letterbox;
- a 24D VLM semantic similarity covering ten coarse vehicle types, eight viewpoints, three plate-visibility states, and three logo-visibility states.

Choose a **Full … criteria and fusion** mode for the four-branch result plus a separate `criteria-v1` report. It calculates HOG shape, LBP texture, gradient detail, dominant colors, hue shift, and scale. Fixed prompt vocabularies describe cargo structure, cage pattern and condition, barrel wrapping and layout, outline, common truck brands, viewpoint, plate presentation, regional colors, and human presence. Every selected label includes confidence and its complete prompt distribution in the JSON download. Exact model lettering and plate characters are always `unknown` because OCR is unavailable, and the report never claims that two plates are identical.

The criteria score uses fixed weights: 35% appearance, 15% shape, 10% texture, 5% detail, 10% color, 10% cargo consistency, and 5% each for vehicle type, brand, and plate presentation. Missing deterministic features are removed and the remaining weights are normalized. Bands are high at 0.80 or above, moderate at 0.60 or above, and low below 0.60. This experimental diagnostic does not replace the normal fusion decision.

Choose a **visual encoder only** mode to load only the stripped image tower. This mode calculates raw cosine between two normalized vectors (512D for CoCa B/32; 768D for CoCa L/14 and SigLIP2) and uses a configurable threshold from -1 to 1. It does not run color, HOG, or text prompts. The same encoder appears in experiments, folder benchmarks, evaluation, and gallery promotion.

The combined score defaults to 60% normalized appearance, 20% color, 15% shape, and 5% semantics. All four editable weights must total 100%. If color or shape cannot be extracted, its weight is removed and the available weights are renormalized. The default same-vehicle threshold is 0.75, equality counts as a match, and the displayed decision is explicitly experimental and uncalibrated. Normal experiment matching continues to use its existing appearance-plus-color policy.

Changing weights or threshold and choosing **Apply scoring without re-embedding** reuses the saved vectors. **Download vectors JSON** contains all four vectors, dimensions/norms, model and scoring fingerprints, semantic and criteria distributions, component scores, effective weights, threshold, decision, and safe relative image names. It excludes local filesystem paths and model weights. Brand output is limited to the documented fixed vocabulary; no OCR is performed.

Pair comparisons use the normal queue, cancellation, retry, restart recovery, and deletion behavior. History is isolated per selected site. Install the desired pairs with `scripts/download-reid-models.ps1 -Encoders coca,coca_visual,coca_l14,coca_l14_visual,siglip2,siglip2_visual`, or recreate a derived tower with its extraction script. Inference remains offline after installation, and each visual-only directory can run without its full parent checkpoint. CoCa L/14 automatically retries its complete embedding or prompt stage on CPU after a CUDA out-of-memory error.

### Complete results package

For a completed experiment, **Download all results (.zip)** exports this portable package:

```text
manifest.json
README.txt
data/
  observations.csv
  observations.json
  cosine_comparisons.csv
  cosine_comparisons.json
  results.csv
  results.json
images/
  queries/<observation-id>.jpg
  references/<vehicle-id>/<reference-observation-id>_<crop-index>.jpg
  annotated/<encoder-id>/<original-filename>.jpg
```

`observations` has one row per uploaded observation and selected encoder. `detected_vehicle_id` is the final site ID shown for the observation. `encoder_assigned_vehicle_id` is the individual encoder's assignment and can be empty for a comparison-only encoder. The decision status is `new`, `automatic`, `review`, or `reviewed`; provenance records whether the result came from enrollment, experimental or calibrated matching, or human confirmation. Candidate, selected-reference, threshold, review, appearance, color, and combined-score fields preserve the decision context.

`cosine_comparisons` has one row for every eligible trusted gallery crop used by the matching policy. Eligibility requires the same site, vehicle class, and encoder fingerprint; self/duplicate-source and simultaneous same-camera identities are excluded. Matching retains the latest eight trusted reference crops per vehicle ID. `cosine_similarity` is the unmodified dot product of L2-normalized query and reference embeddings, and `cosine_rank` orders that value within one query crop and encoder. `combined_score` remains separate because it can include color. Flags identify the winning candidate identity, the assigned identity, and the exact decision reference.

An experiment with no prior gallery still contains observation rows, query and annotated images, and empty comparison files with CSV headers. Image names are sanitized and all paths are relative to the ZIP; local paths, embeddings, and model weights are excluded. **Refresh matching & results** regenerates the package after review or gallery changes. Downloading is blocked while work or deletion is pending and when reference deletion has made results stale.

Inspect candidate references and compare appearance, color, and combined scores. Scores describe similarity, not accuracy probabilities. Default matching combines 75% appearance with 25% color; unavailable color falls back to appearance. Only matching vehicle classes can share identities.

For a completed experiment, select **View comparisons** on an observation to keep its query crop beside every eligible gallery reference scored for the selected encoder. **Experiment image comparison table** shows the same audit across all queries. Filter by query, encoder, candidate vehicle ID, or decision role. Each pair includes raw cosine and rank, color and combined scores, camera/class context, detected and assigned IDs, and flags for the winning identity and exact decision reference. Click either image to inspect the full crop. These views read the saved result package and do not rerun an encoder; use **Refresh matching & results** if the package is stale or predates comparison exports.

**Confirm identity** associates an observation with an existing site ID or creates a new ID. Automatic matches do not become true training labels. Expand **Optional identity labels** to label actual vehicles consistently across cameras or exclude observations from evaluation. Refresh matching after review to update results.

**Site & cameras** controls matching thresholds, color weight, optional travel-time rules, and the active encoder. Defaults are automatic matching at 0.85, a 0.05 lead over the next candidate, and new identity creation below 0.65. These are experimental defaults; evaluate with representative labeled crops. Optional timing rules apply only when capture times are known.

Cancel or retry jobs from the results panel. Retrying retains completed ingestion and embedding stages; selecting additional encoders computes their features without duplicating observations. Interrupted running jobs become failed on restart and can be retried.

## Evaluation and optional training

### Folder benchmark

Open **Accuracy & optional training** and use **Folder benchmark** to select a parent directory such as `chosen crops`. Each immediate child directory is a ground-truth truck identity, so `vehicle_5/*.jpg` is treated as one truck seen from several viewpoints. In the current naming scheme, `t5` repeats the identity and each distinct `fNNNNN` file is an independent viewpoint. The preflight summary reports image, identity, viewpoint, singleton, invalid-file, and duplicate counts before anything is uploaded.

Choose the embedding models and an operational cosine threshold, then run the benchmark. The default threshold is the site's matching threshold. All measurements use only the raw dot product of L2-normalized embeddings; color is intentionally excluded. The model table reports two related tasks:

- Pairwise verification labels every unordered image pair as same-truck or different-truck. A pair is predicted as the same truck when its cosine is at least the configured threshold. Precision is `TP / (TP + FP)`, recall is `TP / (TP + FN)`, F1 is the harmonic mean of precision and recall, and accuracy is `(TP + TN) / all pairs`.
- Leave-one-viewpoint-out identification removes the query image, finds its highest-cosine reference, and predicts that reference folder. The table reports top-1 accuracy, Rank-1, mAP, and macro precision/recall/F1. An identity with only one image cannot be a query because no positive reference remains, but its image is retained as a possible negative reference.

**Best F1 threshold** is an exploratory threshold optimized on the same uploaded images. It is useful for inspecting score separation but is not held-out evidence. Use the configured-threshold columns for the actual site cutoff, and use a separate labeled dataset for final model selection. Change **Recalculate operational threshold** to rebuild all threshold-dependent metrics and downloads from cached cosine scores without running the encoders again.

**Download benchmark (.zip)** contains `summary`, `pairwise_comparisons`, `identification_predictions`, `per_identity_metrics`, and `confusion_matrix` in CSV and JSON forms, plus one normalized copy of every image under `images/<vehicle-folder>/`. It includes model fingerprints and exact cosine values but excludes embeddings, weights, and local filesystem paths. Cancelling and retrying retain completed model embeddings; deleting a benchmark removes only its isolated images, features, and reports.

For a visual audit, choose **Prepare comparison images** after the benchmark completes. This starts a cancellable background export from the saved results and does not rerun any encoder. When ready, **Download model comparison images (.zip)** contains a named folder for every model, one JPEG card for every scored pair in descending cosine order, and a full-precision `comparisons.csv` inside each model folder. Each card matches the website view: both crops, identities, filenames, viewpoints, cosine, ground truth, configured and exploratory decisions and thresholds, metric eligibility, and encoder fingerprint. Exports above 50,000 cards are rejected; use a smaller benchmark folder. Changing the operational threshold invalidates a prepared image export, while repeated preparation at the same threshold reuses it.

Select **View compared images** in completed benchmark results to inspect those pairwise rows visually. It shows both labeled images, their vehicle folders and viewpoints, raw cosine, configured-threshold and exploratory best-threshold predictions, strict-metric eligibility, and encoder fingerprint. Filter by encoder, vehicle folder, same/different-truck ground truth, or strict/excluded status. The viewer is paginated and uses the normalized images saved with that benchmark.

Open **Accuracy & optional training**, select labeled experiments, and create a dataset snapshot. At least six labeled identities are required. Use different crops of each vehicle across cameras so evaluation has both positive and negative examples. Duplicate image content must not carry contradictory labels.

Snapshots divide identities into disjoint training, validation, and test groups. Compare models on the same snapshot and split. Validation supports model and threshold selection; reserve the test split for final evaluation. Appearance metrics and combined color metrics are shown separately.

SigLIP, DINOv2, and FastReID support optional fine-tuning. Training needs at least two identities with two distinct crops each in the training partition and eligible cross-camera pairs in validation. OpenVINO and TransReID support inference and evaluation only. TransReID's unseen-camera mode is experimental.

Train locally or export a standalone training bundle for another machine. Extract the bundle, follow its README, package the resulting checkpoint, and import the ZIP under **Import trained encoder package**. Choose **Use encoder & rebuild site gallery** to promote an encoder. Exported bundles contain crops and labels; the app does not send them to an external service.

## Deletion and troubleshooting

**Delete** previews affected observations, files, and identities before permanent removal. Shared IDs are preserved; dependent experiments may require refreshed matching. Research datasets or trained models can block deletion. If cleanup was interrupted, retry deletion before making other changes.

- A missing model is installed with `scripts/download-reid-models.ps1 -Encoders <name>`; use `scripts/diagnostics-reid.ps1` to verify local checkpoints and dependencies.
- Invalid images, missing vehicle classes, and video files are rejected before creating a job.
- On TransReID CUDA memory exhaustion, embedding restarts on CPU. Fine-tuning memory failures require freeing GPU memory or exporting training to another machine.
- Worker logs appear under `.data/reid/<job-id>/`. Run `scripts/verify.ps1` for automated application checks.

## Experiment API

`POST /api/reid/experiments` accepts multipart `files` and a JSON `config` field:

```json
{
  "name": "Morning crops",
  "site_id": "SITE_ID",
  "encoders": ["dinov2", "siglip"],
  "clips": [{"camera_id": "CAM_01", "class_name": "truck", "media_type": "image"}]
}
```

Each file needs a corresponding `clips` entry. `class_name` is required; `media_type` defaults to `image`. Optional `start_time` requires a timezone, and `offset_seconds` adjusts that timestamp. Detector fields (`model_id`, `confidence`, `vehicle_classes`) and video input are rejected. Existing observation routes retain `/tracks/` in their URLs for API continuity; detector, video, preview-stream, and split routes are absent.
