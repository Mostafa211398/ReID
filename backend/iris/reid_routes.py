from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import shutil
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image, ImageOps
from PIL.Image import open as open_image

from .utils import IMAGE_EXTENSIONS
from .reid_benchmark import parse_folder_paths, safe_part
from .reid_schemas import BenchmarkInput, BenchmarkThresholdInput, PairComparisonInput, PairScoringInput, RetryInput
from .utils import sha256_file
from .reid_schemas import AnnotationInput, DatasetInput, EvaluationInput, ExperimentInput, PromotionInput, ReviewInput, SiteInput, TrainingInput


def reid_router(service, save_upload, safe_filename):
    router = APIRouter(prefix="/api/reid", tags=["Vehicle ReID"])

    @router.get("/status")
    def status():
        return {"environment_ready": Path(service.python).is_file(), "encoders": service.encoders(), "training_optional": True}

    @router.get("/sites")
    def sites():
        return service.all("site")

    @router.post("/sites", status_code=201)
    def create_site(value: SiteInput):
        return service.create_site(value)

    @router.put("/sites/{site_id}")
    def update_site(site_id: str, value: SiteInput):
        return service.update_site(site_id, value)

    @router.get("/sites/{site_id}/identities")
    def identities(site_id: str):
        service.get("site", site_id)
        return [v for v in service.all("identity") if v["site_id"] == site_id]

    @router.post("/sites/{site_id}/encoder", status_code=202)
    def promote(site_id: str, value: PromotionInput):
        return service.promote(site_id, value.encoder_id)

    @router.get("/jobs")
    def jobs():
        return service.all("job")

    @router.get("/jobs/{job_id}")
    def job(job_id: str):
        return service.get("job", job_id)

    @router.get("/experiments/{experiment_id}/deletion-preview")
    def deletion_preview(experiment_id: str):
        return service.deletion_preview(experiment_id)

    @router.delete("/experiments/{experiment_id}")
    def delete_experiment(experiment_id: str):
        return service.delete_experiment(experiment_id)

    @router.post("/experiments", status_code=202)
    async def upload(files: Annotated[list[UploadFile], File()], config: Annotated[str, Form()]):
        try:
            value = ExperimentInput.model_validate_json(config)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if len(files) != len(value.clips):
            raise ValueError("Every uploaded file requires a camera entry")
        service.get("site", value.site_id)
        folder = service.root / "uploads" / uuid4().hex
        created, clips = [], []
        try:
            for file, metadata in zip(files, value.clips):
                name = safe_filename(file.filename or "crop.jpg")
                extensions = IMAGE_EXTENSIONS
                if Path(name).suffix.lower() not in extensions:
                    raise ValueError(f"Select a supported {metadata.media_type} file")
                key = uuid4().hex
                destination = folder / (key + Path(name).suffix.lower())
                await save_upload(file, destination, int(service.settings.max_upload_gb * 1024**3))
                created.append(destination)
                if metadata.media_type == "image":
                    from PIL import Image
                    try:
                        with open_image(destination) as image:
                            image.verify()
                        with open_image(destination) as image:
                            image.load()
                    except (OSError, ValueError, Image.DecompressionBombError) as exc:
                        raise ValueError("Could not decode the uploaded image") from exc
                clips.append({**metadata.model_dump(mode="json"), "id": key, "filename": name, "path": str(destination), "sha256": sha256_file(destination), "start_epoch": metadata.start_time.timestamp() + metadata.offset_seconds if metadata.start_time else None})
            if len({c["sha256"] for c in clips}) != len(clips):
                raise ValueError("The same file was uploaded more than once in this experiment")
            return service.create_experiment(value, clips)
        except Exception:
            for path in created:
                path.unlink(missing_ok=True)
            raise
        finally:
            for file in files:
                await file.close()

    @router.post("/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        return service.cancel(job_id)

    @router.post("/jobs/{job_id}/retry", status_code=202)
    def retry(job_id: str, value: RetryInput | None = None):
        return service.retry(job_id, value.encoders if value else None)

    @router.get("/jobs/{job_id}/events")
    async def events(job_id: str, request: Request):
        service.get("job", job_id)
        async def stream():
            while not await request.is_disconnected():
                value = service.get("job", job_id)
                yield f"data: {json.dumps(value)}\n\n"
                if value["state"] not in ("running", "queued"):
                    break
                await asyncio.sleep(.5)
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @router.get("/jobs/{job_id}/artifacts/{name:path}")
    def artifact(job_id: str, name: str):
        path = service.artifact(job_id, name)
        return FileResponse(path, filename=path.name)

    @router.get("/pair-comparisons")
    def pair_comparisons():
        return service.pair_comparisons()

    @router.post("/pair-comparisons", status_code=202)
    async def create_pair_comparison(files: Annotated[list[UploadFile], File()], config: Annotated[str, Form()]):
        try:
            value = PairComparisonInput.model_validate_json(config)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if len(files) != 2:
            raise ValueError("Select exactly two vehicle images")
        service.get("site", value.site_id)
        job_id = uuid4().hex
        directory, incoming, image_root = service.root / job_id, service.root / job_id / ".incoming", service.root / job_id / "images"
        remaining = int(service.settings.max_upload_gb * 1024**3)
        images, hashes = [], set()
        try:
            for index, file in enumerate(files):
                supplied_name = file.filename or f"image-{index + 1}.jpg"
                if any(character in supplied_name for character in ("/", "\\", "\0")) or supplied_name in (".", ".."):
                    raise ValueError("Image filenames must not contain a path")
                name = safe_filename(supplied_name)
                suffix = Path(name).suffix.lower()
                if suffix not in IMAGE_EXTENSIONS:
                    raise ValueError("Select JPEG, PNG, WebP, or BMP vehicle images")
                raw = incoming / f"{index}{suffix}"
                size = await save_upload(file, raw, remaining)
                remaining -= size
                try:
                    with open_image(raw) as original:
                        image = ImageOps.exif_transpose(original).convert("RGB")
                        pixel_hash = hashlib.sha256(f"{image.width}x{image.height}:RGB:".encode() + image.tobytes()).hexdigest()
                        if pixel_hash in hashes:
                            raise ValueError("Choose two different image files")
                        hashes.add(pixel_hash)
                        side = "first" if index == 0 else "second"
                        image_root.mkdir(parents=True, exist_ok=True)
                        destination = image_root / f"{side}.jpg"
                        image.save(destination, format="JPEG", quality=95)
                except ValueError:
                    raise
                except (OSError, Image.DecompressionBombError) as exc:
                    raise ValueError("Could not decode one of the supplied vehicle images") from exc
                images.append({"side": side, "filename": name, "stored_name": destination.name,
                               "path": str(destination), "sha256": sha256_file(destination)})
            return service.create_pair_comparison(value, images, job_id)
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        finally:
            for file in files:
                await file.close()

    @router.put("/pair-comparisons/{comparison_id}/scoring")
    def update_pair_scoring(comparison_id: str, value: PairScoringInput):
        return service.update_pair_scoring(comparison_id, value)

    @router.get("/pair-comparisons/{comparison_id}/results.json")
    def pair_results(comparison_id: str):
        job, path = service.pair_result(comparison_id)
        return FileResponse(path, filename=f'reid-{safe_filename(job["name"])}-comparison.json', media_type="application/json")

    @router.get("/pair-comparisons/{comparison_id}/images/{side}")
    def pair_image(comparison_id: str, side: Literal["first", "second"]):
        return FileResponse(service.pair_image(comparison_id, side), media_type="image/jpeg")

    @router.get("/pair-comparisons/{comparison_id}")
    def pair_comparison(comparison_id: str):
        return service.pair_comparison(comparison_id)

    @router.delete("/pair-comparisons/{comparison_id}")
    def delete_pair_comparison(comparison_id: str):
        return service.delete_pair_comparison(comparison_id)

    @router.get("/experiments/{experiment_id}/tracks")
    def tracks(experiment_id: str):
        service.get("job", experiment_id)
        return service.tracks(experiment_id)

    @router.get("/experiments/{experiment_id}/results.zip")
    def complete_results(experiment_id: str):
        job, path = service.results_package(experiment_id)
        name = safe_filename(job["name"]).removesuffix(".zip")
        return FileResponse(path, filename=f"reid-{name}-results.zip", media_type="application/zip")

    @router.get("/experiments/{experiment_id}/comparisons")
    def comparisons(experiment_id: str, observation_id: str | None = None, encoder_id: str | None = None,
                    candidate_vehicle_id: str | None = None,
                    decision_flag: Literal["all", "winning", "assigned", "decision-reference"] = "all",
                    offset: Annotated[int, Query(ge=0)] = 0,
                    limit: Annotated[int, Query(ge=1, le=100)] = 24):
        return service.experiment_comparisons(experiment_id, observation_id, encoder_id, candidate_vehicle_id,
                                              decision_flag, offset, limit)

    @router.get("/tracks/{track_id}/crops/{index}")
    def crop(track_id: str, index: int):
        track = service.get("track", track_id)
        if not 0 <= index < len(track["crops"]):
            raise KeyError(index)
        return FileResponse(track["crops"][index]["path"], media_type="image/jpeg")

    @router.get("/tracks/{track_id}/image")
    def track_image(track_id: str):
        track = service.get("track", track_id)
        if track.get("media_type", "image") != "image":
            raise KeyError(track_id)
        return FileResponse(service.root / track["experiment_id"] / track["clip_id"] / "preview.jpg", media_type="image/jpeg")

    @router.put("/tracks/{track_id}/annotation")
    def annotation(track_id: str, value: AnnotationInput):
        return service.annotate(track_id, value)

    @router.put("/tracks/{track_id}/review")
    def review(track_id: str, value: ReviewInput):
        return service.review(track_id, value)

    @router.get("/datasets")
    def datasets():
        return [{k: v for k, v in d.items() if k != "rows"} for d in service.all("dataset")]

    @router.post("/datasets", status_code=201)
    def snapshot(value: DatasetInput):
        dataset = service.create_dataset(value)
        return {k: v for k, v in dataset.items() if k != "rows"}

    @router.post("/evaluations", status_code=202)
    def evaluation(value: EvaluationInput):
        return service.evaluate(value)

    @router.get("/evaluations")
    def evaluations():
        return service.all("evaluation")

    @router.get("/benchmarks")
    def benchmarks():
        return service.benchmarks()

    @router.get("/benchmarks/{benchmark_id}")
    def benchmark(benchmark_id: str):
        return service.benchmark(benchmark_id)

    @router.post("/benchmarks", status_code=202)
    async def create_benchmark(files: Annotated[list[UploadFile], File()], config: Annotated[str, Form()]):
        try:
            value = BenchmarkInput.model_validate_json(config)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if len(files) != len(value.items):
            raise ValueError("Every benchmark image requires one relative path")
        service.get("site", value.site_id)
        metadata = parse_folder_paths([item.relative_path for item in value.items])
        identities = {item["identity"] for item in metadata}
        if len(identities) < 2:
            raise ValueError("A benchmark needs at least two vehicle folders")
        job_id = uuid4().hex
        directory = service.root / job_id
        incoming, image_root = directory / ".incoming", directory / "images"
        remaining = int(service.settings.max_upload_gb * 1024**3)
        items, hashes, archive_names = [], set(), set()
        try:
            for index, (file, item) in enumerate(zip(files, metadata)):
                suffix = Path(item["filename"]).suffix.lower()
                raw = incoming / f"{index}{suffix}"
                size = await save_upload(file, raw, remaining)
                remaining -= size
                try:
                    with open_image(raw) as original:
                        image = ImageOps.exif_transpose(original).convert("RGB")
                        pixel_hash = hashlib.sha256(f"{image.width}x{image.height}:RGB:".encode() + image.tobytes()).hexdigest()
                        if pixel_hash in hashes:
                            raise ValueError("The benchmark contains duplicate image content")
                        hashes.add(pixel_hash)
                        key = uuid4().hex
                        stored_name = key + suffix
                        destination = image_root / stored_name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        options = {"quality": 95} if suffix in (".jpg", ".jpeg", ".webp") else {}
                        image.save(destination, **options)
                        image.close()
                except (OSError, ValueError, Image.DecompressionBombError) as exc:
                    if isinstance(exc, ValueError) and "duplicate image" in str(exc):
                        raise
                    raise ValueError(f'Could not decode benchmark image {item["filename"]}') from exc
                finally:
                    raw.unlink(missing_ok=True)
                filename = safe_filename(item["filename"])
                archive_path = f'images/{safe_part(item["identity"])}/{filename}'
                if archive_path.casefold() in archive_names:
                    archive_path = f'images/{safe_part(item["identity"])}/{key}_{filename}'
                archive_names.add(archive_path.casefold())
                items.append({**item, "id": key, "stored_name": stored_name,
                              "source_group": item["source_group"] or key, "archive_path": archive_path,
                              "sha256": pixel_hash})
            if not any(a["identity"] == b["identity"] for index, a in enumerate(items) for b in items[index + 1:]):
                raise ValueError("A benchmark needs at least one same-vehicle image pair")
            shutil.rmtree(incoming, ignore_errors=True)
            return service.create_benchmark(value, items, job_id)
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        finally:
            for file in files:
                await file.close()

    @router.put("/benchmarks/{benchmark_id}/threshold")
    def update_benchmark_threshold(benchmark_id: str, value: BenchmarkThresholdInput):
        return service.update_benchmark_threshold(benchmark_id, value.threshold)

    @router.get("/benchmarks/{benchmark_id}/results.zip")
    def benchmark_results(benchmark_id: str):
        job, path = service.benchmark_package(benchmark_id)
        name = safe_filename(job["name"]).removesuffix(".zip")
        return FileResponse(path, filename=f"reid-{name}-benchmark.zip", media_type="application/zip")

    @router.post("/benchmarks/{benchmark_id}/comparison-images", status_code=202)
    def prepare_benchmark_comparison_images(benchmark_id: str):
        return service.prepare_benchmark_comparison_images(benchmark_id)

    @router.get("/benchmarks/{benchmark_id}/comparison-images.zip")
    def benchmark_comparison_images(benchmark_id: str):
        job, path = service.benchmark_comparison_images_package(benchmark_id)
        name = safe_filename(job["name"]).removesuffix(".zip")
        return FileResponse(path, filename=f"reid-{name}-model-comparison-images.zip", media_type="application/zip")

    @router.get("/benchmarks/{benchmark_id}/comparisons")
    def benchmark_comparisons(benchmark_id: str, encoder_id: str | None = None, identity: str | None = None,
                              pair_type: Literal["all", "same", "different"] = "all",
                              eligibility: Literal["all", "strict", "excluded"] = "all",
                              offset: Annotated[int, Query(ge=0)] = 0,
                              limit: Annotated[int, Query(ge=1, le=100)] = 24):
        return service.benchmark_comparisons(benchmark_id, encoder_id, identity, pair_type, eligibility, offset, limit)

    @router.get("/benchmarks/{benchmark_id}/images/{item_id}")
    def benchmark_image(benchmark_id: str, item_id: str):
        return FileResponse(service.benchmark_image(benchmark_id, item_id))

    @router.delete("/benchmarks/{benchmark_id}")
    def delete_benchmark(benchmark_id: str):
        return service.delete_benchmark(benchmark_id)

    @router.post("/training", status_code=202)
    def training(value: TrainingInput):
        return service.train(value)

    @router.post("/training/export")
    def export_training(value: TrainingInput):
        return FileResponse(service.export_training(value), filename="iris-reid-training.zip")

    @router.post("/encoders/import", status_code=201)
    async def import_encoder(file: Annotated[UploadFile, File()]):
        destination = service.root / "uploads" / (uuid4().hex + ".zip")
        await save_upload(file, destination, 2 * 1024**3)
        try:
            return service.import_model(destination)
        finally:
            destination.unlink(missing_ok=True)

    return router
