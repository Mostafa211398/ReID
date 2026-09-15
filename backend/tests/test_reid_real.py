"""Opt-in real encoder and training smoke checks on synthetic crops."""
import io
import json
import os
import time
from pathlib import Path
import cv2
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from iris.app import create_app
from iris.config import Settings
from iris.reid_core import fingerprint

pytestmark = pytest.mark.skipif(os.getenv("IRIS_REID_REAL_SMOKE") != "1", reason="Explicit real-model smoke test")
PROJECT = Path(__file__).resolve().parents[2]

def test_isolated_environment():
    import importlib.util
    import iris.app
    assert Path(iris.app.__file__).resolve().is_relative_to(PROJECT)
    assert all(importlib.util.find_spec(name) is None for name in ("ultralytics", "mmdet", "mmcv"))

def completed(client, key, timeout=600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = client.get("/api/reid/jobs/" + key).json()
        if value["state"] not in ("queued", "running"):
            assert value["state"] == "completed", value
            return value
        time.sleep(.25)
    raise AssertionError("Real worker timed out")

@pytest.fixture
def real_client(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", frontend_dist=PROJECT / "frontend/dist",
                        gpu_lock_path=tmp_path / "gpu.lock", reid_model_dir=PROJECT / ".data/models/reid")
    with TestClient(create_app(settings)) as client:
        yield client

def test_real_crop_all_encoders_and_retry(real_client):
    site = real_client.post('/api/reid/sites', json={"name": "Real crops", "cameras": [{"id": "A", "name": "A"}]}).json()
    buffer = io.BytesIO()
    Image.fromarray(np.random.default_rng(42).integers(0, 255, (96, 128, 3), dtype=np.uint8)).save(buffer, format="PNG")
    ids = []
    for index in range(2):
        config = {"name": f"Crop {index}", "site_id": site["id"], "encoders": ["dinov2", "siglip", "fastreid", "openvino", "transreid", "coca", "coca_visual", "coca_l14", "coca_l14_visual", "siglip2", "siglip2_visual"],
                  "clips": [{"camera_id": "A", "class_name": "truck"}]}
        response = real_client.post('/api/reid/experiments', data={"config": json.dumps(config)}, files={"files": ("crop.png", buffer.getvalue())})
        assert response.status_code == 202, response.text
        job = completed(real_client, response.json()["id"])
        tracks = real_client.get('/api/reid/experiments/' + job["id"] + '/tracks').json()
        assert len(tracks) == 1 and len(tracks[0]["assignments"]) == 11
        ids.append(tracks[0]["global_id"])
        for url in job["artifacts"].values():
            assert real_client.get(url).status_code == 200
    assert ids[0] and ids[0] == ids[1]
    assert real_client.post('/api/reid/jobs/' + job["id"] + '/retry').status_code == 202
    completed(real_client, job["id"])
    assert len(real_client.get('/api/reid/experiments/' + job["id"] + '/tracks').json()) == 1


def test_real_chosen_crops_folder_benchmark_all_encoders(real_client):
    site = real_client.post('/api/reid/sites', json={"name": "Chosen crops benchmark"}).json()
    root = PROJECT.parent / "RTMDet-Tiny" / "chosen crops"
    paths = sorted(root.rglob("*.jpg"))
    config = {"name": "Chosen crops", "site_id": site["id"], "threshold": .85,
              "encoders": ["dinov2", "siglip", "fastreid", "openvino", "transreid", "coca", "coca_visual", "coca_l14", "coca_l14_visual", "siglip2", "siglip2_visual"],
              "items": [{"relative_path": path.relative_to(root.parent).as_posix()} for path in paths]}
    files = [("files", (path.name, path.read_bytes(), "image/jpeg")) for path in paths]
    response = real_client.post('/api/reid/benchmarks', data={"config": json.dumps(config)}, files=files)
    assert response.status_code == 202, response.text
    job = completed(real_client, response.json()["id"], 1800)
    benchmark = real_client.get('/api/reid/benchmarks/' + job["id"]).json()
    assert benchmark["counts"] == {"identities": 10, "images": 46, "source_groups": 46,
                                   "all_pairs_per_encoder": 1035, "strict_pairs_per_encoder": 1035,
                                   "eligible_identification_queries": 45}
    assert len(benchmark["encoders"]) == 11
    for result in benchmark["encoders"]:
        values = [result["pairwise_strict"][name] for name in ("precision", "recall", "f1", "accuracy")]
        assert np.isfinite(values).all()
        assert result["identification"]["eligible_queries"] == 45
    package = real_client.get('/api/reid/benchmarks/' + job["id"] + '/results.zip')
    assert package.status_code == 200 and len(package.content) > sum(path.stat().st_size for path in paths)

@pytest.mark.parametrize("family", ["dinov2", "siglip", "fastreid"])
def test_real_optional_training_export_import(real_client, tmp_path, family):
    service = real_client.app.state.services.reid
    site = real_client.post("/api/reid/sites", json={"name": "Synthetic training smoke"}).json()
    rows = []
    for identity in range(6):
        for camera in range(2):
            key = f"i{identity}c{camera}"
            crops = []
            for view in range(2):
                image = np.random.default_rng(identity).integers(20, 235, (96, 128, 3), dtype=np.uint8)
                image = np.roll(image, camera + view, axis=1)
                path = tmp_path / f"{key}v{view}.jpg"; cv2.imwrite(str(path), image)
                crops.append({"path": str(path), "frame": view * 10, "timestamp": float(view), "quality": 1})
            rows.append({"id": key, "source_track_id": key, "identity": str(identity), "camera_id": str(camera), "clip_id": key, "source_hash": key,
                         "start": 0, "end": 1, "absolute_start": None, "absolute_end": None, "crops": crops})
    dataset = {"id": "synthetic", "site_id": site["id"], "rows": rows, "splits": {"train": ["0", "1"], "validation": ["2", "3"], "test": ["4", "5"]}, "seed": 42}
    dataset["fingerprint"] = fingerprint(dataset)
    service.put("dataset", dataset)
    response = real_client.post("/api/reid/training", json={"dataset_id": "synthetic", "encoder_id": family, "epochs": 1, "accumulation": 1})
    job = completed(real_client, response.json()["id"], 600)
    assert job["artifacts"]["model package"]
    package = real_client.get(job["artifacts"]["model package"])
    imported = real_client.post("/api/reid/encoders/import", files={"file": ("model.zip", package.content, "application/zip")})
    assert imported.status_code == 201, imported.text
    # The exact safetensors payload survives export/import, so inference has identical weights.
    trained = service.encoder("trained-" + job["id"])
    assert imported.json()["checksum"] == trained["checksum"]
    evaluation = real_client.post("/api/reid/evaluations", json={"dataset_id": "synthetic", "encoder_id": imported.json()["id"], "split": "test"})
    evaluated = completed(real_client, evaluation.json()["id"])
    assert evaluated["metrics"]["cross_camera"]["eligible_queries"] == 4
