import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from iris.reid_worker import ingest, render
from iris.reid_schemas import ExperimentInput, TrainingInput
from test_reid import client
from test_reid_images import new_site, photo_upload


def test_orientation_rgb_and_whole_crop(tmp_path):
    source = tmp_path / "crop.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (12, 20), "red").save(source, exif=exif)
    config = {"output": str(tmp_path / "out"), "progress_path": str(tmp_path / "progress.json"),
              "clip": {"id": "crop", "camera_id": "A", "class_name": "car", "path": str(source), "sha256": "hash", "start_epoch": 1234}}
    ingest(config)
    tracks = json.loads((tmp_path / "out/tracks.json").read_text())
    assert len(tracks) == 1 and tracks[0]["absolute_start"] == 1234
    assert tracks[0]["class_name"] == "car"
    with Image.open(tracks[0]["crops"][0]["path"]) as image:
        assert image.size == (20, 12) and image.mode == "RGB"
    ingest(config)
    assert json.loads((tmp_path / "out/tracks.json").read_text()) == tracks
    render({**config, "output": str(tmp_path / "result.jpg"), "frames": str(tmp_path / "out/frames.jsonl"), "assignments": {}})
    with Image.open(tmp_path / "result.jpg") as result:
        assert result.height == 48 and result.width > 20


@pytest.mark.parametrize("patch", [
    {"model_id": "old-detector"}, {"confidence": .4}, {"vehicle_classes": ["car"]},
    {"clips": [{"camera_id": "A", "media_type": "video", "class_name": "truck"}]},
    {"clips": [{"camera_id": "A"}]},
])
def test_crop_contract_rejects_old_detector_fields(patch):
    with pytest.raises(ValueError):
        ExperimentInput.model_validate({"name": "crop", "site_id": "site", "clips": [{"camera_id": "A", "class_name": "car"}], **patch})


def test_removed_routes_and_invalid_uploads(client):
    site = new_site(client)
    for path in ("/detectors/yolo/classes", "/tracks/missing/video", "/jobs/missing/preview.jpg"):
        assert client.get("/api/reid" + path).status_code == 404
    config = {"name": "bad", "site_id": site["id"], "clips": [{"camera_id": "A", "class_name": "car"}]}
    for filename, content in [("video.mp4", b"video"), ("broken.png", b"broken")]:
        response = client.post('/api/reid/experiments', data={"config": json.dumps(config)}, files={"files": (filename, content)})
        assert response.status_code == 422
    assert client.reid_service.all("job") == []
    assert not list((client.reid_service.root / "uploads").rglob("*.*"))


def test_export_bundle_contains_standalone_runtime(client):
    service = client.reid_service
    site = new_site(client)
    job = photo_upload(client, site)
    service.put("dataset", {"id": "bundle", "site_id": site["id"], "rows": service.tracks(job["id"]), "splits": {}, "fingerprint": "fixture"})
    path = service.export_training(TrainingInput(dataset_id="bundle"))
    with zipfile.ZipFile(path) as package:
        names = set(package.namelist())
        assert {"iris/reid_worker.py", "iris/reid_setup.py", "iris/reid_extra_setup.py", "iris/reid_vehicle_encoders.py", "requirements-reid.txt"} <= names
        requirements = package.read("requirements-reid.txt").decode()
        assert "torch==" in requirements and "ultralytics" not in requirements
        assert "Yolo11n" not in package.read("README.txt").decode()


def test_reject_corrupt_encoder_package(client):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as package:
        package.writestr("encoder.json", json.dumps({"family": "dinov2", "dataset_fingerprint": "fixture", "checksum": "wrong"}))
        package.writestr("model.safetensors", b"invalid")
    result = client.post('/api/reid/encoders/import', files={"file": ("model.zip", buffer.getvalue())})
    assert result.status_code == 422 and "checksum" in result.text.lower()
