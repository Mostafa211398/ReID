import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from iris.reid_core import BASELINES, ENCODER_METADATA
from iris.reid_encoders import Encoder
from iris.reid_vehicle_encoders import rgb_pixels
from test_reid import client, wait_job
from test_reid_images import new_site, photo_upload
from test_reid_delete import idle


def test_vehicle_rgb_preprocessing_and_transreid_normalization():
    image = Image.new("RGB", (17, 9), (255, 128, 0))
    pixels = rgb_pixels([image], 208)
    assert pixels.shape == (1, 3, 208, 208) and pixels.dtype == np.float32
    np.testing.assert_array_equal(pixels[0, :, 0, 0], [255, 128, 0])
    import torch
    encoder = Encoder.__new__(Encoder)
    encoder.family, encoder.torch, encoder.device = "transreid", torch, torch.device("cpu")
    normalized = encoder.inputs([image]).numpy()
    assert normalized.shape == (1, 3, 256, 256)
    np.testing.assert_allclose(normalized[0, :, 0, 0], [1, 128 / 127.5 - 1, -1], atol=1e-6)


def test_unknown_family_and_invalid_checkpoint_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsupported"):
        Encoder({"family": "unknown"}, tmp_path)
    from iris.reid_extra_setup import _validate_checkpoint
    path = tmp_path / "invalid.pth"
    path.write_text("This is not a model")
    with pytest.raises(ValueError, match="Invalid TransReID checkpoint"):
        _validate_checkpoint(path)


@pytest.mark.parametrize("key", ["openvino", "transreid", "coca", "coca_visual", "coca_l14",
                                  "coca_l14_visual", "siglip2", "siglip2_visual"])
def test_training_capability_is_enforced_in_api(client, monkeypatch, key):
    monkeypatch.setattr(client.reid_service, "encoder", lambda _: {**BASELINES[key], **ENCODER_METADATA[key], "available": True})
    for endpoint in ("training", "training/export"):
        result = client.post('/api/reid/' + endpoint, json={"dataset_id": "unused", "encoder_id": key})
        assert result.status_code == 422 and "inference and evaluation only" in result.text
    encoder = Encoder.__new__(Encoder)
    encoder.family = key
    with pytest.raises(ValueError, match="inference and evaluation only"):
        encoder.trainable_tail()


@pytest.mark.parametrize("media_type", ["image"])
def test_refresh_new_encoders_preserves_ingestion_and_deletes_all_features(client, media_type):
    service, site = client.reid_service, new_site(client)
    job = photo_upload(client, site, media_type)
    initial_tracks = service.tracks(job["id"])
    detection_calls = [s for s in client.stage_calls if s.startswith("ingest-")]
    cache = service.vector_path(job["id"], service.encoder("dinov2"))
    cache_before = cache.read_bytes()
    response = client.post('/api/reid/jobs/' + job["id"] + '/retry', json={"encoders": ["openvino", "transreid"]})
    assert response.status_code == 202
    finished = wait_job(client, job["id"])
    assert finished["config"]["encoders"] == ["dinov2", "openvino", "transreid"]
    assert detection_calls == [s for s in client.stage_calls if s.startswith("ingest-")]
    assert cache.read_bytes() == cache_before
    track = service.tracks(job["id"])[0]
    assert track["id"] == initial_tracks[0]["id"] and track["global_id"] == initial_tracks[0]["global_id"]
    assert set(track["assignments"]) == {"dinov2", "openvino", "transreid"}
    assert all(client.get(url).status_code == 404 for name, url in job["artifacts"].items() if name.startswith("siglip ·"))
    for key in ("openvino", "transreid"):
        assert service.vector_path(job["id"], service.encoder(key)).exists()
    idle(service)
    assert client.delete('/api/reid/experiments/' + job["id"]).status_code == 200
    assert not (service.root / job["id"]).exists()


def test_refresh_validation_does_not_change_job(client, monkeypatch):
    service, site = client.reid_service, new_site(client)
    job = photo_upload(client, site)
    before = service.get("job", job["id"])
    assert client.post('/api/reid/jobs/' + job["id"] + '/retry', json={"encoders": []}).status_code == 422
    existing = service.encoder
    monkeypatch.setattr(service, "encoder", lambda k: {**existing(k), "available": k != "openvino", "availability_reason": "OpenVINO runtime missing"})
    response = client.post('/api/reid/jobs/' + job["id"] + '/retry', json={"encoders": ["openvino"]})
    assert response.status_code == 422 and "runtime missing" in response.text
    assert service.get("job", job["id"]) == before


def test_out_of_memory_restarts_whole_stage_on_cpu(monkeypatch):
    import torch
    from iris import reid_worker
    attempts = []
    def attempt(config):
        attempts.append(config)
        if config.get("device") != "cpu":
            raise torch.cuda.OutOfMemoryError("test")
    monkeypatch.setattr(reid_worker, "embed_once", attempt)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    reid_worker.embed({"encoder": {"family": "transreid"}})
    assert len(attempts) == 2 and attempts[1]["device"] == "cpu"
    assert "restarted on CPU" in attempts[1]["fallback"]
    attempts.clear()
    reid_worker.embed({"encoder": {"family": "coca_l14", "name": "CoCa ViT-L/14", "cpu_fallback": True}})
    assert len(attempts) == 2 and attempts[1]["device"] == "cpu"
    with pytest.raises(torch.cuda.OutOfMemoryError):
        reid_worker.embed({"encoder": {"family": "dinov2"}})


def test_selective_setup_preserves_existing_manifest(tmp_path, monkeypatch):
    from iris import reid_setup, reid_extra_setup
    old = {"dinov2": {"fingerprint": "existing-cache-must-survive"}}
    (tmp_path / "manifest.json").write_text(json.dumps(old))
    monkeypatch.setattr(reid_extra_setup, "install_vehicle", lambda *a: {"fingerprint": "new"})
    reid_setup.install(tmp_path, ["openvino"])
    after = json.loads((tmp_path / "manifest.json").read_text())
    assert after["dinov2"] == old["dinov2"] and after["openvino"]["fingerprint"] == "new"


def test_status_reports_missing_checkpoint_and_runtime(client, tmp_path):
    service = client.reid_service
    root = service.model_root
    (root / "openvino").mkdir(parents=True)
    model = root / "openvino/model.onnx"
    model.write_bytes(b"status fixture")
    (root / "manifest.json").write_text(json.dumps({"openvino": {"fingerprint": "fixture", "checksums": {"model.onnx": "fixture"}}}))
    service.python = tmp_path / "isolated/Scripts/python.exe"
    record = next(e for e in service.encoders() if e["id"] == "openvino")
    assert not record["available"] and "runtime missing" in record["availability_reason"]
    package = tmp_path / "isolated/Lib/site-packages/openvino/__init__.py"
    package.parent.mkdir(parents=True)
    package.write_text("")
    record = next(e for e in service.encoders() if e["id"] == "openvino")
    assert record["available"] and record["embedding_dimension"] == 512 and not record["supports_training"]
    model.unlink()
    record = next(e for e in service.encoders() if e["id"] == "openvino")
    assert not record["available"] and "Install" in record["availability_reason"]
