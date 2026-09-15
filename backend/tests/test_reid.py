import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from iris.app import create_app
from iris.config import Settings
from iris.reid_core import calibrate, candidates, partition, retrieval, simultaneous, timestamp, track_vector, unit
from iris.reid_schemas import ClipInput, SiteInput
from iris.reid_service import ReIDService
from PIL import Image
import io


def row(key, identity, camera="A", start=0):
    return {"id": key, "source_track_id": key, "identity": identity, "camera_id": camera,
            "clip_id": key, "source_hash": key, "start": start, "end": start + 1,
            "absolute_start": None, "absolute_end": None}


def test_retrieval_counts_and_no_self_or_same_camera_leakage():
    rows = [row("a1", "A"), row("a2", "A", "B"), row("b1", "B"), row("b2", "B", "B")]
    vectors = {"a1": [1, 0], "a2": [1, 0], "b1": [0, 1], "b2": [0, 1]}
    metrics = retrieval(rows, vectors)
    assert metrics["rank1"] == metrics["mAP"] == 1
    assert metrics["eligible_queries"] == 4
    assert retrieval(rows, vectors, False)["eligible_queries"] == 0
    rows[1]["source_hash"] = rows[0]["source_hash"]
    assert retrieval(rows, vectors)["eligible_queries"] == 2


def test_invalid_embeddings_and_timestamp_fallback():
    for vector in ([0, 0], [float("nan"), 1], []):
        with pytest.raises(ValueError):
            unit(vector)
    assert np.allclose(track_vector([[2, 0], [0, 2]]), [2**-.5, 2**-.5])
    assert timestamp(float("nan"), 10, 10, .9) == 1
    assert timestamp(0, 11, 10, 1) == 1.1
    with pytest.raises(ValueError):
        ClipInput(camera_id="A", start_time="2026-01-01T10:00:00")


def test_unknown_times_and_same_camera_exclusion():
    a, b = row("one", "A"), row("two", "A")
    assert not simultaneous(a, b)  # Different clips, unknown timing.
    b["clip_id"] = a["clip_id"]
    assert simultaneous(a, b)
    assert candidates(a, [[1, 0]], {"TRUCK_00001": [[1, 0]]}, {"TRUCK_00001": [b]}, []) == []
    b["camera_id"] = "B"
    assert len(candidates(a, [[1, 0]], {"TRUCK_00001": [[1, 0]]}, {"TRUCK_00001": [b]}, [])) == 1


def test_calibration_requires_evidence_and_rejects_ambiguous_scores():
    assert calibrate([{"correct": True, "score": .99, "margin": .2}] * 5) is None
    queries = [{"correct": True, "score": .9, "margin": .2}] * 20 + [{"correct": False, "score": .91, "margin": .01}] * 5
    threshold = calibrate(queries)
    assert threshold["accepted"] == 20 and threshold["precision"] == 1
    assert threshold["margin"] > .01


def test_identity_split_is_deterministic_and_disjoint():
    rows = [row(str(i), str(i // 3)) for i in range(30)]
    first, second = partition(rows), partition(list(reversed(rows)))
    assert first == second
    assert len(set(first["train"]) | set(first["validation"]) | set(first["test"])) == 10
    assert not set(first["train"]) & set(first["validation"])
    with pytest.raises(ValueError):
        partition(rows[:6])
    with pytest.raises(ValueError):
        SiteInput(name="Site", cameras=[{"id": "A", "name": "one"}, {"id": "A", "name": "two"}])


@pytest.fixture
def client(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path / "data", frontend_dist=tmp_path / "missing",
                        gpu_lock_path=tmp_path / "gpu.lock")
    settings.ensure_directories()
    app = create_app(settings)
    service = app.state.services.reid
    monkeypatch.setattr(service, "encoder", lambda key: {"id": key, "name": key,
                        "family": key if key in ("coca", "coca_visual", "coca_l14", "coca_l14_visual",
                                                  "siglip2", "siglip2_visual") else "dinov2",
                        "available": True, "fingerprint": "test-" + key, "origin": "pretrained"})
    calls = []

    def fake_stage(job, stage, config):
        if stage in job["completed_stages"]:
            return
        calls.append(stage)
        out = Path(config["output"])
        out.parent.mkdir(parents=True, exist_ok=True)
        if config["task"] == "ingest":
            from iris.reid_worker import ingest
            ingest({**config, "progress_path": str(out.parent / "progress.json")})
        elif config["task"] == "embed":
            with out.open("wb") as handle:
                np.savez(handle, **{t["id"]: np.array([[1, 0]] * len(t["crops"]), np.float32) for t in config["tracks"]})
        elif config["task"] == "pair_features":
            from iris.reid_core import atomic_json
            from iris.reid_pair import (CRITERIA_PROMPT_GROUPS, CRITERIA_VERSION, PAIR_FEATURE_VERSION,
                                        SEMANTIC_GROUPS, SEMANTIC_VERSION, SHAPE_VERSION)
            if config["encoder"]["family"].endswith("_visual"):
                with out.open("wb") as handle:
                    np.savez(handle, appearance=np.array([[1., 0.], [.8, .6]], np.float32))
                atomic_json(Path(config["metadata_path"]), {"pair_feature_version": PAIR_FEATURE_VERSION,
                            "encoder_only": True})
                job["completed_stages"].append(stage); service.put("job", job); return
            color = np.zeros((2, 72), np.float32); color[:, 0] = 1
            shape = np.zeros((2, 8100), np.float32); shape[:, 0] = 1
            semantic = np.zeros((2, 24), np.float32); semantic[:, 0] = 1
            with out.open("wb") as handle:
                np.savez(handle, appearance=np.array([[1., 0.], [.8, .6]], np.float32), color=color,
                         color_available=np.ones(2, np.bool_), shape=shape,
                         shape_available=np.ones(2, np.bool_), semantic=semantic,
                         texture=np.tile(np.eye(1, 256, dtype=np.float32), (2, 1)),
                         texture_available=np.ones(2, np.bool_), detail=np.tile(np.eye(1, 34, dtype=np.float32), (2, 1)),
                         detail_available=np.ones(2, np.bool_), edge_scale=np.array([.8, .7], np.float32))
            groups = {**SEMANTIC_GROUPS, **CRITERIA_PROMPT_GROUPS}
            attributes = [{group: {"label": values[0][0], "confidence": 1.,
                                    "distribution": {label: float(index == 0) for index, (label, _) in enumerate(values)}}
                           for group, values in groups.items()} for _ in range(2)]
            atomic_json(Path(config["metadata_path"]), {"pair_feature_version": PAIR_FEATURE_VERSION,
                        "shape_version": SHAPE_VERSION, "semantic_version": SEMANTIC_VERSION,
                        "criteria_version": CRITERIA_VERSION,
                        "color": [{"method": "fixture"}, {"method": "fixture"}],
                        "color_summary": [{"dominant_colors": ["red"], "mean_hue_radians": 0.},
                                          {"dominant_colors": ["red"], "mean_hue_radians": 0.}],
                        "semantic_attributes": [{key: value for key, value in row.items() if key in SEMANTIC_GROUPS}
                                                for row in attributes], "criteria_attributes": attributes})
        elif config["task"] == "benchmark_comparison_export":
            from iris.reid_benchmark import write_comparison_images
            write_comparison_images(config)
        elif out.suffix == ".jpg":
            cv2.imwrite(str(out), np.full((64, 64, 3), 120, np.uint8))
        else:
            out.write_bytes(b"video")
        job["completed_stages"].append(stage)
        service.put("job", job)

    monkeypatch.setattr(service, "run_stage", fake_stage)
    with TestClient(app) as connection:
        connection.reid_service = service
        connection.stage_calls = calls
        yield connection


def wait_job(client, key):
    for _ in range(300):
        value = client.get("/api/reid/jobs/" + key).json()
        if value["state"] not in ("queued", "running"):
            assert value["state"] == "completed", value
            return value
        time.sleep(.03)
    pytest.fail("Job did not finish")


def upload(client, site_id, name="first"):
    config = {"name": name, "site_id": site_id, "encoders": ["dinov2", "siglip"],
              "clips": [{"camera_id": "A", "class_name": "truck"}, {"camera_id": "B", "class_name": "truck"}]}
    files = []
    for i in range(2):
        buffer = io.BytesIO()
        Image.new("RGB", (64, 64), (120+i, 120, 120)).save(buffer, format="PNG")
        files.append(("files", (f"{i}.png", buffer.getvalue(), "image/png")))
    response = client.post("/api/reid/experiments", data={"config": json.dumps(config)}, files=files)
    assert response.status_code == 202, response.text
    return wait_job(client, response.json()["id"])


def test_upload_review_persistent_ids_and_optional_training(client):
    site = client.post("/api/reid/sites", json={"name": "Site", "matching": {"mode": "review"}, "cameras": [{"id": "A", "name": "A"}, {"id": "B", "name": "B"}]}).json()
    job = upload(client, site["id"])
    tracks = client.get(f'/api/reid/experiments/{job["id"]}/tracks').json()
    assert len(tracks) == 2 and all(t["local_id"] == 1 for t in tracks)
    assert tracks[0]["id"] != tracks[1]["id"]
    assert tracks[0]["identity"] is None
    assert tracks[0]["assignments"]["dinov2"]["neighbors"]
    assert client.get(job["artifacts"]["results.csv"]).status_code == 200
    first = client.put(f'/api/reid/tracks/{tracks[0]["id"]}/review', json={"new_identity": True}).json()
    assert first["global_id"] == "VEHICLE_00001"
    second = client.put(f'/api/reid/tracks/{tracks[1]["id"]}/review', json={"global_id": first["global_id"]})
    assert second.status_code == 200
    refreshed = client.post(f'/api/reid/jobs/{job["id"]}/retry')
    assert refreshed.status_code == 202
    wait_job(client, job["id"])
    assert len([s for s in client.stage_calls if s.startswith("ingest-")]) == 2
    later = upload(client, site["id"], "later")
    later_tracks = client.get(f'/api/reid/experiments/{later["id"]}/tracks').json()
    assert later_tracks[0]["assignments"]["dinov2"]["candidates"][0]["global_id"] == first["global_id"]
    assert later_tracks[0]["assignments"]["dinov2"]["status"] == "review"  # No calibration or labels invented.
    assert client.get("/api/reid/datasets").json() == []
    assert not any(j["kind"] == "training" for j in client.get("/api/reid/jobs").json())


def test_bad_upload_and_cross_site_review_rejected(client):
    site = client.post("/api/reid/sites", json={"name": "Site", "matching": {"mode": "review"}, "cameras": [{"id": "A", "name": "A"}, {"id": "B", "name": "B"}]}).json()
    job = upload(client, site["id"])
    track = client.get(f'/api/reid/experiments/{job["id"]}/tracks').json()[0]
    assert client.put(f'/api/reid/tracks/{track["id"]}/review', json={"global_id": "TRUCK_99999"}).status_code == 422
    assert client.get(f'/api/reid/jobs/{job["id"]}/artifacts/worker.json').status_code == 404
    assert client.get(f'/api/reid/jobs/{job["id"]}/artifacts/..%2F..%2Firis.db').status_code == 404
    config = {"name": "bad", "site_id": site["id"], "model_id": "yolo11n-coco", "clips": [{"camera_id": "A"}]}
    response = client.post("/api/reid/experiments", data={"config": json.dumps(config)}, files={"files": ("x.txt", b"bad")})
    assert response.status_code == 422


def test_standalone_routes(tmp_path):
    settings = Settings(data_dir=tmp_path, frontend_dist=tmp_path / "none", gpu_lock_path=tmp_path / "gpu")
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/reid/sites").status_code == 200


def test_queued_cancellation_does_not_release_another_gpu_owner(client):
    from iris.reid_schemas import Job
    service = client.reid_service
    assert service.coordinator.acquire("test-owner", wait=False)
    try:
        job = service.submit(Job(name="Wait for GPU", site_id="test", config={}))
        response = client.post("/api/reid/jobs/" + job["id"] + "/cancel")
        assert response.json()["state"] == "cancelled"
        assert "test-owner" in service.coordinator.owner
        assert not client.stage_calls
    finally:
        service.coordinator.release("test-owner")


def test_restart_marks_interrupted_stage_failed_and_preserves_artifacts(client):
    from iris.reid_schemas import Job
    service = client.reid_service
    job = service.put("job", Job(name="Interrupted", site_id="test", state="running", stage="embed-dinov2",
                                 completed_stages=["ingest-camera-a"], artifacts={"json": "/saved.json"}))
    service.close()
    replacement = ReIDService(service.db, service.settings, service.coordinator)
    try:
        recovered = replacement.get("job", job["id"])
        assert recovered["state"] == "failed"
        assert recovered["completed_stages"] == ["ingest-camera-a"]
        assert recovered["artifacts"] == {"json": "/saved.json"}
    finally:
        replacement.close()


def test_identity_registry_is_not_truncated_after_500_records(client):
    service = client.reid_service
    for i in range(510):
        service.put("identity", {"id": str(i), "site_id": "large-site", "global_id": f"TRUCK_{i:05d}"})
    assert len(service.all("identity")) == 510


def test_snapshot_rejects_contradictory_duplicate_crop_labels(client):
    from iris.reid_schemas import DatasetInput
    site = client.post("/api/reid/sites", json={"name": "Site", "matching": {"mode": "review"}, "cameras": [{"id": "A", "name": "A"}, {"id": "B", "name": "B"}]}).json()
    job = upload(client, site["id"])
    service = client.reid_service
    template = service.tracks(job["id"])[0]
    for i in range(6):
        service.put("track", {**template, "id": str(i), "identity": f"truck{i}"})
    with pytest.raises(ValueError, match="Identical crops"):
        service.create_dataset(DatasetInput(site_id=site["id"], experiment_ids=[job["id"]]))
