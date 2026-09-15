import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from iris.reid_core import unit
from iris.reid_color import scoring_spec
from iris.reid_schemas import MatchingSettings, SiteInput
from iris.reid_service import ReIDService
from iris.reid_worker import ingest, read_image, render
from test_reid import client, wait_job


def photo_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (120, 80), "blue").save(buffer, format="PNG")
    return buffer.getvalue()


def photo_upload(client, site, media_type="image", classes=None, data=None):
    config = {"name": "Photo test", "site_id": site["id"],
              "encoders": ["siglip"], 
              "clips": [{"camera_id": "A", "media_type": media_type, "class_name": (classes or ["car"])[0]}]}
    response = client.post("/api/reid/experiments", data={"config": json.dumps(config)},
                           files={"files": ("vehicle.png" if media_type == "image" else "vehicle.mp4", photo_bytes() if data is None else data)})
    assert response.status_code == 202, response.text
    return wait_job(client, response.json()["id"])


def new_site(client):
    return client.post("/api/reid/sites", json={"name": "Photos", "cameras": [{"id": "A", "name": "A"}]}).json()


def test_photo_api_enrollment_single_reference_retry_and_restart(client):
    site = new_site(client)
    assert site["matching"]["mode"] == "automatic"
    first = photo_upload(client, site)
    assert first["config"]["encoders"] == ["dinov2", "siglip"]
    assert len([url for url in first["artifacts"].values() if url.endswith(".jpg")]) == 2
    service = client.reid_service
    track = service.tracks(first["id"])[0]
    assert track["global_id"] == "VEHICLE_00001" and track["gallery_reference"]
    assert track["identity"] is None and not track["reviewed"]
    assert client.get(f'/api/reid/tracks/{track["id"]}/image').status_code == 200
    assert client.post(f'/api/reid/tracks/{track["id"]}/split', json={"frame": 1}).status_code == 404
    assert client.get(first["artifacts"]["results.csv"]).status_code == 200
    client.post(f'/api/reid/jobs/{first["id"]}/retry')
    wait_job(client, first["id"])
    assert len(service.all("identity")) == 1
    second = photo_upload(client, site)
    later = service.tracks(second["id"])[0]
    assert later["global_id"] == track["global_id"]
    assert not later.get("gallery_reference") and later["identity"] is None
    assignment = later["assignments"]["dinov2"]
    assert assignment["status"] == "automatic" and assignment["reference_track_id"] == track["id"]
    assert assignment["encoder_fingerprint"] == "test-dinov2"
    service.close()
    reopened = ReIDService(service.db, service.settings, service.coordinator)
    try:
        reopened._match(second, {"id": "dinov2", "fingerprint": "test-dinov2"}, {later["id"]: [[1, 0]]})
        assert reopened.get("track", later["id"])["global_id"] == track["global_id"]
    finally:
        reopened.close()


def observation(service, site, key, clip=None, class_name="car"):
    job = {"id": key, "site_id": site["id"], "state": "completed", "config": {"clips": [{"id": clip or key}]}}
    service.put("job", job)
    track = {"id": key, "source_track_id": key, "experiment_id": key, "site_id": site["id"],
             "clip_id": clip or key, "camera_id": "A", "media_type": "image", "class_name": class_name,
             "start": 0, "end": 0, "start_frame": 0, "end_frame": 0, "local_id": 1,
             "absolute_start": None, "absolute_end": None, "assignments": {}, "reviewed": False,
             "global_id": None, "identity": None, "crops": [], "excluded": False}
    service.put("track", track)
    return job


def associate(service, job, vector, fingerprint="test-dinov2"):
    spec = {"id": "dinov2", "fingerprint": fingerprint}
    vector = unit([vector])
    path = service.vector_path(job["id"], spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{job["id"]: vector})
    service._match(job, spec, {job["id"]: vector})
    return service.get("track", job["id"])


def test_thresholds_ambiguity_classes_site_and_fingerprint(client):
    service, site = client.reid_service, new_site(client)
    first = associate(service, observation(service, site, "first"), [1, 0])
    ambiguous = associate(service, observation(service, site, "ambiguous"), [.75, (1-.75**2)**.5])
    assert ambiguous["global_id"] is None
    different = associate(service, observation(service, site, "different"), [0, 1])
    assert different["global_id"] != first["global_id"]
    truck = associate(service, observation(service, site, "truck", class_name="truck"), [1, 0])
    assert truck["global_id"] not in (first["global_id"], different["global_id"])
    other_site = new_site(client)
    other = associate(service, observation(service, other_site, "other-site"), [1, 0])
    assert other["assignments"]["dinov2"]["candidates"] == []
    incompatible = associate(service, observation(service, site, "new-encoder"), [1, 0], "different-space")
    assert incompatible["assignments"]["dinov2"]["candidates"] == []
    # Create a second valid reference with near-identical appearance: require separation.
    near_job = observation(service, site, "near")
    near = associate(service, near_job, [.99, .14])
    confirmed = client.put('/api/reid/tracks/near/review', json={"new_identity": True})
    assert confirmed.status_code == 200
    tie = associate(service, observation(service, site, "tie"), [1, 0])
    assert tie["global_id"] is None and tie["assignments"]["dinov2"]["status"] == "review"


def test_distinct_vehicles_same_image_and_calibration_precedence(client):
    service, site = client.reid_service, new_site(client)
    first = associate(service, observation(service, site, "one", "shared-photo"), [1, 0])
    second = associate(service, observation(service, site, "two", "shared-photo"), [1, 0])
    assert first["global_id"] != second["global_id"]
    service.put("calibration", {"id": site["id"] + "_" + scoring_spec("test-dinov2")["fingerprint"], "threshold": .99, "margin": .01})
    third = associate(service, observation(service, site, "third"), [.95, (1-.95**2)**.5])
    assert third["global_id"] is None
    assert third["assignments"]["dinov2"]["provenance"] == "calibrated"


def test_legacy_site_settings_and_validation(client, monkeypatch):
    service, site = client.reid_service, new_site(client)
    site.pop("matching")
    service.put("site", site)
    updated = client.put('/api/reid/sites/' + site["id"], json={"name": "Renamed", "cameras": site["cameras"]})
    assert updated.status_code == 200 and "matching" not in updated.json()
    legacy = associate(service, observation(service, site, "legacy"), [1, 0])
    assert legacy["global_id"] is None
    with pytest.raises(ValueError):
        MatchingSettings(new_threshold=.9, threshold=.8)
    with pytest.raises(AssertionError, match="Could not decode"):
        photo_upload(client, site, data=b"broken")


def test_enrollment_crash_recovery_and_no_automatic_gallery_growth(client):
    service, site = client.reid_service, new_site(client)
    first_job = observation(service, site, "enrolled")
    first = associate(service, first_job, [1, 0])
    # Simulate identity persistence succeeding but the observation write being lost.
    first.update(global_id=None, gallery_reference=False, assignments={})
    service.put("track", first)
    recovered = associate(service, first_job, [1, 0])
    assert len(service.all("identity")) == 1 and recovered["global_id"] == "VEHICLE_00001"
    auto = associate(service, observation(service, site, "auto"), [.9, (1-.9**2)**.5])
    assert auto["global_id"] == recovered["global_id"] and not auto.get("gallery_reference")
    query = associate(service, observation(service, site, "query"), [.75, (1-.75**2)**.5])
    assert query["global_id"] is None  # Would pass if the automatic crop contaminated the gallery.
    assert client.put('/api/reid/tracks/auto/review', json={"global_id": recovered["global_id"]}).status_code == 200
    query = associate(service, service.get("job", "query"), [.75, (1-.75**2)**.5])
    assert query["global_id"] == recovered["global_id"]
    empty_job = observation(service, site, "no-crop")
    service._match(empty_job, {"id": "dinov2", "fingerprint": "test-dinov2"}, {})
    empty = service.get("track", "no-crop")
    assert empty["global_id"] is None and "small" in empty["assignments"]["dinov2"]["reason"]


