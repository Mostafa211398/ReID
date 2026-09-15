import json
import threading
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from iris.reid_delete import journals
from iris.reid_service import ReIDService
from iris.reid_worker import save_vectors
from test_reid import client
from test_reid_images import new_site, photo_upload


def idle(service):
    deadline = time.monotonic() + 3
    while service.queue.unfinished_tasks and time.monotonic() < deadline:
        time.sleep(.01)
    assert service.queue.unfinished_tasks == 0


@pytest.mark.parametrize("media_type", ["image"])
@pytest.mark.parametrize("state", ["completed", "failed", "cancelled"])
def test_permanent_deletion_all_terminal_states(client, media_type, state):
    service, site = client.reid_service, new_site(client)
    job = photo_upload(client, site, media_type)
    idle(service)
    job["state"] = state
    service.put("job", job)
    track = service.tracks(job["id"])[0]
    paths = [Path(c["path"]) for c in job["config"]["clips"]] + [Path(c["path"]) for c in track["crops"]]
    preview = client.get(f'/api/reid/experiments/{job["id"]}/deletion-preview')
    assert preview.status_code == 200 and not preview.json()["blockers"]
    assert preview.json()["observations"] == 1 and preview.json()["files"] > 0
    assert client.delete('/api/reid/experiments/' + job["id"]).status_code == 200
    assert service.db.get("reid_job", job["id"]) is None
    assert service.db.get("reid_track", track["id"]) is None
    assert service.all("identity") == [] and not journals(service)
    assert not (service.root / job["id"]).exists()
    assert all(not path.exists() for path in paths)
    assert not list((service.root / "colors").glob("*/*.json"))
    for url in [f'/api/reid/jobs/{job["id"]}', f'/api/reid/experiments/{job["id"]}/tracks',
                f'/api/reid/tracks/{track["id"]}/crops/0', *job["artifacts"].values()]:
        assert client.get(url).status_code == 404
    assert client.post(f'/api/reid/jobs/{job["id"]}/retry').status_code == 404
    assert client.delete('/api/reid/experiments/' + job["id"]).status_code == 404


def test_shared_id_and_color_cache_survive_and_references_are_scrubbed(client):
    service, site = client.reid_service, new_site(client)
    first, second = photo_upload(client, site), photo_upload(client, site)
    a, b = service.tracks(first["id"])[0], service.tracks(second["id"])[0]
    client.put('/api/reid/tracks/' + b["id"] + '/review', json={"global_id": a["global_id"]})
    client.put('/api/reid/tracks/' + b["id"] + '/annotation', json={"identity": "My vehicle"})
    idle(service)
    cache_paths = list((service.root / "colors").glob("*/*.json"))
    assert client.delete('/api/reid/experiments/' + first["id"]).status_code == 200
    after = service.get("track", b["id"])
    assert after["global_id"] == a["global_id"] and after["reviewed"]
    assert after["identity"] == "My vehicle"
    assert a["id"] not in json.dumps(after)
    assert len(service.all("identity")) == 1
    assert "enrollment_track_id" not in service.all("identity")[0]
    job = service.get("job", second["id"])
    assert job["refresh_required"] and job["artifacts"] == {}
    assert all(path.exists() for path in cache_paths)
    assert service.vector_path(second["id"], service.encoder("dinov2")).exists()
    assert all(Path(c["path"]).exists() for c in b["crops"])
    assert all(client.get(url).status_code == 404 for url in second["artifacts"].values())
    refreshed = client.post('/api/reid/jobs/' + second["id"] + '/retry')
    assert refreshed.status_code == 202


def test_orphan_id_clears_automatic_matches_and_numbers_are_not_reused(client):
    service, site = client.reid_service, new_site(client)
    first, second = photo_upload(client, site), photo_upload(client, site)
    idle(service)
    b = service.tracks(second["id"])[0]
    assert b["global_id"] == "VEHICLE_00001"
    assert client.delete('/api/reid/experiments/' + first["id"]).status_code == 200
    assert service.get("track", b["id"])["global_id"] is None
    assert service.all("identity") == []
    third = photo_upload(client, site)
    assert service.tracks(third["id"])[0]["global_id"] == "VEHICLE_00002"


def test_research_dependencies_block_without_mutation(client):
    service, site = client.reid_service, new_site(client)
    job = photo_upload(client, site)
    idle(service)
    assert not client.get('/api/reid/experiments/' + job["id"] + '/deletion-preview').json()["blockers"]
    service.put("dataset", {"id": "research", "name": "Frozen research snapshot", "site_id": site["id"],
                            "experiment_ids": [job["id"]], "fingerprint": "research-fp", "rows": service.tracks(job["id"])})
    service.put("encoder", {"id": "trained-research", "name": "Trained research model", "dataset_fingerprint": "research-fp"})
    preview = client.get('/api/reid/experiments/' + job["id"] + '/deletion-preview').json()
    assert {b["kind"] for b in preview["blockers"]} == {"dataset", "model"}
    before = service.get("job", job["id"])
    result = client.delete('/api/reid/experiments/' + job["id"])
    assert result.status_code == 409
    assert "Frozen research snapshot" in result.text
    assert service.get("job", job["id"]) == before
    assert all(Path(c["path"]).exists() for c in job["config"]["clips"])
    assert not journals(service)


def test_promotion_caches_are_pruned_and_cannot_restore_deleted_vectors(client):
    service, site = client.reid_service, new_site(client)
    first, second = photo_upload(client, site), photo_upload(client, site)
    idle(service)
    a, b = service.tracks(first["id"])[0], service.tracks(second["id"])[0]
    key = uuid4().hex
    service.put("job", {"id": key, "kind": "promotion", "name": "Use encoder", "site_id": site["id"], "state": "completed",
                        "config": {"encoder_id": "dinov2"}, "completed_stages": ["embed-site-gallery"], "artifacts": {}})
    paths = [service.root / key / "gallery-tracks.npz", service.root / "sites" / site["id"] / "test-dinov2.npz"]
    for path in paths:
        save_vectors(path, {a["id"]: np.array([[1, 0]]), b["id"]: np.array([[0, 1]])})
    worker = service.root / key / "worker.json"
    worker.write_text(json.dumps({"tracks": [a, b]}))
    assert client.delete('/api/reid/experiments/' + first["id"]).status_code == 200
    for path in paths:
        assert set(service.load_vectors(path)) == {b["id"]}
    assert not worker.exists()
    assert service.get("job", key)["completed_stages"] == []
    assert service.get("site", site["id"])["active_encoder"] == "dinov2"


def test_locked_file_retry_and_restart_recovery(client, monkeypatch):
    service, site = client.reid_service, new_site(client)
    job = photo_upload(client, site)
    idle(service)
    target = Path(job["config"]["clips"][0]["path"])
    unlink = Path.unlink
    def locked(path, *args, **kwargs):
        if path == target:
            raise PermissionError("File is in use")
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", locked)
    assert client.delete('/api/reid/experiments/' + job["id"]).status_code == 409
    assert service.get("job", job["id"])["deletion_pending"]
    assert journals(service)
    assert client.post('/api/reid/jobs/' + job["id"] + '/retry').status_code == 409
    assert client.post('/api/reid/sites', json={"name": "Blocked during cleanup"}).status_code == 409
    monkeypatch.setattr(Path, "unlink", unlink)
    service.close()
    restarted = ReIDService(service.db, service.settings, service.coordinator)
    try:
        assert restarted.db.get("reid_job", job["id"]) is None
        assert restarted.tracks(job["id"]) == []
        assert not target.exists() and not journals(restarted)
    finally:
        restarted.close()


def test_database_failure_keeps_journal_and_can_retry(client, monkeypatch):
    service, site = client.reid_service, new_site(client)
    job = photo_upload(client, site)
    idle(service)
    apply = service.db.apply_changes
    monkeypatch.setattr(service.db, "apply_changes", lambda *args: (_ for _ in ()).throw(RuntimeError("Simulated database failure")))
    assert client.delete('/api/reid/experiments/' + job["id"]).status_code == 409
    assert service.db.get("reid_job", job["id"]) is not None
    assert service.tracks(job["id"])
    monkeypatch.setattr(service.db, "apply_changes", apply)
    assert client.delete('/api/reid/experiments/' + job["id"]).status_code == 200
    assert not journals(service)


def test_processing_blocks_deletion_and_outside_path_is_rejected(client, tmp_path):
    service, site = client.reid_service, new_site(client)
    job = photo_upload(client, site)
    idle(service)
    # An unsettled worker blocks deletion even if the public job says cancelled.
    service.active_id = job["id"]
    assert client.delete('/api/reid/experiments/' + job["id"]).status_code == 409
    service.active_id = None
    outside = tmp_path / "keep.png"
    outside.write_bytes(b"keep")
    job["config"]["clips"][0]["path"] = str(outside)
    service.put("job", job)
    assert client.delete('/api/reid/experiments/' + job["id"]).status_code == 422
    assert outside.read_bytes() == b"keep"
    assert service.get("job", job["id"]) and not journals(service)


def test_empty_experiment_and_shared_upload_are_preserved_correctly(client):
    service, site = client.reid_service, new_site(client)
    first, second = photo_upload(client, site), photo_upload(client, site)
    idle(service)
    shared = Path(first["config"]["clips"][0]["path"])
    second["config"]["clips"][0]["path"] = str(shared)
    service.put("job", second)
    for track in service.tracks(first["id"]):
        service.db.delete("reid_track", track["id"])
    assert client.delete('/api/reid/experiments/' + first["id"]).status_code == 200
    assert shared.exists() and service.get("job", second["id"])


def test_database_related_changes_roll_back_together(client):
    db = client.reid_service.db
    db.put("reid_test", "keep", {"id": "keep"})
    with pytest.raises(KeyError):
        db.apply_changes([("reid_test", {"id": "added"}), ("reid_test", {})], [("reid_test", "keep")])
    assert db.get("reid_test", "keep") and db.get("reid_test", "added") is None


def test_concurrent_retry_cannot_recreate_deleted_experiment(client, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    service, site = client.reid_service, new_site(client)
    job = photo_upload(client, site)
    idle(service)
    entered, release = threading.Event(), threading.Event()
    apply = service.db.apply_changes
    def paused(*args):
        entered.set()
        assert release.wait(5)
        return apply(*args)
    monkeypatch.setattr(service.db, "apply_changes", paused)
    with ThreadPoolExecutor(2) as pool:
        deletion = pool.submit(service.delete_experiment, job["id"])
        assert entered.wait(5)
        retry = pool.submit(service.retry, job["id"])
        try:
            assert not retry.done()
        finally:
            release.set()
        assert deletion.result()["deleted"]
        with pytest.raises(KeyError):
            retry.result()
    assert service.db.get("reid_job", job["id"]) is None
    assert service.queue.unfinished_tasks == 0


def test_journal_cleanup_retry_after_database_commit(client, monkeypatch):
    service, site = client.reid_service, new_site(client)
    job = photo_upload(client, site)
    idle(service)
    journal = service.root / ".deletions" / (job["id"] + ".json")
    unlink = Path.unlink
    def locked(path, *args, **kwargs):
        if path == journal:
            raise PermissionError("Journal briefly locked")
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", locked)
    assert client.delete('/api/reid/experiments/' + job["id"]).status_code == 409
    assert service.db.get("reid_job", job["id"]) is None
    assert journal.exists()
    monkeypatch.setattr(Path, "unlink", unlink)
    assert client.delete('/api/reid/experiments/' + job["id"]).status_code == 200
    assert not journal.exists()


def test_site_isolation_and_persistent_database_removal(client):
    from iris.db import Database
    service = client.reid_service
    site_a, site_b = new_site(client), new_site(client)
    a, b = photo_upload(client, site_a), photo_upload(client, site_b)
    idle(service)
    surviving = service.tracks(b["id"])[0]
    removed = service.tracks(a["id"])[0]
    assert surviving["global_id"] == removed["global_id"] == "VEHICLE_00001"
    assert client.delete('/api/reid/experiments/' + a["id"]).status_code == 200
    reopened = Database(service.db.path)
    try:
        assert reopened.get("reid_job", a["id"]) is None
        assert reopened.get("reid_track", removed["id"]) is None
        assert reopened.get("reid_track", surviving["id"]) == surviving
        assert reopened.get_state("reid_identity_counter:" + site_a["id"]) == 1
        assert not service.get("job", b["id"]).get("refresh_required")
        assert len(service.all("identity")) == 1
    finally:
        reopened.close()
