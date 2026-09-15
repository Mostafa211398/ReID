import csv
import io
import json
import zipfile

import numpy as np

from iris.reid_core import unit
from iris.reid_schemas import Job
from test_reid import client
from test_reid_images import associate, new_site, observation, photo_upload


def open_package(response):
    assert response.status_code == 200, response.text
    return zipfile.ZipFile(io.BytesIO(response.content))


def test_complete_package_has_portable_tables_images_and_exact_cosines(client, tmp_path):
    site = new_site(client)
    first = photo_upload(client, site)
    assert "Complete results (.zip)" in first["artifacts"]
    with open_package(client.get(f'/api/reid/experiments/{first["id"]}/results.zip')) as archive:
        names = set(archive.namelist())
        required = {
            "manifest.json", "README.txt", "data/observations.csv", "data/observations.json",
            "data/cosine_comparisons.csv", "data/cosine_comparisons.json", "data/results.csv", "data/results.json",
        }
        assert required <= names
        observations = json.loads(archive.read("data/observations.json"))
        comparisons = json.loads(archive.read("data/cosine_comparisons.json"))
        assert len(observations) == 2 and comparisons == []
        assert {row["decision_status"] for row in observations} == {"new"}
        assert all(row["query_image"] in names for row in observations)
        assert len([name for name in names if name.startswith("images/queries/")]) == 1
        assert len([name for name in names if name.startswith("images/annotated/")]) == 2
        assert list(csv.DictReader(io.StringIO(archive.read("data/cosine_comparisons.csv").decode()))) == []

    second = photo_upload(client, site)
    response = client.get(f'/api/reid/experiments/{second["id"]}/results.zip')
    assert "reid-Photo_test-results.zip" in response.headers["content-disposition"]
    with open_package(response) as archive:
        names = set(archive.namelist())
        comparisons = json.loads(archive.read("data/cosine_comparisons.json"))
        observations = json.loads(archive.read("data/observations.json"))
        csv_rows = list(csv.DictReader(io.StringIO(archive.read("data/cosine_comparisons.csv").decode())))
        assert len(comparisons) == len(csv_rows) == 2
        assert {row["encoder_id"] for row in comparisons} == {"dinov2", "siglip"}
        assert all(row["cosine_similarity"] == 1.0 and row["cosine_rank"] == 1 for row in comparisons)
        assert all(row["query_image"] in names and row["reference_image"] in names for row in comparisons)
        assert len({row["query_image"] for row in comparisons}) == 1
        assert len({row["reference_image"] for row in comparisons}) == 1
        detected = {row["detected_vehicle_id"] for row in observations}
        assert detected == {"VEHICLE_00001"}
        assert {row["decision_status"] for row in observations} == {"automatic", "review"}
        text = b"\n".join(archive.read(name) for name in names if name.endswith((".json", ".csv", ".txt")))
        assert str(tmp_path).encode() not in text and b":\\" not in text
        assert all(not name.startswith(("/", "\\")) and ".." not in name.split("/") for name in names)

    track = client.reid_service.tracks(second["id"])[0]
    assert client.put(f'/api/reid/tracks/{track["id"]}/review', json={"global_id": track["global_id"]}).status_code == 200
    before = response.content
    assert client.post(f'/api/reid/jobs/{second["id"]}/retry').status_code == 202
    from test_reid import wait_job
    wait_job(client, second["id"])
    refreshed = client.get(f'/api/reid/experiments/{second["id"]}/results.zip')
    assert refreshed.content != before
    with open_package(refreshed) as archive:
        rows = json.loads(archive.read("data/observations.json"))
        assert {row["decision_status"] for row in rows} == {"reviewed"}
        assert {row["provenance"] for row in rows} == {"human"}


def test_comparison_audit_ranks_all_eligible_references_and_exclusions(client):
    service, site = client.reid_service, new_site(client)
    first = associate(service, observation(service, site, "first"), [1, 0])
    second = associate(service, observation(service, site, "second"), [0, 1])
    third_job = observation(service, site, "third")
    third = associate(service, third_job, [1, 0])
    client.put('/api/reid/tracks/third/review', json={"global_id": first["global_id"]})
    vector = unit([[0.8, 0.6]])
    spec = {"id": "dinov2", "name": "DINOv2", "fingerprint": "test-dinov2"}

    all_rows = service._match(observation(service, site, "all-query"), spec, {"all-query": vector})
    assert [(row["reference_observation_id"], row["cosine_rank"]) for row in all_rows] == [
        (first["id"], 1), (third["id"], 2), (second["id"], 3),
    ]
    assert np.allclose([row["cosine_similarity"] for row in all_rows], [0.8, 0.8, 0.6])
    assert {row["candidate_vehicle_id"] for row in all_rows} == {first["global_id"], second["global_id"]}

    query_job = observation(service, site, "query", clip="second")
    query = service.get("track", "query")
    query["source_track_id"] = "first"
    service.put("track", query)
    rows = service._match(query_job, spec, {"query": vector})

    # The original first crop shares the source record; the second identity is
    # simultaneous in the same clip. The other trusted crop remains eligible.
    assert [(row["reference_observation_id"], row["cosine_rank"]) for row in rows] == [(third["id"], 1)]
    assert np.isclose(rows[0]["cosine_similarity"], 0.8)
    assert rows[0]["candidate_vehicle_id"] == first["global_id"]
    assert rows[0]["encoder_fingerprint"] == "test-dinov2"

    truck_job = observation(service, site, "truck-query", class_name="truck")
    assert service._match(truck_job, spec, {"truck-query": vector}) == []
    incompatible = {**spec, "fingerprint": "another-space"}
    assert service._match(observation(service, site, "other-space"), incompatible, {"other-space": vector}) == []


def test_comparison_view_api_filters_paginates_and_enriches_images(client):
    site = new_site(client)
    first = photo_upload(client, site)
    empty = client.get(f'/api/reid/experiments/{first["id"]}/comparisons')
    assert empty.status_code == 200 and empty.json()["total"] == 0

    second = photo_upload(client, site)
    query = client.reid_service.tracks(second["id"])[0]
    reference = client.reid_service.tracks(first["id"])[0]
    response = client.get(f'/api/reid/experiments/{second["id"]}/comparisons', params={"limit": 1})
    assert response.status_code == 200
    page = response.json()
    assert page["total"] == 2 and len(page["items"]) == 1 and page["offset"] == 0 and page["limit"] == 1
    assert [item["id"] for item in page["available_encoders"]] == ["dinov2", "siglip"]
    assert page["available_vehicle_ids"] == ["VEHICLE_00001"]
    assert page["available_observations"] == [{"id": query["id"], "filename": "vehicle.png"}]
    row = page["items"][0]
    assert row["query_observation_id"] == query["id"] and row["reference_observation_id"] == reference["id"]
    assert row["query_filename"] == row["reference_filename"] == "vehicle.png"
    assert row["query_image_url"] == f'/api/reid/tracks/{query["id"]}/crops/0'
    assert row["reference_image_url"] == f'/api/reid/tracks/{reference["id"]}/crops/0'
    assert client.get(row["query_image_url"]).status_code == client.get(row["reference_image_url"]).status_code == 200

    filtered = client.get(f'/api/reid/experiments/{second["id"]}/comparisons', params={
        "observation_id": query["id"], "encoder_id": "siglip", "candidate_vehicle_id": "VEHICLE_00001",
        "decision_flag": "winning", "offset": 0, "limit": 24,
    }).json()
    assert filtered["total"] == 1 and filtered["items"][0]["encoder_id"] == "siglip"
    assert client.get(f'/api/reid/experiments/{second["id"]}/comparisons',
                      params={"decision_flag": "assigned"}).json()["total"] == 1
    assert client.get(f'/api/reid/experiments/{second["id"]}/comparisons',
                      params={"offset": 1, "limit": 1}).json()["items"][0]["encoder_id"] == "siglip"


def test_results_zip_state_guards_and_reference_deletion_cleanup(client):
    service, site = client.reid_service, new_site(client)
    assert client.get("/api/reid/experiments/missing/results.zip").status_code == 404
    assert client.get("/api/reid/experiments/missing/comparisons").status_code == 404
    other = service.put("job", Job(name="Evaluation", site_id=site["id"], kind="evaluation", state="completed"))
    assert client.get(f'/api/reid/experiments/{other["id"]}/results.zip').status_code == 404
    assert client.get(f'/api/reid/experiments/{other["id"]}/comparisons').status_code == 404

    first, second = photo_upload(client, site), photo_upload(client, site)
    package = service.root / second["id"] / "complete-results.zip"
    assert package.is_file()
    value = service.get("job", second["id"])
    value["state"] = "running"
    service.put("job", value)
    assert client.get(f'/api/reid/experiments/{second["id"]}/results.zip').status_code == 409
    assert client.get(f'/api/reid/experiments/{second["id"]}/comparisons').status_code == 409
    value["state"] = "completed"
    value["deletion_pending"] = True
    service.put("job", value)
    assert client.get(f'/api/reid/experiments/{second["id"]}/results.zip').status_code == 409
    assert client.get(f'/api/reid/experiments/{second["id"]}/comparisons').status_code == 409
    value["deletion_pending"] = False
    service.put("job", value)

    assert client.delete(f'/api/reid/experiments/{first["id"]}').status_code == 200
    assert service.get("job", second["id"])["refresh_required"]
    assert not package.exists()
    assert client.get(f'/api/reid/experiments/{second["id"]}/results.zip').status_code == 409
    assert client.get(f'/api/reid/experiments/{second["id"]}/comparisons').status_code == 409
