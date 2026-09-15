import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from iris.reid_benchmark import _model_folders, binary_metrics, comparison_card_lines, parse_folder_paths, score_encoder
from test_reid import client, wait_job


def item(key, identity, source, vector):
    return ({"id": key, "identity": identity, "source_group": source, "relative_path": f"{identity}/{key}.jpg",
             "archive_path": f"images/{identity}/{key}.jpg"}, np.asarray([vector], np.float32))


def image_bytes(color):
    value = io.BytesIO()
    Image.new("RGB", (24, 18), color).save(value, "PNG")
    return value.getvalue()


def test_metric_formulas_source_exclusion_identification_and_best_threshold():
    values = [item("a1", "vehicle_a", "t0001", [1, 0]), item("a1-frame", "vehicle_a", "t0001", [1, 0]),
              item("a2", "vehicle_a", "t0002", [.9, np.sqrt(.19)]), item("b1", "vehicle_b", "t0003", [-1, 0]),
              item("b2", "vehicle_b", "t0004", [-.9, np.sqrt(.19)]), item("c1", "vehicle_c", "t0005", [0, -1])]
    items, vectors = [value[0] for value in values], {value[0]["id"]: value[1] for value in values}
    result = score_encoder(items, vectors, {"id": "test", "name": "Test", "fingerprint": "space"}, .85)
    summary = result["summary"]
    assert summary["pairwise_strict"]["pairs"] == 14
    assert summary["pairwise_all_images"]["pairs"] == 15
    assert summary["pairwise_strict"]["tp"] == 3
    assert summary["pairwise_strict"]["fp"] == summary["pairwise_strict"]["fn"] == 0
    assert summary["pairwise_strict"]["precision"] == summary["pairwise_strict"]["recall"] == 1
    assert np.isclose(summary["best_f1_exploratory"]["threshold"], .9, atol=1e-6)
    assert summary["identification"]["eligible_queries"] == 5
    assert summary["identification"]["excluded_queries"] == 1
    assert summary["identification"]["accuracy"] == summary["identification"]["rank1"] == 1
    assert next(row for row in result["per_identity"] if row["identity"] == "vehicle_c")["included_in_macro"] is False
    excluded = [row for row in result["pairs"] if not row["strict_eligible"]]
    assert len(excluded) == 1 and excluded[0]["exclusion_reason"] == "same_source_track"

    rows = [{"actual_same_identity": True, "cosine_similarity": .2, "strict_eligible": True},
            {"actual_same_identity": False, "cosine_similarity": .1, "strict_eligible": True}]
    empty_prediction = binary_metrics(rows, 1)
    assert empty_prediction["precision"] == empty_prediction["recall"] == empty_prediction["f1"] == 0
    assert empty_prediction["accuracy"] == .5
    boundary = binary_metrics(rows, .2)
    assert boundary["tp"] == 1  # cosine == threshold is a positive prediction


def test_chosen_crop_layout_counts_are_stable():
    root = Path(__file__).resolve().parents[3] / "RTMDet-Tiny" / "chosen crops"
    paths = [path.relative_to(root.parent).as_posix() for path in root.rglob("*.jpg")]
    parsed = parse_folder_paths(paths)
    assert len(parsed) == 46
    assert len({row["identity"] for row in parsed}) == 10
    assert len({(row["identity"], row["source_group"]) for row in parsed}) == 46
    all_pairs = len(parsed) * (len(parsed) - 1) // 2
    excluded = sum(a["identity"] == b["identity"] and a["source_group"] == b["source_group"]
                   for index, a in enumerate(parsed) for b in parsed[index + 1:])
    eligible_queries = sum(any(a["identity"] == b["identity"] and a["source_group"] != b["source_group"] for b in parsed) for a in parsed)
    assert (all_pairs, all_pairs - excluded, eligible_queries) == (1035, 1035, 45)


def post_benchmark(client, site, paths=None, colors=None, encoders=None):
    paths = paths or ["chosen crops/vehicle_1/t0001_f00001.png", "chosen crops/vehicle_1/t0002_f00002.png",
                      "chosen crops/vehicle_2/t0003_f00003.png", "chosen crops/vehicle_2/t0004_f00004.png"]
    colors = colors or ["red", "green", "blue", "yellow"]
    config = {"name": "Chosen crops", "site_id": site["id"], "encoders": encoders or ["dinov2", "siglip"],
              "threshold": .8, "items": [{"relative_path": path} for path in paths]}
    files = [("files", (Path(path).name, image_bytes(color), "image/png")) for path, color in zip(paths, colors)]
    return client.post("/api/reid/benchmarks", data={"config": json.dumps(config)}, files=files)


def test_benchmark_api_threshold_export_retry_and_delete(client):
    site = client.post("/api/reid/sites", json={"name": "Benchmarks"}).json()
    response = post_benchmark(client, site)
    assert response.status_code == 202, response.text
    job = wait_job(client, response.json()["id"])
    assert job["kind"] == "benchmark" and len([stage for stage in client.stage_calls if stage.startswith("embed-benchmark")]) == 2
    record = client.get(f'/api/reid/benchmarks/{job["id"]}').json()
    assert record["state"] == "completed"
    assert record["counts"] == {"identities": 2, "images": 4, "source_groups": 4,
                                 "all_pairs_per_encoder": 6, "strict_pairs_per_encoder": 6,
                                 "eligible_identification_queries": 4}
    assert len(record["encoders"]) == 2
    metrics = record["encoders"][0]["pairwise_strict"]
    assert metrics["tp"] == 2 and metrics["fp"] == 4 and metrics["recall"] == 1

    comparisons = client.get(f'/api/reid/benchmarks/{job["id"]}/comparisons', params={"limit": 2}).json()
    assert comparisons["total"] == 12 and len(comparisons["items"]) == 2
    assert [row["id"] for row in comparisons["available_encoders"]] == ["dinov2", "siglip"]
    assert comparisons["available_identities"] == ["vehicle_1", "vehicle_2"]
    assert comparisons["items"][0]["cosine_similarity"] >= comparisons["items"][1]["cosine_similarity"]
    pair = comparisons["items"][0]
    assert pair["left_filename"].endswith(".png") and pair["right_filename"].endswith(".png")
    assert client.get(pair["left_image_url"]).status_code == client.get(pair["right_image_url"]).status_code == 200
    assert client.get(f'/api/reid/benchmarks/{job["id"]}/comparisons', params={"encoder_id": "siglip"}).json()["total"] == 6
    assert client.get(f'/api/reid/benchmarks/{job["id"]}/comparisons', params={"identity": "vehicle_1"}).json()["total"] == 10
    assert client.get(f'/api/reid/benchmarks/{job["id"]}/comparisons', params={"pair_type": "same"}).json()["total"] == 4
    assert client.get(f'/api/reid/benchmarks/{job["id"]}/comparisons', params={"pair_type": "different"}).json()["total"] == 8
    assert client.get(f'/api/reid/benchmarks/{job["id"]}/comparisons', params={"eligibility": "excluded"}).json()["total"] == 0

    package = client.get(f'/api/reid/benchmarks/{job["id"]}/results.zip')
    assert package.status_code == 200 and "reid-Chosen_crops-benchmark.zip" in package.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(package.content)) as archive:
        names = set(archive.namelist())
        assert {"manifest.json", "data/summary.csv", "data/pairwise_comparisons.json",
                "data/identification_predictions.csv", "data/per_identity_metrics.json",
                "data/confusion_matrix.csv"} <= names
        assert len([name for name in names if name.startswith("images/")]) == 4
        assert b":\\" not in b"\n".join(archive.read(name) for name in names if name.endswith((".json", ".csv", ".txt")))

    stages_before_export = list(client.stage_calls)
    prepared = client.post(f'/api/reid/benchmarks/{job["id"]}/comparison-images')
    assert prepared.status_code == 202
    export_job = wait_job(client, prepared.json()["id"])
    assert export_job["kind"] == "benchmark_export"
    assert not any(stage.startswith("embed") for stage in client.stage_calls[len(stages_before_export):])
    status = client.get(f'/api/reid/benchmarks/{job["id"]}').json()["comparison_export"]
    assert status["ready"] and status["comparison_count"] == 12 and status["progress"] == 1
    assert client.post(f'/api/reid/benchmarks/{job["id"]}/comparison-images').json()["id"] == export_job["id"]
    rendered = client.get(f'/api/reid/benchmarks/{job["id"]}/comparison-images.zip')
    assert rendered.status_code == 200 and "model-comparison-images.zip" in rendered.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(rendered.content)) as archive:
        names = archive.namelist()
        assert names[:2] == ["manifest.json", "README.txt"]
        assert len([name for name in names if name.startswith("dinov2/") and name.endswith(".jpg")]) == 6
        assert len([name for name in names if name.startswith("siglip/") and name.endswith(".jpg")]) == 6
        assert {"dinov2/comparisons.csv", "siglip/comparisons.csv"} <= set(names)
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["comparison_count"] == 12 and [row["folder"] for row in manifest["models"]] == ["dinov2", "siglip"]
        assert b"cosine_similarity" in archive.read("dinov2/comparisons.csv")
        assert b"1.0" in archive.read("dinov2/comparisons.csv")
        first_jpeg = next(name for name in names if name.endswith(".jpg"))
        with Image.open(io.BytesIO(archive.read(first_jpeg))) as image:
            assert image.size == (1400, 850) and image.format == "JPEG"
        assert b":\\" not in b"\n".join(archive.read(name) for name in names if name.endswith((".json", ".csv", ".txt")))

    stored_export = client.reid_service.get("job", export_job["id"])
    stored_export["state"] = "running"; client.reid_service.put("job", stored_export)
    assert client.put(f'/api/reid/benchmarks/{job["id"]}/threshold', json={"threshold": .9}).status_code == 409
    assert client.post(f'/api/reid/jobs/{export_job["id"]}/cancel').json()["state"] == "cancelled"
    assert client.get(f'/api/reid/benchmarks/{job["id"]}').json()["comparison_export"]["state"] == "cancelled"
    assert client.get(f'/api/reid/benchmarks/{job["id"]}/comparison-images.zip').status_code == 409
    assert client.post(f'/api/reid/jobs/{export_job["id"]}/retry').status_code == 202
    wait_job(client, export_job["id"])
    assert client.get(f'/api/reid/benchmarks/{job["id"]}').json()["comparison_export"]["ready"]
    (client.reid_service.root / export_job["id"] / "benchmark-comparison-images.zip").unlink()
    replacement = client.post(f'/api/reid/benchmarks/{job["id"]}/comparison-images').json()
    assert replacement["id"] != export_job["id"]
    wait_job(client, replacement["id"])

    stages = list(client.stage_calls)
    updated = client.put(f'/api/reid/benchmarks/{job["id"]}/threshold', json={"threshold": 1}).json()
    assert updated["threshold"] == 1 and client.stage_calls == stages
    assert updated["encoders"][0]["pairwise_strict"]["tp"] == 2
    assert client.get(f'/api/reid/benchmarks/{job["id"]}/comparison-images.zip').status_code == 409
    assert client.get(f'/api/reid/benchmarks/{job["id"]}').json()["comparison_export"]["state"] == "not_prepared"
    assert client.post(f'/api/reid/jobs/{job["id"]}/retry').status_code == 202
    wait_job(client, job["id"])
    assert client.stage_calls == stages
    folder = client.reid_service.root / job["id"]
    assert client.delete(f'/api/reid/benchmarks/{job["id"]}').status_code == 200
    assert not folder.exists()
    assert client.get(f'/api/reid/benchmarks/{job["id"]}').status_code == 404
    assert client.get(f'/api/reid/jobs/{export_job["id"]}').status_code == 404
    assert client.get(f'/api/reid/jobs/{replacement["id"]}').status_code == 404


def test_comparison_card_contract_and_collision_safe_model_folders():
    row = {"encoder_id": "first", "encoder_name": "Same model", "encoder_fingerprint": "1234567890abcdef",
           "cosine_similarity": .87654, "actual_same_identity": True, "predicted_same_configured": True,
           "predicted_same_best_f1": False, "best_f1_threshold": .9, "strict_eligible": False,
           "exclusion_reason": "same_source_track"}
    assert comparison_card_lines(row, .8) == [
        "Same model | Cosine 0.877", "Ground truth: Same truck", "Configured: same", "Best F1: different",
        "Configured threshold: 0.800", "Exploratory best threshold: 0.900",
        "Strict metric: Excluded · same_source_track", "Encoder fingerprint: 1234567890ab"]
    folders = _model_folders([row, {**row, "encoder_id": "second"}])
    assert folders == {"first": "Same_model__first", "second": "Same_model__second"}


def test_benchmark_comparison_export_limit(client, monkeypatch):
    from iris import reid_benchmark
    site = client.post("/api/reid/sites", json={"name": "Export limit"}).json()
    job = wait_job(client, post_benchmark(client, site, encoders=["dinov2"]).json()["id"])
    monkeypatch.setattr(reid_benchmark, "COMPARISON_EXPORT_LIMIT", 5)
    response = client.post(f'/api/reid/benchmarks/{job["id"]}/comparison-images')
    assert response.status_code == 422 and "limited to 5 images" in response.text


def test_benchmark_comparison_export_rejects_corrupt_source_atomically(client):
    site = client.post("/api/reid/sites", json={"name": "Corrupt export"}).json()
    job = wait_job(client, post_benchmark(client, site, encoders=["dinov2"]).json()["id"])
    _, package = client.reid_service.benchmark_package(job["id"])
    package.write_bytes(b"not a zip")
    response = client.post(f'/api/reid/benchmarks/{job["id"]}/comparison-images')
    assert response.status_code == 202
    export_id = response.json()["id"]
    for _ in range(100):
        export = client.get(f"/api/reid/jobs/{export_id}").json()
        if export["state"] not in ("queued", "running"):
            break
        time.sleep(.02)
    assert export["state"] == "failed" and "corrupt" in export["error"]
    directory = client.reid_service.root / export_id
    assert not (directory / "benchmark-comparison-images.zip").exists()
    assert not (directory / "benchmark-comparison-images.zip.partial").exists()
    assert client.get(f'/api/reid/benchmarks/{job["id"]}/comparison-images.zip').status_code == 409


def test_benchmark_result_state_guards(client):
    site = client.post("/api/reid/sites", json={"name": "Benchmark guards"}).json()
    response = post_benchmark(client, site, encoders=["dinov2"])
    job = wait_job(client, response.json()["id"])
    service = client.reid_service
    stored = service.get("job", job["id"])
    stored["state"] = "running"
    service.put("job", stored)
    assert client.get(f'/api/reid/benchmarks/{job["id"]}/results.zip').status_code == 409
    assert client.get(f'/api/reid/benchmarks/{job["id"]}/comparisons').status_code == 409
    assert client.post(f'/api/reid/benchmarks/{job["id"]}/comparison-images').status_code == 409
    assert client.get(f'/api/reid/benchmarks/{job["id"]}/comparison-images.zip').status_code == 409
    assert client.put(f'/api/reid/benchmarks/{job["id"]}/threshold', json={"threshold": .7}).status_code == 409
    assert client.delete(f'/api/reid/benchmarks/{job["id"]}').status_code == 409
    stored["state"] = "completed"
    stored["deletion_pending"] = True
    service.put("job", stored)
    assert client.get(f'/api/reid/benchmarks/{job["id"]}/results.zip').status_code == 409
    assert client.get(f'/api/reid/benchmarks/{job["id"]}/comparisons').status_code == 409
    assert client.get("/api/reid/benchmarks/does-not-exist").status_code == 404
    assert client.get("/api/reid/benchmarks/does-not-exist/results.zip").status_code == 404
    assert client.get("/api/reid/benchmarks/does-not-exist/comparisons").status_code == 404


def test_benchmark_rejects_bad_layout_duplicates_and_traversal(client):
    site = client.post("/api/reid/sites", json={"name": "Invalid benchmarks"}).json()
    assert post_benchmark(client, site, paths=["root/only/a.png", "root/only/b.png"], colors=["red", "blue"]).status_code == 422
    assert post_benchmark(client, site, colors=["red", "red", "blue", "yellow"]).status_code == 422
    traversal = ["../vehicle_1/a.png", "../vehicle_1/b.png", "../vehicle_2/c.png"]
    assert post_benchmark(client, site, paths=traversal, colors=["red", "green", "blue"]).status_code == 422
