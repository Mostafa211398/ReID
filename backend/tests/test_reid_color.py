import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from iris.reid_color import COLOR_VERSION, ColorCache, best_pair, extract_color, scoring_spec
from iris.reid_core import candidates, retrieval, track_vector
from test_reid import client, row, wait_job
from test_reid_images import associate, new_site, observation


def solid(bgr):
    return np.full((80, 100, 3), bgr, np.uint8)


def color(bgr):
    return extract_color(solid(bgr))


def test_color_vectors_hue_gray_and_brightness():
    red, blue = color((0, 0, 255)), color((255, 0, 0))
    for value in (red, blue):
        assert len(value["vector"]) == 72
        assert np.linalg.norm(value["vector"]) == pytest.approx(1)
    assert np.dot(red["vector"], blue["vector"]) < .1
    assert np.dot(red["vector"], color((0, 0, 185))["vector"]) > .95
    white, gray, black = [color((v, v, v))["vector"] for v in (245, 128, 20)]
    assert np.dot(white, gray) < .2 and np.dot(gray, black) < .2 and np.dot(white, black) < .2
    assert extract_color(None)["method"] == "unavailable"


def test_foreground_estimation_background_and_fallback(monkeypatch):
    images = []
    for background in ((110, 110, 110), (230, 30, 20)):
        image = solid(background)
        image[15:65, 20:80] = (0, 0, 255)
        images.append(extract_color(image))
    assert all(value["method"] == "grabcut" for value in images)
    assert np.dot(images[0]["vector"], images[1]["vector"]) > .95
    repeated = extract_color(solid((0, 0, 255)))
    assert repeated == extract_color(solid((0, 0, 255)))
    monkeypatch.setattr(cv2, "grabCut", lambda *args: (_ for _ in ()).throw(cv2.error("failed")))
    assert color((0, 0, 255))["method"] == "center-weighted"


def test_weighted_scoring_and_same_pair_constraint():
    red, blue = color((0, 0, 255)), color((255, 0, 0))
    different_color = best_pair([[1, 0]], [[.8, .6]], [red], [blue], .25)
    assert different_color["appearance_similarity"] == pytest.approx(.8)
    assert different_color["combined_score"] == pytest.approx(.6)
    assert best_pair([[1, 0]], [[.8, .6]], [red], [blue], 0)["similarity"] == pytest.approx(.8)
    same_color_different_shape = best_pair([[1, 0]], [[.2, np.sqrt(.96)]], [red], [red], .25)
    assert same_color_different_shape["similarity"] == pytest.approx(.4)
    # Appearance is best in the blue reference; color is best in the red reference.
    # Illegally combining these separate views would give 1.0 instead of 0.75.
    result = best_pair([[1, 0]], [[1, 0], [0, 1]], [red], [blue, red], .25)
    assert result["similarity"] == pytest.approx(.75)
    assert result["reference_crop_index"] == 0 and result["color_similarity"] < .1
    missing = best_pair([[1, 0]], [[.8, .6]], [], [red], .25)
    assert missing["similarity"] == pytest.approx(.8) and missing["color_similarity"] is None
    assert missing["effective_color_weight"] == 0


def test_color_cache_reuse_content_change_and_version(tmp_path, monkeypatch):
    path = tmp_path / "crop.png"
    cv2.imwrite(str(path), solid((0, 0, 255)))
    cache = ColorCache(tmp_path / "cache")
    first = cache.crop({"path": str(path)})
    import iris.reid_color as module
    original = module.extract_color
    monkeypatch.setattr(module, "extract_color", lambda _: pytest.fail("Cached crop was recomputed"))
    assert cache.crop({"path": str(path)}) == first
    monkeypatch.setattr(module, "extract_color", original)
    cv2.imwrite(str(path), solid((255, 0, 0)))
    second = cache.crop({"path": str(path)})
    assert second["source_hash"] != first["source_hash"]
    stored = cache.root / (second["source_hash"] + ".json")
    stored.write_text(json.dumps({**second, "color_version": "old-version", "vector": [0] * 72}))
    assert cache.crop({"path": str(path)}) == second
    assert cache.crop({"path": str(tmp_path / "missing")})["method"] == "unavailable"
    assert second["color_version"] == COLOR_VERSION


def test_ranks_all_identities_before_top_five():
    query = row("query", "q")
    galleries = {str(index): [[1, 0]] for index in range(8)}
    options = candidates(query, [[1, 0]], galleries, {}, [], lambda gid: {"similarity": int(gid) / 10})
    assert [option["global_id"] for option in options] == ["7", "6", "5", "4", "3"]


def test_live_color_penalty_calibration_and_refresh(client, tmp_path):
    service, site = client.reid_service, new_site(client)
    jobs = []
    for key, paint in (("red-truck", (0, 0, 255)), ("blue-truck", (255, 0, 0))):
        job = observation(service, site, key)
        track = service.get("track", key)
        crop = tmp_path / (key + ".png")
        cv2.imwrite(str(crop), solid(paint))
        track["crops"] = [{"path": str(crop), "frame": 0, "timestamp": 0}]
        service.put("track", track)
        jobs.append(job)
    first = associate(service, jobs[0], [1, 0])
    service.put("calibration", {"id": site["id"] + "_test-dinov2", "threshold": .1, "margin": 0})
    second = associate(service, jobs[1], [.99, np.sqrt(1-.99**2)])
    assignment = second["assignments"]["dinov2"]
    assert second["global_id"] is None  # High appearance similarity no longer automatically merges these colors.
    assert assignment["provenance"] == "experimental"
    assert assignment["appearance_similarity"] == pytest.approx(.99)
    assert assignment["combined_score"] < .85
    version = scoring_spec("test-dinov2", .25)
    assert assignment["scoring_version"] == version["fingerprint"]
    service.put("calibration", {"id": site["id"] + "_" + version["fingerprint"], "threshold": .98, "margin": 0})
    assert service.match_calibration(site["id"], version)["threshold"] == .98
    assert service.match_calibration(site["id"], scoring_spec("test-dinov2", .3)) is None
    site["matching"]["color_weight"] = 0
    service.put("site", site)
    refreshed = associate(service, jobs[1], [.99, np.sqrt(1-.99**2)])
    assert refreshed["global_id"] == first["global_id"]
    assert service.get("track", "red-truck")["global_id"] == first["global_id"]


def test_combined_evaluation_retains_appearance_benchmark(client):
    service = client.reid_service
    rows = [row("a1", "A"), row("a2", "A", "B"), row("b1", "B"), row("b2", "B", "B")]
    for track in rows:
        track["class_name"] = "truck"
    red, blue = color((0, 0, 255)), color((255, 0, 0))
    colors = {r["id"]: [red if r["identity"] == "A" else blue] for r in rows}
    crops = {"a1": [[1, 0]], "a2": [[.9, np.sqrt(.19)]], "b1": [[.9, np.sqrt(.19)]], "b2": [[1, 0]]}
    vectors = {k: track_vector(v) for k, v in crops.items()}
    original = retrieval(rows, vectors)
    combined = retrieval(rows, vectors, pair_score=lambda a, b: best_pair(crops[a["id"]], crops[b["id"]], colors[a["id"]], colors[b["id"]], .25)["similarity"])
    assert original["rank1"] == 0 and combined["rank1"] == 1
    association = service.association_queries(rows, crops, colors, .25)
    assert all(q["correct"] for q in association)


def test_evaluation_api_freezes_color_policy_and_exports_both_metrics(client, tmp_path):
    service, site = client.reid_service, new_site(client)
    rows = [row("a1", "A"), row("a2", "A", "B"), row("b1", "B"), row("b2", "B", "B")]
    for track in rows:
        path = tmp_path / (track["id"] + ".png")
        cv2.imwrite(str(path), solid((0, 0, 255) if track["identity"] == "A" else (255, 0, 0)))
        track["crops"] = [{"path": str(path), "frame": 0, "timestamp": 0}]
    service.put("dataset", {"id": "color-evaluation", "site_id": site["id"], "rows": rows,
                            "fingerprint": "frozen-test", "splits": {"validation": ["A", "B"], "test": ["A", "B"]}})
    response = client.post('/api/reid/evaluations', json={"dataset_id": "color-evaluation", "encoder_id": "dinov2", "split": "validation"})
    assert response.status_code == 202
    job = wait_job(client, response.json()["id"])
    metrics = job["metrics"]
    assert metrics["combined_cross_camera"]["rank1"] == 1
    assert metrics["color_crops_available"] == metrics["color_crops_total"] == 4
    assert "rank1" in metrics["cross_camera"]
    exported = client.get(job["artifacts"]["evaluation.json"]).json()
    assert exported["scoring"]["color_weight"] == .25
    site["matching"]["color_weight"] = .4
    service.put("site", site)
    client.post('/api/reid/jobs/' + job["id"] + '/retry')
    repeated = wait_job(client, job["id"])
    assert repeated["metrics"]["scoring"] == metrics["scoring"]
