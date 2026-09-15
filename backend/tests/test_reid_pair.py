import io
import json
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from iris.reid_pair import (CRITERIA_PROMPT_GROUPS, DEFAULT_WEIGHTS, SEMANTIC_GROUPS, build_criteria_report,
                            build_result, color_summary, detail_feature, edge_scale, shape_feature,
                            texture_feature, validate_weights)
from test_reid import client, wait_job


def image_bytes(color):
    value = io.BytesIO()
    Image.new("RGB", (80, 52), color).save(value, "PNG")
    return value.getvalue()


def test_shape_and_pair_scoring_contract():
    rgb = np.zeros((60, 100, 3), np.uint8)
    rgb[10:50, 20:80] = 255
    shape = shape_feature(rgb)
    assert shape.shape == (8100,)
    assert np.isclose(np.linalg.norm(shape), 1)
    with pytest.raises(ValueError, match="add up"):
        validate_weights({"appearance": .6, "color": .2, "shape": .2, "semantic": .2})
    arrays = {
        "appearance": np.array([[1., 0.], [.8, .6]], np.float32),
        "color": np.array([[1., 0.], [1., 0.]], np.float32),
        "color_available": np.ones(2, np.bool_),
        "shape": np.array([[1., 0.], [1., 0.]], np.float32),
        "shape_available": np.ones(2, np.bool_),
        "semantic": np.array([[1., 0.], [1., 0.]], np.float32),
    }
    comparison = {"id": "pair", "name": "Pair", "site_id": "site", "threshold": .94,
                  "weights": DEFAULT_WEIGHTS, "images": [{"side": "first", "filename": "a.jpg"},
                                                            {"side": "second", "filename": "b.jpg"}]}
    metadata = {"color": [{"method": "test"}, {"method": "test"}],
                "semantic_attributes": [{}, {}]}
    result = build_result(comparison, {"id": "coca", "name": "CoCa", "family": "coca", "fingerprint": "fp"}, arrays, metadata)
    assert np.isclose(result["similarities"]["appearance_cosine_raw"], .8)
    assert np.isclose(result["similarities"]["appearance"], .9)
    assert np.isclose(result["similarities"]["combined"], .94)
    assert result["decision"]["same_vehicle"]  # equality at the threshold is positive
    first_scoring_fingerprint = result["scoring"]["fingerprint"]
    comparison["threshold"] = .95
    assert build_result(comparison, {"id": "coca", "name": "CoCa", "family": "coca", "fingerprint": "fp"}, arrays, metadata)["scoring"]["fingerprint"] != first_scoring_fingerprint
    comparison["threshold"] = .94
    arrays["color_available"][1] = False
    missing = build_result(comparison, {"id": "coca", "name": "CoCa", "family": "coca", "fingerprint": "fp"}, arrays, metadata)
    assert missing["similarities"]["color"] is None
    assert missing["scoring"]["effective_weights"]["color"] == 0
    assert np.isclose(sum(missing["scoring"]["effective_weights"].values()), 1)


def test_deterministic_criteria_metrics_and_report():
    rgb = np.zeros((80, 120, 3), np.uint8); rgb[10:70, 15:105] = (220, 30, 30)
    assert texture_feature(rgb).shape == (256,)
    assert detail_feature(rgb).shape == (34,)
    assert 0 < edge_scale(rgb) <= 1
    assert "red" in color_summary(rgb)["dominant_colors"]
    groups = {**SEMANTIC_GROUPS, **CRITERIA_PROMPT_GROUPS}
    attributes = [{group: {"label": values[0][0], "confidence": 1.,
                            "distribution": {label: float(index == 0) for index, (label, _) in enumerate(values)}}
                   for group, values in groups.items()} for _ in range(2)]
    arrays = {"texture": np.stack([texture_feature(rgb)] * 2), "texture_available": np.ones(2, bool),
              "detail": np.stack([detail_feature(rgb)] * 2), "detail_available": np.ones(2, bool),
              "edge_scale": np.array([.8, .4])}
    report = build_criteria_report(arrays, {"criteria_attributes": attributes,
                                           "color_summary": [color_summary(rgb), color_summary(rgb)]}, 1., 1., 1.)
    assert report["criteria_similarity_score"] == pytest.approx(1.)
    assert report["qualitative_band"] == "high" and report["shape_extractor"]["scale_ratio"] == .5
    assert report["plate_logo_type_viewpoint"]["model_text"]["image_1"]["label"] == "unknown"
    assert report["plate_logo_type_viewpoint"]["license_plate"]["identity_claim"] == "not evaluated"
    arrays["texture_available"][1] = False
    missing = build_criteria_report(arrays, {"criteria_attributes": attributes,
                                             "color_summary": [color_summary(rgb), color_summary(rgb)]}, 1., 1., 1.)
    assert missing["availability"]["texture"] is False
    assert sum(missing["effective_weights"].values()) == pytest.approx(1.)


def test_coca_download_is_revision_pinned_and_rejects_bad_checkpoint(monkeypatch, tmp_path):
    import huggingface_hub
    from iris import reid_setup

    payload = b"invalid checkpoint"

    def fake_snapshot(repository, *, revision, local_dir, allow_patterns):
        assert repository == "laion/CoCa-ViT-B-32-laion2B-s13B-b90k"
        assert revision == "47aff38863cd40aa76d915bb04e4a3d8edf5824c"
        assert allow_patterns == ["open_clip_pytorch_model.bin"]
        destination = Path(local_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "open_clip_pytorch_model.bin").write_bytes(payload)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    monkeypatch.setattr(reid_setup, "COCA_SIZE", len(payload))
    with pytest.raises(ValueError, match="size or SHA-256"):
        reid_setup.install(tmp_path / "bad-hash", ["coca"])
    monkeypatch.setattr(reid_setup, "COCA_CHECKSUM", reid_setup.digest(tmp_path / "bad-hash" / "coca" / "open_clip_pytorch_model.bin"))
    monkeypatch.setattr(reid_setup, "COCA_SIZE", len(payload) + 1)
    with pytest.raises(ValueError, match="size or SHA-256"):
        reid_setup.install(tmp_path / "bad-size", ["coca"])
    with pytest.raises(ValueError, match="pinned full CoCa"):
        reid_setup.extract_coca_visual(tmp_path / "missing-parent", {
            "revision": "47aff38863cd40aa76d915bb04e4a3d8edf5824c"})


def test_coca_l14_download_is_revision_pinned_and_rejects_bad_checkpoint(monkeypatch, tmp_path):
    import huggingface_hub
    from iris import reid_setup

    payload = b"invalid l14 checkpoint"

    def fake_snapshot(repository, *, revision, local_dir, allow_patterns):
        assert repository == "laion/CoCa-ViT-L-14-laion2B-s13B-b90k"
        assert revision == "74207cb7fde8eafc9864451ebd332fa8e75b150f"
        assert allow_patterns == ["open_clip_pytorch_model.bin"]
        destination = Path(local_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "open_clip_pytorch_model.bin").write_bytes(payload)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    monkeypatch.setattr(reid_setup, "COCA_L14_SIZE", len(payload))
    with pytest.raises(ValueError, match="size or SHA-256"):
        reid_setup.install(tmp_path / "bad-hash", ["coca_l14"])
    monkeypatch.setattr(reid_setup, "COCA_L14_CHECKSUM",
                        reid_setup.digest(tmp_path / "bad-hash" / "coca_l14" / "open_clip_pytorch_model.bin"))
    monkeypatch.setattr(reid_setup, "COCA_L14_SIZE", len(payload) + 1)
    with pytest.raises(ValueError, match="size or SHA-256"):
        reid_setup.install(tmp_path / "bad-size", ["coca_l14"])
    with pytest.raises(ValueError, match="pinned full CoCa ViT-L/14"):
        reid_setup.extract_coca_l14_visual(tmp_path / "missing-parent", {
            "revision": reid_setup.HF_REVISIONS["coca_l14"]})


def test_siglip2_download_is_revision_pinned_and_rejects_bad_checkpoint(monkeypatch, tmp_path):
    import huggingface_hub
    from iris import reid_setup

    payload = b"test siglip2 checkpoint"

    def fake_snapshot(repository, *, revision, local_dir, allow_patterns):
        assert repository == "google/siglip2-base-patch16-384"
        assert revision == "f775b65a79762255128c981547af89addcfe0f88"
        assert allow_patterns == reid_setup.SIGLIP2_FILES
        destination = Path(local_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for name in allow_patterns:
            (destination / name).write_bytes(payload if name == "model.safetensors" else b"{}")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    monkeypatch.setattr(reid_setup, "SIGLIP2_SIZE", len(payload))
    with pytest.raises(ValueError, match="size or SHA-256"):
        reid_setup.install(tmp_path / "bad-hash", ["siglip2"])
    monkeypatch.setattr(reid_setup, "SIGLIP2_CHECKSUM",
                        reid_setup.digest(tmp_path / "bad-hash" / "siglip2" / "model.safetensors"))
    manifest_root = tmp_path / "valid"
    reid_setup.install(manifest_root, ["siglip2"])
    manifest = json.loads((manifest_root / "manifest.json").read_text())
    assert manifest["siglip2"]["revision"] == reid_setup.HF_REVISIONS["siglip2"]
    assert manifest["siglip2"]["sizes"]["model.safetensors"] == len(payload)


def test_siglip2_visual_extraction_matches_full_tiny_model(monkeypatch, tmp_path):
    import torch
    from transformers import SiglipConfig, SiglipModel, SiglipTextConfig, SiglipVisionConfig, SiglipVisionModel
    from iris import reid_setup

    full_dir = tmp_path / "siglip2"
    text = SiglipTextConfig(vocab_size=20, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                            num_attention_heads=2, max_position_embeddings=4)
    vision = SiglipVisionConfig(hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                                num_attention_heads=2, image_size=8, patch_size=4)
    full = SiglipModel(SiglipConfig.from_text_vision_configs(text, vision)).eval()
    full.save_pretrained(full_dir, safe_serialization=True)
    (full_dir / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    source = full_dir / "model.safetensors"
    monkeypatch.setattr(reid_setup, "SIGLIP2_CHECKSUM", reid_setup.digest(source))
    monkeypatch.setattr(reid_setup, "SIGLIP2_SIZE", source.stat().st_size)
    parent = {"revision": reid_setup.HF_REVISIONS["siglip2"], "checksums": {
        name: reid_setup.digest(full_dir / name)
        for name in ("model.safetensors", "config.json", "preprocessor_config.json")}}
    record = reid_setup.extract_siglip2_visual(tmp_path, parent)
    visual = SiglipVisionModel.from_pretrained(tmp_path / "siglip2_visual", local_files_only=True).eval()
    pixels = torch.rand(2, 3, 8, 8)
    with torch.inference_mode():
        expected = full.get_image_features(pixel_values=pixels)
        actual = visual(pixel_values=pixels).pooler_output
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert record["parent_sha256"] == reid_setup.digest(source)


def post_pair(client, site, colors=("red", "blue"), file_count=2):
    config = {"name": "Two trucks", "site_id": site["id"], "encoder_id": "coca", "threshold": .75,
              "weights": DEFAULT_WEIGHTS}
    files = [("files", (f"truck-{index}.png", image_bytes(color), "image/png"))
             for index, color in enumerate(colors[:file_count])]
    return client.post("/api/reid/pair-comparisons", data={"config": json.dumps(config)}, files=files)


def test_pair_comparison_api_recalculation_retry_export_and_delete(client):
    site = client.post("/api/reid/sites", json={"name": "Pair site"}).json()
    response = post_pair(client, site)
    assert response.status_code == 202, response.text
    job = wait_job(client, response.json()["id"])
    comparison_id = job["id"]
    assert job["kind"] == "comparison"
    assert len([stage for stage in client.stage_calls if stage.startswith("pair-features-")]) == 1
    record = client.get(f"/api/reid/pair-comparisons/{comparison_id}").json()
    assert record["state"] == "completed" and record["result"]["decision"]["same_vehicle"]
    assert all("path" not in image and image["image_url"].startswith("/api/reid/") for image in record["images"])
    assert all(client.get(image["image_url"]).status_code == 200 for image in record["images"])
    download = client.get(f"/api/reid/pair-comparisons/{comparison_id}/results.json")
    assert download.status_code == 200
    result = download.json()
    assert len(result["images"][0]["vectors"]["appearance"]) == 2
    assert len(result["images"][0]["vectors"]["color"]) == 72
    assert len(result["images"][0]["vectors"]["shape"]) == 8100
    assert len(result["images"][0]["vectors"]["semantic"]) == 24
    assert result["encoder"]["checkpoint"]["filename"] == "open_clip_pytorch_model.bin"
    assert result["criteria_report"]["version"] == "criteria-v1"
    assert str(client.reid_service.root) not in json.dumps(result)

    stages = list(client.stage_calls)
    updated = client.put(f"/api/reid/pair-comparisons/{comparison_id}/scoring", json={
        "threshold": .99, "weights": {"appearance": 1, "color": 0, "shape": 0, "semantic": 0}
    }).json()
    assert updated["threshold"] == .99 and not updated["result"]["decision"]["same_vehicle"]
    assert client.stage_calls == stages
    assert client.post(f"/api/reid/jobs/{comparison_id}/retry").status_code == 202
    wait_job(client, comparison_id)
    assert client.stage_calls == stages
    folder = client.reid_service.root / comparison_id
    assert client.delete(f"/api/reid/pair-comparisons/{comparison_id}").status_code == 200
    assert not folder.exists()
    assert client.delete(f"/api/reid/pair-comparisons/{comparison_id}").status_code == 200
    assert client.get(f"/api/reid/pair-comparisons/{comparison_id}").status_code == 404


def test_pair_upload_and_state_validation(client):
    site = client.post("/api/reid/sites", json={"name": "Pair validation"}).json()
    assert post_pair(client, site, file_count=1).status_code == 422
    duplicate = post_pair(client, site, colors=("red", "red"))
    assert duplicate.status_code == 422 and "different" in duplicate.text
    invalid = client.post("/api/reid/pair-comparisons", data={"config": json.dumps({
        "name": "Bad", "site_id": site["id"], "encoder_id": "coca", "threshold": .75,
        "weights": DEFAULT_WEIGHTS})}, files=[("files", ("a.txt", b"x", "text/plain")),
                                            ("files", ("b.png", image_bytes("blue"), "image/png"))])
    assert invalid.status_code == 422
    invalid_content = client.post("/api/reid/pair-comparisons", data={"config": json.dumps({
        "name": "Bad image", "site_id": site["id"], "encoder_id": "coca", "threshold": .75,
        "weights": DEFAULT_WEIGHTS})}, files=[("files", ("a.jpg", b"not an image", "image/jpeg")),
                                            ("files", ("b.png", image_bytes("blue"), "image/png"))])
    assert invalid_content.status_code == 422
    unsafe_name = client.post("/api/reid/pair-comparisons", data={"config": json.dumps({
        "name": "Unsafe", "site_id": site["id"], "encoder_id": "coca", "threshold": .75,
        "weights": DEFAULT_WEIGHTS})}, files=[("files", ("../a.png", image_bytes("red"), "image/png")),
                                            ("files", ("b.png", image_bytes("blue"), "image/png"))])
    assert unsafe_name.status_code == 422

    response = post_pair(client, site)
    comparison_id = response.json()["id"]
    wait_job(client, comparison_id)
    job = client.reid_service.get("job", comparison_id)
    job["state"] = "running"; client.reid_service.put("job", job)
    assert client.get(f"/api/reid/pair-comparisons/{comparison_id}/results.json").status_code == 409
    assert client.delete(f"/api/reid/pair-comparisons/{comparison_id}").status_code == 409
    job["state"] = "completed"; client.reid_service.put("job", job)
    client.delete(f"/api/reid/pair-comparisons/{comparison_id}")


@pytest.mark.parametrize("encoder_id", ["coca_visual", "coca_l14_visual", "siglip2_visual"])
def test_visual_only_pair_uses_raw_cosine_and_only_appearance_vector(client, encoder_id):
    site = client.post("/api/reid/sites", json={"name": "Visual only"}).json()
    config = {"name": "Visual cosine", "site_id": site["id"], "encoder_id": encoder_id, "threshold": .8,
              "weights": DEFAULT_WEIGHTS}
    response = client.post("/api/reid/pair-comparisons", data={"config": json.dumps(config)}, files=[
        ("files", ("a.png", image_bytes("red"), "image/png")),
        ("files", ("b.png", image_bytes("blue"), "image/png"))])
    job = wait_job(client, response.json()["id"])
    result = client.get(f'/api/reid/pair-comparisons/{job["id"]}/results.json').json()
    assert result["encoder_only"] is True
    assert result["similarities"]["combined"] == pytest.approx(.8)
    assert result["decision"]["same_vehicle"] is True
    assert set(result["images"][0]["vectors"]) == {"appearance"}
    assert "criteria_report" not in result
    assert client.get(f'/api/reid/pair-comparisons/{job["id"]}').json()["weights"] == {
        "appearance": 1., "color": 0., "shape": 0., "semantic": 0.}


def test_l14_visual_result_exports_extraction_provenance():
    arrays = {"appearance": np.array([[1., 0.], [.5, np.sqrt(.75)]], np.float32)}
    comparison = {"id": "pair", "name": "Pair", "site_id": "site", "threshold": .5,
                  "weights": DEFAULT_WEIGHTS, "images": [{"side": "first", "filename": "a.jpg"},
                                                            {"side": "second", "filename": "b.jpg"}]}
    encoder = {"id": "coca_l14_visual", "name": "CoCa L/14 visual", "family": "coca_l14_visual",
               "model_name": "coca_ViT-L-14", "repository_name": "laion/repository", "fingerprint": "fp",
               "pair_mode": "visual_cosine", "parent_encoder": "coca_l14", "parent_sha256": "parent-sha",
               "extraction_version": "coca-l14-visual-v1", "embedding_dimension": 768,
               "checkpoint_filename": "model.safetensors", "checksums": {"model.safetensors": "visual-sha"},
               "sizes": {"model.safetensors": 123}}
    exported = build_result(comparison, encoder, arrays, {})["encoder"]
    assert exported["repository_name"] == "laion/repository"
    assert exported["pair_mode"] == "visual_cosine" and exported["parent_encoder"] == "coca_l14"
    assert exported["parent_sha256"] == "parent-sha"
    assert exported["extraction_version"] == "coca-l14-visual-v1"
    assert exported["checkpoint"] == {"filename": "model.safetensors", "sha256": "visual-sha", "size_bytes": 123}


@pytest.mark.parametrize("encoder_id,checkpoint", [("siglip2", "model.safetensors"),
                                                    ("coca_l14", "open_clip_pytorch_model.bin")])
def test_full_modern_vlm_pair_uses_fusion(client, monkeypatch, encoder_id, checkpoint):
    from iris.reid_core import BASELINES, ENCODER_METADATA
    monkeypatch.setattr(client.reid_service, "encoder", lambda _: {
        **BASELINES[encoder_id], **ENCODER_METADATA[encoder_id], "available": True,
        "fingerprint": f"test-{encoder_id}", "revision": "pinned", "checksums": {checkpoint: "sha"},
        "sizes": {checkpoint: 123}})
    site = client.post("/api/reid/sites", json={"name": f"{encoder_id} full"}).json()
    config = {"name": f"{encoder_id} fusion", "site_id": site["id"], "encoder_id": encoder_id,
              "threshold": .75, "weights": DEFAULT_WEIGHTS}
    response = client.post("/api/reid/pair-comparisons", data={"config": json.dumps(config)}, files=[
        ("files", ("a.png", image_bytes("red"), "image/png")),
        ("files", ("b.png", image_bytes("blue"), "image/png"))])
    job = wait_job(client, response.json()["id"])
    result = client.get(f'/api/reid/pair-comparisons/{job["id"]}/results.json').json()
    assert result["encoder_only"] is False
    assert result["decision"]["score_source"] == "combined-fusion"
    assert result["encoder"]["embedding_dimension"] == 768
    assert result["encoder"]["checkpoint"] == {"filename": checkpoint, "sha256": "sha", "size_bytes": 123}
    assert len(result["images"][0]["vectors"]["semantic"]) == 24
    assert result["criteria_report"]["version"] == "criteria-v1"
