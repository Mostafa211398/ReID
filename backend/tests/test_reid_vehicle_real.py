"""Explicit checks of every installed embedding space."""
import json
import os
import shutil
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from iris.reid_core import BASELINES, ENCODER_METADATA
from iris.reid_encoders import Encoder
from iris.reid_pair import SEMANTIC_GROUPS

@pytest.mark.skipif(os.getenv("IRIS_REID_REAL_SMOKE") != "1", reason="Explicit real-model smoke test")
@pytest.mark.parametrize("family", list(BASELINES))
def test_real_encoder_dimensions(family):
    root = Path(__file__).resolve().parents[2] / ".data/models/reid"
    manifest = json.loads((root / "manifest.json").read_text())
    encoder = Encoder({**BASELINES[family], **manifest[family]}, root, device="cpu")
    crop = Image.fromarray(np.random.default_rng(42).integers(0, 255, (96, 128, 3), dtype=np.uint8))
    result = encoder.encode([crop])
    assert result.shape == (1, ENCODER_METADATA[family]["embedding_dimension"])
    assert np.isfinite(result).all()
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1, atol=1e-5)
    np.testing.assert_allclose(result, encoder.encode([crop]), atol=1e-5)


@pytest.mark.skipif(os.getenv("IRIS_REID_REAL_SMOKE") != "1", reason="Explicit real-model smoke test")
@pytest.mark.parametrize("family,dimension", [("coca", 512), ("coca_l14", 768)])
def test_real_coca_semantic_pair_features(family, dimension):
    root = Path(__file__).resolve().parents[2] / ".data/models/reid"
    manifest = json.loads((root / "manifest.json").read_text())
    encoder = Encoder({**BASELINES[family], **manifest[family]}, root, device="cpu")
    image_root = Path(__file__).resolve().parents[3] / "RTMDet-Tiny" / "chosen crops" / "vehicle_2"
    paths = sorted(image_root.glob("*.jpg"))[:2]
    images = []
    try:
        for path in paths:
            with Image.open(path) as source:
                images.append(source.convert("RGB"))
        appearance, semantic, attributes = encoder.encode_with_semantics(images, SEMANTIC_GROUPS)
        assert appearance.shape == (2, dimension) and semantic.shape == (2, 24)
        assert np.isfinite(appearance).all() and np.isfinite(semantic).all()
        np.testing.assert_allclose(np.linalg.norm(appearance, axis=1), 1, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(semantic, axis=1), 1, atol=1e-5)
        assert all(set(value) == set(SEMANTIC_GROUPS) for value in attributes)
    finally:
        for image in images:
            image.close()


@pytest.mark.skipif(os.getenv("IRIS_REID_REAL_SMOKE") != "1", reason="Explicit real-model smoke test")
def test_real_standalone_visual_matches_full_coca_without_parent(tmp_path):
    root = Path(__file__).resolve().parents[2] / ".data/models/reid"
    manifest = json.loads((root / "manifest.json").read_text())
    crop = Image.fromarray(np.random.default_rng(7).integers(0, 255, (96, 128, 3), dtype=np.uint8))
    full = Encoder({**BASELINES["coca"], **manifest["coca"]}, root, device="cpu")
    expected = full.encode([crop])
    del full
    isolated = tmp_path / "models"
    shutil.copytree(root / "coca_visual", isolated / "coca_visual")
    visual = Encoder({**BASELINES["coca_visual"], **manifest["coca_visual"]}, isolated, device="cpu")
    actual = visual.encode([crop])
    np.testing.assert_allclose(actual, expected, atol=1e-5)
    assert not (isolated / "coca").exists()


@pytest.mark.skipif(os.getenv("IRIS_REID_REAL_SMOKE") != "1", reason="Explicit real-model smoke test")
def test_real_standalone_visual_matches_full_coca_l14_without_parent(tmp_path):
    root = Path(__file__).resolve().parents[2] / ".data/models/reid"
    manifest = json.loads((root / "manifest.json").read_text())
    crop = Image.fromarray(np.random.default_rng(14).integers(0, 255, (96, 128, 3), dtype=np.uint8))
    full = Encoder({**BASELINES["coca_l14"], **manifest["coca_l14"]}, root, device="cpu")
    expected = full.encode([crop])
    del full
    isolated = tmp_path / "models"
    shutil.copytree(root / "coca_l14_visual", isolated / "coca_l14_visual")
    visual = Encoder({**BASELINES["coca_l14_visual"], **manifest["coca_l14_visual"]}, isolated, device="cpu")
    actual = visual.encode([crop])
    np.testing.assert_allclose(actual, expected, atol=1e-5)
    assert not (isolated / "coca_l14").exists()


@pytest.mark.skipif(os.getenv("IRIS_REID_REAL_SMOKE") != "1", reason="Explicit real-model smoke test")
def test_real_standalone_visual_matches_full_siglip2_without_parent(tmp_path):
    root = Path(__file__).resolve().parents[2] / ".data/models/reid"
    manifest = json.loads((root / "manifest.json").read_text())
    crop = Image.fromarray(np.random.default_rng(8).integers(0, 255, (120, 180, 3), dtype=np.uint8))
    full = Encoder({**BASELINES["siglip2"], **manifest["siglip2"]}, root, device="cpu")
    expected = full.encode([crop])
    del full
    isolated = tmp_path / "models"
    shutil.copytree(root / "siglip2_visual", isolated / "siglip2_visual")
    visual = Encoder({**BASELINES["siglip2_visual"], **manifest["siglip2_visual"]}, isolated, device="cpu")
    actual = visual.encode([crop])
    np.testing.assert_allclose(actual, expected, atol=1e-5)
    assert not (isolated / "siglip2").exists()


@pytest.mark.skipif(os.getenv("IRIS_REID_REAL_SMOKE") != "1", reason="Explicit real-model smoke test")
def test_real_siglip2_semantic_pair_features():
    root = Path(__file__).resolve().parents[2] / ".data/models/reid"
    manifest = json.loads((root / "manifest.json").read_text())
    encoder = Encoder({**BASELINES["siglip2"], **manifest["siglip2"]}, root, device="cpu")
    images = [Image.fromarray(np.random.default_rng(seed).integers(0, 255, (120, 180, 3), dtype=np.uint8))
              for seed in (9, 10)]
    appearance, semantic, attributes = encoder.encode_with_semantics(images, SEMANTIC_GROUPS)
    assert appearance.shape == (2, 768) and semantic.shape == (2, 24)
    assert np.isfinite(appearance).all() and np.isfinite(semantic).all()
    assert all(set(value) == set(SEMANTIC_GROUPS) for value in attributes)


@pytest.mark.skipif(os.getenv("IRIS_REID_REAL_SMOKE") != "1", reason="Explicit real-model smoke test")
@pytest.mark.parametrize("family", ["coca", "coca_l14"])
def test_real_structured_coca_criteria(tmp_path, family):
    from iris.reid_pair import DEFAULT_WEIGHTS, build_result
    from iris.reid_worker import pair_features_once
    root = Path(__file__).resolve().parents[2] / ".data/models/reid"
    manifest = json.loads((root / "manifest.json").read_text())
    spec = {**BASELINES[family], **manifest[family]}
    image_root = Path(__file__).resolve().parents[3] / "RTMDet-Tiny" / "chosen crops" / "vehicle_2"
    paths = sorted(image_root.glob("*.jpg"))[:2]
    images = [{"side": side, "filename": path.name, "path": str(path)}
              for side, path in zip(("first", "second"), paths)]
    output, metadata_path = tmp_path / "features.npz", tmp_path / "features.json"
    pair_features_once({"encoder": spec, "model_root": str(root), "device": "cpu", "images": images,
                        "output": str(output), "metadata_path": str(metadata_path),
                        "progress_path": str(tmp_path / "progress.json")})
    with np.load(output, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    result = build_result({"id": "real", "name": "real", "site_id": "site", "images": images,
                           "threshold": .75, "weights": DEFAULT_WEIGHTS}, spec, arrays,
                          json.loads(metadata_path.read_text()))
    report = result["criteria_report"]
    assert 0 <= report["criteria_similarity_score"] <= 1
    assert report["qualitative_band"] in ("high", "moderate", "low")
    assert report["plate_logo_type_viewpoint"]["model_text"]["image_1"]["label"] == "unknown"
    assert report["plate_logo_type_viewpoint"]["license_plate"]["identity_claim"] == "not evaluated"
