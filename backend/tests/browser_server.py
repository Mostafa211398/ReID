"""Disposable browser-test app; only neural inference is replaced."""
import os
import threading
import io
import json
import zipfile
from pathlib import Path
import numpy as np
import uvicorn
from PIL import Image
from iris.app import create_app
from iris.config import Settings, PROJECT_ROOT
from iris.reid_core import BASELINES, ENCODER_METADATA
from iris.reid_service import ReIDService
from iris.reid_worker import ingest, render, save_vectors
from iris.reid_core import atomic_json
from iris.reid_pair import (CRITERIA_PROMPT_GROUPS, CRITERIA_VERSION, PAIR_FEATURE_VERSION,
                            SEMANTIC_GROUPS, SEMANTIC_VERSION, SHAPE_VERSION)


server = None


def test_stage(self, job, stage, config):
    if stage in job["completed_stages"]:
        return
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    config["progress_path"] = str(output.parent / "progress.json")
    if config["task"] == "ingest":
        ingest(config)
    elif config["task"] == "embed":
        save_vectors(output, {t["id"]: np.array([[1., 0.]]) for t in config["tracks"]})
    elif config["task"] == "pair_features":
        if config["encoder"].get("pair_mode", "visual_cosine" if config["encoder"]["family"].endswith("_visual") else "vlm_fusion") == "visual_cosine":
            save_vectors(output, {"appearance": np.array([[1., 0.], [.8, .6]], dtype=np.float32)})
            atomic_json(Path(config["metadata_path"]), {"pair_feature_version": PAIR_FEATURE_VERSION,
                        "encoder_only": True})
            job["completed_stages"].append(stage); self.put("job", job); return
        color = np.zeros((2, 72), dtype=np.float32); color[:, 0] = 1
        shape = np.zeros((2, 8100), dtype=np.float32); shape[:, 0] = 1
        semantic = np.zeros((2, 24), dtype=np.float32); semantic[:, 0] = 1
        save_vectors(output, {"appearance": np.array([[1., 0.], [.8, .6]], dtype=np.float32),
                     "color": color, "color_available": np.ones(2, dtype=np.bool_),
                     "shape": shape, "shape_available": np.ones(2, dtype=np.bool_), "semantic": semantic,
                     "texture": np.tile(np.eye(1, 256, dtype=np.float32), (2, 1)),
                     "texture_available": np.ones(2, dtype=np.bool_),
                     "detail": np.tile(np.eye(1, 34, dtype=np.float32), (2, 1)),
                     "detail_available": np.ones(2, dtype=np.bool_), "edge_scale": np.array([.8, .7])})
        groups = {**SEMANTIC_GROUPS, **CRITERIA_PROMPT_GROUPS}
        attributes = []
        for _ in range(2):
            attributes.append({group: {"label": values[0][0], "confidence": 1.,
                                      "distribution": {label: float(index == 0) for index, (label, _) in enumerate(values)}}
                               for group, values in groups.items()})
        atomic_json(Path(config["metadata_path"]), {"pair_feature_version": PAIR_FEATURE_VERSION,
                    "shape_version": SHAPE_VERSION, "semantic_version": SEMANTIC_VERSION,
                    "criteria_version": CRITERIA_VERSION,
                    "color": [{"method": "fixture"}, {"method": "fixture"}],
                    "color_summary": [{"dominant_colors": ["red"], "mean_hue_radians": 0.},
                                      {"dominant_colors": ["red"], "mean_hue_radians": 0.}],
                    "semantic_attributes": [{key: value for key, value in row.items() if key in SEMANTIC_GROUPS}
                                            for row in attributes], "criteria_attributes": attributes})
    elif config["task"] == "render":
        render(config)
    elif config["task"] == "benchmark_comparison_export":
        output = Path(config["output"]); output.parent.mkdir(parents=True, exist_ok=True)
        image = io.BytesIO(); Image.new("RGB", (80, 50), "navy").save(image, "JPEG")
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("manifest.json", json.dumps({"comparison_count": config["comparison_count"]}))
            archive.writestr("README.txt", "Browser fixture")
            for encoder_id in config["encoders"]:
                archive.writestr(f"{encoder_id}/comparisons.csv", "encoder_id,cosine_similarity\n")
                archive.writestr(f"{encoder_id}/000001__same__fixture.jpg", image.getvalue())
    else:
        raise ValueError("Unexpected browser-test worker task")
    job["completed_stages"].append(stage)
    self.put("job", job)


def create_test_app():
    ReIDService.run_stage = test_stage
    ReIDService.encoders = lambda self: [{**spec, **ENCODER_METADATA[key], "available": True, "origin": "pretrained", "fingerprint": "browser-" + key} for key, spec in BASELINES.items()]
    data_dir = Path(os.environ["IRIS_BROWSER_DATA"]).resolve()
    app = create_app(Settings(data_dir=data_dir, frontend_dist=PROJECT_ROOT / "frontend/dist", gpu_lock_path=data_dir / "gpu.lock"))

    @app.post("/api/system/test-shutdown")
    def test_shutdown():
        def stop():
            if server is not None:
                server.should_exit = True

        timer = threading.Timer(.2, stop)
        timer.daemon = True
        timer.start()
        return {"stopping": True}

    # The production static mount owns "/", so keep this test-only API route ahead of it.
    shutdown_route = app.router.routes.pop()
    app.router.routes.insert(0, shutdown_route)

    return app


if __name__ == "__main__":
    app = create_test_app()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8013))
    server.run()
