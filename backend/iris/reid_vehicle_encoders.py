"""Vehicle-specific inference adapters. Inputs are saved RGB vehicle crops."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image


def rgb_pixels(images, size):
    return np.stack([np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32).transpose(2, 0, 1) for image in images])


class OpenVINOVehicle:
    def __init__(self, location):
        import openvino as ov
        core = ov.Core()
        model = core.read_model(str(location / "model.onnx"))
        if not model.input().partial_shape.compatible(ov.PartialShape([1, 3, 208, 208])):
            raise ValueError("Expected the official vehicle-reid-0001 input shape")
        model.reshape([1, 3, 208, 208])
        if list(model.output().shape) != [1, 512]:
            raise ValueError("Expected the official vehicle-reid-0001 ONNX input/output shapes")
        self.model = core.compile_model(model, "CPU", {"INFERENCE_PRECISION_HINT": "f32", "PERFORMANCE_HINT": "LATENCY"})

    def encode(self, images):
        # The original ONNX includes normalization and expects RGB 0..255.
        # BGR reversal applies only to OMZ's converted IR, which we do not load.
        if not images:
            return np.empty((0, 512), dtype=np.float32)
        vectors = np.concatenate([np.asarray(self.model(rgb_pixels([image], 208))[self.model.output()]).copy() for image in images]).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if vectors.shape != (len(images), 512) or not np.isfinite(vectors).all() or np.any(norms <= 1e-12):
            raise ValueError("OpenVINO returned invalid vehicle features")
        return vectors / norms


def transreid_model(location: Path, checkpoint_path=None):
    import torch
    source = location / "source"
    sys.path.insert(0, str(source))
    try:
        from config import cfg as defaults
        from model.make_model import make_model
        cfg = defaults.clone()
        cfg.merge_from_file(str(location / "vehicle.yml"))
        cfg.MODEL.PRETRAIN_CHOICE = "none"
        cfg.MODEL.PRETRAIN_PATH = ""
        state = torch.load(checkpoint_path or location / "model.pth", map_location="cpu", weights_only=True)
        state = state.get("state_dict", state.get("model", state))
        state = {k.removeprefix("module."): v for k, v in state.items()}
        if tuple(state["base.pos_embed"].shape) != (1, 442, 768) or tuple(state["classifier.weight"].shape) != (576, 768):
            raise ValueError("Not the VeRi-776 stride-12 ViT checkpoint")
        # Allocate the exact trained table for strict loading, then bypass its
        # contribution. Arbitrary site camera strings are not VeRi camera IDs.
        cfg.MODEL.SIE_CAMERA = True
        cfg.MODEL.SIE_VIEW = False
        model = make_model(cfg, 576, int(state["base.sie_embed"].shape[0]), 0)
        model.load_state_dict(state, strict=True)
        model.base.cam_num = 0
        model.base.view_num = 0
        return model
    except (KeyError, RuntimeError, ValueError, EOFError) as exc:
        raise ValueError("Incompatible TransReID checkpoint. Install the official VeRi TransReID ViT stride-12 model; use -TransReIDCheckpoint for a local copy. " + str(exc)) from exc
    finally:
        sys.path.remove(str(source))
