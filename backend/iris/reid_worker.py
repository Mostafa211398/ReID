"""Subprocess entrypoint. Files form the protocol, stdout is diagnostics only."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
from PIL.Image import open as open_image

from .reid_core import atomic_json, atomic_replace, retrieval, timestamp, track_vector


def report(config, progress, **metrics):
    atomic_json(Path(config["progress_path"]), {"progress": progress, "metrics": metrics})


def save_vectors(path: Path, arrays: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".partial")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    atomic_replace(temp, path)


def benchmark_comparison_export(config):
    from .reid_benchmark import write_comparison_images
    write_comparison_images(config, lambda progress, **metrics: report(config, progress, **metrics))


def read_image(path):
    from PIL import ImageOps
    with open_image(path) as image:
        return cv2.cvtColor(np.asarray(ImageOps.exif_transpose(image).convert("RGB")), cv2.COLOR_RGB2BGR)


def ingest(config):
    """One supplied crop is one observation; never infer boxes or classes."""
    from PIL import ImageOps
    root, clip = Path(config["output"]), config["clip"]
    root.mkdir(parents=True, exist_ok=True)
    key = clip["id"] + "_1"
    crop_path = root / (key + "_0.jpg")
    with open_image(clip["path"]) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        width, height = image.size
        image.save(crop_path, quality=95)
        image.save(root / "preview.jpg", quality=95)
        image.close()
    track = {"id": key, "source_track_id": key, "clip_id": clip["id"], "camera_id": clip["camera_id"],
             "source_hash": clip["sha256"], "media_type": "image", "class_name": clip["class_name"],
             "local_id": 1, "start": 0, "end": 0, "start_frame": 0, "end_frame": 0,
             "absolute_start": clip.get("start_epoch"), "absolute_end": clip.get("start_epoch"),
             "crops": [{"path": str(crop_path), "frame": 0, "timestamp": 0, "quality": 1.0}]}
    atomic_json(root / "tracks.json", [track])
    atomic_json(root / "frames.jsonl", {"frame": 0, "timestamp": 0, "objects": [{"track_id": key, "bbox": [0, 0, width - 1, height - 1]}]})
    report(config, 1, observations=1)


def embed(config):
    import gc
    import torch
    try:
        return embed_once(config)
    except torch.cuda.OutOfMemoryError:
        if (config["encoder"]["family"] != "transreid" and not config["encoder"].get("cpu_fallback")) or config.get("device") == "cpu":
            raise
    # Leave the exception scope before collecting so its traceback cannot retain
    # CUDA tensors. Restart all crops; never save a mixture of partial attempts.
    gc.collect()
    torch.cuda.empty_cache()
    print(f'{config["encoder"].get("name", config["encoder"]["family"])} CUDA memory exhausted; restarting embedding on CPU', flush=True)
    return embed_once({**config, "device": "cpu", "fallback": "CUDA out of memory; embedding stage restarted on CPU"})


def embed_once(config):
    from PIL import Image
    from .reid_encoders import Encoder
    encoder = Encoder(config["encoder"], Path(config["model_root"]), config.get("device", "auto"))
    arrays = {}
    started = time.perf_counter()
    for index, track in enumerate(config["tracks"]):
        images = []
        try:
            for crop in track["crops"]:
                with Image.open(crop["path"]) as image:
                    images.append(image.convert("RGB"))
            if images:
                # Single-image inference bounds memory independently of gallery size.
                arrays[track["id"]] = np.concatenate([encoder.encode([image]) for image in images])
        finally:
            for image in images:
                image.close()
        report(config, (index + 1) / max(1, len(config["tracks"])), tracks=index + 1)
    save_vectors(Path(config["output"]), arrays)
    report(config, 1, tracks=len(arrays), seconds=time.perf_counter() - started,
           **encoder.runtime_metrics(), fallback=config.get("fallback"), embedding_dimension=next(iter(arrays.values())).shape[-1] if arrays else config["encoder"].get("embedding_dimension"))


def pair_features(config):
    import gc
    import torch
    try:
        return pair_features_once(config)
    except torch.cuda.OutOfMemoryError:
        if config.get("device") == "cpu":
            raise
    gc.collect()
    torch.cuda.empty_cache()
    print("VLM CUDA memory exhausted; restarting pair extraction on CPU", flush=True)
    return pair_features_once({**config, "device": "cpu", "fallback": "CUDA out of memory; pair extraction restarted on CPU"})


def pair_features_once(config):
    from PIL import Image
    from .reid_color import extract_color
    from .reid_encoders import Encoder
    from .reid_pair import (CRITERIA_PROMPT_GROUPS, CRITERIA_VERSION, PAIR_FEATURE_VERSION, SEMANTIC_GROUPS,
                            SEMANTIC_VERSION, SHAPE_VERSION, color_summary, detail_feature, edge_scale,
                            shape_feature, texture_feature)
    encoder = Encoder(config["encoder"], Path(config["model_root"]), config.get("device", "auto"))
    images, rgb_values = [], []
    started = time.perf_counter()
    try:
        for item in config["images"]:
            with Image.open(item["path"]) as source:
                image = source.convert("RGB")
                images.append(image.copy())
                rgb_values.append(np.asarray(image, dtype=np.uint8).copy())
        if len(images) != 2:
            raise ValueError("A pair comparison requires exactly two images")
        if encoder.spec.get("pair_mode", "visual_cosine" if encoder.family.endswith("_visual") else "vlm_fusion") == "visual_cosine":
            appearance = encoder.encode(images)
            save_vectors(Path(config["output"]), {"appearance": appearance})
            atomic_json(Path(config["metadata_path"]), {"pair_feature_version": PAIR_FEATURE_VERSION,
                        "encoder_only": True})
            report(config, 1, seconds=time.perf_counter() - started, **encoder.runtime_metrics(),
                   fallback=config.get("fallback"), embedding_dimension=appearance.shape[-1])
            return
        groups = {**SEMANTIC_GROUPS, **CRITERIA_PROMPT_GROUPS}
        appearance, semantic, attributes = encoder.encode_with_semantics(images, groups, set(SEMANTIC_GROUPS))
        color_records = [extract_color(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)) for rgb in rgb_values]
        color_available = np.array([record.get("vector") is not None for record in color_records], dtype=np.bool_)
        color = np.stack([np.asarray(record["vector"], dtype=np.float32) if record.get("vector") is not None else np.zeros(72, dtype=np.float32) for record in color_records])
        shape_records = [shape_feature(rgb) for rgb in rgb_values]
        shape_available = np.array([value is not None for value in shape_records], dtype=np.bool_)
        shape = np.stack([value if value is not None else np.zeros(8100, dtype=np.float32) for value in shape_records])
        texture_records = [texture_feature(rgb) for rgb in rgb_values]
        texture_available = np.array([value is not None for value in texture_records], dtype=np.bool_)
        texture = np.stack([value if value is not None else np.zeros(256, dtype=np.float32) for value in texture_records])
        detail_records = [detail_feature(rgb) for rgb in rgb_values]
        detail_available = np.array([value is not None for value in detail_records], dtype=np.bool_)
        detail = np.stack([value if value is not None else np.zeros(34, dtype=np.float32) for value in detail_records])
        scale_records = [edge_scale(rgb) for rgb in rgb_values]
        scales = np.asarray([value if value is not None else np.nan for value in scale_records], np.float32)
        save_vectors(Path(config["output"]), {"appearance": appearance, "color": color,
                     "color_available": color_available, "shape": shape, "shape_available": shape_available,
                     "semantic": semantic, "texture": texture, "texture_available": texture_available,
                     "detail": detail, "detail_available": detail_available, "edge_scale": scales})
        atomic_json(Path(config["metadata_path"]), {"pair_feature_version": PAIR_FEATURE_VERSION,
                    "shape_version": SHAPE_VERSION, "semantic_version": SEMANTIC_VERSION,
                    "criteria_version": CRITERIA_VERSION,
                    "color": [{key: value for key, value in record.items() if key != "vector"} for record in color_records],
                    "color_summary": [color_summary(rgb) for rgb in rgb_values],
                    "semantic_attributes": [{key: value for key, value in row.items() if key in SEMANTIC_GROUPS}
                                            for row in attributes],
                    "criteria_attributes": attributes})
        report(config, 1, seconds=time.perf_counter() - started, **encoder.runtime_metrics(),
               fallback=config.get("fallback"), embedding_dimension=appearance.shape[-1])
    finally:
        for image in images:
            image.close()


def render(config):
    frame = read_image(config["clip"]["path"])
    record = json.loads(Path(config["frames"]).read_text())
    # Keep captions readable even for small supplied crops, without covering pixels.
    labels = []
    for obj in record["objects"]:
        assignment = config["assignments"].get(obj["track_id"], {})
        label = assignment.get("global_id") or "Needs review"
        if assignment.get("similarity") is not None:
            label += f' combined={assignment["similarity"]:.3f}' if assignment.get("scoring_version") else f' {assignment["similarity"]:.3f}'
        labels.append(label)
    caption_width = max((cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .6, 1)[0][0] + 24 for label in labels), default=0)
    canvas = np.full((frame.shape[0] + 36 * len(labels), max(frame.shape[1], caption_width), 3), (24, 18, 17), dtype=np.uint8)
    left = (canvas.shape[1] - frame.shape[1]) // 2
    canvas[:frame.shape[0], left:left + frame.shape[1]] = frame
    for index, label in enumerate(labels):
        cv2.putText(canvas, label, (12, frame.shape[0] + 24 + index * 36), cv2.FONT_HERSHEY_SIMPLEX, .6, (143, 217, 98), 1, cv2.LINE_AA)
    path = Path(config["output"])
    temporary = path.with_name(path.stem + ".partial.jpg")
    if not cv2.imwrite(str(temporary), canvas):
        raise RuntimeError("Could not save annotated image")
    atomic_replace(temporary, path)
    report(config, 1, frames=1)
    return


def train(config):
    import importlib.metadata
    import torch
    from PIL import Image, ImageEnhance
    from safetensors.torch import save_file
    from .reid_encoders import Encoder
    settings = config["training"]
    random.seed(settings["seed"]); np.random.seed(settings["seed"]); torch.manual_seed(settings["seed"])
    encoder = Encoder(config["encoder"], Path(config["model_root"]), config.get("device", "auto"))
    parameters = encoder.trainable_tail()
    rows = config["dataset"]["rows"]
    train_ids = config["dataset"]["splits"]["train"]
    grouped = {identity: [c for r in rows if r["identity"] == identity for c in r["crops"]] for identity in train_ids}
    if len(grouped) < 2 or any(len(crops) < 2 for crops in grouped.values()):
        raise ValueError("Training needs at least two identities with two distinct crops each")
    validation = [r for r in rows if r["identity"] in config["dataset"]["splits"]["validation"]]
    first = next(iter(grouped.values()))[0]
    with Image.open(first["path"]) as image:
        dimension = encoder.encode([image.convert("RGB")]).shape[-1]
    classifier = torch.nn.Linear(dimension, len(train_ids)).to(encoder.device)
    optimizer = torch.optim.AdamW(parameters + list(classifier.parameters()), lr=settings["learning_rate"], weight_decay=.01)
    best, stale, history = -1, 0, []
    output = Path(config["output"]); output.mkdir(parents=True, exist_ok=True)
    steps = max(1, math.ceil(sum(map(len, grouped.values())) / 4))
    for epoch in range(settings["epochs"]):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step in range(steps):
            chosen = random.sample(train_ids, 2)
            images, targets = [], []
            for identity in chosen:
                for crop in random.sample(grouped[identity], 2):
                    with Image.open(crop["path"]) as image:
                        image = image.convert("RGB")
                        if random.random() < .5:
                            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                        images.append(ImageEnhance.Brightness(image).enhance(random.uniform(.85, 1.15)))
                    targets.append(train_ids.index(identity))
            try:
                features = encoder.features(encoder.inputs(images))
                labels = torch.tensor(targets, device=encoder.device)
                classification = torch.nn.functional.cross_entropy(classifier(features), labels)
                distances = torch.cdist(features, features)
                same = labels[:, None] == labels[None, :]
                positive = distances.masked_fill(~same, 0).max(dim=1).values
                negative = distances.masked_fill(same, float("inf")).min(dim=1).values
                loss = classification + torch.relu(positive - negative + .3).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("Training produced a non-finite loss")
                group_start = (step // settings["accumulation"]) * settings["accumulation"]
                group_size = min(settings["accumulation"], steps - group_start)
                (loss / group_size).backward()
                if (step + 1) % settings["accumulation"] == 0 or step + 1 == steps:
                    torch.nn.utils.clip_grad_norm_(parameters + list(classifier.parameters()), 1)
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach()))
            except torch.cuda.OutOfMemoryError as exc:
                raise RuntimeError("The FP32 two-identity batch does not fit available VRAM. Stop other GPU work or export this training bundle to a larger GPU.") from exc
            finally:
                for image in images:
                    image.close()
            report(config, (epoch + (step + 1) / steps) / settings["epochs"], epoch=epoch + 1, loss=losses[-1])
        vectors = {}
        for row in validation:
            values = []
            for crop in row["crops"]:
                with Image.open(crop["path"]) as image:
                    values.append(encoder.encode([image.convert("RGB")])[0])
            if values:
                vectors[row["id"]] = track_vector(values)
        metric = retrieval(validation, vectors)
        if metric["mAP"] is None:
            raise ValueError("Validation needs positive cross-camera track pairs and negative identities")
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "mAP": metric["mAP"], "rank1": metric["rank1"]})
        if metric["mAP"] > best:
            best, stale = metric["mAP"], 0
            temp = output / "model.partial"
            save_file({k: v.detach().cpu().contiguous() for k, v in encoder.model.state_dict().items()}, str(temp))
            atomic_replace(temp, output / "model.safetensors")
        else:
            stale += 1
        atomic_json(output / "history.json", history)
        report(config, (epoch + 1) / settings["epochs"], history=history, best_validation_mAP=best)
        if stale >= settings["patience"]:
            break
    versions = {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "transformers", "safetensors", "numpy", "Pillow")}
    atomic_json(output / "training.json", {"config": settings, "encoder": config["encoder"], "history": history, "dataset_fingerprint": config["dataset"]["fingerprint"], "environment": versions, "best_validation_mAP": best})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    config = json.loads(Path(parser.parse_args().config).read_text(encoding="utf-8"))
    try:
        {"ingest": ingest, "embed": embed, "pair_features": pair_features, "render": render, "train": train,
         "benchmark_comparison_export": benchmark_comparison_export}[config["task"]](config)
    except Exception as exc:
        atomic_json(Path(config["progress_path"]), {"error": str(exc)})
        raise


if __name__ == "__main__":
    main()
