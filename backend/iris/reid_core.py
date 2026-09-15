"""Deterministic, framework-independent matching and evaluation.

Predictions never become ground truth. Every embedding space is addressed by
an encoder fingerprint, including preprocessing and trained weights.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np


BASELINES = {
    "siglip": {"id": "siglip", "name": "SigLIP Base", "family": "siglip", "model_name": "google/siglip-base-patch16-224"},
    "dinov2": {"id": "dinov2", "name": "DINOv2 Small", "family": "dinov2", "model_name": "facebook/dinov2-small"},
    "fastreid": {"id": "fastreid", "name": "FastReID VeRi SBS R50-IBN", "family": "fastreid", "model_name": "veri_sbs_R50-ibn"},
    "openvino": {"id": "openvino", "name": "OpenVINO vehicle-reid-0001", "family": "openvino", "model_name": "vehicle-reid-0001"},
    "transreid": {"id": "transreid", "name": "TransReID ViT · VeRi-776", "family": "transreid", "model_name": "veri_vit_transreid_stride"},
    "coca": {"id": "coca", "name": "CoCa ViT-B/32 · LAION2B", "family": "coca", "model_name": "coca_ViT-B-32",
             "repository_name": "laion/CoCa-ViT-B-32-laion2B-s13B-b90k"},
    "coca_visual": {"id": "coca_visual", "name": "CoCa ViT-B/32 · visual encoder only", "family": "coca_visual",
                    "model_name": "coca_ViT-B-32", "parent_encoder": "coca"},
    "coca_l14": {"id": "coca_l14", "name": "CoCa ViT-L/14 · LAION2B", "family": "coca_l14",
                 "model_name": "coca_ViT-L-14",
                 "repository_name": "laion/CoCa-ViT-L-14-laion2B-s13B-b90k"},
    "coca_l14_visual": {"id": "coca_l14_visual", "name": "CoCa ViT-L/14 · visual encoder only",
                        "family": "coca_l14_visual", "model_name": "coca_ViT-L-14",
                        "repository_name": "laion/CoCa-ViT-L-14-laion2B-s13B-b90k",
                        "parent_encoder": "coca_l14"},
    "siglip2": {"id": "siglip2", "name": "SigLIP2 Base Patch16 384", "family": "siglip2",
                "model_name": "google/siglip2-base-patch16-384",
                "repository_name": "google/siglip2-base-patch16-384"},
    "siglip2_visual": {"id": "siglip2_visual", "name": "SigLIP2 Base Patch16 384 · visual encoder only",
                       "family": "siglip2_visual", "model_name": "google/siglip2-base-patch16-384",
                       "parent_encoder": "siglip2"},
}

# Keep presentation/capability metadata outside legacy checkpoint fingerprints.
ENCODER_METADATA = {
    "siglip": {"runtime": "PyTorch", "embedding_dimension": 768, "supports_training": True},
    "dinov2": {"runtime": "PyTorch", "embedding_dimension": 384, "supports_training": True},
    "fastreid": {"runtime": "PyTorch", "embedding_dimension": 2048, "supports_training": True},
    "openvino": {"runtime": "OpenVINO CPU", "embedding_dimension": 512, "supports_training": False},
    "transreid": {"runtime": "PyTorch CUDA / CPU", "embedding_dimension": 3840, "supports_training": False,
                  "adaptation": "Experimental unseen-camera mode: training camera/viewpoint embeddings bypassed"},
    "coca": {"runtime": "OpenCLIP PyTorch CUDA / CPU", "embedding_dimension": 512, "supports_training": False,
             "adaptation": "General-purpose image/text encoder; vehicle similarity requires local evaluation",
             "pair_mode": "vlm_fusion", "checkpoint_filename": "open_clip_pytorch_model.bin",
             "checkpoint_size_bytes": 1_014_488_932},
    "coca_visual": {"runtime": "OpenCLIP visual tower CUDA / CPU", "embedding_dimension": 512, "supports_training": False,
                    "adaptation": "Stripped image encoder; mathematically equivalent to CoCa encode_image",
                    "pair_mode": "visual_cosine", "checkpoint_filename": "model.safetensors"},
    "coca_l14": {"runtime": "OpenCLIP PyTorch CUDA / CPU", "embedding_dimension": 768, "supports_training": False,
                 "adaptation": "Large general-purpose image/text encoder; automatic CPU retry on CUDA memory exhaustion",
                 "pair_mode": "vlm_fusion", "checkpoint_filename": "open_clip_pytorch_model.bin",
                 "checkpoint_size_bytes": 2_554_109_637, "cpu_fallback": True},
    "coca_l14_visual": {"runtime": "OpenCLIP visual tower CUDA / CPU", "embedding_dimension": 768,
                        "supports_training": False,
                        "adaptation": "Stripped image encoder; mathematically equivalent to CoCa ViT-L/14 encode_image",
                        "pair_mode": "visual_cosine", "checkpoint_filename": "model.safetensors",
                        "cpu_fallback": True},
    "siglip2": {"runtime": "Transformers PyTorch CUDA / CPU", "embedding_dimension": 768, "supports_training": False,
                "adaptation": "Multilingual image/text encoder; vehicle similarity requires local evaluation",
                "pair_mode": "vlm_fusion", "checkpoint_filename": "model.safetensors",
                "checkpoint_size_bytes": 1_501_968_264},
    "siglip2_visual": {"runtime": "Transformers visual tower CUDA / CPU", "embedding_dimension": 768,
                       "supports_training": False,
                       "adaptation": "Stripped image encoder; mathematically equivalent to SigLIP2 get_image_features",
                       "pair_mode": "visual_cosine", "checkpoint_filename": "model.safetensors"},
}


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding="utf-8")
    atomic_replace(temporary, path)


def atomic_replace(source: Path, destination: Path) -> None:
    # Windows readers briefly prevent replace even after a successful write.
    for attempt in range(40):
        try:
            source.replace(destination)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.05)


def unit(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Embeddings must be finite and nonempty")
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    if (norms < 1e-12).any():
        raise ValueError("Zero embeddings cannot be compared")
    return array / norms


def track_vector(vectors) -> np.ndarray:
    return unit(unit(vectors).mean(axis=0))


def timestamp(raw: float, frame: int, fps: float, previous: float) -> float:
    fallback = frame / fps
    return raw if math.isfinite(raw) and raw >= 0 and (frame == 0 or raw > previous) else max(fallback, previous + 1 / fps if frame else 0)


def simultaneous(a: dict, b: dict) -> bool:
    if a["camera_id"] != b["camera_id"]:
        return False
    if a["clip_id"] == b["clip_id"]:
        return max(a["start"], b["start"]) <= min(a["end"], b["end"])
    if a.get("absolute_start") is None or b.get("absolute_start") is None:
        return False
    return max(a["absolute_start"], b["absolute_start"]) <= min(a["absolute_end"], b["absolute_end"])


def candidate_context(query: dict, sightings: list[dict], transitions: list[dict]) -> float:
    """Return the travel-time tie-breaker used for one candidate identity."""
    if query.get("absolute_start") is None:
        return 0.0
    earlier = [s for s in sightings if s.get("absolute_end") is not None and s["absolute_end"] <= query["absolute_start"]]
    if not earlier:
        return 0.0
    last = max(earlier, key=lambda s: s["absolute_end"])
    rule = next((r for r in transitions if r["source"] == last["camera_id"] and r["destination"] == query["camera_id"]), None)
    if not rule:
        return 0.0
    delta = query["absolute_start"] - last["absolute_end"]
    return 1.0 if rule["min_seconds"] <= delta <= rule["max_seconds"] else -1.0


def candidates(query: dict, vectors, galleries: dict, observations: dict, transitions: list[dict], score_gallery=None) -> list[dict]:
    q = unit(vectors)
    result = []
    for global_id, gallery in galleries.items():
        sightings = observations.get(global_id, [])
        if any(s["id"] != query["id"] and simultaneous(query, s) for s in sightings):
            continue
        details = score_gallery(global_id) if score_gallery else {"similarity": float(np.max(q @ unit(gallery).T))}
        context = candidate_context(query, sightings, transitions)
        result.append({**details, "global_id": global_id, "context": context})
    if score_gallery:
        return sorted(result, key=lambda r: (-r["similarity"], -r["context"], r["global_id"]))[:5]
    # Context only breaks appearance ties to two decimal places; never bypasses a threshold.
    return sorted(result, key=lambda r: (-round(r["similarity"], 2), -r["context"], -r["similarity"], r["global_id"]))[:5]


def eligible_pair(a: dict, b: dict, cross_camera: bool) -> bool:
    if a["id"] == b["id"] or a.get("source_track_id", a["id"]) == b.get("source_track_id", b["id"]):
        return False
    if a.get("source_hash") and a.get("source_hash") == b.get("source_hash"):
        if max(a["start"], b["start"]) <= min(a["end"], b["end"]):
            return False
    if simultaneous(a, b):
        return False
    return a["camera_id"] != b["camera_id"] if cross_camera else a["camera_id"] == b["camera_id"]


def retrieval(rows: list[dict], vectors: dict, cross_camera: bool = True, pair_score=None, same_class=False) -> dict:
    rank1, rank5, aps, queries = [], [], [], []
    labeled = [r for r in rows if r.get("identity") and not r.get("excluded") and r["id"] in vectors]
    for query in labeled:
        gallery = [g for g in labeled if eligible_pair(query, g, cross_camera) and (not same_class or g.get("class_name", "truck") == query.get("class_name", "truck"))]
        positives = sum(g["identity"] == query["identity"] for g in gallery)
        negatives = len(gallery) - positives
        if not positives or not negatives:
            continue
        scored = sorted((((pair_score(query, g) if pair_score else float(unit(vectors[query["id"]]) @ unit(vectors[g["id"]]))), g) for g in gallery), key=lambda pair: (-pair[0], pair[1]["id"]))
        hits = np.array([g["identity"] == query["identity"] for _, g in scored], dtype=np.float64)
        rank1.append(float(hits[0])); rank5.append(float(hits[:5].any()))
        aps.append(float((np.cumsum(hits) / np.arange(1, len(hits) + 1) * hits).sum() / positives))
        queries.append({"track_id": query["id"], "correct": bool(hits[0]), "score": scored[0][0], "margin": scored[0][0] - scored[1][0]})
    return {"rank1": float(np.mean(rank1)) if rank1 else None, "rank5": float(np.mean(rank5)) if rank5 else None,
            "mAP": float(np.mean(aps)) if aps else None, "eligible_queries": len(aps), "excluded_queries": len(rows) - len(aps), "queries": queries}


def calibrate(queries: list[dict], precision_target: float = .99) -> dict | None:
    # Keep tiny validation sets review-only; the target is empirical, not a guarantee.
    if len(queries) < 20 or not any(not q["correct"] for q in queries):
        return None
    best = None
    for threshold in sorted({q["score"] for q in queries}):
        for margin in sorted({0.0, *(q["margin"] for q in queries)}):
            accepted = [q for q in queries if q["score"] >= threshold and q["margin"] >= margin]
            if len(accepted) < 10:
                continue
            precision = sum(q["correct"] for q in accepted) / len(accepted)
            if precision >= precision_target and (best is None or len(accepted) > best["accepted"]):
                best = {"threshold": threshold, "margin": margin, "precision": precision, "accepted": len(accepted), "target": precision_target}
    return best


def partition(rows: list[dict], seed: int = 42) -> dict[str, list[str]]:
    identities = sorted({r["identity"] for r in rows if r.get("identity") and not r.get("excluded")}, key=lambda name: fingerprint([seed, name]))
    # Two identities in each evaluation partition are needed for positive/negative retrieval.
    if len(identities) < 6:
        raise ValueError("A training/evaluation snapshot needs at least six labeled identities (two per split). You can still run and review pretrained experiments.")
    validation = max(2, round(len(identities) * .2))
    test = max(2, round(len(identities) * .2))
    train = len(identities) - validation - test
    return {"train": identities[:train], "validation": identities[train:train + validation], "test": identities[train + validation:]}
