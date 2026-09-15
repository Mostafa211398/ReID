"""Versioned feature extraction and scoring for isolated two-image comparisons."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .reid_color import COLOR_VERSION
from .reid_core import fingerprint, unit


WEIGHT_KEYS = ("appearance", "color", "shape", "semantic")
DEFAULT_WEIGHTS = {"appearance": .60, "color": .20, "shape": .15, "semantic": .05}
SHAPE_CONFIG = {"algorithm": "hog-letterbox-v1", "size": 128, "block": 16, "stride": 8, "cell": 8, "bins": 9}
SHAPE_VERSION = fingerprint(SHAPE_CONFIG)

SEMANTIC_GROUPS = {
    "vehicle_type": [
        ("passenger car", "a cropped photo of a passenger car"),
        ("pickup truck", "a cropped photo of a pickup truck"),
        ("box truck", "a cropped photo of a box truck"),
        ("dump truck", "a cropped photo of a dump truck"),
        ("tanker truck", "a cropped photo of a tanker truck"),
        ("concrete mixer", "a cropped photo of a concrete mixer truck"),
        ("tractor trailer", "a cropped photo of a semi truck with a trailer"),
        ("bus", "a cropped photo of a bus"),
        ("van", "a cropped photo of a van"),
        ("construction vehicle", "a cropped photo of a construction vehicle"),
    ],
    "viewpoint": [
        ("front", "a cropped vehicle viewed directly from the front"),
        ("rear", "a cropped vehicle viewed directly from the rear"),
        ("left side", "a cropped vehicle viewed directly from the left side"),
        ("right side", "a cropped vehicle viewed directly from the right side"),
        ("front left", "a cropped vehicle viewed from the front left"),
        ("front right", "a cropped vehicle viewed from the front right"),
        ("rear left", "a cropped vehicle viewed from the rear left"),
        ("rear right", "a cropped vehicle viewed from the rear right"),
    ],
    "plate_visibility": [
        ("clearly visible", "a cropped vehicle with a clearly visible license plate"),
        ("partly visible", "a cropped vehicle with a partly visible or blurred license plate"),
        ("not visible", "a cropped vehicle with no visible license plate"),
    ],
    "logo_visibility": [
        ("clearly visible", "a cropped vehicle with a clearly visible company logo or branding"),
        ("unclear", "a cropped vehicle with small or unclear company branding"),
        ("not visible", "a cropped vehicle with no visible company logo or branding"),
    ],
}
CRITERIA_PROMPT_GROUPS = {
    "cargo_structure": [(label, f"a cropped truck with {description}") for label, description in (
        ("flatbed", "a flat cargo bed"), ("box body", "an enclosed box cargo body"),
        ("tanker", "a liquid tanker body"), ("dump body", "a tipper or dump cargo body"),
        ("cage carrying barrels", "a metal cage carrying barrels or cylinders"),
        ("covered cargo", "cargo covered by a tarp or net"), ("shipping container", "a shipping container"),
        ("concrete mixer", "a concrete mixer drum"), ("empty open bed", "an empty open cargo bed"),
        ("other or unclear", "an unclear cargo structure"))],
    "cage_pattern": [(label, f"a cropped truck cargo cage with {description}") for label, description in (
        ("rectangular grid", "rectangular grid cells"), ("diamond or hexagonal grid", "diamond or hexagonal cells"),
        ("vertical rails", "mainly vertical rails"), ("solid panels", "solid side panels"),
        ("net or tarp covered", "a net or tarp covering it"), ("no visible cage", "no visible cage"),
        ("unclear", "an unclear pattern"))],
    "cage_condition": [(label, f"a cropped truck with {description}") for label, description in (
        ("painted and intact cage", "a painted intact cargo cage"), ("rusted cage", "a visibly rusted cargo cage"),
        ("damaged cage", "a bent or damaged cargo cage"), ("net-covered cage", "a cargo cage covered by netting"),
        ("no visible cage", "no visible cargo cage"), ("unclear", "an unclear cargo cage condition"))],
    "barrel_wrapping": [(label, f"a cropped truck with {description}") for label, description in (
        ("plastic-wrapped barrels", "plastic-wrapped barrels or cylinders"), ("bare metal barrels", "bare metal barrels"),
        ("mixed wrapped and bare barrels", "both wrapped and bare barrels"), ("no visible barrels", "no visible barrels"),
        ("unclear", "barrels whose wrapping is unclear"))],
    "barrel_shape": [(label, f"a cropped truck carrying {description}") for label, description in (
        ("horizontal cylinders", "horizontal cylindrical barrels"), ("vertical cylinders", "upright cylindrical barrels"),
        ("mixed cylinders", "cylinders in mixed orientations"), ("no visible barrels", "no visible barrels"),
        ("unclear", "objects of unclear shape"))],
    "cargo_layout": [(label, f"a cropped truck with {description}") for label, description in (
        ("single tier", "cargo stacked in one tier"), ("two tiers", "cargo stacked in two tiers"),
        ("three or more tiers", "cargo stacked in three or more tiers"), ("irregular stack", "irregularly stacked cargo"),
        ("empty", "an empty cargo area"), ("unclear", "an unclear cargo layout"))],
    "truck_outline": [(label, f"a cropped photo of {description}") for label, description in (
        ("cab-over with extended bed", "a cab-over truck with an extended cargo bed"),
        ("conventional cab with extended bed", "a conventional long-hood truck with an extended bed"),
        ("short rigid truck", "a short rigid truck"), ("tractor and trailer", "a tractor with a separate trailer"),
        ("van body", "a van-shaped commercial vehicle"), ("unclear", "a truck with an unclear outline"))],
    "brand": [(brand, f"a cropped photo of a {brand} truck") for brand in
              ("Chevrolet", "Isuzu", "Mercedes-Benz", "MAN", "Volvo", "Scania", "Iveco", "Renault",
               "Hino", "Mitsubishi Fuso", "Toyota", "Nissan", "Ford", "GMC", "Tata", "unknown")],
    "human_presence": [(label, f"a cropped vehicle with {description}") for label, description in (
        ("no visible person", "no visible person"), ("driver only", "a driver visible inside and nobody outside"),
        ("external person only", "a person outside but no visible driver"),
        ("driver and external person", "a driver inside and another person outside"), ("unclear", "unclear human presence"))],
    "plate_presentation": [(label, f"a cropped vehicle with {description}") for label, description in (
        ("front red plate", "a red front license plate"), ("front white plate", "a white front license plate"),
        ("rear red plate", "a red rear license plate"), ("rear white plate", "a white rear license plate"),
        ("other visible plate", "a visible license plate of another color or position"),
        ("no visible plate", "no visible license plate"), ("unclear", "an unclear license plate"))],
}
COLOR_NAMES = {"white": (245, 245, 245), "black": (20, 20, 20), "gray": (128, 128, 128),
               "red": (200, 35, 35), "orange": (230, 120, 25), "yellow": (225, 205, 35),
               "green": (45, 145, 65), "blue": (45, 90, 190), "brown": (115, 75, 45), "beige": (205, 185, 145)}
for region, phrase in (("cab_color", "truck cab"), ("cargo_color", "cargo area"),
                       ("barrel_color", "barrels or cylinders"), ("background_color", "background")):
    CRITERIA_PROMPT_GROUPS[region] = [(color, f"a cropped truck where the {phrase} is mainly {color}") for color in COLOR_NAMES]

SEMANTIC_VERSION = fingerprint(SEMANTIC_GROUPS)
CRITERIA_VERSION = fingerprint({"name": "criteria-v1", "prompts": CRITERIA_PROMPT_GROUPS})
PAIR_FEATURE_SPEC = {
    "appearance": "coca-image-l2-v1",
    "color": COLOR_VERSION,
    "shape": SHAPE_VERSION,
    "semantic": SEMANTIC_VERSION,
    "criteria": CRITERIA_VERSION,
    "texture": "lbp-8-neighbor-v1",
    "detail": "gradient-histogram-v1",
    "scale": "canny-edge-bounds-v1",
}
PAIR_FEATURE_VERSION = fingerprint(PAIR_FEATURE_SPEC)
CRITERIA_WEIGHTS = {"appearance": .35, "shape": .15, "texture": .10, "detail": .05, "color": .10,
                    "cargo": .10, "vehicle_type": .05, "brand": .05, "plate": .05}


def validate_weights(weights: dict) -> dict[str, float]:
    if set(weights) != set(WEIGHT_KEYS):
        raise ValueError("Weights must contain appearance, color, shape, and semantic")
    values = {key: float(weights[key]) for key in WEIGHT_KEYS}
    if any(not np.isfinite(value) or value < 0 or value > 1 for value in values.values()):
        raise ValueError("Every comparison weight must be between 0 and 1")
    if not np.isclose(sum(values.values()), 1., atol=1e-6):
        raise ValueError("Comparison weights must add up to 1")
    return values


def shape_feature(rgb: np.ndarray):
    if rgb is None or rgb.ndim != 3 or min(rgb.shape[:2]) < 8:
        return None
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    size = SHAPE_CONFIG["size"]
    scale = min(size / gray.shape[1], size / gray.shape[0])
    width, height = max(1, round(gray.shape[1] * scale)), max(1, round(gray.shape[0] * scale))
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((size, size), dtype=np.uint8)
    left, top = (size - width) // 2, (size - height) // 2
    canvas[top:top + height, left:left + width] = resized
    hog = cv2.HOGDescriptor((size, size), (16, 16), (8, 8), (8, 8), 9)
    vector = hog.compute(canvas).reshape(-1).astype(np.float32)
    if vector.shape != (8100,) or not np.isfinite(vector).all() or np.linalg.norm(vector) <= 1e-12:
        return None
    return unit(vector).astype(np.float32)


def texture_feature(rgb: np.ndarray):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    center = gray[1:-1, 1:-1]
    if center.size == 0:
        return None
    code = np.zeros(center.shape, np.uint8)
    neighbors = (gray[:-2, :-2], gray[:-2, 1:-1], gray[:-2, 2:], gray[1:-1, 2:],
                 gray[2:, 2:], gray[2:, 1:-1], gray[2:, :-2], gray[1:-1, :-2])
    for bit, neighbor in enumerate(neighbors):
        code |= ((neighbor >= center).astype(np.uint8) << bit)
    histogram = np.bincount(code.ravel(), minlength=256).astype(np.float32)
    return unit(histogram).astype(np.float32) if histogram.sum() else None


def detail_feature(rgb: np.ndarray):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    if float(magnitude.sum()) <= 1e-8:
        return None
    magnitude_hist, _ = np.histogram(np.clip(magnitude, 0, 4), bins=16, range=(0, 4), weights=magnitude)
    angle_hist, _ = np.histogram(angle % 180, bins=18, range=(0, 180), weights=magnitude)
    return unit(np.concatenate([magnitude_hist, angle_hist]).astype(np.float32)).astype(np.float32)


def edge_scale(rgb: np.ndarray):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    points = cv2.findNonZero(edges)
    if points is None or len(points) < 20:
        return None
    _, _, width, height = cv2.boundingRect(points)
    return float(np.sqrt((width * height) / (rgb.shape[0] * rgb.shape[1])))


def color_summary(rgb: np.ndarray):
    pixels = rgb.reshape(-1, 3).astype(np.float32)
    palette = np.asarray(list(COLOR_NAMES.values()), np.float32)
    assignments = np.argmin(((pixels[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2), axis=1)
    counts = np.bincount(assignments, minlength=len(palette))
    order = np.argsort(-counts)
    dominant = [list(COLOR_NAMES)[index] for index in order[:5] if counts[index] / len(pixels) >= .02]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation = hsv[..., 1] / 255.
    angles = hsv[..., 0] * (2 * np.pi / 180.)
    weight = saturation.sum()
    hue = None if weight <= 1e-6 else float(np.arctan2((np.sin(angles) * saturation).sum(),
                                                       (np.cos(angles) * saturation).sum()) % (2 * np.pi))
    return {"dominant_colors": dominant, "mean_hue_radians": hue}


def _cosine(vectors, available=True):
    if not available:
        return None
    values = np.asarray(vectors, dtype=np.float32)
    if values.shape[0] != 2:
        raise ValueError("A comparison requires exactly two feature vectors")
    return float(np.clip(unit(values[0]) @ unit(values[1]), -1, 1))


def _attribute(attributes, group, index):
    value = dict(attributes[index][group])
    if group == "brand":
        ordered = sorted(value["distribution"].values(), reverse=True)
        margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
        if value["label"] == "unknown" or value["confidence"] < .35 or margin < .05:
            value.update(label="unknown", confidence=float(value["distribution"].get("unknown", 0.)))
    return value


def _group_similarity(attributes, group):
    labels = [label for label, _ in ({**SEMANTIC_GROUPS, **CRITERIA_PROMPT_GROUPS})[group]]
    left = np.asarray([attributes[0][group]["distribution"][label] for label in labels], np.float32)
    right = np.asarray([attributes[1][group]["distribution"][label] for label in labels], np.float32)
    return _cosine(np.stack([left, right]))


def _comparison_attribute(attributes, group):
    return {"image_1": _attribute(attributes, group, 0), "image_2": _attribute(attributes, group, 1),
            "consistency_score": _group_similarity(attributes, group)}


def build_criteria_report(arrays, metadata, appearance_normalized, color_similarity, shape_similarity):
    attributes = metadata["criteria_attributes"]
    texture = _cosine(arrays["texture"], bool(np.asarray(arrays["texture_available"]).all()))
    detail = _cosine(arrays["detail"], bool(np.asarray(arrays["detail_available"]).all()))
    scale_values = np.asarray(arrays["edge_scale"], dtype=float)
    scale_ratio = None if not np.isfinite(scale_values).all() or np.max(scale_values) <= 0 else float(np.min(scale_values) / np.max(scale_values))
    colors = metadata["color_summary"]
    hues = [value.get("mean_hue_radians") for value in colors]
    hue_shift = None
    if all(value is not None for value in hues):
        delta = abs(hues[0] - hues[1]) % (2 * np.pi)
        hue_shift = float(min(delta, 2 * np.pi - delta) / np.pi)
    cargo_groups = ("cargo_structure", "cage_pattern", "cage_condition", "barrel_wrapping", "barrel_shape", "cargo_layout")
    cargo_score = float(np.mean([_group_similarity(attributes, group) for group in cargo_groups]))
    plate_score = float(np.mean([_group_similarity(attributes, "plate_visibility"),
                                 _group_similarity(attributes, "plate_presentation")]))
    components = {"appearance": appearance_normalized, "shape": shape_similarity, "texture": texture,
                  "detail": detail, "color": color_similarity, "cargo": cargo_score,
                  "vehicle_type": _group_similarity(attributes, "vehicle_type"),
                  "brand": _group_similarity(attributes, "brand"), "plate": plate_score}
    active = {key: weight for key, weight in CRITERIA_WEIGHTS.items() if components[key] is not None}
    total = sum(active.values())
    effective = {key: (active.get(key, 0.) / total if total else 0.) for key in CRITERIA_WEIGHTS}
    score = float(sum(effective[key] * (components[key] or 0.) for key in components))
    band = "high" if score >= .80 else "moderate" if score >= .60 else "low"
    viewpoint = _comparison_attribute(attributes, "viewpoint")
    angles = {"front": 0, "front right": 45, "right side": 90, "rear right": 135, "rear": 180,
              "rear left": -135, "left side": -90, "front left": -45}
    left_angle, right_angle = angles[viewpoint["image_1"]["label"]], angles[viewpoint["image_2"]["label"]]
    angle_difference = min(abs(left_angle - right_angle), 360 - abs(left_angle - right_angle))
    unavailable_text = {"image_1": {"label": "unknown", "confidence": None, "distribution": {}},
                        "image_2": {"label": "unknown", "confidence": None, "distribution": {}},
                        "availability": False, "reason": "OCR is not available"}
    region_colors = {name.removesuffix("_color"): _comparison_attribute(attributes, name)
                     for name in ("cab_color", "cargo_color", "barrel_color", "background_color")}
    return {
        "version": "criteria-v1", "fingerprint": CRITERIA_VERSION,
        "appearance_embedding": {"appearance_similarity": appearance_normalized,
                                 "shape_similarity": shape_similarity, "texture_consistency": texture,
                                 "detail_preservation": detail,
                                 "cargo_structure": _comparison_attribute(attributes, "cargo_structure"),
                                 "cage_condition": _comparison_attribute(attributes, "cage_condition"),
                                 "barrel_wrapping": _comparison_attribute(attributes, "barrel_wrapping"),
                                 "human_presence": _comparison_attribute(attributes, "human_presence")},
        "color_extractor": {"dominant_colors": {"image_1": colors[0]["dominant_colors"],
                                                   "image_2": colors[1]["dominant_colors"],
                                                   "shared": sorted(set(colors[0]["dominant_colors"]) & set(colors[1]["dominant_colors"]))},
                            "color_distribution": region_colors,
                            "color_variance": None if color_similarity is None else float(1 - color_similarity),
                            "hue_shift": hue_shift},
        "shape_extractor": {"truck_outline": _comparison_attribute(attributes, "truck_outline"),
                            "cage_grid_pattern": _comparison_attribute(attributes, "cage_pattern"),
                            "barrel_shape": _comparison_attribute(attributes, "barrel_shape"),
                            "spatial_layout": _comparison_attribute(attributes, "cargo_layout"),
                            "viewpoint_angle_diff": float(angle_difference), "scale_ratio": scale_ratio},
        "plate_logo_type_viewpoint": {"brand_logo": _comparison_attribute(attributes, "brand"),
                                      "logo_visibility": _comparison_attribute(attributes, "logo_visibility"),
                                      "model_text": unavailable_text,
                                      "license_plate": {"presentation": _comparison_attribute(attributes, "plate_presentation"),
                                                        "visibility": _comparison_attribute(attributes, "plate_visibility"),
                                                        "characters": unavailable_text,
                                                        "identity_claim": "not evaluated"},
                                      "vehicle_type": _comparison_attribute(attributes, "vehicle_type"),
                                      "viewpoint": viewpoint,
                                      "human_activity": _comparison_attribute(attributes, "human_presence")},
        "criteria_components": components, "configured_weights": CRITERIA_WEIGHTS,
        "effective_weights": effective, "criteria_similarity_score": score,
        "qualitative_band": band, "status": "experimental-uncalibrated",
        "availability": {key: components[key] is not None for key in components},
        "extractor_versions": {"criteria": CRITERIA_VERSION, "shape": SHAPE_VERSION,
                               "texture": "lbp-8-neighbor-v1", "detail": "gradient-histogram-v1",
                               "color": COLOR_VERSION, "scale": "canny-edge-bounds-v1"},
    }


def build_result(comparison: dict, encoder: dict, arrays: dict, metadata: dict) -> dict:
    appearance_raw = _cosine(arrays["appearance"])
    appearance_normalized = float(np.clip((appearance_raw + 1.) / 2., 0, 1))
    encoder_only = encoder.get("pair_mode", "visual_cosine" if encoder["family"].endswith("_visual") else "vlm_fusion") == "visual_cosine"
    weights = {"appearance": 1., "color": 0., "shape": 0., "semantic": 0.} if encoder_only else validate_weights(comparison["weights"])
    available = {
        "appearance": True,
        "color": not encoder_only and bool(np.asarray(arrays["color_available"]).all()),
        "shape": not encoder_only and bool(np.asarray(arrays["shape_available"]).all()),
        "semantic": not encoder_only,
    }
    components = {
        "appearance": appearance_normalized,
        "color": _cosine(arrays.get("color", np.zeros((2, 1))), available["color"]),
        "shape": _cosine(arrays.get("shape", np.zeros((2, 1))), available["shape"]),
        "semantic": _cosine(arrays.get("semantic", np.zeros((2, 1))), available["semantic"]),
    }
    denominator = sum(weights[key] for key in WEIGHT_KEYS if available[key])
    if denominator <= 0:
        raise ValueError("At least one available comparison feature needs a positive weight")
    effective = {key: weights[key] / denominator if available[key] else 0. for key in WEIGHT_KEYS}
    combined = appearance_raw if encoder_only else float(sum(effective[key] * (components[key] or 0.) for key in WEIGHT_KEYS))
    images = []
    for index, item in enumerate(comparison["images"]):
        vector_keys = ("appearance",) if encoder_only else WEIGHT_KEYS
        images.append({
            "side": item["side"],
            "filename": item["filename"],
            "image_path": f'images/{item["side"]}.jpg',
            "vectors": {key: np.asarray(arrays[key][index], dtype=float).tolist() for key in vector_keys},
            "norms": {key: float(np.linalg.norm(arrays[key][index])) if available[key] else None for key in WEIGHT_KEYS},
            "color_method": metadata.get("color", [{}, {}])[index].get("method", "unavailable"),
            "semantic_attributes": metadata.get("semantic_attributes", [{}, {}])[index],
        })
    threshold = float(comparison["threshold"])
    feature_spec = ({"appearance": f'{encoder["family"]}-l2-v1'} if encoder_only else
                    {**PAIR_FEATURE_SPEC, "appearance": f'{encoder["family"]}-image-l2-v1'})
    scoring = {
        "weights": weights,
        "effective_weights": effective,
        "threshold": threshold,
        "feature_spec": feature_spec,
        "fingerprint": fingerprint({"encoder": encoder["fingerprint"], "features": feature_spec,
                                    "weights": weights, "threshold": threshold}),
    }
    checkpoint_name = encoder.get("checkpoint_filename", "open_clip_pytorch_model.bin"
                                  if encoder["family"] in ("coca", "coca_l14") else "model.safetensors")
    criteria = None
    if not encoder_only and "criteria_attributes" in metadata and "texture" in arrays:
        criteria = build_criteria_report(arrays, metadata, appearance_normalized, components["color"], components["shape"])
    result = {
        "comparison_id": comparison["id"], "name": comparison["name"], "site_id": comparison["site_id"],
        "encoder": {"id": encoder["id"], "name": encoder["name"], "family": encoder["family"],
                    "model_name": encoder.get("model_name"), "repository_name": encoder.get("repository_name"),
                    "revision": encoder.get("revision"), "fingerprint": encoder["fingerprint"],
                    "pair_mode": encoder.get("pair_mode"), "parent_encoder": encoder.get("parent_encoder"),
                    "parent_sha256": encoder.get("parent_sha256"),
                    "extraction_version": encoder.get("extraction_version"),
                    "embedding_dimension": encoder.get("embedding_dimension", np.asarray(arrays["appearance"]).shape[-1]),
                    "checkpoint": {"filename": checkpoint_name,
                                   "sha256": encoder.get("checksums", {}).get(checkpoint_name),
                                   "size_bytes": encoder.get("sizes", {}).get(
                                       checkpoint_name, encoder.get("checkpoint_size_bytes"))}},
        "images": images,
        "similarities": {"appearance_cosine_raw": appearance_raw, "appearance": components["appearance"],
                         "color": components["color"], "shape": components["shape"],
                         "semantic": components["semantic"], "combined": combined},
        "availability": available, "scoring": scoring, "encoder_only": encoder_only,
        "decision": {"same_vehicle": combined >= threshold, "threshold": threshold,
                     "score_source": "raw-cosine" if encoder_only else "combined-fusion",
                     "status": "experimental-uncalibrated"},
    }
    if criteria is not None:
        result["criteria_report"] = criteria
        result["criteria_similarity_score"] = criteria["criteria_similarity_score"]
    return result


def public_summary(result: dict) -> dict:
    summary = {**{key: result[key] for key in ("encoder", "similarities", "availability", "scoring", "decision", "encoder_only")},
            "images": [{"side": image["side"], "filename": image["filename"], "norms": image["norms"],
                        "color_method": image["color_method"], "semantic_attributes": image["semantic_attributes"]}
                       for image in result["images"]]}
    if "criteria_report" in result:
        summary.update(criteria_report=result["criteria_report"],
                       criteria_similarity_score=result["criteria_similarity_score"])
    return summary
