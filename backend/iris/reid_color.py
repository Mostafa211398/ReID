"""CPU color descriptors and shared, versioned appearance/color scoring."""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import cv2
import numpy as np

from .reid_core import atomic_json, fingerprint, unit


COLOR_CONFIG = {"algorithm": "hsv-foreground-v1", "max_side": 192, "iterations": 3,
                "hue_bins": 16, "saturation_bins": 4, "gray_bins": 8,
                "gray_saturation": 48, "dark_value": 40, "seed": 42,
                "foreground_min": .1, "foreground_max": .98, "center_sigma": .55}
COLOR_VERSION = fingerprint(COLOR_CONFIG)
SCORING_POLICY = "same-crop-pair-linear-v1"
_grabcut_lock = threading.Lock()


def scoring_spec(encoder_fingerprint, color_weight=.25):
    value = {"encoder_fingerprint": encoder_fingerprint, "color_version": COLOR_VERSION,
             "color_weight": color_weight, "policy": SCORING_POLICY, "missing_color": "appearance-only"}
    return {**value, "fingerprint": fingerprint(value)}


def extract_color(bgr):
    if bgr is None or bgr.ndim != 3 or min(bgr.shape[:2]) < 8:
        return {"vector": None, "method": "unavailable", "reason": "Missing or undersized crop"}
    height, width = bgr.shape[:2]
    scale = min(1., COLOR_CONFIG["max_side"] / max(height, width))
    bgr = cv2.resize(bgr, (max(8, round(width * scale)), max(8, round(height * scale))), interpolation=cv2.INTER_AREA)
    height, width = bgr.shape[:2]
    yy, xx = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
    weights = np.exp(-(xx * xx + yy * yy) / (2 * COLOR_CONFIG["center_sigma"] ** 2))
    method = "center-weighted"
    try:
        mask = np.zeros((height, width), np.uint8)
        border_x, border_y = max(1, round(width * .02)), max(1, round(height * .02))
        with _grabcut_lock:
            cv2.setRNGSeed(COLOR_CONFIG["seed"])
            cv2.grabCut(bgr, mask, (border_x, border_y, width - 2 * border_x, height - 2 * border_y),
                        np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
                        COLOR_CONFIG["iterations"], cv2.GC_INIT_WITH_RECT)
        foreground = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
        coverage = float(foreground.mean())
        if COLOR_CONFIG["foreground_min"] <= coverage <= COLOR_CONFIG["foreground_max"] and foreground.sum() >= 64:
            weights *= foreground
            method = "grabcut"
    except cv2.error:
        pass
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = [hsv[:, :, index] for index in range(3)]
    gray = (s < COLOR_CONFIG["gray_saturation"]) | (v < COLOR_CONFIG["dark_value"])
    chromatic = np.histogram2d(h[~gray], s[~gray], bins=(16, 4), range=((0, 180), (0, 256)), weights=weights[~gray])[0]
    # Smooth adjacent hue bins across the red wrap-around and adjacent gray values.
    chromatic = .5 * chromatic + .25 * np.roll(chromatic, 1, axis=0) + .25 * np.roll(chromatic, -1, axis=0)
    achromatic = np.histogram(v[gray], bins=8, range=(0, 256), weights=weights[gray])[0]
    padded = np.pad(achromatic, (1, 1), mode="edge")
    achromatic = .25 * padded[:-2] + .5 * padded[1:-1] + .25 * padded[2:]
    hist = np.concatenate([chromatic.ravel(), achromatic])
    if not hist.sum() or not np.isfinite(hist).all():
        return {"vector": None, "method": "unavailable", "reason": "No usable color pixels"}
    return {"vector": np.sqrt(hist / hist.sum()).tolist(), "method": method}


class ColorCache:
    def __init__(self, root):
        self.root = Path(root) / COLOR_VERSION

    def crop(self, crop):
        try:
            content = Path(crop["path"]).read_bytes()
        except (OSError, KeyError):
            return {"vector": None, "method": "unavailable", "color_version": COLOR_VERSION, "reason": "Saved crop is missing"}
        digest = hashlib.sha256(content).hexdigest()
        path = self.root / (digest + ".json")
        if path.exists():
            try:
                result = json.loads(path.read_text())
                vector = np.asarray(result.get("vector"), dtype=float)
                if result.get("source_hash") == digest and result.get("color_version") == COLOR_VERSION and (result.get("vector") is None or (vector.shape == (72,) and np.isfinite(vector).all() and np.isclose(np.linalg.norm(vector), 1))):
                    return result
            except (OSError, ValueError, TypeError):
                pass
        try:
            decoded = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        except cv2.error:
            decoded = None
        result = {**extract_color(decoded),
                  "source_hash": digest, "color_version": COLOR_VERSION}
        atomic_json(path, result)
        return result

    def tracks(self, rows):
        return {row["id"]: [self.crop(crop) for crop in row.get("crops", [])] for row in rows}


def best_pair(query, reference, query_colors, reference_colors, weight):
    """Both components and the displayed crop indices come from one winning pair."""
    appearance = unit(query) @ unit(reference).T
    best = None
    for qi in range(appearance.shape[0]):
        for ri in range(appearance.shape[1]):
            qc = query_colors[qi] if qi < len(query_colors) else {}
            rc = reference_colors[ri] if ri < len(reference_colors) else {}
            color = float(np.clip(unit(qc["vector"]) @ unit(rc["vector"]), 0, 1)) if qc.get("vector") is not None and rc.get("vector") is not None else None
            effective = weight if color is not None else 0.
            raw = float(appearance[qi, ri])
            score = (1 - effective) * raw + effective * (color or 0.)
            result = {"similarity": score, "combined_score": score, "appearance_similarity": raw,
                      "color_similarity": color, "color_weight": weight, "effective_color_weight": effective,
                      "query_crop_index": qi, "reference_crop_index": ri,
                      "color_methods": {"query": qc.get("method", "unavailable"), "reference": rc.get("method", "unavailable")}}
            if best is None or score > best["similarity"]:
                best = result
    return best
