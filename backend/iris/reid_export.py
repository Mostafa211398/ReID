"""Portable, path-safe experiment result packages."""
from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from pathlib import Path

from .reid_core import atomic_replace
from .utils import utc_now

OBSERVATION_FIELDS = [
    "experiment_id", "experiment_name", "site_id", "observation_id", "original_filename", "query_image",
    "camera_id", "class_name", "capture_time", "encoder_id", "encoder_name", "encoder_fingerprint",
    "detected_vehicle_id", "encoder_assigned_vehicle_id", "decision_status", "provenance", "reviewed",
    "top_candidate_vehicle_id", "selected_reference_observation_id", "selected_reference_crop_index",
    "selected_reference_image", "appearance_cosine", "color_similarity", "combined_score", "matching_mode",
    "decision_threshold", "decision_margin", "new_vehicle_threshold", "scoring_version",
]

COMPARISON_FIELDS = [
    "experiment_id", "query_observation_id", "query_crop_index", "query_image", "query_camera_id",
    "candidate_vehicle_id", "reference_observation_id", "reference_crop_index", "reference_image",
    "reference_camera_id", "query_capture_time", "reference_capture_time", "class_name", "query_class_name",
    "reference_class_name", "encoder_id", "encoder_name", "encoder_fingerprint",
    "cosine_similarity", "cosine_rank", "color_similarity", "combined_score", "effective_color_weight",
    "context_score", "detected_vehicle_id", "assigned_vehicle_id", "decision_status", "is_winning_identity",
    "is_assigned_identity", "is_decision_reference",
]


def safe_part(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")[:120] or "item"


def crop_name(track, index, reference=False):
    if reference:
        return f'images/references/{safe_part(track.get("global_id") or "unassigned")}/{safe_part(track["id"])}_{index}.jpg'
    suffix = "" if index == 0 else f"_{index}"
    return f'images/queries/{safe_part(track["id"])}{suffix}.jpg'


def csv_text(rows, fields):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def portable_results(job, tracks, query_names):
    config = json.loads(json.dumps(job["config"]))
    for clip in config.get("clips", []):
        clip.pop("path", None)
        clip["archive_path"] = query_names.get((clip["id"] + "_1", 0))
    records = json.loads(json.dumps(tracks))
    for track in records:
        for index, crop in enumerate(track.get("crops", [])):
            crop.pop("path", None)
            crop["archive_path"] = query_names.get((track["id"], index))
    return {"experiment_id": job["id"], "config": config, "metrics": job["metrics"], "tracks": records}


def build_results_package(service, job, encoder_specs, comparisons):
    tracks = service.tracks(job["id"])
    by_id = {track["id"]: track for track in service.tracks(site_id=job["site_id"])}
    clips = {clip["id"]: clip for clip in job["config"].get("clips", [])}
    query_names = {(track["id"], index): crop_name(track, index) for track in tracks for index, _ in enumerate(track.get("crops", []))}
    reference_keys = {(row["reference_observation_id"], row["reference_crop_index"]) for row in comparisons}
    reference_names = {key: query_names.get(key) or crop_name(by_id[key[0]], key[1], True) for key in reference_keys}
    specs = {spec["id"]: spec for spec in encoder_specs}
    observations = []
    for track in tracks:
        clip = clips.get(track["clip_id"], {})
        for encoder_id in job["config"].get("encoders", []):
            assignment = track.get("assignments", {}).get(encoder_id, {})
            reference_key = (assignment.get("reference_track_id"), assignment.get("reference_crop_index", 0))
            observations.append({
                "experiment_id": job["id"], "experiment_name": job["name"], "site_id": job["site_id"],
                "observation_id": track["id"], "original_filename": clip.get("filename"),
                "query_image": query_names.get((track["id"], 0)), "camera_id": track["camera_id"],
                "class_name": track.get("class_name", "truck"), "capture_time": track.get("absolute_start"),
                "encoder_id": encoder_id, "encoder_name": specs.get(encoder_id, {}).get("name", encoder_id),
                "encoder_fingerprint": assignment.get("encoder_fingerprint") or specs.get(encoder_id, {}).get("fingerprint"),
                "detected_vehicle_id": track.get("global_id"), "encoder_assigned_vehicle_id": assignment.get("global_id"),
                "decision_status": assignment.get("status"), "provenance": assignment.get("provenance"),
                "reviewed": bool(track.get("reviewed")), "top_candidate_vehicle_id": assignment.get("comparison_global_id"),
                "selected_reference_observation_id": assignment.get("reference_track_id"),
                "selected_reference_crop_index": assignment.get("reference_crop_index"),
                "selected_reference_image": reference_names.get(reference_key),
                "appearance_cosine": assignment.get("appearance_similarity"), "color_similarity": assignment.get("color_similarity"),
                "combined_score": assignment.get("combined_score", assignment.get("similarity")),
                "matching_mode": assignment.get("matching_mode"), "decision_threshold": assignment.get("decision_threshold"),
                "decision_margin": assignment.get("decision_margin"), "new_vehicle_threshold": assignment.get("new_vehicle_threshold"),
                "scoring_version": assignment.get("scoring_version"),
            })
    exported_comparisons = []
    for row in comparisons:
        value = dict(row)
        value["query_image"] = query_names[(row["query_observation_id"], row["query_crop_index"])]
        value["reference_image"] = reference_names[(row["reference_observation_id"], row["reference_crop_index"])]
        exported_comparisons.append(value)
    manifest = {
        "schema_version": "1.0", "generated_at": utc_now().isoformat(), "experiment_id": job["id"],
        "experiment_name": job["name"], "site_id": job["site_id"], "matching": service.get("site", job["site_id"]).get("matching"),
        "encoders": [{k: spec.get(k) for k in ("id", "name", "family", "fingerprint", "embedding_dimension", "runtime")} for spec in encoder_specs],
        "counts": {"observations": len(tracks), "observation_encoder_rows": len(observations),
                   "cosine_comparisons": len(exported_comparisons), "query_images": len(query_names),
                   "reference_images": len(reference_names)},
        "score_definitions": {
            "cosine_similarity": "L2-normalized query embedding dot L2-normalized reference embedding",
            "combined_score": "Appearance cosine blended with color similarity when color is available",
            "cosine_rank": "Descending raw cosine rank within one query crop and encoder",
        },
    }
    path = service.root / job["id"] / "complete-results.zip"
    temporary = path.with_suffix(".zip.partial")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, allow_nan=False))
        archive.writestr("README.txt", """IRIS ReID complete results

data/observations.csv and .json contain one row per query observation and encoder.
data/cosine_comparisons.csv and .json contain every eligible trusted-gallery crop comparison.
data/results.csv and .json are portable versions of the standard experiment results.
images/queries contains uploaded crops, images/references contains gallery crops, and images/annotated contains rendered labels per encoder.

cosine_similarity is the raw dot product of L2-normalized embeddings. combined_score may blend that appearance score with color_similarity. cosine_rank is calculated separately for each query crop and encoder. detected_vehicle_id is the final site vehicle ID; encoder_assigned_vehicle_id is that encoder's decision. decision_status and provenance distinguish automatic, new, review, and human-reviewed outcomes. The comparison flags identify the top candidate identity, assigned identity, and exact reference used for the decision.

All paths are relative to this ZIP. Local paths, embeddings, and model weights are excluded.
""")
        archive.writestr("data/observations.csv", csv_text(observations, OBSERVATION_FIELDS))
        archive.writestr("data/observations.json", json.dumps(observations, indent=2, allow_nan=False))
        archive.writestr("data/cosine_comparisons.csv", csv_text(exported_comparisons, COMPARISON_FIELDS))
        archive.writestr("data/cosine_comparisons.json", json.dumps(exported_comparisons, indent=2, allow_nan=False))
        archive.writestr("data/results.csv", (service.root / job["id"] / "results.csv").read_bytes())
        archive.writestr("data/results.json", json.dumps(portable_results(job, tracks, query_names), indent=2, allow_nan=False))
        written = set()
        for names in (query_names, reference_names):
            for key, name in names.items():
                if name not in written:
                    archive.write(by_id[key[0]]["crops"][key[1]]["path"], name)
                    written.add(name)
        used = set()
        for encoder_id in job["config"].get("encoders", []):
            for clip in job["config"].get("clips", []):
                source = service.root / job["id"] / clip["id"] / (encoder_id + ".jpg")
                if not source.is_file():
                    continue
                base = safe_part(Path(clip.get("filename") or clip["id"]).stem) + ".jpg"
                name = f"images/annotated/{safe_part(encoder_id)}/{base}"
                if name in used:
                    name = f"images/annotated/{safe_part(encoder_id)}/{safe_part(clip['id'])}_{base}"
                archive.write(source, name)
                used.add(name)
    atomic_replace(temporary, path)
    return path
