from __future__ import annotations

import csv
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from functools import wraps
from urllib.parse import quote
from uuid import uuid4

import numpy as np

from .utils import sha256_file
from .reid_core import BASELINES, ENCODER_METADATA, atomic_json, calibrate, fingerprint, partition, retrieval, simultaneous, track_vector
from .reid_schemas import BenchmarkInput, Job, MatchingSettings, PairComparisonInput, Site, SiteInput, TrainingInput
from .reid_color import ColorCache, best_pair, scoring_spec
from .utils import utc_now


def serialized_mutation(method):
    @wraps(method)
    def locked(self, *args, **kwargs):
        with self.lock:
            from .reid_delete import journals
            if journals(self):
                raise RuntimeError("Finish the pending experiment deletion before changing ReID data")
            return method(self, *args, **kwargs)
    return locked


class Cancelled(Exception):
    pass


class ReIDService:
    def __init__(self, db, settings, coordinator):
        self.db, self.settings, self.coordinator = db, settings, coordinator
        self.root = (settings.data_dir / "reid").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        project = Path(__file__).resolve().parents[2]
        self.python = settings.reid_python or project / ".runtime/reid-env/Scripts/python.exe"
        self.model_root = (settings.reid_model_dir or settings.data_dir / "models/reid").resolve()
        self.lock = threading.RLock()
        self.closed = threading.Event()
        self.events = {}
        self.process = None
        self.active_id = None
        self.queue = queue.Queue()
        from .reid_delete import recover, journals
        recover(self)
        for job in self.all("job"):
            if job["state"] == "running":
                job.update(state="failed", error="Application stopped during this stage. Retry preserves completed stages.")
                self.put("job", job)
            elif job["state"] == "queued":
                if journals(self):
                    job.update(state="failed", error="Finish pending deletion, then retry this job.")
                    self.put("job", job)
                else:
                    self.queue.put(job["id"])
        self.thread = threading.Thread(target=self._worker, name="iris-reid", daemon=True)
        self.thread.start()

    def all(self, kind):
        result, offset = [], 0
        while True:
            page = self.db.list("reid_" + kind, limit=500, offset=offset)
            result.extend(page)
            if len(page) < 500:
                return result
            offset += len(page)

    def get(self, kind, key):
        value = self.db.get("reid_" + kind, key)
        if value is None:
            raise KeyError(key)
        return value

    def put(self, kind, value):
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        self.db.put("reid_" + kind, value["id"], value)
        return value

    def encoders(self):
        manifest_path = self.model_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        entries = []
        for key, value in BASELINES.items():
            record = {**value, **manifest.get(key, {}), **ENCODER_METADATA[key], "origin": "pretrained"}
            files = record.get("checksums", {})
            record["available"] = bool(record.get("fingerprint")) and bool(files) and all((self.model_root / key / name).is_file() for name in files)
            record["availability_reason"] = None if record["available"] else f"Install with download-reid-models.ps1 -Encoders {key}"
            if key == "openvino" and record["available"]:
                runtime_package = Path(self.python).parent.parent / "Lib/site-packages/openvino/__init__.py"
                if not runtime_package.is_file():
                    record.update(available=False, availability_reason="OpenVINO runtime missing. Run setup-reid.ps1 in the ReID environment.")
            if key in ("coca", "coca_visual", "coca_l14", "coca_l14_visual") and record["available"]:
                runtime_package = Path(self.python).parent.parent / "Lib/site-packages/open_clip/__init__.py"
                if not runtime_package.is_file():
                    record.update(available=False, availability_reason="OpenCLIP runtime missing. Run setup-reid.ps1 in the ReID environment.")
            entries.append(record)
        for record in self.all("encoder"):
            record.update(ENCODER_METADATA.get(record["family"], {}))
            record["available"] = Path(record["checkpoint"]).is_file()
            entries.append(record)
        return entries

    def encoder(self, key):
        entry = next((e for e in self.encoders() if e["id"] == key), None)
        if entry is None:
            raise ValueError("Unknown ReID encoder")
        return entry

    @serialized_mutation
    def create_site(self, value: SiteInput):
        data = value.model_dump()
        data["matching"] = (value.matching or MatchingSettings()).model_dump()
        return self.put("site", Site(**data))

    @serialized_mutation
    def update_site(self, site_id, value):
        with self.lock:
            site = self.get("site", site_id)
            removed = {c["id"] for c in site["cameras"]} - {c.id for c in value.cameras}
            used = {t["camera_id"] for t in self.all("track") if t["site_id"] == site_id}
            if removed & used:
                raise ValueError("Cameras referenced by existing tracks cannot be removed")
            site.update(value.model_dump(mode="json", exclude={"matching"}))
            if value.matching is not None:
                site["matching"] = value.matching.model_dump()
            return self.put("site", site)

    @serialized_mutation
    def submit(self, job):
        job = self.put("job", job)
        self.events[job["id"]] = threading.Event()
        self.queue.put(job["id"])
        return job

    @serialized_mutation
    def create_experiment(self, value, clips):
        site = self.get("site", value.site_id)
        if any(c.camera_id not in {c["id"] for c in site["cameras"]} for c in value.clips):
            raise ValueError("Register every camera in this site before uploading")
        encoders = list(dict.fromkeys([site["active_encoder"], *value.encoders]))
        for key in encoders:
            self.encoder(key)
        return self.submit(Job(name=value.name, site_id=value.site_id, config={**value.model_dump(mode="json"), "encoders": encoders, "clips": clips}))

    def tracks(self, experiment_id=None, site_id=None):
        return [t for t in self.all("track") if (experiment_id is None or t["experiment_id"] == experiment_id) and (site_id is None or t["site_id"] == site_id)]

    def editable(self, experiment_id):
        job = self.get("job", experiment_id)
        if job["state"] in ("running", "queued"):
            raise RuntimeError("Wait for this experiment to stop before editing its tracks")
        return job

    @serialized_mutation
    def annotate(self, track_id, value):
        with self.lock:
            track = self.get("track", track_id)
            self.editable(track["experiment_id"])
            track.update(value.model_dump())
            return self.put("track", track)

    @serialized_mutation
    def review(self, track_id, value):
        with self.lock:
            track = self.get("track", track_id)
            self.editable(track["experiment_id"])
            identities = [v for v in self.all("identity") if v["site_id"] == track["site_id"]]
            if value.new_identity:
                global_id = self.next_identity(track["site_id"])
                self.put("identity", {"id": f'{track["site_id"]}_{global_id}', "site_id": track["site_id"], "global_id": global_id, "class_name": track.get("class_name", "truck"), "created_at": utc_now().isoformat()})
            else:
                global_id = value.global_id
                if not any(v["global_id"] == global_id for v in identities):
                    raise ValueError("Vehicle identity does not belong to this site")
                identity = next(v for v in identities if v["global_id"] == global_id)
                if identity.get("class_name", "truck") != track.get("class_name", "truck"):
                    raise ValueError("Vehicle identity belongs to a different vehicle class")
                if any(t["id"] != track["id"] and t.get("global_id") == global_id and simultaneous(track, t) for t in self.tracks(site_id=track["site_id"])):
                    raise ValueError("This identity is already assigned to a simultaneous distinct track on the same camera")
            track.update(global_id=global_id, reviewed=True, gallery_reference=True)
            for assignment in track.get("assignments", {}).values():
                assignment.update(global_id=global_id, status="reviewed", provenance="human")
            return self.put("track", track)

    @serialized_mutation
    def cancel(self, job_id):
        with self.lock:
            job = self.get("job", job_id)
            if job["state"] in ("queued", "running"):
                self.events.setdefault(job_id, threading.Event()).set()
                if self.active_id == job_id and self.process and self.process.poll() is None:
                    self.process.kill()
                job.update(state="cancelled", finished_at=utc_now().isoformat())
                self.put("job", job)
            return job

    @serialized_mutation
    def retry(self, job_id, encoders=None):
        with self.lock:
            job = self.editable(job_id)
            if self.active_id == job_id or (job["state"] == "cancelled" and self.queue.unfinished_tasks):
                raise RuntimeError("Wait for this experiment's cancelled worker to stop before retrying")
            if encoders is not None:
                if job["kind"] != "experiment":
                    raise ValueError("Encoder selection is supported only for experiment refresh")
                active = self.get("site", job["site_id"])["active_encoder"]
                selected = list(dict.fromkeys([active, *encoders]))
                for key in selected:
                    spec = self.encoder(key)
                    if not spec["available"]:
                        raise ValueError(spec.get("availability_reason") or f'{spec["name"]} is not installed')
                removed = set(job["config"]["encoders"]) - set(selected)
                for clip in job["config"]["clips"]:
                    for key in removed:
                        for extension in (".jpg", ".mp4"):
                            name = clip["id"] + "/" + key + extension
                            try:
                                self.artifact(job_id, name).unlink()
                            except KeyError:
                                pass
                job["config"]["encoders"] = selected
            job.update(state="queued", error=None, finished_at=None)
            job["artifacts"] = {}
            # Refresh matching and image exports after labels/review. Expensive stages stay cached.
            job["completed_stages"] = [s for s in job["completed_stages"] if not s.startswith(("match-", "render-"))]
            return self.submit(job)

    def check(self, job_id):
        if self.closed.is_set() or self.events.setdefault(job_id, threading.Event()).is_set():
            raise Cancelled()

    def _worker(self):
        while not self.closed.is_set():
            job_id = self.queue.get()
            if job_id is None:
                return
            acquired = False
            owner = "reid:" + job_id
            try:
                self.check(job_id)
                if self.get("job", job_id)["state"] != "queued":
                    continue
                acquired = self.coordinator.acquire(owner, cancel_event=self.events.setdefault(job_id, threading.Event()))
                self.check(job_id)
                if not acquired:
                    continue
                self.active_id = job_id
                job = self.get("job", job_id)
                job.update(state="running", error=None)
                self.put("job", job)
                {"experiment": self._experiment, "training": self._training, "evaluation": self._evaluate,
                 "promotion": self._promote, "benchmark": self._benchmark, "benchmark_export": self._benchmark_export,
                 "comparison": self._comparison}[job["kind"]](job)
                self.check(job_id)
                job.update(state="completed", progress=1, stage="completed", finished_at=utc_now().isoformat())
                self.put("job", job)
            except Cancelled:
                job = self.get("job", job_id)
                job.update(state="cancelled", finished_at=utc_now().isoformat())
                self.put("job", job)
            except Exception as exc:
                job = self.get("job", job_id)
                job.update(state="failed", error=str(exc), finished_at=utc_now().isoformat())
                self.put("job", job)
            finally:
                if self.process and self.process.poll() is None:
                    self.process.kill(); self.process.wait()
                self.process = None
                self.active_id = None
                if acquired:
                    self.coordinator.release(owner)
                self.queue.task_done()

    def run_stage(self, job, stage, config):
        self.check(job["id"])
        if stage in job["completed_stages"]:
            return
        if not Path(self.python).is_file():
            raise RuntimeError("ReID environment is missing. Run scripts/setup-reid.ps1 and download-reid-models.ps1.")
        directory = self.root / job["id"]
        directory.mkdir(parents=True, exist_ok=True)
        progress_path = directory / "progress.json"
        progress_path.unlink(missing_ok=True)
        config.update(progress_path=str(progress_path), model_root=str(self.model_root))
        config_path = directory / "worker.json"
        atomic_json(config_path, config)
        job.update(stage=stage, progress=0)
        self.put("job", job)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        env["PYTHONUNBUFFERED"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["PATH"] = str(Path(self.python).parent) + os.pathsep + str(Path(self.python).parent / "Library/bin") + os.pathsep + env.get("PATH", "")
        with (directory / f"{stage}.log").open("w", encoding="utf-8") as log:
            self.process = subprocess.Popen([str(self.python), "-m", "iris.reid_worker", str(config_path)], env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            while self.process.poll() is None:
                self.check(job["id"])
                if progress_path.exists():
                    try:
                        progress = json.loads(progress_path.read_text())
                        job["progress"] = progress.get("progress", 0)
                        job["metrics"][stage] = progress.get("metrics", {})
                        self.put("job", job)
                    except (OSError, json.JSONDecodeError):
                        pass
                self.events[job["id"]].wait(.25)
            self.check(job["id"])
            if self.process.returncode:
                progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
                raise RuntimeError(progress.get("error") or f"Worker failed in {stage}; see its downloaded log")
        job["completed_stages"].append(stage)
        if progress_path.exists():
            job["metrics"][stage] = json.loads(progress_path.read_text()).get("metrics", {})
        self.put("job", job)

    def vector_path(self, experiment_id, encoder):
        return self.root / experiment_id / f'features-{encoder["fingerprint"]}.npz'

    def load_vectors(self, path):
        if not path.exists():
            return {}
        with np.load(path, allow_pickle=False) as archive:
            return {k: archive[k] for k in archive.files}

    def _experiment(self, job):
        cfg = job["config"]
        active_encoder = self.get("site", job["site_id"])["active_encoder"]
        cfg["encoders"] = list(dict.fromkeys([active_encoder, *cfg["encoders"]]))
        for clip in cfg["clips"]:
            folder = self.root / job["id"] / clip["id"]
            self.run_stage(job, "ingest-" + clip["id"], {"task": "ingest", "clip": clip, "output": str(folder)})
            existing = {t["source_track_id"] for t in self.tracks(job["id"])}
            for track in json.loads((folder / "tracks.json").read_text()):
                if track["id"] not in existing:
                    track.update(experiment_id=job["id"], site_id=job["site_id"], identity=None, excluded=False, global_id=None, reviewed=False, assignments={})
                    self.put("track", track)
        tracks = self.tracks(job["id"])
        for track in tracks:
            track["assignments"] = {key: value for key, value in track.get("assignments", {}).items() if key in cfg["encoders"]}
            self.put("track", track)
        comparison_rows, encoder_specs = [], []
        for key in cfg["encoders"]:
            spec = self.encoder(key)
            encoder_specs.append(spec)
            if not spec["available"]:
                raise RuntimeError(f'{spec["name"]} is not installed. Run download-reid-models.ps1. Completed ingestion stages are retained.')
            path = self.vector_path(job["id"], spec)
            self.run_stage(job, "embed-" + key + "-" + spec["fingerprint"][:12], {"task": "embed", "encoder": spec, "tracks": tracks, "output": str(path)})
            vectors = self.load_vectors(path)
            comparison_rows.extend(self._match(job, spec, vectors))
            assignments = {t["id"]: t.get("assignments", {}).get(key, {}) for t in self.tracks(job["id"])}
            for clip in cfg["clips"]:
                folder = self.root / job["id"] / clip["id"]
                output = folder / (key + ".jpg")
                self.run_stage(job, f'render-{key}-{clip["id"]}', {"task": "render", "clip": clip, "assignments": assignments, "frames": str(folder / "frames.jsonl"), "output": str(output)})
                job["artifacts"][f'{key} · {clip["filename"]}'] = self.artifact_url(job["id"], output.relative_to(self.root / job["id"]).as_posix())
        job["refresh_required"] = False
        rows = self.tracks(job["id"])
        atomic_json(self.root / job["id"] / "results.json", {"experiment_id": job["id"], "config": job["config"], "metrics": job["metrics"], "tracks": rows})
        with (self.root / job["id"] / "results.csv.partial").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["track_id", "camera_id", "start", "end", "encoder", "global_id", "status", "similarity", "human_identity", "media_type", "class_name", "reference_track_id", "encoder_fingerprint", "provenance", "appearance_similarity", "color_similarity", "combined_score", "color_weight", "effective_color_weight", "color_methods", "scoring_version", "reference_crop_index", "query_crop_index", "comparison_global_id"])
            for row in rows:
                for key, assignment in row["assignments"].items():
                    writer.writerow([row["id"], row["camera_id"], row["start"], row["end"], key, assignment.get("global_id"), assignment["status"], assignment.get("similarity"), row.get("identity"), row.get("media_type", "image"), row.get("class_name", "truck"), assignment.get("reference_track_id"), assignment.get("encoder_fingerprint"), assignment.get("provenance"), assignment.get("appearance_similarity"), assignment.get("color_similarity"), assignment.get("combined_score"), assignment.get("color_weight"), assignment.get("effective_color_weight"), json.dumps(assignment.get("color_methods")), assignment.get("scoring_version"), assignment.get("reference_crop_index"), assignment.get("query_crop_index"), assignment.get("comparison_global_id")])
        (self.root / job["id"] / "results.csv.partial").replace(self.root / job["id"] / "results.csv")
        for name in ("results.json", "results.csv"):
            job["artifacts"][name] = self.artifact_url(job["id"], name)
        from .reid_export import build_results_package
        build_results_package(self, job, encoder_specs, comparison_rows)
        job["artifacts"]["Complete results (.zip)"] = self.artifact_url(job["id"], "complete-results.zip")
        self.put("job", job)

    def _match(self, job, spec, vectors):
        from .reid_matching import match
        with self.lock:
            return match(self, job, spec, vectors)

    def results_package(self, experiment_id):
        job = self.get("job", experiment_id)
        if job.get("kind") != "experiment":
            raise KeyError(experiment_id)
        if job.get("deletion_pending"):
            raise RuntimeError("Deletion cleanup is pending; retry Delete experiment")
        if job.get("refresh_required"):
            raise RuntimeError("Refresh matching and results before downloading this package")
        if job.get("state") != "completed":
            raise RuntimeError("The complete results package is available after the experiment completes")
        try:
            path = self.artifact(experiment_id, "complete-results.zip")
        except KeyError as exc:
            raise RuntimeError("Refresh matching and results to generate the complete results package") from exc
        return job, path

    def experiment_comparisons(self, experiment_id, observation_id=None, encoder_id=None,
                               candidate_vehicle_id=None, decision_flag="all", offset=0, limit=24):
        job, package = self.results_package(experiment_id)
        try:
            with zipfile.ZipFile(package) as archive:
                rows = json.loads(archive.read("data/cosine_comparisons.json"))
        except (OSError, KeyError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
            raise RuntimeError("Refresh matching and results to rebuild the comparison audit") from exc

        tracks = {track["id"]: track for track in self.tracks(site_id=job["site_id"])}
        required = {row[key] for row in rows for key in ("query_observation_id", "reference_observation_id")}
        if not required.issubset(tracks):
            raise RuntimeError("Refresh matching and results because a comparison image is no longer available")
        jobs, filenames = {}, {}
        for track_id in required:
            track = tracks[track_id]
            source_job = jobs.setdefault(track["experiment_id"], self.get("job", track["experiment_id"]))
            clips = {clip["id"]: clip for clip in source_job.get("config", {}).get("clips", [])}
            filenames[track_id] = clips.get(track.get("clip_id"), {}).get("filename") or track_id + ".jpg"

        encoder_order = {key: index for index, key in enumerate(job.get("config", {}).get("encoders", []))}
        query_order = {track["id"]: index for index, track in enumerate(self.tracks(experiment_id))}
        rows.sort(key=lambda row: (query_order.get(row["query_observation_id"], len(query_order)),
                                   encoder_order.get(row["encoder_id"], len(encoder_order)),
                                   row.get("query_crop_index", 0), row.get("cosine_rank", 0),
                                   row["candidate_vehicle_id"], row["reference_observation_id"],
                                   row.get("reference_crop_index", 0)))
        available_encoders = []
        for row in rows:
            if not any(value["id"] == row["encoder_id"] for value in available_encoders):
                available_encoders.append({"id": row["encoder_id"], "name": row.get("encoder_name", row["encoder_id"]),
                                           "fingerprint": row.get("encoder_fingerprint")})
        available_vehicle_ids = sorted({row["candidate_vehicle_id"] for row in rows})
        available_observations = [{"id": track_id, "filename": filenames[track_id]}
                                  for track_id in query_order if track_id in {row["query_observation_id"] for row in rows}]

        flag_key = {"winning": "is_winning_identity", "assigned": "is_assigned_identity",
                    "decision-reference": "is_decision_reference"}.get(decision_flag)
        filtered = [row for row in rows
                    if (observation_id is None or row["query_observation_id"] == observation_id)
                    and (encoder_id is None or row["encoder_id"] == encoder_id)
                    and (candidate_vehicle_id is None or row["candidate_vehicle_id"] == candidate_vehicle_id)
                    and (flag_key is None or row.get(flag_key) is True)]
        total = len(filtered)
        items = []
        for row in filtered[offset:offset + limit]:
            query_id, reference_id = row["query_observation_id"], row["reference_observation_id"]
            items.append({**row, "query_filename": filenames[query_id], "reference_filename": filenames[reference_id],
                          "query_image_url": f'/api/reid/tracks/{quote(query_id, safe="")}/crops/{row.get("query_crop_index", 0)}',
                          "reference_image_url": f'/api/reid/tracks/{quote(reference_id, safe="")}/crops/{row.get("reference_crop_index", 0)}'})
        return {"experiment_id": experiment_id, "items": items, "total": total, "offset": offset, "limit": limit,
                "available_encoders": available_encoders, "available_vehicle_ids": available_vehicle_ids,
                "available_observations": available_observations}

    def next_identity(self, site_id):
        with self.lock:
            key = "reid_identity_counter:" + site_id
            used = [int(i["global_id"].split("_")[1]) for i in self.all("identity") if i["site_id"] == site_id]
            number = max([self.db.get_state(key, 0), *used]) + 1
            self.db.set_state(key, number)
            return f"VEHICLE_{number:05d}"

    def deletion_preview(self, experiment_id):
        from .reid_delete import build
        with self.lock:
            return build(self, experiment_id)["preview"]

    def delete_experiment(self, experiment_id):
        from .reid_delete import delete
        return delete(self, experiment_id)

    def color_features(self, rows):
        return ColorCache(self.root / "colors").tracks(rows)

    def match_calibration(self, site_id, scoring):
        record = self.db.get("reid_calibration", f'{site_id}_{scoring["fingerprint"]}')
        if record is None and scoring["color_weight"] == 0:
            record = self.db.get("reid_calibration", f'{site_id}_{scoring["encoder_fingerprint"]}')
        return record

    @serialized_mutation
    def create_dataset(self, value):
        self.get("site", value.site_id)
        rows = []
        for experiment_id in dict.fromkeys(value.experiment_ids):
            job = self.editable(experiment_id)
            if job["site_id"] != value.site_id:
                raise ValueError("Dataset experiments must belong to one site")
            rows.extend(t for t in self.tracks(experiment_id) if t.get("identity") and not t.get("excluded") and t["crops"])
        splits = partition(rows)
        crop_labels = {}
        for row in rows:
            for crop in row["crops"]:
                digest = sha256_file(Path(crop["path"]))
                if digest in crop_labels and crop_labels[digest] != row["identity"]:
                    raise ValueError("Identical crops have conflicting identity labels; correct these before creating a snapshot")
                crop_labels[digest] = row["identity"]
        # Retain immutable copied crops so later review/split actions cannot alter the snapshot.
        key = uuid4().hex
        directory = self.root / "datasets" / key
        for row in rows:
            for crop in row["crops"]:
                dest = directory / "crops" / row["id"] / Path(crop["path"]).name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(crop["path"], dest)
                crop["path"] = str(dest)
                crop["sha256"] = sha256_file(dest)
        value = {"id": key, "site_id": value.site_id, "experiment_ids": value.experiment_ids, "rows": rows, "splits": splits, "seed": 42}
        value["fingerprint"] = fingerprint({"splits": splits, "rows": [{**r, "crops": [{k: v for k, v in c.items() if k != "path"} for c in r["crops"]]} for r in rows]})
        self.put("dataset", value)
        atomic_json(directory / "dataset.json", value)
        return value

    @serialized_mutation
    def evaluate(self, value):
        dataset = self.get("dataset", value.dataset_id)
        spec = self.encoder(value.encoder_id)
        if spec.get("dataset_fingerprint") and spec["dataset_fingerprint"] != dataset["fingerprint"]:
            raise ValueError("Evaluate a tuned encoder on the frozen snapshot used for its training")
        settings = self.get("site", dataset["site_id"]).get("matching") or {}
        scoring = scoring_spec(spec["fingerprint"], settings.get("color_weight", .25))
        return self.submit(Job(kind="evaluation", name=f'{value.split.title()} evaluation', site_id=dataset["site_id"], config={**value.model_dump(), "scoring": scoring}))

    def _evaluate(self, job):
        cfg = job["config"]; dataset = self.get("dataset", cfg["dataset_id"]); spec = self.encoder(cfg["encoder_id"])
        rows = [r for r in dataset["rows"] if r["identity"] in dataset["splits"][cfg["split"]]]
        path = self.root / job["id"] / "features.npz"
        self.run_stage(job, "embed-evaluation", {"task": "embed", "encoder": spec, "tracks": rows, "output": str(path)})
        crops = self.load_vectors(path)
        vectors = {key: track_vector(values) for key, values in crops.items()}
        cross, same = retrieval(rows, vectors), retrieval(rows, vectors, False)
        if not cross["eligible_queries"] and not same["eligible_queries"]:
            raise ValueError("This split needs separate matching tracks and negative identities; cross-camera evaluation needs at least two cameras")
        scoring = cfg.setdefault("scoring", scoring_spec(spec["fingerprint"], (self.get("site", job["site_id"]).get("matching") or {}).get("color_weight", .25)))
        if scoring != scoring_spec(spec["fingerprint"], scoring["color_weight"]):
            raise ValueError("Scoring policy changed. Submit a new evaluation to keep results reproducible.")
        colors = self.color_features(rows)
        def pair_score(a, b):
            return best_pair(crops[a["id"]], crops[b["id"]], colors.get(a["id"], []), colors.get(b["id"], []), scoring["color_weight"])["similarity"]
        combined_cross = retrieval(rows, vectors, pair_score=pair_score, same_class=True)
        combined_same = retrieval(rows, vectors, False, pair_score=pair_score, same_class=True)
        association = self.association_queries(rows, crops, colors, scoring["color_weight"])
        calibration = calibrate(association) if cfg["split"] == "validation" else self.match_calibration(job["site_id"], scoring)
        if cfg["split"] == "validation" and calibration:
            self.put("calibration", {"id": f'{job["site_id"]}_{scoring["fingerprint"]}', "scoring": scoring, "dataset_fingerprint": dataset["fingerprint"], **calibration})
        accepted = [q for q in association if calibration and q["score"] >= calibration["threshold"] and q["margin"] >= calibration["margin"]]
        metrics = {"cross_camera": cross, "same_camera": same, "combined_cross_camera": combined_cross, "combined_same_camera": combined_same,
                   "scoring": scoring, "color_crops_available": sum(c.get("vector") is not None for values in colors.values() for c in values),
                   "color_crops_total": sum(len(values) for values in colors.values()), "calibration": calibration,
                   "false_merges": sum(not q["correct"] for q in accepted), "missed_matches": len(association) - sum(q["correct"] for q in accepted)}
        record = {"id": job["id"], "site_id": job["site_id"], **cfg, "encoder_fingerprint": spec["fingerprint"], "dataset_fingerprint": dataset["fingerprint"], "metrics": metrics}
        self.put("evaluation", record)
        atomic_json(self.root / job["id"] / "evaluation.json", record)
        job["metrics"] = metrics
        job["artifacts"]["evaluation.json"] = self.artifact_url(job["id"], "evaluation.json")

    def association_queries(self, rows, crops, colors=None, color_weight=0):
        from .reid_core import eligible_pair
        queries = []
        colors = colors or {}
        for row in rows:
            if row["id"] not in crops:
                continue
            galleries = {}
            for other in rows:
                if other["id"] in crops and eligible_pair(row, other, True) and row.get("class_name", "truck") == other.get("class_name", "truck"):
                    for index, vector in enumerate(crops[other["id"]]):
                        values = colors.get(other["id"], [])
                        galleries.setdefault(other["identity"], []).append((vector, values[index] if index < len(values) else {}))
            scores = {identity: best_pair(crops[row["id"]], [v for v, _ in values[-8:]], colors.get(row["id"], []), [c for _, c in values[-8:]], color_weight)["similarity"] for identity, values in galleries.items()}
            if row["identity"] not in scores or len(scores) < 2:
                continue
            ranked = sorted(scores.items(), key=lambda p: (-p[1], p[0]))
            queries.append({"correct": ranked[0][0] == row["identity"], "score": ranked[0][1], "margin": ranked[0][1] - ranked[1][1]})
        return queries

    @serialized_mutation
    def create_benchmark(self, value: BenchmarkInput, items, job_id):
        site = self.get("site", value.site_id)
        encoders = list(dict.fromkeys(value.encoders))
        for key in encoders:
            spec = self.encoder(key)
            if not spec.get("available"):
                raise ValueError(spec.get("availability_reason") or f'{spec["name"]} is not installed')
        threshold = value.threshold if value.threshold is not None else (site.get("matching") or {}).get("threshold", .85)
        job = Job(id=job_id, kind="benchmark", name=value.name, site_id=value.site_id,
                  config={"encoders": encoders, "threshold": threshold, "items": items})
        self.put("benchmark", {"id": job_id, "job_id": job_id, "name": value.name, "site_id": value.site_id,
                               "threshold": threshold, "requested_encoders": encoders, "counts": {
                                   "identities": len({item["identity"] for item in items}), "images": len(items),
                                   "source_groups": len({(item["identity"], item["source_group"]) for item in items})},
                               "created_at": job.created_at.isoformat(), "updated_at": job.created_at.isoformat()})
        return self.submit(job)

    def benchmark_tracks(self, job):
        directory = self.root / job["id"] / "images"
        return [{"id": item["id"], "identity": item["identity"], "source_track_id": item["source_group"],
                 "crops": [{"path": str(directory / item["stored_name"]), "frame": 0, "timestamp": 0, "quality": 1.0}]}
                for item in job["config"]["items"]]

    def rebuild_benchmark(self, job, specs):
        from .reid_benchmark import score_encoder, write_results
        results = []
        for spec in specs:
            vectors = self.load_vectors(self.vector_path(job["id"], spec))
            results.append(score_encoder(job["config"]["items"], vectors, spec, job["config"]["threshold"]))
        record = write_results(self, job, results)
        self.put("benchmark", record)
        job["config"]["package_name"] = record["package_name"]
        job["metrics"]["benchmark"] = [result["summary"] for result in results]
        job["artifacts"].update({"Benchmark summary (.json)": self.artifact_url(job["id"], "summary.json"),
                                 "Benchmark summary (.csv)": self.artifact_url(job["id"], "summary.csv"),
                                 "Benchmark results (.zip)": f'/api/reid/benchmarks/{job["id"]}/results.zip'})
        self.put("job", job)
        return record

    def _benchmark(self, job):
        tracks, specs = self.benchmark_tracks(job), []
        for key in job["config"]["encoders"]:
            spec = self.encoder(key)
            if not spec.get("available"):
                raise RuntimeError(f'{spec["name"]} is not installed. Completed benchmark embeddings are retained.')
            specs.append(spec)
            path = self.vector_path(job["id"], spec)
            self.run_stage(job, "embed-benchmark-" + key + "-" + spec["fingerprint"][:12],
                           {"task": "embed", "encoder": spec, "tracks": tracks, "output": str(path)})
        self.rebuild_benchmark(job, specs)

    def _benchmark_export(self, job):
        benchmark_id = job["config"]["benchmark_id"]
        benchmark_job, package = self.benchmark_package(benchmark_id)
        record = self.get("benchmark", benchmark_id)
        from .reid_benchmark import benchmark_result_fingerprint
        if benchmark_result_fingerprint(record, benchmark_job) != job["config"]["source_result_fingerprint"]:
            raise RuntimeError("Benchmark results changed; prepare a new comparison-image export")
        output = self.root / job["id"] / "benchmark-comparison-images.zip"
        self.run_stage(job, "render-benchmark-comparisons", {
            "task": "benchmark_comparison_export", "benchmark_id": benchmark_id,
            "source_package": str(package), "source_result_fingerprint": job["config"]["source_result_fingerprint"],
            "comparison_count": job["config"]["comparison_count"], "threshold": record["threshold"],
            "encoders": benchmark_job["config"]["encoders"], "items": benchmark_job["config"]["items"],
            "image_root": str(self.root / benchmark_id / "images"), "output": str(output)})
        job["artifacts"]["Model comparison images (.zip)"] = self.artifact_url(job["id"], output.name)
        self.put("job", job)

    def _benchmark_export_status(self, record, benchmark_job):
        from .reid_benchmark import COMPARISON_EXPORT_LIMIT, benchmark_result_fingerprint
        count = int(record.get("counts", {}).get("all_pairs_per_encoder", 0)) * len(record.get("encoders", []))
        value = {"job_id": None, "state": "not_prepared", "progress": 0., "error": None,
                 "comparison_count": count, "limit": COMPARISON_EXPORT_LIMIT, "ready": False,
                 "source_result_fingerprint": benchmark_result_fingerprint(record, benchmark_job)}
        pointer = record.get("comparison_export")
        if not pointer:
            return value
        try:
            export_job = self.get("job", pointer["job_id"])
        except KeyError:
            return value
        stale = pointer.get("source_result_fingerprint") != value["source_result_fingerprint"]
        output = self.root / export_job["id"] / "benchmark-comparison-images.zip"
        value.update(job_id=export_job["id"], state="stale" if stale else export_job["state"],
                     progress=export_job.get("progress", 0), error=export_job.get("error"),
                     ready=not stale and export_job["state"] == "completed" and output.is_file())
        return value

    @serialized_mutation
    def prepare_benchmark_comparison_images(self, benchmark_id):
        benchmark_job, _ = self.benchmark_package(benchmark_id)
        record = self.get("benchmark", benchmark_id)
        from .reid_benchmark import COMPARISON_EXPORT_LIMIT, benchmark_result_fingerprint
        count = int(record.get("counts", {}).get("all_pairs_per_encoder", 0)) * len(record.get("encoders", []))
        if count > COMPARISON_EXPORT_LIMIT:
            raise ValueError(f"Comparison-image exports are limited to {COMPARISON_EXPORT_LIMIT:,} images; use a smaller benchmark folder")
        source_fingerprint = benchmark_result_fingerprint(record, benchmark_job)
        pointer = record.get("comparison_export")
        if pointer and pointer.get("source_result_fingerprint") == source_fingerprint:
            try:
                existing = self.get("job", pointer["job_id"])
                output = self.root / existing["id"] / "benchmark-comparison-images.zip"
                if existing["state"] != "completed" or output.is_file():
                    return existing
            except KeyError:
                pass
        export_job = Job(id=uuid4().hex, kind="benchmark_export", name=f'{record["name"]} comparison images',
                         site_id=record["site_id"], config={"benchmark_id": benchmark_id,
                         "source_result_fingerprint": source_fingerprint, "comparison_count": count})
        record["comparison_export"] = {"job_id": export_job.id, "source_result_fingerprint": source_fingerprint}
        self.put("benchmark", record)
        return self.submit(export_job)

    def benchmark_comparison_images_package(self, benchmark_id):
        benchmark_job, _ = self.benchmark_package(benchmark_id)
        record = self.get("benchmark", benchmark_id)
        status = self._benchmark_export_status(record, benchmark_job)
        if not status["ready"]:
            raise RuntimeError("Prepare the current benchmark comparison images before downloading")
        return benchmark_job, self.artifact(status["job_id"], "benchmark-comparison-images.zip")

    def benchmarks(self):
        records = self.all("benchmark")
        jobs = {job["id"]: job for job in self.all("job") if job.get("kind") == "benchmark"}
        return [{**record, "state": jobs.get(record["id"], {}).get("state", "missing"),
                 "comparison_export": self._benchmark_export_status(record, jobs[record["id"]])}
                if record["id"] in jobs else {**record, "state": "missing"} for record in records]

    def benchmark(self, benchmark_id):
        record = self.get("benchmark", benchmark_id)
        job = self.get("job", benchmark_id)
        return {**record, "state": job["state"], "comparison_export": self._benchmark_export_status(record, job)}

    @serialized_mutation
    def update_benchmark_threshold(self, benchmark_id, threshold):
        job = self.editable(benchmark_id)
        if job.get("kind") != "benchmark":
            raise KeyError(benchmark_id)
        if any(value.get("kind") == "benchmark_export"
               and value.get("config", {}).get("benchmark_id") == benchmark_id
               and value["state"] in ("queued", "running") for value in self.all("job")):
            raise RuntimeError("Cancel or wait for the comparison-image export before changing the threshold")
        record = self.get("benchmark", benchmark_id)
        if not record.get("encoders"):
            raise RuntimeError("Complete or retry this benchmark before changing its threshold")
        job["config"]["threshold"] = threshold
        specs = [{"id": row["encoder_id"], "name": row["encoder_name"], "fingerprint": row["encoder_fingerprint"]}
                 for row in record["encoders"]]
        return self.rebuild_benchmark(job, specs)

    def benchmark_package(self, benchmark_id):
        job = self.get("job", benchmark_id)
        if job.get("kind") != "benchmark":
            raise KeyError(benchmark_id)
        if job.get("deletion_pending"):
            raise RuntimeError("Benchmark deletion is pending")
        if job.get("state") != "completed":
            raise RuntimeError("Benchmark results are available after the job completes")
        try:
            return job, self.artifact(benchmark_id, job["config"].get("package_name", "benchmark-results.zip"))
        except KeyError as exc:
            raise RuntimeError("Retry the benchmark to regenerate its results package") from exc

    def benchmark_comparisons(self, benchmark_id, encoder_id=None, identity=None, pair_type="all",
                              eligibility="all", offset=0, limit=24):
        job, package = self.benchmark_package(benchmark_id)
        try:
            with zipfile.ZipFile(package) as archive:
                rows = json.loads(archive.read("data/pairwise_comparisons.json"))
        except (OSError, KeyError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
            raise RuntimeError("Retry the benchmark to rebuild its pairwise comparison audit") from exc
        items = {item["id"]: item for item in job.get("config", {}).get("items", [])}
        required = {row[key] for row in rows for key in ("left_id", "right_id")}
        if not required.issubset(items):
            raise RuntimeError("Retry the benchmark because a comparison image is no longer available")
        encoder_order = {key: index for index, key in enumerate(job.get("config", {}).get("encoders", []))}
        rows.sort(key=lambda row: (encoder_order.get(row["encoder_id"], len(encoder_order)),
                                   -row["cosine_similarity"], row["left_identity"], row["left_path"],
                                   row["right_identity"], row["right_path"]))
        available_encoders = []
        for row in rows:
            if not any(value["id"] == row["encoder_id"] for value in available_encoders):
                available_encoders.append({"id": row["encoder_id"], "name": row.get("encoder_name", row["encoder_id"]),
                                           "fingerprint": row.get("encoder_fingerprint")})
        available_identities = sorted({row[key] for row in rows for key in ("left_identity", "right_identity")}, key=str.casefold)
        filtered = [row for row in rows
                    if (encoder_id is None or row["encoder_id"] == encoder_id)
                    and (identity is None or identity in (row["left_identity"], row["right_identity"]))
                    and (pair_type == "all" or row["actual_same_identity"] is (pair_type == "same"))
                    and (eligibility == "all" or row["strict_eligible"] is (eligibility == "strict"))]
        total, values = len(filtered), []
        for row in filtered[offset:offset + limit]:
            left, right = items[row["left_id"]], items[row["right_id"]]
            values.append({**row, "left_filename": left["filename"], "right_filename": right["filename"],
                           "left_image_url": f'/api/reid/benchmarks/{quote(benchmark_id, safe="")}/images/{quote(left["id"], safe="")}',
                           "right_image_url": f'/api/reid/benchmarks/{quote(benchmark_id, safe="")}/images/{quote(right["id"], safe="")}'})
        return {"benchmark_id": benchmark_id, "items": values, "total": total, "offset": offset, "limit": limit,
                "available_encoders": available_encoders, "available_identities": available_identities}

    def benchmark_image(self, benchmark_id, item_id):
        job, _ = self.benchmark_package(benchmark_id)
        item = next((value for value in job.get("config", {}).get("items", []) if value["id"] == item_id), None)
        if item is None:
            raise KeyError(item_id)
        base = (self.root / benchmark_id / "images").resolve()
        path = (base / item["stored_name"]).resolve()
        if not path.is_relative_to(base) or not path.is_file():
            raise KeyError(item_id)
        return path

    @serialized_mutation
    def delete_benchmark(self, benchmark_id):
        job = self.get("job", benchmark_id)
        if job.get("kind") != "benchmark":
            raise KeyError(benchmark_id)
        if job["state"] in ("queued", "running"):
            raise RuntimeError("Cancel or wait for the benchmark before deleting it")
        export_jobs = [value for value in self.all("job")
                       if value.get("kind") == "benchmark_export" and value.get("config", {}).get("benchmark_id") == benchmark_id]
        if any(value["state"] in ("queued", "running") for value in export_jobs):
            raise RuntimeError("Cancel or wait for the comparison-image export before deleting this benchmark")
        job["deletion_pending"] = True
        self.put("job", job)
        directory = (self.root / benchmark_id).resolve()
        if not directory.is_relative_to(self.root) or not re.fullmatch(r"[0-9a-f]{32}", benchmark_id):
            raise ValueError("Invalid benchmark ID")
        try:
            shutil.rmtree(directory, ignore_errors=False) if directory.exists() else None
            for export_job in export_jobs:
                export_directory = (self.root / export_job["id"]).resolve()
                if export_directory.is_relative_to(self.root) and re.fullmatch(r"[0-9a-f]{32}", export_job["id"]):
                    shutil.rmtree(export_directory, ignore_errors=False) if export_directory.exists() else None
            self.db.apply_changes([], [("reid_benchmark", benchmark_id), ("reid_job", benchmark_id),
                                       *(("reid_job", value["id"]) for value in export_jobs)])
        except Exception as exc:
            job.update(deletion_error=str(exc))
            self.put("job", job)
            raise RuntimeError("Benchmark cleanup was interrupted; retry deletion") from exc
        return {"deleted": True, "benchmark_id": benchmark_id}

    @serialized_mutation
    def create_pair_comparison(self, value: PairComparisonInput, images, job_id):
        self.get("site", value.site_id)
        spec = self.encoder(value.encoder_id)
        if not spec.get("available"):
            raise ValueError(spec.get("availability_reason") or "Install the selected vision-language encoder before comparing images")
        encoder_only = spec.get("pair_mode", "visual_cosine" if spec["family"].endswith("_visual") else "vlm_fusion") == "visual_cosine"
        if not encoder_only and value.threshold < 0:
            raise ValueError("Full VLM fusion thresholds must be between 0 and 1")
        weights = ({"appearance": 1., "color": 0., "shape": 0., "semantic": 0.}
                   if encoder_only else value.weights.model_dump())
        config = {**value.model_dump(mode="json"), "weights": weights, "images": images}
        record = {"id": job_id, "job_id": job_id, "name": value.name, "site_id": value.site_id,
                  "encoder_id": value.encoder_id, "threshold": value.threshold,
                  "weights": weights, "images": images,
                  "created_at": utc_now().isoformat(), "updated_at": utc_now().isoformat()}
        self.put("pair", record)
        return self.submit(Job(id=job_id, kind="comparison", name=value.name, site_id=value.site_id, config=config))

    def pair_comparisons(self):
        jobs = {job["id"]: job for job in self.all("job") if job.get("kind") == "comparison"}
        return [{**self._public_pair(record), "state": jobs.get(record["id"], {}).get("state", "missing"),
                 "stage": jobs.get(record["id"], {}).get("stage", "missing"),
                 "progress": jobs.get(record["id"], {}).get("progress", 0),
                 "error": jobs.get(record["id"], {}).get("error")}
                for record in self.all("pair")]

    def _public_pair(self, record):
        images = [{"side": item["side"], "filename": item["filename"],
                   "image_url": f'/api/reid/pair-comparisons/{quote(record["id"], safe="")}/images/{item["side"]}'}
                  for item in record.get("images", [])]
        return {**{key: value for key, value in record.items() if key != "images"}, "images": images}

    def pair_comparison(self, comparison_id):
        record = self.get("pair", comparison_id)
        job = self.get("job", comparison_id)
        if job.get("kind") != "comparison":
            raise KeyError(comparison_id)
        return {**self._public_pair(record), "state": job["state"], "stage": job["stage"], "progress": job["progress"],
                "error": job.get("error"), "deletion_pending": job.get("deletion_pending", False)}

    def _comparison(self, job):
        from .reid_pair import PAIR_FEATURE_VERSION
        spec = self.encoder(job["config"]["encoder_id"])
        if not spec.get("available"):
            raise RuntimeError(spec.get("availability_reason") or "The selected vision-language encoder is not installed")
        directory = self.root / job["id"]
        features = directory / f'pair-features-{spec["fingerprint"]}-{PAIR_FEATURE_VERSION}.npz'
        metadata = directory / f'pair-features-{spec["fingerprint"]}-{PAIR_FEATURE_VERSION}.json'
        self.run_stage(job, f'pair-features-{spec["fingerprint"][:12]}-{PAIR_FEATURE_VERSION[:12]}',
                       {"task": "pair_features", "encoder": spec, "images": job["config"]["images"],
                        "output": str(features), "metadata_path": str(metadata)})
        self.rebuild_pair_comparison(job, spec, features, metadata)

    def rebuild_pair_comparison(self, job, spec=None, features=None, metadata=None):
        from .reid_pair import PAIR_FEATURE_VERSION, build_result, public_summary
        spec = spec or self.encoder(job["config"]["encoder_id"])
        directory = self.root / job["id"]
        features = features or directory / f'pair-features-{spec["fingerprint"]}-{PAIR_FEATURE_VERSION}.npz'
        metadata = metadata or directory / f'pair-features-{spec["fingerprint"]}-{PAIR_FEATURE_VERSION}.json'
        arrays = self.load_vectors(features)
        required = ({"appearance"} if spec.get("pair_mode", "visual_cosine" if spec["family"].endswith("_visual") else "vlm_fusion") == "visual_cosine" else
                    {"appearance", "color", "color_available", "shape", "shape_available", "semantic",
                     "texture", "texture_available", "detail", "detail_available", "edge_scale"})
        if set(arrays) != required or not metadata.is_file():
            raise RuntimeError("Retry this comparison to rebuild its feature cache")
        details = json.loads(metadata.read_text(encoding="utf-8"))
        if details.get("pair_feature_version") != PAIR_FEATURE_VERSION:
            raise RuntimeError("Retry this comparison because its feature definition changed")
        comparison = {"id": job["id"], "name": job["name"], "site_id": job["site_id"],
                      "images": job["config"]["images"], "threshold": job["config"]["threshold"],
                      "weights": job["config"]["weights"]}
        result = build_result(comparison, spec, arrays, details)
        atomic_json(directory / "comparison-results.json", result)
        record = self.get("pair", job["id"])
        record.update(threshold=comparison["threshold"], weights=comparison["weights"],
                      result=public_summary(result), updated_at=utc_now().isoformat())
        self.put("pair", record)
        job["metrics"]["comparison"] = public_summary(result)
        job["artifacts"]["Comparison vectors and scores (.json)"] = self.artifact_url(job["id"], "comparison-results.json")
        self.put("job", job)
        return {**self._public_pair(record), "state": job["state"], "stage": job["stage"], "progress": job["progress"]}

    @serialized_mutation
    def update_pair_scoring(self, comparison_id, value):
        job = self.editable(comparison_id)
        if job.get("kind") != "comparison" or job.get("state") != "completed":
            raise RuntimeError("Complete or retry this comparison before changing its scoring")
        spec = self.encoder(job["config"]["encoder_id"])
        encoder_only = spec.get("pair_mode", "visual_cosine" if spec["family"].endswith("_visual") else "vlm_fusion") == "visual_cosine"
        if not encoder_only and value.threshold < 0:
            raise ValueError("Full VLM fusion thresholds must be between 0 and 1")
        job["config"]["threshold"] = value.threshold
        job["config"]["weights"] = ({"appearance": 1., "color": 0., "shape": 0., "semantic": 0.}
                                     if encoder_only else value.weights.model_dump())
        self.put("job", job)
        return self.rebuild_pair_comparison(job)

    def pair_result(self, comparison_id):
        job = self.get("job", comparison_id)
        if job.get("kind") != "comparison":
            raise KeyError(comparison_id)
        if job.get("deletion_pending"):
            raise RuntimeError("Comparison deletion is pending")
        if job.get("state") != "completed":
            raise RuntimeError("Comparison results are available after the job completes")
        path = self.root / comparison_id / "comparison-results.json"
        if not path.is_file():
            raise RuntimeError("Retry this comparison to regenerate its results")
        return job, path

    def pair_image(self, comparison_id, side):
        if side not in ("first", "second"):
            raise KeyError(side)
        record = self.get("pair", comparison_id)
        item = next((image for image in record["images"] if image["side"] == side), None)
        if item is None:
            raise KeyError(side)
        base = (self.root / comparison_id / "images").resolve()
        path = (base / item["stored_name"]).resolve()
        if not path.is_relative_to(base) or not path.is_file():
            raise KeyError(side)
        return path

    @serialized_mutation
    def delete_pair_comparison(self, comparison_id):
        try:
            job = self.get("job", comparison_id)
        except KeyError:
            return {"deleted": True, "comparison_id": comparison_id}
        if job.get("kind") != "comparison":
            raise KeyError(comparison_id)
        if job["state"] in ("queued", "running"):
            raise RuntimeError("Cancel or wait for the comparison before deleting it")
        job["deletion_pending"] = True
        self.put("job", job)
        directory = (self.root / comparison_id).resolve()
        if not directory.is_relative_to(self.root) or not re.fullmatch(r"[0-9a-f]{32}", comparison_id):
            raise ValueError("Invalid comparison ID")
        try:
            shutil.rmtree(directory, ignore_errors=False) if directory.exists() else None
            self.db.apply_changes([], [("reid_pair", comparison_id), ("reid_job", comparison_id)])
        except Exception as exc:
            job.update(deletion_error=str(exc))
            self.put("job", job)
            raise RuntimeError("Comparison cleanup was interrupted; retry deletion") from exc
        return {"deleted": True, "comparison_id": comparison_id}

    @serialized_mutation
    def train(self, value: TrainingInput):
        self.require_training(value.encoder_id)
        dataset = self.get("dataset", value.dataset_id)
        self.encoder(value.encoder_id)
        return self.submit(Job(kind="training", name=f'Fine-tune {value.encoder_id}', site_id=dataset["site_id"], config=value.model_dump()))

    def _training(self, job):
        cfg = job["config"]; dataset = self.get("dataset", cfg["dataset_id"]); spec = self.encoder(cfg["encoder_id"])
        output = self.root / job["id"]
        self.run_stage(job, "train", {"task": "train", "training": cfg, "dataset": dataset, "encoder": spec, "output": str(output)})
        model = {**spec, "id": "trained-" + job["id"], "name": f'{spec["name"]} · site tuned', "origin": "trained", "checkpoint": str(output / "model.safetensors"), "dataset_fingerprint": dataset["fingerprint"]}
        model["checksum"] = sha256_file(Path(model["checkpoint"]))
        model["fingerprint"] = fingerprint([spec["fingerprint"], model["checksum"]])
        self.put("encoder", model)
        atomic_json(output / "encoder.json", model)
        for name in ("model.safetensors", "history.json", "training.json", "encoder.json"):
            job["artifacts"][name] = self.artifact_url(job["id"], name)
        self.export_model(model["id"])
        job["artifacts"]["model package"] = self.artifact_url(job["id"], "model.zip")

    @serialized_mutation
    def promote(self, site_id, key):
        self.get("site", site_id); spec = self.encoder(key)
        if spec.get("origin") != "pretrained" and not any(e["site_id"] == site_id and e["encoder_fingerprint"] == spec["fingerprint"] and e["dataset_fingerprint"] == spec.get("dataset_fingerprint") and e["split"] == "test" for e in self.all("evaluation")):
            raise ValueError("Evaluate this tuned encoder on its held-out test split before promotion")
        return self.submit(Job(kind="promotion", name=f'Use {spec["name"]} for site', site_id=site_id, config={"encoder_id": key}))

    def _promote(self, job):
        spec = self.encoder(job["config"]["encoder_id"])
        tracks = [t for t in self.tracks(site_id=job["site_id"]) if (t.get("reviewed") or t.get("gallery_reference")) and t.get("global_id")]
        path = self.root / job["id"] / "gallery-tracks.npz"
        self.run_stage(job, "embed-site-gallery", {"task": "embed", "encoder": spec, "tracks": tracks, "output": str(path)})
        values = self.load_vectors(path)
        from .reid_worker import save_vectors
        # Cache by track, not global ID: later review corrections must not keep
        # obsolete crop-to-identity links alive in the promoted gallery.
        save_vectors(self.root / "sites" / job["site_id"] / f'{spec["fingerprint"]}.npz', values)
        with self.lock:
            site = self.get("site", job["site_id"]); site["active_encoder"] = spec["id"]; self.put("site", site)

    @serialized_mutation
    def export_training(self, value):
        self.require_training(value.encoder_id)
        dataset = json.loads(json.dumps(self.get("dataset", value.dataset_id)))
        spec = self.encoder(value.encoder_id)
        directory = self.root / "exports" / uuid4().hex
        directory.mkdir(parents=True)
        path = directory / "training.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for row in dataset["rows"]:
                for crop in row["crops"]:
                    name = f'crops/{row["id"]}/{Path(crop["path"]).name}'
                    archive.write(crop["path"], name); crop["path"] = name
            if spec.get("checkpoint"):
                archive.write(spec["checkpoint"], "base.safetensors"); spec["checkpoint"] = "base.safetensors"
            config = {"task": "train", "dataset": dataset, "encoder": spec, "training": value.model_dump(), "model_root": "models", "output": "output", "progress_path": "progress.json"}
            archive.writestr("training.json", json.dumps(config, indent=2))
            for name in ("__init__.py", "reid_worker.py", "reid_encoders.py", "reid_core.py", "reid_vehicle_encoders.py", "reid_extra_setup.py"):
                archive.write(Path(__file__).parent / name, "iris/" + name)
            project = Path(__file__).resolve().parents[2]
            archive.write(project / "requirements-reid.txt", "requirements-reid.txt")
            archive.write(Path(__file__).parent / "reid_setup.py", "iris/reid_setup.py")
            archive.writestr("README.txt", "Use Python 3.10. Install a suitable PyTorch/torchvision pair, then pip install -r requirements-reid.txt.\nRun python -m iris.reid_setup --models models\nFrom this extracted directory run python -m iris.reid_worker training.json\nPackage output with: python -m iris.reid_setup --package output --config training.json\nImport output/model.zip in IRIS. Training does not upload footage to any service.\n")
        return path

    def require_training(self, encoder_id):
        spec = self.encoder(encoder_id)
        if not ENCODER_METADATA.get(spec["family"], {}).get("supports_training", False):
            raise ValueError(f'{spec["name"]} supports inference and evaluation only; fine-tuning and training bundles are unavailable')

    def export_model(self, encoder_id):
        spec = self.encoder(encoder_id)
        if not spec.get("checkpoint"):
            raise ValueError("Pretrained baselines are installed through the download script")
        path = Path(spec["checkpoint"]).parent / "model.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(spec["checkpoint"], "model.safetensors")
            archive.writestr("encoder.json", json.dumps({**spec, "checkpoint": "model.safetensors"}))
        return path

    @serialized_mutation
    def import_model(self, path):
        target = self.root / "imported" / uuid4().hex
        target.mkdir(parents=True)
        with zipfile.ZipFile(path) as archive:
            if sorted(archive.namelist()) != ["encoder.json", "model.safetensors"]:
                raise ValueError("Model package must contain only encoder.json and model.safetensors")
            if sum(i.file_size for i in archive.infolist()) > 2 * 1024**3:
                raise ValueError("Model package exceeds the 2 GB extracted limit")
            spec = json.loads(archive.read("encoder.json"))
            if not ENCODER_METADATA.get(spec.get("family"), {}).get("supports_training") or not spec.get("dataset_fingerprint"):
                raise ValueError("Model package requires a supported family and training dataset fingerprint")
            with archive.open("model.safetensors") as src, (target / "model.safetensors").open("wb") as dest:
                shutil.copyfileobj(src, dest)
        checksum = sha256_file(target / "model.safetensors")
        if checksum != spec.get("checksum"):
            raise ValueError("Model checksum does not match its manifest")
        baseline = self.encoder(spec["family"])
        spec = {**baseline, "id": "imported-" + target.name, "name": str(spec.get("name", "Imported encoder"))[:100], "origin": "imported", "checkpoint": str(target / "model.safetensors"), "checksum": checksum, "dataset_fingerprint": spec["dataset_fingerprint"], "fingerprint": fingerprint([baseline.get("fingerprint"), checksum])}
        return self.put("encoder", spec)

    def artifact_url(self, job_id, name):
        from urllib.parse import quote
        return f"/api/reid/jobs/{job_id}/artifacts/{quote(name)}"

    def artifact(self, job_id, name):
        job = self.get("job", job_id)
        if job.get("deletion_pending"):
            raise RuntimeError("Deletion cleanup is pending; retry Delete experiment")
        base = self.root / job_id
        path = (base / name).resolve()
        if not path.is_relative_to(base) or not path.is_file() or path.suffix not in (".json", ".csv", ".jpg", ".log", ".safetensors", ".zip"):
            raise KeyError(name)
        # Worker configs carry filesystem locations and aren't result artifacts.
        if path.name in ("worker.json", "progress.json"):
            raise KeyError(name)
        return path

    def close(self):
        self.closed.set()
        for job in self.all("job"):
            if job["state"] in ("running", "queued"):
                self.events.setdefault(job["id"], threading.Event()).set()
        if self.process and self.process.poll() is None:
            self.process.kill()
        self.queue.put(None)
        self.thread.join(timeout=30)
