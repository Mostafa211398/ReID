"""Journaled permanent experiment deletion, coordinated by ReIDService.lock."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .utils import sha256_file
from .reid_core import atomic_replace


def write_journal(path, plan):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(plan, handle, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    atomic_replace(temporary, path)


def managed(service, path):
    path = Path(path).resolve()
    if path == service.root or not path.is_relative_to(service.root):
        raise ValueError("Deletion refused: a file is outside ReID-managed storage")
    return path


def journals(service):
    return list((service.root / ".deletions").glob("*.json"))


def references(value, keys):
    if isinstance(value, str):
        return value in keys
    if isinstance(value, dict):
        return any(references(v, keys) for v in value.values())
    if isinstance(value, list):
        return any(references(v, keys) for v in value)
    return False


def build(service, experiment_id):
    job = service.get("job", experiment_id)
    if job["kind"] != "experiment":
        raise ValueError("Only experiments can be deleted here")
    if not re.fullmatch(r"[0-9a-f]{32}", experiment_id):
        raise ValueError("Invalid experiment storage identifier")
    folder = managed(service, service.root / experiment_id)
    tracks = service.all("track")
    removed = [t for t in tracks if t["experiment_id"] == experiment_id]
    remaining = [t for t in tracks if t["experiment_id"] != experiment_id]
    removed_ids = {t["id"] for t in removed}
    datasets = service.all("dataset")
    dependencies = [d for d in datasets if experiment_id in d.get("experiment_ids", []) or any(r.get("experiment_id") == experiment_id or r["id"] in removed_ids for r in d.get("rows", []))]
    fingerprints = {d.get("fingerprint") for d in dependencies} - {None}
    blockers = [{"kind": "dataset", "id": d["id"], "name": d.get("name") or "Dataset " + d["id"][:8]} for d in dependencies]
    blockers.extend({"kind": "model", "id": e["id"], "name": e.get("name", e["id"])} for e in service.all("encoder") if e.get("dataset_fingerprint") in fingerprints)
    jobs = service.all("job")
    dependency_ids = {d["id"] for d in dependencies}
    blockers.extend({"kind": j["kind"], "id": j["id"], "name": j["name"]} for j in jobs if j.get("config", {}).get("dataset_id") in dependency_ids)
    if job["state"] not in ("completed", "failed", "cancelled") or service.queue.unfinished_tasks or service.active_id:
        blockers.append({"kind": "processing", "id": experiment_id, "name": "Wait for ReID processing to finish, or cancel jobs and wait for their workers to stop"})
    other_journals = [p for p in journals(service) if p.stem != experiment_id]
    if other_journals:
        blockers.append({"kind": "cleanup", "id": other_journals[0].stem, "name": "Finish the pending experiment deletion first"})
    identities = [i for i in service.all("identity") if i["site_id"] == job["site_id"]]
    touched = {t.get("global_id") for t in removed} - {None}
    touched.update(i["global_id"] for i in identities if i.get("enrollment_track_id") in removed_ids)
    supported = {t.get("global_id") for t in remaining if t["site_id"] == job["site_id"] and (t.get("reviewed") or t.get("gallery_reference"))}
    orphaned = touched - supported
    upserts, deletes = [], [("reid_job", experiment_id)] + [("reid_track", t["id"]) for t in removed]
    for identity in identities:
        if identity["global_id"] in orphaned:
            deletes.append(("reid_identity", identity["id"]))
        elif identity.get("enrollment_track_id") in removed_ids:
            identity.pop("enrollment_track_id")
            upserts.append(("reid_identity", identity))
    affected = set()
    for track in remaining:
        if track["site_id"] != job["site_id"]:
            continue
        assignments = track.get("assignments", {})
        unsupported = track.get("global_id") in orphaned
        chosen_removed = any(a.get("status") == "automatic" and a.get("reference_track_id") in removed_ids for a in assignments.values())
        if unsupported or references(assignments, removed_ids | orphaned):
            if (unsupported or chosen_removed) and not (track.get("reviewed") or track.get("gallery_reference")):
                track["global_id"] = None
            status = "reviewed" if track.get("reviewed") else "new" if track.get("gallery_reference") else "review"
            track["assignments"] = {key: {"global_id": track.get("global_id"), "status": status, "similarity": None, "candidates": [], "neighbors": [], "reason": "Reference removed; refresh matching required"} for key in assignments}
            affected.add(track["experiment_id"])
            upserts.append(("reid_track", track))
    files, prune = set(), set()
    protected_paths = set()
    for other in jobs:
        if other["id"] != experiment_id:
            protected_paths.update(Path(c["path"]).resolve() for c in other.get("config", {}).get("clips", []) if c.get("path"))
    protected_rows = remaining + [r for d in datasets for r in d.get("rows", [])]
    protected_paths.update(Path(c["path"]).resolve() for t in protected_rows for c in t.get("crops", []) if c.get("path"))
    if folder.exists():
        files.update(managed(service, p) for p in folder.rglob("*") if p.is_file())
    for clip in job.get("config", {}).get("clips", []):
        if clip.get("path"):
            files.add(managed(service, clip["path"]))
    for track in removed:
        for crop in track.get("crops", []):
            files.add(managed(service, crop["path"]))
    files -= protected_paths
    for other in jobs:
        if other["id"] in affected:
            base = managed(service, service.root / other["id"])
            # Result files can embed reference IDs, paths, or thumbnails.
            files.update(managed(service, base / name) for name in ("results.json", "results.csv", "complete-results.zip", "complete-results.zip.partial", "worker.json", "results.json.partial", "results.csv.partial", "worker.json.partial"))
            for clip in other.get("config", {}).get("clips", []):
                for encoder in other["config"].get("encoders", []):
                    files.update(managed(service, base / clip["id"] / (encoder + ext)) for ext in (".mp4", ".jpg", ".partial.mp4", ".partial.jpg"))
            other.update(artifacts={}, refresh_required=True)
            other["completed_stages"] = [s for s in other.get("completed_stages", []) if not s.startswith(("render-", "match-"))]
            upserts.append(("reid_job", other))
        elif other["kind"] == "promotion" and other["site_id"] == job["site_id"]:
            base = managed(service, service.root / other["id"])
            archive = base / "gallery-tracks.npz"
            config_path = base / "worker.json"
            config_uses_tracks = config_path.exists() and references(json.loads(config_path.read_text()), removed_ids)
            if removed_ids.intersection(service.load_vectors(archive)) or config_uses_tracks:
                prune.add(managed(service, archive))
                files.update(managed(service, base / name) for name in ("worker.json", "worker.json.partial", "progress.json", "progress.json.partial", "embed-site-gallery.log", "gallery-tracks.partial"))
                other["completed_stages"] = []
                other["artifacts"] = {}
                upserts.append(("reid_job", other))
    site_folder = managed(service, service.root / "sites" / job["site_id"])
    for path in site_folder.glob("*.npz"):
        path = managed(service, path)
        if removed_ids.intersection(service.load_vectors(path)):
            prune.add(path)
            files.add(managed(service, path.with_suffix(".partial")))
    # Color cache entries are shared across encoders, experiments and snapshots.
    removed_hashes = {sha256_file(Path(c["path"])) for t in removed for c in t.get("crops", []) if Path(c["path"]).is_file()}
    protected_hashes = {sha256_file(Path(c["path"])) for t in protected_rows for c in t.get("crops", []) if Path(c["path"]).is_file()}
    for digest in removed_hashes - protected_hashes:
        files.update(managed(service, p) for p in (service.root / "colors").glob("*/" + digest + ".json"))
    files -= protected_paths
    counter_key = "reid_identity_counter:" + job["site_id"]
    high_water = max([service.db.get_state(counter_key, 0), *[int(i["global_id"].split("_")[1]) for i in identities]])
    preview = {"experiment_id": experiment_id, "name": job["name"], "observations": len(removed),
               "files": sum(p.is_file() for p in files), "identities_removed": sorted(orphaned),
               "identities_preserved": sorted(touched - orphaned), "experiments_to_refresh": sorted(affected), "blockers": blockers}
    plan = {"experiment_id": experiment_id, "preview": preview, "files": sorted(str(p.relative_to(service.root)) for p in files),
            "prune": sorted(str(p.relative_to(service.root)) for p in prune), "removed_tracks": sorted(removed_ids),
            "upserts": upserts, "deletes": deletes, "states": [(counter_key, high_water)]}
    return plan


def execute(service, plan, journal):
    # If the DB commit succeeded before a crash, only the journal remains to remove.
    if service.db.get("reid_job", plan["experiment_id"]) is None:
        journal.unlink(missing_ok=True)
        return
    from .reid_worker import save_vectors
    # Validate every target again before any file operation, including on restart.
    for name in [*plan["files"], *plan["prune"]]:
        managed(service, service.root / name)
    parents = {managed(service, service.root / plan["experiment_id"])}
    for name in plan["files"]:
        path = managed(service, service.root / name)
        path.unlink(missing_ok=True)
        parents.add(path.parent)
    for name in plan["prune"]:
        path = managed(service, service.root / name)
        if path.exists():
            values = service.load_vectors(path)
            values = {key: value for key, value in values.items() if key not in plan["removed_tracks"]}
            save_vectors(path, values)
    # Remove empty owned folders, without recursively removing shared content.
    for parent in sorted(parents, key=lambda p: len(p.parts), reverse=True):
        while parent != service.root:
            managed(service, parent)
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
    service.db.apply_changes(plan["upserts"], plan["deletes"], plan["states"])
    journal.unlink(missing_ok=True)
    service.events.pop(plan["experiment_id"], None)


def recover(service):
    for journal in journals(service):
        try:
            plan = json.loads(journal.read_text())
            execute(service, plan, journal)
        except Exception as exc:
            job = service.db.get("reid_job", journal.stem)
            if job:
                job.update(deletion_pending=True, deletion_error=f"Cleanup incomplete: {exc}. Retry Delete experiment.")
                service.put("job", job)


def delete(service, experiment_id):
    with service.lock:
        if not re.fullmatch(r"[0-9a-f]{32}", experiment_id):
            raise ValueError("Invalid experiment storage identifier")
        if service.queue.unfinished_tasks or service.active_id:
            raise RuntimeError("Wait for ReID jobs to finish or cancel them and wait for their workers to stop")
        journal = managed(service, service.root / ".deletions" / (experiment_id + ".json"))
        if journal.exists():
            plan = json.loads(journal.read_text())
        else:
            plan = build(service, experiment_id)
            if plan["preview"]["blockers"]:
                raise RuntimeError("Deletion blocked: " + "; ".join(b["name"] for b in plan["preview"]["blockers"]))
            write_journal(journal, plan)
        try:
            job = service.db.get("reid_job", experiment_id)
            if job is not None:
                job.update(deletion_pending=True, deletion_error=None)
                service.put("job", job)
            execute(service, plan, journal)
        except Exception as exc:
            job = service.db.get("reid_job", experiment_id)
            if job:
                job.update(deletion_pending=True, deletion_error=f"Cleanup incomplete: {exc}. Retry Delete experiment.")
                service.put("job", job)
            raise RuntimeError("Deletion cleanup is incomplete. Retry Delete experiment; processing is paused until cleanup finishes.") from exc
        return {"deleted": True, "experiment_id": experiment_id}
