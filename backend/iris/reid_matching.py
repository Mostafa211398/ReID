"""Persistent, provenance-aware association for cropped vehicle observations."""
from .reid_core import candidate_context, candidates, simultaneous, unit
from .utils import utc_now
from .reid_color import best_pair, scoring_spec


def match(service, job, spec, vectors):
    site = service.get("site", job["site_id"])
    settings = site.get("matching") or {"mode": "review"}
    weight = settings.get("color_weight", .25)
    scoring = scoring_spec(spec["fingerprint"], weight)
    automatic = settings["mode"] == "automatic"
    active = spec["id"] == site["active_encoder"]
    history = service.tracks(site_id=site["id"])
    colors = service.color_features(history)
    cached, references, sightings = {}, {}, {}
    promoted = service.load_vectors(service.root / "sites" / site["id"] / f'{spec["fingerprint"]}.npz')
    for track in history:
        gid = track.get("global_id")
        trusted = track.get("reviewed") or track.get("gallery_reference")
        if gid and (track["experiment_id"] != job["id"] or trusted):
            sightings.setdefault(gid, []).append(track)
        if not gid or not trusted:
            continue
        experiment = track["experiment_id"]
        if experiment not in cached:
            cached[experiment] = service.load_vectors(service.vector_path(experiment, spec))
        values = cached[experiment].get(track["id"], promoted.get(track["id"]))
        if values is not None:
            references.setdefault(gid, []).extend((track, vector, index) for index, vector in enumerate(values))
    references = {gid: refs[-8:] for gid, refs in references.items()}
    calibration = service.match_calibration(site["id"], scoring)
    current = service.tracks(job["id"])
    comparisons = []
    clip_order = {clip["id"]: index for index, clip in enumerate(job["config"].get("clips", []))}
    for track in sorted(current, key=lambda t: (clip_order.get(t["clip_id"], 0), t["start"], t["local_id"])):
        service.check(job["id"])
        def eligible_reference(reference):
            if reference["id"] == track["id"] or reference.get("source_track_id", reference["id"]) == track.get("source_track_id", track["id"]):
                return False
            if reference.get("class_name", "truck") != track.get("class_name", "truck"):
                return False
            return True
        eligible = {gid: [(r, v, index) for r, v, index in refs if eligible_reference(r)] for gid, refs in references.items()}
        eligible = {gid: refs for gid, refs in eligible.items() if refs and not any(s["id"] != track["id"] and simultaneous(track, s) for s in sightings.get(gid, []))}
        eligible_contexts = {gid: candidate_context(track, sightings.get(gid, []), site["transitions"]) for gid in eligible}
        galleries = {gid: [v for _, v, _ in refs] for gid, refs in eligible.items() if refs}
        def score_gallery(gid):
            refs = eligible[gid]
            reference_colors = [colors.get(r["id"], [])[index] if index < len(colors.get(r["id"], [])) else {} for r, _, index in refs]
            result = best_pair(vectors[track["id"]], galleries[gid], colors.get(track["id"], []), reference_colors, weight)
            reference, _, index = refs[result["reference_crop_index"]]
            result.update(reference_track_id=reference["id"], reference_crop_index=index, scoring_version=scoring["fingerprint"])
            return result
        options = candidates(track, vectors[track["id"]], galleries, sightings, site["transitions"], score_gallery) if track["id"] in vectors else []
        assignment = {"global_id": None, "status": "review", "candidates": options, "similarity": options[0]["similarity"] if options else None,
                      "encoder_fingerprint": spec["fingerprint"], "provenance": "calibrated" if calibration else "experimental", "reference_track_id": None}
        assignment.update(scoring_version=scoring["fingerprint"], color_weight=weight, color_version=scoring["color_version"])
        if options:
            assignment.update({k: v for k, v in options[0].items() if k not in ("global_id", "context")})
            assignment["comparison_global_id"] = options[0]["global_id"]
        if track["id"] not in vectors:
            assignment["reason"] = "Vehicle crop is too small to embed"
        neighbors = []
        if track["id"] in vectors:
            for other in current:
                if other["id"] == track["id"] or other["id"] not in vectors or simultaneous(track, other) or other.get("class_name", "truck") != track.get("class_name", "truck"):
                    continue
                neighbors.append({"track_id": other["id"], **best_pair(vectors[track["id"]], vectors[other["id"]], colors.get(track["id"], []), colors.get(other["id"], []), weight), "scoring_version": scoring["fingerprint"]})
        assignment["neighbors"] = sorted(neighbors, key=lambda n: (-n["similarity"], n["track_id"]))[:5]
        policy = calibration or settings
        assignment.update(matching_mode=settings.get("mode", "review"), decision_threshold=policy.get("threshold", .85),
                          decision_margin=policy.get("margin", .05), new_vehicle_threshold=settings.get("new_threshold", .65))
        if track.get("reviewed"):
            assignment.update(global_id=track["global_id"], status="reviewed", provenance="human")
        elif track.get("gallery_reference") and track.get("global_id"):
            assignment.update(global_id=track["global_id"], status="new", provenance="enrollment")
        elif active and track["id"] in vectors:
            threshold, margin = policy.get("threshold", .85), policy.get("margin", .05)
            allowed = automatic or (calibration and len(options) >= 2)
            if allowed and options and options[0]["similarity"] >= threshold and (len(options) == 1 or options[0]["similarity"] - options[1]["similarity"] >= margin):
                assignment.update(global_id=options[0]["global_id"], status="automatic", reference_track_id=options[0]["reference_track_id"])
            elif automatic and (not options or options[0]["similarity"] < min(settings.get("new_threshold", .65), threshold)):
                # Recover a crash between identity creation and observation persistence.
                identities = [i for i in service.all("identity") if i["site_id"] == site["id"]]
                identity = next((i for i in identities if i.get("enrollment_track_id") == track["id"]), None)
                if identity is None:
                    gid = service.next_identity(site["id"])
                    identity = service.put("identity", {"id": f'{site["id"]}_{gid}', "site_id": site["id"], "global_id": gid,
                                                       "class_name": track.get("class_name", "truck"), "enrollment_track_id": track["id"], "created_at": utc_now().isoformat()})
                assignment.update(global_id=identity["global_id"], status="new", provenance="enrollment")
                track["gallery_reference"] = True
                references[identity["global_id"]] = [(track, vector, index) for index, vector in enumerate(vectors[track["id"]])][-8:]
            track["global_id"] = assignment["global_id"]
        if active and assignment["global_id"]:
            sightings.setdefault(assignment["global_id"], []).append(track)
        track["assignments"][spec["id"]] = assignment
        service.put("track", track)
        if track["id"] in vectors:
            query_colors = colors.get(track["id"], [])
            rows = []
            for global_id, refs in eligible.items():
                context = eligible_contexts[global_id]
                for query_index, query_vector in enumerate(vectors[track["id"]]):
                    query_color = query_colors[query_index] if query_index < len(query_colors) else {}
                    for reference, reference_vector, reference_index in refs:
                        reference_colors = colors.get(reference["id"], [])
                        reference_color = reference_colors[reference_index] if reference_index < len(reference_colors) else {}
                        pair = best_pair([query_vector], [reference_vector], [query_color], [reference_color], weight)
                        rows.append({"experiment_id": job["id"], "query_observation_id": track["id"], "query_crop_index": query_index,
                                     "reference_observation_id": reference["id"], "reference_crop_index": reference_index,
                                     "candidate_vehicle_id": global_id, "encoder_id": spec["id"],
                                     "encoder_name": spec.get("name", spec["id"]),
                                     "encoder_fingerprint": spec["fingerprint"], "cosine_similarity": float(unit(query_vector) @ unit(reference_vector)),
                                     "color_similarity": pair.get("color_similarity"), "combined_score": pair["similarity"],
                                     "effective_color_weight": pair.get("effective_color_weight", 0), "context_score": context,
                                     "query_camera_id": track["camera_id"], "reference_camera_id": reference["camera_id"],
                                     "query_capture_time": track.get("absolute_start"),
                                     "reference_capture_time": reference.get("absolute_start"),
                                     "class_name": track.get("class_name", "truck"),
                                     "query_class_name": track.get("class_name", "truck"),
                                     "reference_class_name": reference.get("class_name", "truck")})
            rows.sort(key=lambda row: (row["query_crop_index"], -row["cosine_similarity"], row["candidate_vehicle_id"], row["reference_observation_id"], row["reference_crop_index"]))
            ranks = {}
            for row in rows:
                query_index = row["query_crop_index"]
                ranks[query_index] = ranks.get(query_index, 0) + 1
                row.update(cosine_rank=ranks[query_index], assigned_vehicle_id=assignment.get("global_id"),
                           detected_vehicle_id=track.get("global_id"), decision_status=assignment["status"],
                           is_winning_identity=row["candidate_vehicle_id"] == assignment.get("comparison_global_id"),
                           is_assigned_identity=row["candidate_vehicle_id"] == assignment.get("global_id"),
                           is_decision_reference=(row["reference_observation_id"] == assignment.get("reference_track_id") and
                                                  row["reference_crop_index"] == assignment.get("reference_crop_index") and
                                                  row["query_crop_index"] == assignment.get("query_crop_index")))
            comparisons.extend(rows)
    return comparisons
