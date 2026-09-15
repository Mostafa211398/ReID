"""Labeled-folder embedding benchmark metrics and portable exports."""
from __future__ import annotations

import csv
import io
import json
import re
import textwrap
import zipfile
from pathlib import Path, PurePosixPath
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .reid_core import atomic_replace, fingerprint, unit
from .utils import utc_now

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
COMPARISON_EXPORT_LIMIT = 50_000
COMPARISON_CARD_SIZE = (1400, 850)


def safe_part(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")[:120] or "item"


def benchmark_result_fingerprint(record, job):
    return fingerprint({"package": record.get("package_name") or job.get("config", {}).get("package_name"),
                        "threshold": record.get("threshold"),
                        "encoders": [(row.get("encoder_id"), row.get("encoder_fingerprint"))
                                     for row in record.get("encoders", [])],
                        "pairs_per_encoder": record.get("counts", {}).get("all_pairs_per_encoder")})


def comparison_card_lines(row, configured_threshold):
    return [
        f'{row["encoder_name"]} | Cosine {row["cosine_similarity"]:.3f}',
        "Ground truth: " + ("Same truck" if row["actual_same_identity"] else "Different trucks"),
        "Configured: " + ("same" if row["predicted_same_configured"] else "different"),
        "Best F1: " + ("same" if row["predicted_same_best_f1"] else "different"),
        f"Configured threshold: {configured_threshold:.3f}",
        f'Exploratory best threshold: {row["best_f1_threshold"]:.3f}',
        "Strict metric: " + ("Eligible" if row["strict_eligible"] else
                              f'Excluded · {row.get("exclusion_reason") or "unspecified"}'),
        f'Encoder fingerprint: {row["encoder_fingerprint"][:12]}',
    ]


def _font(size, bold=False):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _fit_image(path, size):
    with Image.open(path) as source:
        return ImageOps.contain(ImageOps.exif_transpose(source).convert("RGB"), size,
                                method=Image.Resampling.LANCZOS)


def render_comparison_card(row, items, image_root, configured_threshold):
    canvas = Image.new("RGB", COMPARISON_CARD_SIZE, "#0b111a")
    draw = ImageDraw.Draw(canvas)
    title, body, small = _font(30, True), _font(22), _font(17)
    draw.text((30, 22), comparison_card_lines(row, configured_threshold)[0], font=title, fill="#f4f7fb")
    for column, item_id, identity_key, filename_key, source_key, side in (
            (30, row["left_id"], "left_identity", "left_filename", "left_source_group", "First image"),
            (710, row["right_id"], "right_identity", "right_filename", "right_source_group", "Second image")):
        item = items[item_id]
        path = (image_root / item["stored_name"]).resolve()
        if not path.is_relative_to(image_root.resolve()) or not path.is_file():
            raise ValueError("A benchmark comparison image is missing")
        image = _fit_image(path, (640, 420))
        x, y = column + (640 - image.width) // 2, 90 + (420 - image.height) // 2
        canvas.paste(image, (x, y)); image.close()
        draw.rounded_rectangle((column, 80, column + 660, 570), radius=10, outline="#33465e", width=2)
        draw.text((column + 14, 520), side, font=small, fill="#8fa7c2")
        label = f'{row[identity_key]} · {row[filename_key]}'
        draw.text((column + 14, 548), label[:72], font=small, fill="#f4f7fb")
        draw.text((column + 14, 578), f'Viewpoint {row[source_key]}'[:78], font=small, fill="#aebed0")
    lines = comparison_card_lines(row, configured_threshold)[1:]
    for index, line in enumerate(lines):
        x = 30 if index < 4 else 710
        y = 630 + (index if index < 4 else index - 4) * 42
        color = "#8fd6a5" if "Same truck" in line or line.endswith("Eligible") else "#f4f7fb"
        draw.text((x, y), "\n".join(textwrap.wrap(line, width=62))[:140], font=body, fill=color)
    return canvas


def _model_folders(rows):
    models = []
    for row in rows:
        if not any(value[0] == row["encoder_id"] for value in models):
            models.append((row["encoder_id"], row.get("encoder_name", row["encoder_id"])))
    bases = [safe_part(name) for _, name in models]
    counts = {base.casefold(): sum(other.casefold() == base.casefold() for other in bases) for base in bases}
    return {encoder_id: base if counts[base.casefold()] == 1 else f"{base}__{safe_part(encoder_id)}"
            for (encoder_id, _), base in zip(models, bases)}


def write_comparison_images(config, progress=None):
    source_package = Path(config["source_package"])
    output = Path(config["output"])
    image_root = Path(config["image_root"]).resolve()
    try:
        with zipfile.ZipFile(source_package) as source:
            rows = json.loads(source.read("data/pairwise_comparisons.json"))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise ValueError("The saved benchmark package is corrupt; retry the benchmark") from exc
    if len(rows) != config["comparison_count"]:
        raise ValueError("The saved benchmark comparison count changed; prepare a new export")
    if len(rows) > COMPARISON_EXPORT_LIMIT:
        raise ValueError(f"Comparison-image exports are limited to {COMPARISON_EXPORT_LIMIT:,} images; use a smaller benchmark folder")
    items = {item["id"]: item for item in config["items"]}
    if any(row.get(key) not in items for row in rows for key in ("left_id", "right_id")):
        raise ValueError("The saved benchmark package does not match its images")
    for row in rows:
        row["left_filename"] = items[row["left_id"]]["filename"]
        row["right_filename"] = items[row["right_id"]]["filename"]
    encoder_order = {key: index for index, key in enumerate(config["encoders"])}
    rows.sort(key=lambda row: (encoder_order.get(row["encoder_id"], len(encoder_order)),
                               -row["cosine_similarity"], row["left_identity"].casefold(), row["left_path"].casefold(),
                               row["right_identity"].casefold(), row["right_path"].casefold()))
    folders = _model_folders(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".zip.partial")
    temporary.unlink(missing_ok=True)
    manifest = {"schema_version": "1.0", "generated_at": utc_now().isoformat(),
                "benchmark_id": config["benchmark_id"], "source_result_fingerprint": config["source_result_fingerprint"],
                "configured_threshold": config["threshold"], "comparison_count": len(rows),
                "models": [{"encoder_id": key, "encoder_name": next(row["encoder_name"] for row in rows if row["encoder_id"] == key),
                            "folder": folder, "comparison_count": sum(row["encoder_id"] == key for row in rows)}
                           for key, folder in folders.items()]}
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, indent=2, allow_nan=False))
            archive.writestr("README.txt", "IRIS ReID model-grouped benchmark comparison images\n\nEach JPEG reproduces the criteria shown by the Accuracy-section benchmark comparison card. Model folders contain every scored pair in descending cosine order. comparisons.csv retains full-precision saved values. No embeddings or model inference are included.\n")
            for encoder_id, folder in folders.items():
                model_rows = [row for row in rows if row["encoder_id"] == encoder_id]
                archive.writestr(f"{folder}/comparisons.csv", csv_text(model_rows))
            ranks = {key: 0 for key in folders}
            for index, row in enumerate(rows):
                ranks[row["encoder_id"]] += 1
                rank = ranks[row["encoder_id"]]
                status = "same" if row["actual_same_identity"] else "different"
                left_stem, right_stem = Path(row["left_filename"]).stem, Path(row["right_filename"]).stem
                name = (f'{rank:06d}__{status}__{safe_part(row["left_identity"])[:28]}--'
                        f'{safe_part(row["right_identity"])[:28]}__{safe_part(left_stem)[:28]}--{safe_part(right_stem)[:28]}.jpg')
                card = render_comparison_card(row, items, image_root, config["threshold"])
                payload = io.BytesIO(); card.save(payload, "JPEG", quality=88, subsampling=2); card.close()
                archive.writestr(f'{folders[row["encoder_id"]]}/{name}', payload.getvalue())
                if progress and (index % 10 == 0 or index + 1 == len(rows)):
                    progress((index + 1) / max(1, len(rows)), rendered=index + 1, total=len(rows))
        atomic_replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return manifest


def parse_folder_paths(values):
    parsed = []
    for value in values:
        normalized = str(value).replace("\\", "/")
        path = PurePosixPath(normalized)
        if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
            raise ValueError("Benchmark paths must be safe relative paths")
        parsed.append(path.parts)
    strip_root = bool(parsed) and all(len(parts) >= 3 for parts in parsed) and len({parts[0].casefold() for parts in parsed}) == 1
    result = []
    seen = set()
    for parts in parsed:
        parts = parts[1:] if strip_root else parts
        if len(parts) != 2:
            raise ValueError("Select one parent folder containing one immediate folder per vehicle")
        identity, filename = parts
        if not identity.strip() or len(identity) > 100 or Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError("Every benchmark file must be a supported image inside a named vehicle folder")
        relative = f"{identity}/{filename}"
        if relative.casefold() in seen:
            raise ValueError("Benchmark relative paths must be unique")
        seen.add(relative.casefold())
        match = re.match(r"^(t\d+)_f\d+$", Path(filename).stem, re.IGNORECASE)
        result.append({"identity": identity, "filename": filename, "relative_path": relative,
                       "truck_code": match.group(1).lower() if match else None,
                       "source_group": Path(filename).stem.casefold()})
    return result


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def binary_metrics(rows, threshold, strict=True):
    selected = [row for row in rows if not strict or row["strict_eligible"]]
    tp = fp = tn = fn = 0
    for row in selected:
        predicted = row["cosine_similarity"] >= threshold
        actual = row["actual_same_identity"]
        tp += int(predicted and actual)
        fp += int(predicted and not actual)
        tn += int(not predicted and not actual)
        fn += int(not predicted and actual)
    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    return {"threshold": float(threshold), "pairs": len(selected), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall, "f1": ratio(2 * precision * recall, precision + recall),
            "accuracy": ratio(tp + tn, len(selected))}


def best_f1(rows):
    eligible = [row for row in rows if row["strict_eligible"]]
    candidates = {-1.0, 1.0, *(float(row["cosine_similarity"]) for row in eligible)}
    return max((binary_metrics(eligible, threshold) for threshold in candidates),
               key=lambda metrics: (metrics["f1"], metrics["precision"], metrics["threshold"]))


def identification(items, similarities):
    predictions = []
    by_id = {item["id"]: item for item in items}
    ordered = sorted(items, key=lambda item: (item["identity"].casefold(), item["relative_path"].casefold(), item["id"]))
    indexes = {item["id"]: index for index, item in enumerate(items)}
    for query in ordered:
        qi = indexes[query["id"]]
        gallery = []
        for reference in ordered:
            if reference["id"] == query["id"]:
                continue
            same_source = reference["identity"] == query["identity"] and reference["source_group"] == query["source_group"]
            if not same_source:
                gallery.append((float(similarities[qi, indexes[reference["id"]]]), reference))
        ranked = sorted(gallery, key=lambda pair: (-pair[0], pair[1]["identity"].casefold(), pair[1]["relative_path"].casefold(), pair[1]["id"]))
        eligible = any(reference["identity"] == query["identity"] for _, reference in ranked)
        winner = ranked[0] if ranked else (None, None)
        hits = np.asarray([reference["identity"] == query["identity"] for _, reference in ranked], dtype=np.float64)
        positives = int(hits.sum())
        average_precision = float(((np.cumsum(hits) / np.arange(1, len(hits) + 1)) * hits).sum() / positives) if positives else None
        predictions.append({"query_id": query["id"], "query_path": query["archive_path"], "actual_identity": query["identity"],
                            "source_group": query["source_group"], "eligible": eligible,
                            "exclusion_reason": None if eligible else "no_cross_source_positive_reference",
                            "predicted_identity": winner[1]["identity"] if winner[1] else None,
                            "winning_reference_id": winner[1]["id"] if winner[1] else None,
                            "winning_reference_path": winner[1]["archive_path"] if winner[1] else None,
                            "cosine_similarity": winner[0], "correct": bool(eligible and winner[1]["identity"] == query["identity"]) if winner[1] else False,
                            "rank1": bool(hits[0]) if eligible and len(hits) else False, "average_precision": average_precision})
    eligible_rows = [row for row in predictions if row["eligible"]]
    identities = sorted({item["identity"] for item in items}, key=str.casefold)
    per_identity, confusion = [], []
    for actual in identities:
        support = sum(row["actual_identity"] == actual for row in eligible_rows)
        for predicted in identities:
            count = sum(row["actual_identity"] == actual and row["predicted_identity"] == predicted for row in eligible_rows)
            confusion.append({"actual_identity": actual, "predicted_identity": predicted, "count": count})
        tp = sum(row["actual_identity"] == actual and row["predicted_identity"] == actual for row in eligible_rows)
        fp = sum(row["actual_identity"] != actual and row["predicted_identity"] == actual for row in eligible_rows)
        fn = sum(row["actual_identity"] == actual and row["predicted_identity"] != actual for row in eligible_rows)
        tn = len(eligible_rows) - tp - fp - fn
        precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
        per_identity.append({"identity": actual, "support": support, "included_in_macro": support > 0,
                             "tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision,
                             "recall": recall, "f1": ratio(2 * precision * recall, precision + recall)})
    supported = [row for row in per_identity if row["included_in_macro"]]
    metrics = {"eligible_queries": len(eligible_rows), "excluded_queries": len(predictions) - len(eligible_rows),
               "accuracy": ratio(sum(row["correct"] for row in eligible_rows), len(eligible_rows)),
               "rank1": ratio(sum(row["rank1"] for row in eligible_rows), len(eligible_rows)),
               "mAP": float(np.mean([row["average_precision"] for row in eligible_rows])) if eligible_rows else 0.0,
               "macro_precision": float(np.mean([row["precision"] for row in supported])) if supported else 0.0,
               "macro_recall": float(np.mean([row["recall"] for row in supported])) if supported else 0.0,
               "macro_f1": float(np.mean([row["f1"] for row in supported])) if supported else 0.0}
    return metrics, predictions, per_identity, confusion


def score_encoder(items, vectors, spec, threshold):
    missing = [item["id"] for item in items if item["id"] not in vectors or not len(vectors[item["id"]])]
    if missing:
        raise ValueError(f"Encoder did not produce embeddings for {len(missing)} benchmark images")
    matrix = unit(np.stack([np.asarray(vectors[item["id"]][0], dtype=np.float32) for item in items]))
    similarities = matrix @ matrix.T
    pairs = []
    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            a, b = items[left], items[right]
            actual = a["identity"] == b["identity"]
            same_source = actual and a["source_group"] == b["source_group"]
            pairs.append({"left_id": a["id"], "left_path": a["archive_path"], "left_identity": a["identity"], "left_source_group": a["source_group"],
                          "right_id": b["id"], "right_path": b["archive_path"], "right_identity": b["identity"], "right_source_group": b["source_group"],
                          "cosine_similarity": float(np.clip(similarities[left, right], -1, 1)), "actual_same_identity": actual,
                          "strict_eligible": not same_source, "exclusion_reason": "same_source_track" if same_source else None})
    best = best_f1(pairs)
    configured = binary_metrics(pairs, threshold)
    all_images = binary_metrics(pairs, threshold, strict=False)
    for row in pairs:
        row.update(encoder_id=spec["id"], encoder_name=spec.get("name", spec["id"]), encoder_fingerprint=spec["fingerprint"],
                   predicted_same_configured=row["cosine_similarity"] >= threshold,
                   predicted_same_best_f1=row["cosine_similarity"] >= best["threshold"], best_f1_threshold=best["threshold"])
    id_metrics, predictions, per_identity, confusion = identification(items, similarities)
    metadata = {"encoder_id": spec["id"], "encoder_name": spec.get("name", spec["id"]),
                "encoder_fingerprint": spec["fingerprint"], "embedding_dimension": int(matrix.shape[1]),
                "configured_threshold": float(threshold), "pairwise_strict": configured,
                "pairwise_all_images": all_images, "best_f1_exploratory": best, "identification": id_metrics}
    for collection in (predictions, per_identity, confusion):
        for row in collection:
            row.update(encoder_id=spec["id"], encoder_name=spec.get("name", spec["id"]), encoder_fingerprint=spec["fingerprint"])
    return {"summary": metadata, "pairs": pairs, "predictions": predictions, "per_identity": per_identity, "confusion": confusion}


def csv_text(rows):
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def flat_summary(result):
    summary = result["summary"]
    row = {key: value for key, value in summary.items() if not isinstance(value, dict)}
    for group in ("pairwise_strict", "pairwise_all_images", "best_f1_exploratory", "identification"):
        row.update({f"{group}_{key}": value for key, value in summary[group].items()})
    return row


def write_results(service, job, results):
    directory = service.root / job["id"]
    previous_package = job["config"].get("package_name")
    package_name = "benchmark-results-" + uuid4().hex + ".zip"
    summaries = [{**result["summary"], "per_identity": result["per_identity"], "confusion": result["confusion"]} for result in results]
    pairs = [row for result in results for row in result["pairs"]]
    predictions = [row for result in results for row in result["predictions"]]
    per_identity = [row for result in results for row in result["per_identity"]]
    confusion = [row for result in results for row in result["confusion"]]
    counts = {"identities": len({item["identity"] for item in job["config"]["items"]}), "images": len(job["config"]["items"]),
              "source_groups": len({(item["identity"], item["source_group"]) for item in job["config"]["items"]}),
              "all_pairs_per_encoder": len(results[0]["pairs"]) if results else 0,
              "strict_pairs_per_encoder": sum(row["strict_eligible"] for row in results[0]["pairs"]) if results else 0,
              "eligible_identification_queries": results[0]["summary"]["identification"]["eligible_queries"] if results else 0}
    record = {"id": job["id"], "job_id": job["id"], "name": job["name"], "site_id": job["site_id"],
              "threshold": job["config"]["threshold"], "encoders": summaries, "counts": counts,
              "package_name": package_name, "created_at": job.get("created_at"), "updated_at": utc_now().isoformat()}
    from .reid_core import atomic_json
    atomic_json(directory / "summary.json", record)
    (directory / "summary.csv.partial").write_text(csv_text(flat_summary(result) for result in results), encoding="utf-8")
    atomic_replace(directory / "summary.csv.partial", directory / "summary.csv")
    final_package = directory / package_name
    temporary = final_package.with_suffix(".zip.partial")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        manifest = {"schema_version": "1.0", "generated_at": record["updated_at"], "benchmark_id": job["id"],
                    "name": job["name"], "site_id": job["site_id"], "counts": counts,
                    "encoders": [{key: value for key, value in summary.items() if key in ("encoder_id", "encoder_name", "encoder_fingerprint", "embedding_dimension")} for summary in summaries]}
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, allow_nan=False))
        archive.writestr("README.txt", "IRIS ReID labeled-folder benchmark\n\nFolders and tNN labels are ground-truth truck identities; each distinct fNN image is treated as an independent viewpoint. Raw L2-normalized embedding cosine is used without color. Identification excludes only the query itself unless duplicate source metadata is explicitly present. The best-F1 threshold is exploratory because it is selected on these same images. All paths are relative; embeddings, weights, and local paths are excluded.\n")
        payloads = {"summary": record, "pairwise_comparisons": pairs, "identification_predictions": predictions,
                    "per_identity_metrics": per_identity, "confusion_matrix": confusion}
        for name, rows in payloads.items():
            archive.writestr(f"data/{name}.json", json.dumps(rows, indent=2, allow_nan=False))
            archive.writestr(f"data/{name}.csv", csv_text([flat_summary(result) for result in results] if name == "summary" else rows))
        written = set()
        for item in job["config"]["items"]:
            if item["archive_path"] in written:
                continue
            archive.write(directory / "images" / item["stored_name"], item["archive_path"])
            written.add(item["archive_path"])
    atomic_replace(temporary, final_package)
    if previous_package and previous_package != package_name:
        try:
            (directory / previous_package).unlink(missing_ok=True)
        except PermissionError:
            pass
    return record
