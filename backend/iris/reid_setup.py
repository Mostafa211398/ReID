"""Explicit model installation and diagnostics; inference never downloads weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

from .reid_core import BASELINES, ENCODER_METADATA, atomic_json, atomic_replace, fingerprint

FASTREID_REVISION = "c9bc3ceb2f7a6438b62fb515ea3df6d1e999e95d"
HF_REVISIONS = {"siglip": "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed", "dinov2": "ed25f3a31f01632728cabb09d1542f84ab7b0056",
                "coca": "47aff38863cd40aa76d915bb04e4a3d8edf5824c",
                "coca_l14": "74207cb7fde8eafc9864451ebd332fa8e75b150f",
                "siglip2": "f775b65a79762255128c981547af89addcfe0f88"}
COCA_CHECKSUM = "df49af430529127b5f019825ddab06867429a63cec4006e34c3cb0bc4c8b4a04"
COCA_SIZE = 1_014_488_932
COCA_VISUAL_EXTRACTION_VERSION = "coca-visual-v1"
COCA_VISUAL_CHECKSUM = "84f1c68835ce933fa3ee7b0ea3ec0830adb5c31c5a7e7c810772afbc5cb337b9"
COCA_VISUAL_SIZE = 356_673_624
COCA_VISUAL_CONFIG = {
    "embed_dim": 512,
    "vision_cfg": {"image_size": 224, "layers": 12, "width": 768, "patch_size": 32,
                   "attentional_pool": True, "attn_pooler_heads": 8, "output_tokens": True},
    "preprocess": {"size": [224, 224], "mean": [0.48145466, 0.4578275, 0.40821073],
                   "std": [0.26862954, 0.26130258, 0.27577711], "interpolation": "bicubic",
                   "resize_mode": "shortest", "fill_color": 0},
}
COCA_L14_CHECKSUM = "73725652298ad76ed2162caffdae96d8653a05d7a29b6281103e4df81d0ff8ea"
COCA_L14_SIZE = 2_554_109_637
COCA_L14_VISUAL_EXTRACTION_VERSION = "coca-l14-visual-v1"
COCA_L14_VISUAL_CHECKSUM = "51e977d10c02b43c4c1dcbce07d3699c14c77ad4903291aba117b6c3ea45c728"
COCA_L14_VISUAL_SIZE = 1_226_933_392
COCA_L14_VISUAL_CONFIG = {
    "embed_dim": 768,
    "vision_cfg": {"image_size": 224, "layers": 24, "width": 1024, "patch_size": 14,
                   "attentional_pool": True, "attn_pooler_heads": 8, "output_tokens": True},
    "preprocess": {"size": [224, 224], "mean": [0.48145466, 0.4578275, 0.40821073],
                   "std": [0.26862954, 0.26130258, 0.27577711], "interpolation": "bicubic",
                   "resize_mode": "shortest", "fill_color": 0},
}
SIGLIP2_CHECKSUM = "ed72c0ace85020ae610fc817c2538b9cae5a477b012a50859c60af5b3ad30857"
SIGLIP2_SIZE = 1_501_968_264
SIGLIP2_VISUAL_EXTRACTION_VERSION = "siglip2-visual-v1"
SIGLIP2_FILES = ["config.json", "preprocessor_config.json", "model.safetensors", "special_tokens_map.json",
                 "tokenizer.json", "tokenizer.model", "tokenizer_config.json"]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        return
    temporary = path.with_suffix(".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "IRIS-ReID"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    temporary.replace(path)


def install(root, selected=None, transreid_checkpoint=None):
    root.mkdir(parents=True, exist_ok=True)
    selected = ["siglip", "dinov2", "fastreid", "coca", "coca_visual", "coca_l14", "coca_l14_visual",
                "siglip2", "siglip2_visual"] if selected is None else selected
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for key in selected:
        if key in ("openvino", "transreid"):
            from .reid_extra_setup import install_vehicle
            manifest[key] = install_vehicle(root, key, transreid_checkpoint)
            atomic_json(manifest_path, manifest)
    hf_selected = set(selected)
    if "coca_visual" in hf_selected:
        hf_selected.add("coca")
    if "coca_l14_visual" in hf_selected:
        hf_selected.add("coca_l14")
    if "siglip2_visual" in hf_selected:
        hf_selected.add("siglip2")
    for key in ("siglip", "dinov2", "coca", "coca_l14", "siglip2"):
        if key not in hf_selected:
            continue
        if key == "coca" and key not in selected:
            installed = root / "coca" / "open_clip_pytorch_model.bin"
            if (key in manifest and installed.is_file() and installed.stat().st_size == COCA_SIZE
                    and digest(installed) == COCA_CHECKSUM):
                continue
        if key == "coca_l14" and key not in selected:
            installed = root / key / "open_clip_pytorch_model.bin"
            if (key in manifest and installed.is_file() and installed.stat().st_size == COCA_L14_SIZE
                    and digest(installed) == COCA_L14_CHECKSUM):
                continue
        if key == "siglip2" and key not in selected:
            installed = root / "siglip2" / "model.safetensors"
            if (key in manifest and installed.is_file() and installed.stat().st_size == SIGLIP2_SIZE
                    and digest(installed) == SIGLIP2_CHECKSUM):
                continue
        from huggingface_hub import snapshot_download
        spec = BASELINES[key]
        revision = HF_REVISIONS[key]
        print(f'Downloading {spec["name"]} at {revision}', flush=True)
        location = root / key
        patterns = (["open_clip_pytorch_model.bin"] if key in ("coca", "coca_l14") else SIGLIP2_FILES
                    if key == "siglip2" else ["config.json", "preprocessor_config.json", "model.safetensors"])
        if key == "siglip" and (location / "vision-only.json").exists():
            patterns.remove("model.safetensors")
        snapshot_download(spec.get("repository_name", spec["model_name"]), revision=revision, local_dir=str(location), allow_patterns=patterns)
        if key == "siglip":
            compact_siglip(location)
        files = {p.name: digest(p) for p in location.iterdir()
                 if p.is_file() and p.suffix in (".json", ".safetensors", ".bin", ".model")}
        if key == "coca":
            checkpoint = location / "open_clip_pytorch_model.bin"
            if files.get(checkpoint.name) != COCA_CHECKSUM or checkpoint.stat().st_size != COCA_SIZE:
                raise ValueError("Downloaded CoCa checkpoint failed its pinned size or SHA-256 verification")
        if key == "coca_l14":
            checkpoint = location / "open_clip_pytorch_model.bin"
            if files.get(checkpoint.name) != COCA_L14_CHECKSUM or checkpoint.stat().st_size != COCA_L14_SIZE:
                raise ValueError("Downloaded CoCa ViT-L/14 checkpoint failed its pinned size or SHA-256 verification")
        if key == "siglip2":
            checkpoint = location / "model.safetensors"
            if files.get(checkpoint.name) != SIGLIP2_CHECKSUM or checkpoint.stat().st_size != SIGLIP2_SIZE:
                raise ValueError("Downloaded SigLIP2 checkpoint failed its pinned size or SHA-256 verification")
        manifest[key] = {"revision": revision, "checksums": files,
                         "sizes": {p.name: p.stat().st_size for p in location.iterdir() if p.name in files},
                         "fingerprint": fingerprint([spec, revision, files, "native-preprocess-v1"])}
        atomic_json(manifest_path, manifest)
    if "coca_visual" in selected:
        manifest["coca_visual"] = extract_coca_visual(root, manifest["coca"])
        atomic_json(manifest_path, manifest)
    if "coca_l14_visual" in selected:
        manifest["coca_l14_visual"] = extract_coca_l14_visual(root, manifest["coca_l14"])
        atomic_json(manifest_path, manifest)
    if "siglip2_visual" in selected:
        manifest["siglip2_visual"] = extract_siglip2_visual(root, manifest["siglip2"])
        atomic_json(manifest_path, manifest)
    if "fastreid" not in selected:
        print("Selected ReID models installed. Run diagnostics-reid.ps1 to verify.", flush=True)
        return
    archive = root / "fast-reid.zip"
    print("Downloading pinned FastReID source and official VeRi weights", flush=True)
    download(f"https://codeload.github.com/JDAI-CV/fast-reid/zip/{FASTREID_REVISION}", archive)
    source = root / "fast-reid"
    if not source.exists():
        source.mkdir()
        with zipfile.ZipFile(archive) as package:
            for item in package.infolist():
                parts = Path(item.filename).parts[1:]
                if not parts or item.is_dir():
                    continue
                destination = source.joinpath(*parts).resolve()
                if not destination.is_relative_to(source.resolve()):
                    raise ValueError("Invalid FastReID source archive path")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with package.open(item) as src, destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        # Upstream's model-only imports need these Python 3.10 compatibility changes.
        for path in (source / "fastreid").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            updated = text.replace("from collections import Mapping", "from collections.abc import Mapping")
            updated = updated.replace("from torch._six import container_abcs", "import collections.abc as container_abcs")
            if updated != text:
                path.write_text(updated, encoding="utf-8")
    checkpoint = root / "fastreid/model.pth"
    download("https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veri_sbs_R50-ibn.pth", checkpoint)
    files = {"model.pth": digest(checkpoint)}
    manifest["fastreid"] = {"revision": FASTREID_REVISION, "checksums": files, "source_checksum": digest(archive),
                           "fingerprint": fingerprint([BASELINES["fastreid"], FASTREID_REVISION, files, "native-preprocess-v1"])}
    atomic_json(root / "manifest.json", manifest)
    print("Model installation complete. Run diagnostics-reid.ps1 to verify installed encoders.", flush=True)


def extract_coca_visual(root: Path, parent: dict):
    import torch
    from open_clip.model import CLIPVisionCfg, _build_vision_tower
    from safetensors.torch import save_file
    source = root / "coca" / "open_clip_pytorch_model.bin"
    if not source.is_file() or digest(source) != COCA_CHECKSUM or source.stat().st_size != COCA_SIZE:
        raise ValueError("Install the pinned full CoCa checkpoint before extracting its visual encoder")
    print("Extracting the standalone CoCa visual tower", flush=True)
    state = torch.load(source, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    visual = {name.removeprefix("visual."): value.contiguous() for name, value in state.items() if name.startswith("visual.")}
    expected_model = _build_vision_tower(COCA_VISUAL_CONFIG["embed_dim"], CLIPVisionCfg(**COCA_VISUAL_CONFIG["vision_cfg"]))
    if set(visual) != set(expected_model.state_dict()):
        raise ValueError("The CoCa checkpoint visual tower does not match the pinned architecture")
    del expected_model, state
    location = root / "coca_visual"
    location.mkdir(parents=True, exist_ok=True)
    checkpoint = location / "model.safetensors"
    temporary = location / "model.partial"
    # Keep the artifact byte-reproducible. Provenance is stored in the separately
    # checksummed config and manifest; Safetensors metadata map order is unstable.
    save_file(visual, str(temporary))
    atomic_replace(temporary, checkpoint)
    if checkpoint.stat().st_size != COCA_L14_VISUAL_SIZE or digest(checkpoint) != COCA_L14_VISUAL_CHECKSUM:
        raise ValueError("Extracted CoCa ViT-L/14 visual checkpoint does not match the pinned artifact")
    if checkpoint.stat().st_size != COCA_VISUAL_SIZE or digest(checkpoint) != COCA_VISUAL_CHECKSUM:
        raise ValueError("Extracted CoCa visual checkpoint does not match the pinned artifact")
    config = {**COCA_VISUAL_CONFIG, "parent_revision": parent["revision"],
              "parent_sha256": COCA_CHECKSUM, "extraction_version": COCA_VISUAL_EXTRACTION_VERSION}
    atomic_json(location / "config.json", config)
    (location / "LICENSE.txt").write_text(
        "Derived visual-tower weights from laion/CoCa-ViT-B-32-laion2B-s13B-b90k.\n"
        "Model and OpenCLIP license notices: https://huggingface.co/laion/CoCa-ViT-B-32-laion2B-s13B-b90k\n"
        "https://github.com/mlfoundations/open_clip/blob/main/LICENSE\n", encoding="utf-8")
    files = {path.name: digest(path) for path in location.iterdir() if path.is_file()}
    return {"revision": parent["revision"], "parent_sha256": COCA_CHECKSUM,
            "extraction_version": COCA_VISUAL_EXTRACTION_VERSION, "checksums": files,
            "sizes": {path.name: path.stat().st_size for path in location.iterdir() if path.is_file()},
            "fingerprint": fingerprint([BASELINES["coca_visual"], config, files])}


def extract_coca_l14_visual(root: Path, parent: dict):
    import torch
    from open_clip.model import CLIPVisionCfg, _build_vision_tower
    from safetensors.torch import save_file
    source = root / "coca_l14" / "open_clip_pytorch_model.bin"
    if (not source.is_file() or digest(source) != COCA_L14_CHECKSUM or source.stat().st_size != COCA_L14_SIZE
            or parent.get("revision") != HF_REVISIONS["coca_l14"]
            or parent.get("checksums", {}).get(source.name) != COCA_L14_CHECKSUM):
        raise ValueError("Install the pinned full CoCa ViT-L/14 checkpoint before extracting its visual encoder")
    print("Extracting the standalone CoCa ViT-L/14 visual tower", flush=True)
    state = torch.load(source, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    # This official distributed-training checkpoint retains OpenCLIP's DDP
    # wrapper prefix, unlike the B/32 checkpoint.
    state = {name.removeprefix("module."): value for name, value in state.items()}
    visual = {name.removeprefix("visual."): value.contiguous() for name, value in state.items()
              if name.startswith("visual.")}
    expected_model = _build_vision_tower(
        COCA_L14_VISUAL_CONFIG["embed_dim"], CLIPVisionCfg(**COCA_L14_VISUAL_CONFIG["vision_cfg"]))
    if set(visual) != set(expected_model.state_dict()):
        raise ValueError("The CoCa ViT-L/14 checkpoint visual tower does not match the pinned architecture")
    del expected_model, state
    location = root / "coca_l14_visual"
    location.mkdir(parents=True, exist_ok=True)
    checkpoint = location / "model.safetensors"
    temporary = location / "model.partial"
    save_file(visual, str(temporary))
    atomic_replace(temporary, checkpoint)
    config = {**COCA_L14_VISUAL_CONFIG, "parent_revision": parent["revision"],
              "parent_sha256": COCA_L14_CHECKSUM, "extraction_version": COCA_L14_VISUAL_EXTRACTION_VERSION}
    atomic_json(location / "config.json", config)
    (location / "LICENSE.txt").write_text(
        "Derived visual-tower weights from laion/CoCa-ViT-L-14-laion2B-s13B-b90k.\n"
        "Model and OpenCLIP license notices: https://huggingface.co/laion/CoCa-ViT-L-14-laion2B-s13B-b90k\n"
        "https://github.com/mlfoundations/open_clip/blob/main/LICENSE\n", encoding="utf-8")
    files = {path.name: digest(path) for path in location.iterdir() if path.is_file()}
    return {"revision": parent["revision"], "parent_sha256": COCA_L14_CHECKSUM,
            "extraction_version": COCA_L14_VISUAL_EXTRACTION_VERSION, "checksums": files,
            "sizes": {path.name: path.stat().st_size for path in location.iterdir() if path.is_file()},
            "fingerprint": fingerprint([BASELINES["coca_l14_visual"], config, files])}


def compact_siglip(location):
    from safetensors import safe_open
    from safetensors.numpy import save_file
    checkpoint = location / "model.safetensors"
    with safe_open(checkpoint, framework="numpy") as source:
        if not any(k.startswith("text_model.") for k in source.keys()):
            return
        vision = {k: source.get_tensor(k) for k in source.keys() if k.startswith("vision_model.")}
        temp = location / "vision.partial"
        save_file(vision, str(temp), metadata={"format": "pt"})
    atomic_replace(temp, checkpoint)
    atomic_json(location / "vision-only.json", {"revision": HF_REVISIONS["siglip"], "encoder": "image"})


def extract_siglip2_visual(root: Path, parent: dict):
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import SiglipVisionConfig, SiglipVisionModel
    source = root / "siglip2" / "model.safetensors"
    if (not source.is_file() or digest(source) != SIGLIP2_CHECKSUM
            or source.stat().st_size != SIGLIP2_SIZE or parent.get("revision") != HF_REVISIONS["siglip2"]
            or parent.get("checksums", {}).get("model.safetensors") != SIGLIP2_CHECKSUM):
        raise ValueError("Install the pinned full SigLIP2 checkpoint before extracting its visual encoder")
    print("Extracting the standalone SigLIP2 visual tower", flush=True)
    for filename in ("config.json", "preprocessor_config.json"):
        expected_checksum = parent.get("checksums", {}).get(filename)
        if not expected_checksum or digest(root / "siglip2" / filename) != expected_checksum:
            raise ValueError(f"Pinned SigLIP2 parent metadata failed checksum verification: {filename}")
    parent_config = json.loads((root / "siglip2" / "config.json").read_text(encoding="utf-8"))
    vision_config = parent_config["vision_config"]
    expected = SiglipVisionModel(SiglipVisionConfig(**vision_config))
    with safe_open(source, framework="pt", device="cpu") as checkpoint:
        visual = {name: checkpoint.get_tensor(name).contiguous() for name in checkpoint.keys()
                  if name.startswith("vision_model.")}
    if set(visual) != set(expected.state_dict()):
        raise ValueError("The SigLIP2 checkpoint visual tower does not match the pinned architecture")
    del expected
    location = root / "siglip2_visual"
    location.mkdir(parents=True, exist_ok=True)
    temporary = location / "model.partial"
    save_file(visual, str(temporary), metadata={"format": "pt"})
    atomic_replace(temporary, location / "model.safetensors")
    config = {**vision_config, "architectures": ["SiglipVisionModel"],
              "parent_revision": parent["revision"], "parent_sha256": SIGLIP2_CHECKSUM,
              "extraction_version": SIGLIP2_VISUAL_EXTRACTION_VERSION}
    atomic_json(location / "config.json", config)
    shutil.copyfile(root / "siglip2" / "preprocessor_config.json", location / "preprocessor_config.json")
    (location / "LICENSE.txt").write_text(
        "Derived visual-tower weights from google/siglip2-base-patch16-384.\n"
        "Model license and notices: https://huggingface.co/google/siglip2-base-patch16-384\n",
        encoding="utf-8")
    files = {path.name: digest(path) for path in location.iterdir() if path.is_file()}
    return {"revision": parent["revision"], "parent_sha256": SIGLIP2_CHECKSUM,
            "extraction_version": SIGLIP2_VISUAL_EXTRACTION_VERSION, "checksums": files,
            "sizes": {path.name: path.stat().st_size for path in location.iterdir() if path.is_file()},
            "fingerprint": fingerprint([BASELINES["siglip2_visual"], config, files])}


def verify(root, selected=None):
    import numpy as np
    from PIL import Image
    from .reid_encoders import Encoder
    manifest = json.loads((root / "manifest.json").read_text())
    for key, spec in BASELINES.items():
        if key not in (selected or manifest):
            continue
        if key not in manifest:
            raise ValueError(f"{key} is not installed; run download-reid-models.ps1 -Encoders {key}")
        for name, expected in manifest[key]["checksums"].items():
            path = root / key / name
            actual = digest(path)
            if actual != expected:
                raise ValueError(f"Checksum mismatch: {key}/{name}")
            if key == "coca" and name == "open_clip_pytorch_model.bin":
                if actual != COCA_CHECKSUM or path.stat().st_size != COCA_SIZE:
                    raise ValueError("CoCa checkpoint does not match its pinned size and SHA-256")
            if key == "coca_l14" and name == "open_clip_pytorch_model.bin":
                if actual != COCA_L14_CHECKSUM or path.stat().st_size != COCA_L14_SIZE:
                    raise ValueError("CoCa ViT-L/14 checkpoint does not match its pinned size and SHA-256")
            if key == "siglip2" and name == "model.safetensors":
                if actual != SIGLIP2_CHECKSUM or path.stat().st_size != SIGLIP2_SIZE:
                    raise ValueError("SigLIP2 checkpoint does not match its pinned size and SHA-256")
        if key == "coca_visual":
            config = json.loads((root / key / "config.json").read_text(encoding="utf-8"))
            if (manifest[key].get("parent_sha256") != COCA_CHECKSUM
                    or manifest[key].get("revision") != HF_REVISIONS["coca"]
                    or manifest[key].get("extraction_version") != COCA_VISUAL_EXTRACTION_VERSION
                    or digest(root / key / "model.safetensors") != COCA_VISUAL_CHECKSUM
                    or (root / key / "model.safetensors").stat().st_size != COCA_VISUAL_SIZE
                    or config != {**COCA_VISUAL_CONFIG, "parent_revision": HF_REVISIONS["coca"],
                                  "parent_sha256": COCA_CHECKSUM,
                                  "extraction_version": COCA_VISUAL_EXTRACTION_VERSION}):
                raise ValueError("CoCa visual encoder provenance or architecture does not match the pinned definition")
        if key == "coca_l14_visual":
            config = json.loads((root / key / "config.json").read_text(encoding="utf-8"))
            if (manifest[key].get("parent_sha256") != COCA_L14_CHECKSUM
                    or manifest[key].get("revision") != HF_REVISIONS["coca_l14"]
                    or manifest[key].get("extraction_version") != COCA_L14_VISUAL_EXTRACTION_VERSION
                    or digest(root / key / "model.safetensors") != COCA_L14_VISUAL_CHECKSUM
                    or (root / key / "model.safetensors").stat().st_size != COCA_L14_VISUAL_SIZE
                    or config != {**COCA_L14_VISUAL_CONFIG, "parent_revision": HF_REVISIONS["coca_l14"],
                                  "parent_sha256": COCA_L14_CHECKSUM,
                                  "extraction_version": COCA_L14_VISUAL_EXTRACTION_VERSION}):
                raise ValueError("CoCa ViT-L/14 visual encoder provenance or architecture does not match the pinned definition")
        if key == "siglip2_visual":
            config = json.loads((root / key / "config.json").read_text(encoding="utf-8"))
            if (manifest[key].get("parent_sha256") != SIGLIP2_CHECKSUM
                    or manifest[key].get("revision") != HF_REVISIONS["siglip2"]
                    or manifest[key].get("extraction_version") != SIGLIP2_VISUAL_EXTRACTION_VERSION
                    or config.get("parent_revision") != HF_REVISIONS["siglip2"]
                    or config.get("parent_sha256") != SIGLIP2_CHECKSUM
                    or config.get("extraction_version") != SIGLIP2_VISUAL_EXTRACTION_VERSION):
                raise ValueError("SigLIP2 visual encoder provenance does not match the pinned definition")
        # Large encoders are verified on CPU so diagnostics remain reliable on
        # the supported 4 GB GPU; queued production stages still try CUDA first.
        verify_device = "cpu" if ENCODER_METADATA[key].get("cpu_fallback") else "auto"
        model = Encoder({**spec, **manifest[key]}, root, device=verify_device)
        crop = Image.fromarray(np.random.default_rng(42).integers(0, 255, (160, 240, 3), dtype=np.uint8))
        first = model.encode([crop])
        second = model.encode([crop])
        assert first.shape == (1, ENCODER_METADATA[key]["embedding_dimension"])
        assert np.isfinite(first).all() and np.allclose(first, second, atol=1e-5)
        assert np.allclose(np.linalg.norm(first, axis=1), 1, atol=1e-5)
        print(f"{key}: {first.shape[-1]} dimensions, finite normalized features, {model.device}", flush=True)
        import gc
        import torch
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def package(output, config_path):
    config = json.loads(config_path.read_text())
    spec = config["encoder"]
    checksum = digest(output / "model.safetensors")
    record = {**spec, "name": spec["name"] + " · site tuned", "checkpoint": "model.safetensors",
              "checksum": checksum, "dataset_fingerprint": config["dataset"]["fingerprint"], "fingerprint": fingerprint([spec["fingerprint"], checksum])}
    with zipfile.ZipFile(output / "model.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(output / "model.safetensors", "model.safetensors")
        archive.writestr("encoder.json", json.dumps(record))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--package", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--encoders", nargs="+", choices=list(BASELINES))
    parser.add_argument("--transreid-checkpoint", type=Path)
    args = parser.parse_args()
    if args.package:
        package(args.package, args.config)
    elif args.verify:
        verify(args.models.resolve(), args.encoders)
    else:
        install(args.models.resolve(), args.encoders, args.transreid_checkpoint)


if __name__ == "__main__":
    main()
