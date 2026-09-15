"""Pinned vehicle model installation; invoked explicitly, never during inference."""
import hashlib
import shutil
import zipfile
from urllib.error import URLError

from .reid_core import BASELINES, fingerprint
from .reid_setup import digest, download

TRANSREID_REVISION = "dec55046fcdfadee14e2c28e2df89305d8f7557a"
TRANSREID_DRIVE_ID = "1SquTlBhl_pahsa5752KoGDBPY-AZpoSg"
OPENVINO_URL = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/vehicle-reid-0001/osnet_ain_x1_0_vehicle_reid.onnx"
OPENVINO_SHA384 = "0515ce72f653c39780d5b87dfed7255d396dd2b1e8b6e91fbaacdfad1da189166343157273c02f3b0fede3050ef7abb7"


def install_vehicle(root, key, local_checkpoint=None):
    location = root / key
    location.mkdir(parents=True, exist_ok=True)
    if key == "openvino":
        try:
            download(OPENVINO_URL, location / "model.onnx")
        except URLError:
            # Original source listed in OMZ's model.yml; same published SHA384.
            import gdown
            temporary = location / "model.download"
            gdown.download(id="1MEtaIr_9mWuntGD9edydFl_T5waloNRm", output=str(temporary), quiet=False, use_cookies=False)
            if hashlib.sha384(temporary.read_bytes()).hexdigest() != OPENVINO_SHA384:
                raise ValueError("vehicle-reid-0001 original-source checksum mismatch")
            temporary.replace(location / "model.onnx")
        if hashlib.sha384((location / "model.onnx").read_bytes()).hexdigest() != OPENVINO_SHA384:
            raise ValueError("vehicle-reid-0001 checksum mismatch; replace model.onnx with the official download")
        from .reid_vehicle_encoders import OpenVINOVehicle
        OpenVINOVehicle(location)  # Validate runtime and the graph before publishing availability.
        policy = "onnx-rgb-0-255-bilinear208-fp32-cpu-l2-v1"
        revision = "omz-public-2022.1"
    else:
        revision = TRANSREID_REVISION
        archive = location / "source.zip"
        download(f"https://codeload.github.com/damo-cv/TransReID/zip/{revision}", archive)
        # Re-extract the pinned inference sources on retry to repair partial setup.
        with zipfile.ZipFile(archive) as package:
            for item in package.infolist():
                parts = item.filename.split("/")[1:]
                if item.is_dir() or not parts:
                    continue
                if parts[0] not in ("model", "config", "loss", "LICENSE") and parts != ["configs", "VeRi", "vit_transreid_stride.yml"]:
                    continue
                target = (location / "source").joinpath(*parts).resolve()
                if not target.is_relative_to((location / "source").resolve()):
                    raise ValueError("Invalid TransReID source archive path")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(package.read(item))
        backbone = location / "source/model/backbones/vit_pytorch.py"
        backbone.write_text(backbone.read_text(encoding="utf-8").replace("from torch._six import container_abcs", "import collections.abc as container_abcs"), encoding="utf-8")
        shutil.copyfile(location / "source/configs/VeRi/vit_transreid_stride.yml", location / "vehicle.yml")
        checkpoint = location / "model.pth"
        temporary = location / "model.download"
        from .reid_vehicle_encoders import transreid_model
        if local_checkpoint:
            shutil.copyfile(local_checkpoint, temporary)
            _validate_checkpoint(temporary)
            transreid_model(location, temporary)
            temporary.replace(checkpoint)
        elif not checkpoint.exists():
            import gdown
            try:
                gdown.download(id=TRANSREID_DRIVE_ID, output=str(temporary), quiet=False, use_cookies=False)
                _validate_checkpoint(temporary)
                transreid_model(location, temporary)
                temporary.replace(checkpoint)
            except Exception as exc:
                raise RuntimeError("Could not install the official TransReID VeRi checkpoint. Download it from the upstream model link and rerun with -TransReIDCheckpoint <path>.") from exc
        transreid_model(location)  # Strict architecture/weight compatibility check.
        policy = "veri-vit-base-stride12-rgb256-bilinear-meanstd0.5-jpm-before-localdiv4-sie-bypass-fp32-l2-v1"
    files = {p.relative_to(location).as_posix(): digest(p) for p in location.rglob("*") if p.is_file() and (p.suffix in (".onnx", ".pth", ".yml", ".py") or p.name == "LICENSE")}
    return {"revision": revision, "checksums": files, "preprocessing": policy,
            "fingerprint": fingerprint([BASELINES[key], revision, files, policy])}


def _validate_checkpoint(path):
    import torch
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state.get("model", state))
        state = {k.removeprefix("module."): v for k, v in state.items()}
        if tuple(state["base.pos_embed"].shape) != (1, 442, 768) or "b2.0.attn.qkv.weight" not in state:
            raise ValueError("Expected the vehicle TransReID ViT stride-12 checkpoint with JPM")
    except Exception as exc:
        raise ValueError("Invalid TransReID checkpoint; select the official VeRi TransReID ViT model, not a person or ImageNet checkpoint") from exc
