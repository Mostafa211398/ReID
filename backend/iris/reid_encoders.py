"""Optional heavy dependencies are imported only inside the isolated worker."""
from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np


class Encoder:
    def __init__(self, spec: dict, model_root: Path, device: str = "auto"):
        os.environ["HF_HOME"] = str(model_root / ".cache")
        import torch
        self.torch = torch
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device)
        self.spec = spec
        self.family = spec["family"]
        if self.family not in ("siglip", "dinov2", "fastreid", "openvino", "transreid", "coca", "coca_visual",
                               "coca_l14", "coca_l14_visual", "siglip2", "siglip2_visual"):
            raise ValueError("Unsupported ReID encoder family: " + self.family)
        location = model_root / self.family
        from .reid_setup import digest
        for filename, checksum in spec.get("checksums", {}).items():
            if digest(location / filename) != checksum:
                raise ValueError(f"Installed {self.family} file failed checksum verification: {filename}")
        if spec.get("checkpoint") and digest(spec["checkpoint"]) != spec.get("checksum"):
            raise ValueError("Trained encoder checksum does not match its manifest")
        if self.family in ("siglip", "dinov2", "siglip2", "siglip2_visual"):
            from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, SiglipVisionModel
            self.processor = AutoImageProcessor.from_pretrained(str(location), local_files_only=True, use_fast=False)
            if self.family == "siglip2":
                # The fixed-resolution SigLIP2 checkpoints use a Gemma tokenizer;
                # AutoProcessor in Transformers 4.49 incorrectly forces SiglipTokenizer.
                self.tokenizer = AutoTokenizer.from_pretrained(str(location), local_files_only=True, use_fast=False)
            cls = SiglipVisionModel if self.family in ("siglip", "siglip2_visual") else AutoModel
            self.model = cls.from_pretrained(str(location), local_files_only=True, use_safetensors=True)
        elif self.family in ("coca", "coca_l14"):
            import open_clip
            checkpoint = location / "open_clip_pytorch_model.bin"
            self.model, _, self.processor = open_clip.create_model_and_transforms(
                spec["model_name"], pretrained=str(checkpoint), device=str(self.device)
            )
            self.tokenizer = open_clip.get_tokenizer(spec["model_name"])
        elif self.family in ("coca_visual", "coca_l14_visual"):
            import json
            from open_clip.model import CLIPVisionCfg, _build_vision_tower
            from open_clip.transform import PreprocessCfg, image_transform_v2
            from safetensors.torch import load_file
            config = json.loads((location / "config.json").read_text(encoding="utf-8"))
            self.model = _build_vision_tower(config["embed_dim"], CLIPVisionCfg(**config["vision_cfg"]))
            self.model.load_state_dict(load_file(location / "model.safetensors"), strict=True)
            preprocess = {**config["preprocess"], "size": tuple(config["preprocess"]["size"])}
            self.processor = image_transform_v2(PreprocessCfg(**preprocess), is_train=False)
        elif self.family == "openvino":
            from .reid_vehicle_encoders import OpenVINOVehicle
            self.adapter = OpenVINOVehicle(location)
            self.device = torch.device("cpu")
            return
        elif self.family == "transreid":
            from .reid_vehicle_encoders import transreid_model
            self.model = transreid_model(location)
        elif self.family == "fastreid":
            sys.path.insert(0, str(model_root / "fast-reid"))
            from fastreid.config import get_cfg
            from fastreid.modeling import build_model
            cfg = get_cfg()
            cfg.merge_from_file(str(model_root / "fast-reid/configs/VeRi/sbs_R50-ibn.yml"))
            cfg.MODEL.BACKBONE.PRETRAIN = False
            cfg.MODEL.HEADS.NUM_CLASSES = 575  # Official VeRi release classifier size.
            cfg.MODEL.DEVICE = str(self.device)
            self.model = build_model(cfg)
            # Only official, checksummed weights downloaded by setup enter this path.
            state = torch.load(location / "model.pth", map_location="cpu", weights_only=True)
            state = state.get("model", state)
            # Official v0.1.1 release predates the pinned source's head rename.
            if "heads.classifier.weight" in state:
                state["heads.weight"] = state.pop("heads.classifier.weight")
            for key in ("pixel_mean", "pixel_std", "heads.bnneck.num_batches_tracked"):
                state.pop(key, None)
            self.model.load_state_dict(state, strict=True)
        if spec.get("checkpoint"):
            from safetensors.torch import load_file
            self.model.load_state_dict(load_file(spec["checkpoint"]), strict=True)
        self.model.float().to(self.device).eval()

    def inputs(self, images):
        if self.family == "transreid":
            from .reid_vehicle_encoders import rgb_pixels
            return self.torch.from_numpy(rgb_pixels(images, 256) / 127.5 - 1).to(self.device)
        if self.family in ("coca", "coca_visual", "coca_l14", "coca_l14_visual"):
            return self.torch.stack([self.processor(image.convert("RGB")) for image in images]).to(self.device)
        if self.family != "fastreid":
            return self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
        # FastReID VeRi configuration uses RGB 0..255 and normalizes in Baseline.
        return self.torch.stack([self.torch.from_numpy(np.asarray(image.resize((256, 256))).copy()).permute(2, 0, 1).float() for image in images]).to(self.device)

    def features(self, pixels):
        if self.family == "siglip":
            vectors = self.model(pixel_values=pixels).pooler_output
        elif self.family == "siglip2":
            vectors = self.model.get_image_features(pixel_values=pixels)
        elif self.family == "siglip2_visual":
            vectors = self.model(pixel_values=pixels).pooler_output
        elif self.family == "dinov2":
            vectors = self.model(pixel_values=pixels).last_hidden_state[:, 0]
        elif self.family in ("coca", "coca_l14"):
            vectors = self.model.encode_image(pixels, normalize=True)
        elif self.family in ("coca_visual", "coca_l14_visual"):
            vectors = self.model(pixels)
            if isinstance(vectors, tuple):
                vectors = vectors[0]
        else:
            # Keep heads in inference mode even while gradients update the backbone.
            vectors = self.model(pixels.clone())
        return self.torch.nn.functional.normalize(vectors.float(), dim=-1)

    def encode(self, images):
        if self.family == "openvino":
            return self.adapter.encode(images)
        with self.torch.inference_mode():
            result = self.features(self.inputs(images)).cpu().numpy()
        if not np.isfinite(result).all():
            raise RuntimeError("Encoder produced non-finite features; inference requires FP32")
        if np.any(np.linalg.norm(result, axis=1) < .99):
            raise RuntimeError("Encoder produced an empty feature vector")
        return result

    def encode_with_semantics(self, images, groups, vector_groups=None):
        if self.family not in ("coca", "coca_l14", "siglip2"):
            raise ValueError("Semantic pair features require a full vision-language encoder")
        labels, prompts, ranges = [], [], []
        for group, values in groups.items():
            start = len(prompts)
            labels.extend((group, label) for label, _ in values)
            prompts.extend(prompt for _, prompt in values)
            ranges.append((group, start, len(prompts)))
        with self.torch.inference_mode():
            appearance = self.features(self.inputs(images))
            if self.family in ("coca", "coca_l14"):
                tokens = self.tokenizer(prompts).to(self.device)
                text = self.model.encode_text(tokens, normalize=True).float()
            else:
                tokens = self.tokenizer(prompts, padding="max_length", truncation=True,
                                        max_length=self.model.config.text_config.max_position_embeddings,
                                        return_tensors="pt")
                tokens = {key: value.to(self.device) for key, value in tokens.items()}
                text = self.torch.nn.functional.normalize(self.model.get_text_features(**tokens).float(), dim=-1)
            scale = float(self.model.logit_scale.exp().detach().clamp(max=100))
            semantic_parts, attributes = [], [dict() for _ in images]
            for group, start, end in ranges:
                probabilities = self.torch.softmax(scale * appearance @ text[start:end].T, dim=-1)
                if vector_groups is None or group in vector_groups:
                    semantic_parts.append(probabilities)
                group_labels = [label for current, label in labels[start:end] if current == group]
                for index, row in enumerate(probabilities):
                    best = int(row.argmax())
                    attributes[index][group] = {"label": group_labels[best], "confidence": float(row[best]),
                                                "distribution": {label: float(row[position]) for position, label in enumerate(group_labels)}}
            semantic = self.torch.nn.functional.normalize(self.torch.cat(semantic_parts, dim=-1), dim=-1)
        return appearance.cpu().numpy(), semantic.cpu().numpy(), attributes

    def runtime_metrics(self):
        return {"device": str(self.device), "peak_vram_mb": self.torch.cuda.max_memory_allocated() / 2**20 if self.device.type == "cuda" else 0}

    def trainable_tail(self):
        if self.family in ("openvino", "transreid", "coca", "coca_visual", "coca_l14", "coca_l14_visual",
                           "siglip2", "siglip2_visual"):
            raise ValueError("This encoder supports inference and evaluation only; fine-tuning is unavailable")
        for param in self.model.parameters():
            param.requires_grad_(False)
        if self.family == "siglip":
            tail = self.model.vision_model.encoder.layers[-1]
        elif self.family == "dinov2":
            tail = self.model.encoder.layer[-1]
        else:
            tail = self.model.backbone.layer4
        for param in tail.parameters():
            param.requires_grad_(True)
        # eval() avoids corrupting frozen batchnorm statistics with small batches.
        return list(tail.parameters())
