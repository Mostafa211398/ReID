"""Public contracts for the optional vehicle identity workspace."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .utils import utc_now


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)


class CameraSpec(Contract):
    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    name: str = Field(min_length=1, max_length=80)


class Transition(Contract):
    source: str
    destination: str
    min_seconds: float = Field(default=0, ge=0)
    max_seconds: float = Field(default=3600, gt=0)

    @model_validator(mode="after")
    def ordered(self):
        if self.max_seconds < self.min_seconds:
            raise ValueError("Maximum travel time must follow minimum travel time")
        return self


class MatchingSettings(Contract):
    mode: Literal["review", "automatic"] = "automatic"
    threshold: float = Field(default=.85, ge=-1, le=1)
    margin: float = Field(default=.05, ge=0, le=2)
    new_threshold: float = Field(default=.65, ge=-1, le=1)
    color_weight: float = Field(default=.25, ge=0, le=.5)

    @model_validator(mode="after")
    def ordered(self):
        if self.new_threshold > self.threshold:
            raise ValueError("New-vehicle threshold must not exceed the match threshold")
        return self


class SiteInput(Contract):
    name: str = Field(min_length=1, max_length=100)
    cameras: list[CameraSpec] = Field(default_factory=list, max_length=100)
    transitions: list[Transition] = Field(default_factory=list, max_length=1000)
    matching: MatchingSettings | None = None

    @model_validator(mode="after")
    def unique_cameras(self):
        ids = [c.id for c in self.cameras]
        if len(ids) != len(set(ids)):
            raise ValueError("Camera IDs must be unique within a site")
        if any(t.source not in ids or t.destination not in ids for t in self.transitions):
            raise ValueError("Transitions must reference registered cameras")
        return self


class Site(SiteInput):
    id: str = Field(default_factory=lambda: uuid4().hex)
    active_encoder: str = "dinov2"
    created_at: datetime = Field(default_factory=utc_now)


class ClipInput(Contract):
    camera_id: str
    media_type: Literal["image"] = "image"
    class_name: Literal["car", "truck"]
    start_time: datetime | None = None
    offset_seconds: float = 0

    @model_validator(mode="after")
    def timezone_required(self):
        if self.start_time is not None and self.start_time.tzinfo is None:
            raise ValueError("Recording start time requires a timezone (for example +02:00)")
        return self


class ExperimentInput(Contract):
    name: str = Field(min_length=1, max_length=100)
    site_id: str
    encoders: list[str] = Field(default_factory=lambda: ["siglip", "dinov2", "fastreid"], min_length=1, max_length=11)
    clips: list[ClipInput] = Field(min_length=1, max_length=32)


class Job(Contract):
    id: str = Field(default_factory=lambda: uuid4().hex)
    kind: Literal["experiment", "training", "promotion", "evaluation", "benchmark", "benchmark_export", "comparison"] = "experiment"
    name: str
    site_id: str
    state: Literal["queued", "running", "completed", "failed", "cancelled"] = "queued"
    stage: str = "queued"
    progress: float = 0
    config: dict[str, Any] = Field(default_factory=dict)
    completed_stages: list[str] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None
    refresh_required: bool = False
    deletion_pending: bool = False
    deletion_error: str | None = None


class AnnotationInput(Contract):
    identity: str | None = Field(default=None, min_length=1, max_length=100)
    excluded: bool = False


class ReviewInput(Contract):
    global_id: str | None = Field(default=None, pattern=r"^(TRUCK|VEHICLE)_[0-9]+$")
    new_identity: bool = False

    @model_validator(mode="after")
    def exactly_one(self):
        if bool(self.global_id) == self.new_identity:
            raise ValueError("Choose an existing vehicle or create a new identity")
        return self


class DatasetInput(Contract):
    site_id: str
    experiment_ids: list[str] = Field(min_length=1)


class EvaluationInput(Contract):
    dataset_id: str
    encoder_id: str
    split: Literal["validation", "test"] = "validation"


class BenchmarkFileInput(Contract):
    relative_path: str = Field(min_length=3, max_length=500)


class BenchmarkInput(Contract):
    name: str = Field(min_length=1, max_length=100)
    site_id: str
    encoders: list[str] = Field(default_factory=lambda: ["siglip", "dinov2", "fastreid", "openvino", "transreid"], min_length=1, max_length=11)
    threshold: float | None = Field(default=None, ge=-1, le=1)
    items: list[BenchmarkFileInput] = Field(min_length=2, max_length=500)


class BenchmarkThresholdInput(Contract):
    threshold: float = Field(ge=-1, le=1)


class PairWeights(Contract):
    appearance: float = Field(default=.60, ge=0, le=1)
    color: float = Field(default=.20, ge=0, le=1)
    shape: float = Field(default=.15, ge=0, le=1)
    semantic: float = Field(default=.05, ge=0, le=1)

    @model_validator(mode="after")
    def totals_one(self):
        if abs(self.appearance + self.color + self.shape + self.semantic - 1.) > 1e-6:
            raise ValueError("Comparison weights must add up to 1")
        return self


class PairComparisonInput(Contract):
    name: str = Field(min_length=1, max_length=100)
    site_id: str
    encoder_id: Literal["coca", "coca_visual", "coca_l14", "coca_l14_visual", "siglip2", "siglip2_visual"] = "coca"
    threshold: float = Field(default=.75, ge=-1, le=1)
    weights: PairWeights = Field(default_factory=PairWeights)


class PairScoringInput(Contract):
    threshold: float = Field(default=.75, ge=-1, le=1)
    weights: PairWeights = Field(default_factory=PairWeights)


class TrainingInput(Contract):
    dataset_id: str
    encoder_id: str = "dinov2"
    epochs: int = Field(default=20, ge=1, le=300)
    learning_rate: float = Field(default=0.0001, gt=0, le=0.01)
    accumulation: int = Field(default=4, ge=1, le=64)
    patience: int = Field(default=5, ge=1, le=50)
    seed: int = 42


class PromotionInput(Contract):
    encoder_id: str


class RetryInput(Contract):
    encoders: list[str] | None = Field(default=None, min_length=1, max_length=11)
