"""Persistent configurations, queued runs, responses and metric analyses."""

from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, StringConstraints, field_validator, model_validator
from sqlalchemy import Column, JSON, UniqueConstraint
from sqlmodel import Field, SQLModel

from .settings import orbital_path

Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
RunID = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Schema(SQLModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    @field_validator("*", mode="after")
    @classmethod
    def validate_json_numbers(cls, value):
        # Reject NaN/Infinity in nested options, metrics and provenance too.
        if isinstance(value, (dict, list)):
            import json
            json.dumps(value, allow_nan=False)
        if isinstance(value, datetime) and value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


class ModelConfigCreate(Schema):
    name: Name
    version: Name = "v1"
    model_id: Text
    backend: Name
    revision: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    seed: int = Field(default=42, ge=0)
    max_new_tokens: int = Field(default=128, gt=0)
    enabled: bool = False

    @field_validator("options")
    @classmethod
    def protect_archive(cls, values):
        from pathlib import PureWindowsPath
        for key, value in values.items():
            if key.endswith(("_path", "_dir", "_repo")) and isinstance(value, str):
                if "data_rover" in {part.casefold() for part in PureWindowsPath(value).parts}:
                    raise ValueError("Model assets cannot use the protected archive")
        return values


class ModelConfig(ModelConfigCreate, table=True):
    __tablename__ = "model_configs"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_model_config_version"),)
    id: int | None = Field(default=None, primary_key=True)
    options: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    config_sha256: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ModelConfigRead(ModelConfigCreate):
    id: int
    config_sha256: str
    created_at: datetime
    updated_at: datetime


class DatasetConfigCreate(Schema):
    name: Name
    version: Name = "v1"
    planet: Literal["moon", "mars"]
    task_id: Name
    manifest_path: Text
    requests_path: str | None = None
    targets_path: str | None = None
    splits: list[Literal["train", "val", "test"]] = Field(default_factory=lambda: ["val", "test"])
    conditions: list[Name] = Field(default_factory=lambda: ["clean"])
    preprocessing: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("manifest_path", "requests_path", "targets_path")
    @classmethod
    def validate_path(cls, value):
        return str(orbital_path(value)) if value is not None else None

    @field_validator("splits", "conditions")
    @classmethod
    def distinct_nonempty(cls, value):
        if not value or len(value) != len(set(value)):
            raise ValueError("Values must be nonempty and distinct")
        return value


class DatasetConfig(DatasetConfigCreate, table=True):
    __tablename__ = "dataset_configs"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_dataset_config_version"),)
    id: int | None = Field(default=None, primary_key=True)
    planet: str
    splits: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    conditions: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    preprocessing: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    provenance: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    config_sha256: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class DatasetConfigRead(DatasetConfigCreate):
    id: int
    config_sha256: str
    created_at: datetime
    updated_at: datetime


class BuiltinDatasetRequest(Schema):
    subset: Literal["segmentation", "height", "imp"] = "segmentation"


class ModelResultCreate(Schema):
    model_config_id: int = Field(gt=0)
    dataset_config_id: int = Field(gt=0)
    run_id: RunID
    request_id: Text
    source_sample_id: str | None = None
    scene_group_id: str | None = None
    condition_id: Name = "clean"
    source_split: Literal["train", "val", "test"] | None = None
    image_paths: list[str] = Field(default_factory=list)
    prompt: str = ""
    output_type: Literal["text", "imp_segmentation"] = "text"
    status: Literal["ok", "error", "unsupported"]
    raw_response: str = ""
    elapsed_seconds: float = Field(ge=0)
    error_message: str | None = None
    artifacts: dict[str, Any] = Field(default_factory=dict)
    run_metadata: dict[str, Any] = Field(default_factory=dict)
    mock: bool = False

    @field_validator("elapsed_seconds", mode="before")
    @classmethod
    def validate_timing(cls, value):
        if isinstance(value, bool):
            raise ValueError("elapsed_seconds must be a number, not a boolean")
        return value

    @field_validator("image_paths")
    @classmethod
    def image_paths_are_orbital(cls, values):
        return [str(orbital_path(value)) for value in values]


class ModelResult(ModelResultCreate, table=True):
    __tablename__ = "model_results"
    __table_args__ = (UniqueConstraint("model_config_id", "dataset_config_id", "run_id", "request_id",
                                     name="uq_model_result_request"),)
    id: int | None = Field(default=None, primary_key=True)
    model_config_id: int = Field(foreign_key="model_configs.id", index=True)
    dataset_config_id: int = Field(foreign_key="dataset_configs.id", index=True)
    run_id: str = Field(index=True)
    status: str
    output_type: str = "text"
    source_split: str | None = None
    image_paths: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    artifacts: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    run_metadata: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now)


class ModelResultRead(ModelResultCreate):
    id: int
    created_at: datetime


class MetricAnalysisCreate(Schema):
    model_config_id: int = Field(gt=0)
    dataset_config_id: int = Field(gt=0)
    run_id: RunID
    task_id: Name
    condition_id: str | None = None
    name: Name
    method_version: Name
    result_ids: list[int]
    metrics: dict[str, Any]
    notes: str = ""

    @field_validator("result_ids")
    @classmethod
    def validate_result_ids(cls, values):
        if not values or any(value < 1 for value in values) or len(set(values)) != len(values):
            raise ValueError("result_ids must be distinct positive IDs")
        return sorted(values)


class MetricAnalysis(MetricAnalysisCreate, table=True):
    __tablename__ = "metric_analyses"
    id: int | None = Field(default=None, primary_key=True)
    model_config_id: int = Field(foreign_key="model_configs.id", index=True)
    dataset_config_id: int = Field(foreign_key="dataset_configs.id", index=True)
    run_id: str = Field(index=True)
    result_ids: list[int] = Field(sa_column=Column(JSON, nullable=False))
    metrics: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
    results_sha256: str
    mock: bool
    created_at: datetime = Field(default_factory=utc_now)


class MetricAnalysisRead(MetricAnalysisCreate):
    id: int
    results_sha256: str
    mock: bool
    created_at: datetime


class ImportRun(Schema):
    model_config_id: int = Field(gt=0)
    dataset_config_id: int = Field(gt=0)
    run_id: RunID


Distortion = Literal["shear", "radiation", "lens_flare", "hard_shadows"]


class RunCreate(Schema):
    model_name: Name
    model_version: Name = "v1"
    dataset_id: int = Field(gt=0)
    image_count: int = Field(default=10, ge=1, le=10000)
    split: Literal["val", "test"] = "test"
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    include_clean: bool = True
    distortions: list[list[Distortion]] = Field(default_factory=lambda: [
        ["shear"], ["radiation"], ["lens_flare"], ["hard_shadows"],
        ["shear", "hard_shadows", "lens_flare", "radiation"]])
    parameters: dict[str, dict[str, Any]] = Field(default_factory=dict)
    prompt: Text = "Describe the visible surface features and state uncertainty."
    allowed_answers: list[Name] = Field(default_factory=list)
    reuse_run_id: RunID | None = None

    @model_validator(mode="after")
    def valid_protocol(self):
        if len(self.distortions) > 32:
            raise ValueError("At most 32 distortion variants")
        if not self.include_clean and not self.distortions:
            raise ValueError("Select clean images or at least one distortion")
        if any(not chain or len(set(chain)) != len(chain) for chain in self.distortions):
            raise ValueError("Each chain must be nonempty with distinct effects")
        if len({tuple(chain) for chain in self.distortions}) != len(self.distortions):
            raise ValueError("Duplicate distortion chains")
        from .transforms import validate_parameters
        validate_parameters(self.parameters)
        normalized = [" ".join(item.split()).casefold() for item in self.allowed_answers]
        if len(set(normalized)) != len(normalized):
            raise ValueError("Answer labels must be distinct")
        return self


class Run(SQLModel, table=True):
    __tablename__ = "runs"
    id: str = Field(primary_key=True)
    model_config_id: int = Field(foreign_key="model_configs.id", index=True)
    dataset_config_id: int = Field(foreign_key="dataset_configs.id", index=True)
    kind: str
    status: str = Field(default="queued", index=True)
    phase: str = "waiting"
    cancel_requested: bool = False
    total: int
    completed: int = 0
    error_message: str | None = None
    protocol: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
    samples: list[dict[str, Any]] = Field(sa_column=Column(JSON, nullable=False))
    protocol_sha256: str
    output_dir: str
    loading: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class DatasetURL(Schema):
    url: Text
    name: Name
    version: Name = "v1"
    planet: Literal["moon", "mars"]
    task_id: Name = "description"
    provenance: dict[str, Any] = Field(default_factory=dict)
    preprocessing: dict[str, Any] = Field(default_factory=dict)


class ComparisonCreate(Schema):
    run_ids: list[RunID] = Field(min_length=2, max_length=2)

    @field_validator("run_ids")
    @classmethod
    def different_runs(cls, values):
        if values[0] == values[1]:
            raise ValueError("Choose two different runs")
        return values
