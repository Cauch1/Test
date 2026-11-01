"""Pydantic models for the Stage A summary JSON.

The summary is the only source of truth for numbers. Stage B (the model) reads
this file and nothing else, so every field here must be safe to send: test ID
and generic channel names only, no client names and no raw samples.
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Facility = Literal["CWT1", "CWT2", "VTEOS", "VCC"]
PhaseName = Literal["soak", "pull-down", "stabilised", "heat-up"]
Status = Literal["PASS", "FAIL"]


class _Strict(BaseModel):
    # extra="forbid" makes pydantic reject unknown keys, so a typo in a field
    # name fails loudly instead of silently disappearing from the JSON.
    model_config = ConfigDict(extra="forbid")


class Phase(_Strict):
    id: str = Field(description="Unique key, for example 'pull-down' or 'pull-down_2'")
    name: PhaseName
    start_min: float
    end_min: float


class TestInfo(_Strict):
    test_id: str
    facility: Facility
    spec_ref: str
    run_date: date
    phases: list[Phase]


class Counts(_Strict):
    """Explicit counts, so the number validator can trace them in the commentary."""

    n_channels: int
    n_samples: int
    duration_min: float
    sample_interval_s: float


class ChannelQuality(_Strict):
    nan_fraction: float
    flags: list[str] = Field(default_factory=list)


class ChannelSummary(_Strict):
    name: str
    unit: str
    location_group: str
    min: Optional[float]
    max: Optional[float]
    phase_means: dict[str, Optional[float]] = Field(description="Keyed by Phase.id")
    quality: ChannelQuality


class TimeWindow(_Strict):
    start_min: float
    end_min: float


class Metric(_Strict):
    name: str
    value: Optional[float] = Field(description="None when the metric could not be computed")
    unit: str
    method: str
    window: Optional[TimeWindow]


class Criterion(_Strict):
    name: str
    metric: str
    comparator: Literal["<=", ">=", "<", ">"]
    use_abs: bool
    limit: float
    unit: str
    measured: Optional[float]
    margin: Optional[float] = Field(description="Positive means inside the limit")
    status: Status
    note: Optional[str] = None


class Anomaly(_Strict):
    id: str
    channel: str
    type: Literal["flatline", "drift", "spike", "missing", "out_of_range"]
    start_min: float
    end_min: float
    evidence: dict[str, float]


class Summary(_Strict):
    schema_version: str = "1.0"
    test: TestInfo
    counts: Counts
    channels: list[ChannelSummary]
    metrics: list[Metric]
    criteria: list[Criterion]
    anomalies: list[Anomaly] = Field(default_factory=list)
