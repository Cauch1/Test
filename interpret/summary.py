"""Stage A: DataFrame of mapped channels -> validated Summary -> summary.json."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel

from .anomalies import detect_anomalies, suspect_spans
from .channels import ChannelInfo, select
from .criteria import evaluate_criteria
from .metrics import compute_metrics, group_mean
from .phases import detect_phases
from .schema import (ChannelQuality, ChannelSummary, Counts, Summary, TestInfo)

DECIMALS = 2  # every float in the summary is rounded to this many places


def _r(x):
    """Round floats (and None-safe) so the model sees the same digits the report uses."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), DECIMALS)


def _channel_summary(df, ch: ChannelInfo, phases) -> ChannelSummary:
    s = df[ch.name]
    t_min = df["time_s"] / 60.0
    phase_means = {}
    for p in phases:
        m = (t_min >= p.start_min) & (t_min < p.end_min)
        phase_means[p.id] = _r(s[m].mean())
    return ChannelSummary(
        name=ch.name, unit=ch.unit, location_group=ch.location_group,
        min=_r(s.min()), max=_r(s.max()), phase_means=phase_means,
        quality=ChannelQuality(nan_fraction=_r(s.isna().mean())),
    )


def _round_model(model):
    """Round every float field in place (metrics, criteria, phases, windows)."""
    for name, value in model:
        if isinstance(value, float):
            setattr(model, name, _r(value))
        elif isinstance(value, BaseModel):
            _round_model(value)
    return model


def build_summary(df: pd.DataFrame, channels: list[ChannelInfo], *, test_id: str,
                  facility: str, run_date: date, spec: dict, phase_cfg: dict,
                  anomaly_cfg: dict | None = None) -> Summary:
    """Compute everything the commentary may quote. df needs 'time_s' plus one
    column per ChannelInfo.name."""
    missing = [c.name for c in channels if c.name not in df.columns]
    if missing:
        raise ValueError(f"Channels missing from data: {missing}")

    ref = group_mean(df, select(channels, spec["cabin_reference"]))
    phases = detect_phases(df["time_s"], ref, phase_cfg)
    anomalies, notes, suspect, auto = [], [], {}, False
    if anomaly_cfg:
        anomalies, notes = detect_anomalies(df, channels, anomaly_cfg, phases)
        suspect = suspect_spans(anomalies, anomaly_cfg.get("suspect_types", []))
        auto = bool(anomaly_cfg.get("auto_exclude", False))
    metrics = compute_metrics(df, channels, phases, spec, suspect)
    criteria = evaluate_criteria(metrics, spec, auto_exclude=auto)

    t = df["time_s"].to_numpy()
    summary = Summary(
        test=TestInfo(test_id=test_id, facility=facility, spec_ref=spec["spec_ref"],
                      run_date=run_date, phases=phases),
        counts=Counts(n_channels=len(channels), n_samples=len(df),
                      duration_min=(t[-1] - t[0]) / 60,
                      sample_interval_s=float(np.median(np.diff(t)))),
        channels=[_channel_summary(df, c, phases) for c in channels],
        metrics=metrics,
        criteria=criteria,
        anomalies=anomalies,
        notes=notes,
    )
    for ch in summary.channels:
        ch.quality.flags = sorted({a.type for a in anomalies if a.channel in (ch.name, "all")})
    for group in (summary.test.phases, summary.metrics, summary.criteria):
        for item in group:
            _round_model(item)
    _round_model(summary.counts)
    # Re-validate after rounding so the written file is guaranteed schema-valid.
    return Summary.model_validate(summary.model_dump())


def write_summary(summary: Summary, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{summary.test.test_id}_summary.json"
    path.write_text(summary.model_dump_json(indent=2))
    return path


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())
