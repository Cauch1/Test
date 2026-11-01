"""Metric calculators. Each metric type is a function; the spec YAML picks the
type and parameters, so a new spec needs YAML only, not code.

Time in the DataFrame is seconds (column 'time_s'); every time in the summary
is minutes, because that is the unit the report uses. The model is told not to
convert units, so the summary must already hold the numbers the report needs.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from .channels import ChannelInfo, select
from .schema import Metric, Phase, TimeWindow


def group_mean(df: pd.DataFrame, names: list[str]) -> pd.Series:
    """Row-wise mean of the selected channels, ignoring NaNs."""
    return df[names].mean(axis=1, skipna=True)


def resolve_window(phases: list[Phase], window: dict) -> Optional[TimeWindow]:
    """Turn {phase: stabilised, last_min: 30} into absolute start/end minutes.

    Returns None when the phase does not exist in this run.
    """
    match = [p for p in phases if p.id == window["phase"]]
    if not match:
        return None
    start, end = match[0].start_min, match[0].end_min
    if "last_min" in window:
        start = max(start, end - window["last_min"])
    if "first_min" in window:
        end = min(end, start + window["first_min"])
    return TimeWindow(start_min=start, end_min=end)


def _mask(df: pd.DataFrame, win: TimeWindow) -> pd.Series:
    t_min = df["time_s"] / 60.0
    return (t_min >= win.start_min) & (t_min <= win.end_min)


def time_to_threshold(df, channels, phases, p) -> tuple[Optional[float], Optional[TimeWindow], str]:
    """Minutes from the start of from_phase until the group mean crosses threshold."""
    names = select(channels, p["channels"])
    phase = next((ph for ph in phases if ph.id == p["from_phase"]), None)
    method = (f"Minutes from {p['from_phase']} start until mean of {p['channels']} "
              f"first goes {p['direction']} {p['threshold']} "
              f"({p.get('smoothing_s', 60)} s centred mean, linear interpolation)")
    if phase is None:
        return None, None, method

    t = df["time_s"].to_numpy() / 60.0
    # Smooth before the crossing test: with raw noisy data the first sample
    # past the threshold is reached early (first-passage bias).
    dt_s = float(np.median(np.diff(df["time_s"])))
    win = max(1, int(round(p.get("smoothing_s", 60) / dt_s)))
    y = group_mean(df, names).rolling(win, center=True, min_periods=1).mean().to_numpy()
    after = t >= phase.start_min
    crossed = (y <= p["threshold"]) if p["direction"] == "below" else (y >= p["threshold"])
    idx = np.flatnonzero(after & crossed)
    if idx.size == 0:
        return None, TimeWindow(start_min=phase.start_min, end_min=t[-1]), method

    i = int(idx[0])
    t_cross = t[i]
    if i > 0 and y[i] != y[i - 1]:
        # Interpolate between the last sample outside and the first inside the threshold.
        t_cross = t[i - 1] + (p["threshold"] - y[i - 1]) * (t[i] - t[i - 1]) / (y[i] - y[i - 1])
    return t_cross - phase.start_min, TimeWindow(start_min=phase.start_min, end_min=t_cross), method


def window_mean(df, channels, phases, p):
    """Time-mean of the group mean over a window."""
    names = select(channels, p["channels"])
    win = resolve_window(phases, p["window"])
    method = f"Mean of {p['channels']} over {p['window']}"
    if win is None:
        return None, None, method
    return float(group_mean(df, names)[_mask(df, win)].mean()), win, method


def group_difference(df, channels, phases, p):
    """Mean of group a minus mean of group b over a window."""
    a, b = select(channels, p["a"]), select(channels, p["b"])
    win = resolve_window(phases, p["window"])
    method = f"Mean of {p['a']} minus mean of {p['b']} over {p['window']}"
    if win is None:
        return None, None, method
    m = _mask(df, win)
    return float(group_mean(df, a)[m].mean() - group_mean(df, b)[m].mean()), win, method


METRIC_TYPES: dict[str, Callable] = {
    "time_to_threshold": time_to_threshold,
    "window_mean": window_mean,
    "group_difference": group_difference,
}


def compute_metrics(df: pd.DataFrame, channels: list[ChannelInfo],
                    phases: list[Phase], spec: dict) -> list[Metric]:
    out = []
    for name, p in spec["metrics"].items():
        if p["type"] not in METRIC_TYPES:
            raise ValueError(f"Unknown metric type '{p['type']}' for metric '{name}'")
        value, win, method = METRIC_TYPES[p["type"]](df, channels, phases, p)
        out.append(Metric(name=name, value=value, unit=p["unit"], method=method, window=win))
    return out
