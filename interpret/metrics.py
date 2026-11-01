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


def stratified_mean(df: pd.DataFrame, names: list[str], channels: list[ChannelInfo],
                    by: list[str] | None) -> pd.Series:
    """Mean within each stratum (for example each row), then mean across strata.

    With a 3 degC front-to-rear gradient, a plain mean shifts by about 0.5 degC
    when one front sensor is excluded, because the rear row then has more
    weight. Averaging per row first keeps each row's weight fixed.
    """
    if not by:
        return group_mean(df, names)
    info = {c.name: c for c in channels}
    strata: dict[tuple, list[str]] = {}
    for n in names:
        strata.setdefault(tuple(getattr(info[n], k) for k in by), []).append(n)
    return pd.concat([group_mean(df, ns) for ns in strata.values()], axis=1).mean(axis=1)


def resolve_window(phases: list[Phase], window: dict) -> Optional[TimeWindow]:
    """Turn a window spec into absolute start/end minutes. Two forms:
    {phase: stabilised, last_min: 30} (or first_min), relative to a detected phase;
    {start_min: 60, end_min: 70}, absolute minutes from the start of the record.

    Returns None when the phase does not exist in this run.
    """
    if "phase" not in window:
        return TimeWindow(start_min=window["start_min"], end_min=window["end_min"])
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
    y = stratified_mean(df, names, channels, p.get("stratify_by"))
    y = y.rolling(win, center=True, min_periods=1).mean().to_numpy()
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
    """Time-mean of the (optionally stratified) group mean over a window."""
    names = select(channels, p["channels"])
    win = resolve_window(phases, p["window"])
    by = p.get("stratify_by")
    method = f"Mean of {p['channels']} over {p['window']}" + (f", averaged per {by} first" if by else "")
    if win is None:
        return None, None, method
    return float(stratified_mean(df, names, channels, by)[_mask(df, win)].mean()), win, method


def group_difference(df, channels, phases, p):
    """Group a minus group b over a window.

    With pair_by (for example [row, side]) each a channel is matched with the b
    channel at the same location and the result is the mean of the pairwise
    differences. Without pairing, dropping one channel from an unbalanced layout
    shifts a group mean by the row-to-row offset; pairing removes the matching
    partner too, so the gradient stays unbiased.
    """
    a, b = select(channels, p["a"]), select(channels, p["b"])
    win = resolve_window(phases, p["window"])
    pair_by = p.get("pair_by")
    how = f"paired by {pair_by}, mean of differences" if pair_by else "difference of group means"
    method = f"{p['a']} minus {p['b']} over {p['window']}, {how}"
    if win is None:
        return None, None, method
    m = _mask(df, win)
    if not pair_by:
        return float(group_mean(df, a)[m].mean() - group_mean(df, b)[m].mean()), win, method

    info = {c.name: c for c in channels}
    key = lambda n: tuple(getattr(info[n], k) for k in pair_by)
    b_by_key = {key(n): n for n in b}
    pairs = [(n, b_by_key[key(n)]) for n in a if key(n) in b_by_key]
    if not pairs:
        raise ValueError(f"No {pair_by} pairs between {p['a']} and {p['b']}")
    diffs = [df[x][m].mean() - df[y][m].mean() for x, y in pairs]
    return float(np.mean(diffs)), win, method


METRIC_TYPES: dict[str, Callable] = {
    "time_to_threshold": time_to_threshold,
    "window_mean": window_mean,
    "group_difference": group_difference,
}


def _used_channels(channels: list[ChannelInfo], p: dict) -> set[str]:
    """Every channel any selector in this metric picks."""
    used: set[str] = set()
    for key in ("channels", "a", "b"):
        if key in p:
            used.update(select(channels, p[key]))
    return used


def compute_metrics(df: pd.DataFrame, channels: list[ChannelInfo], phases: list[Phase],
                    spec: dict, suspect: dict | None = None) -> list[Metric]:
    """suspect maps channel -> [(start_min, end_min)] anomaly spans.

    value is always the all-channel result. When a used channel's anomaly span
    overlaps the metric window, value_excluding_suspect is computed as well;
    the criteria step decides what to do with the two.
    """
    suspect = suspect or {}
    out = []
    for name, p in spec["metrics"].items():
        if p["type"] not in METRIC_TYPES:
            raise ValueError(f"Unknown metric type '{p['type']}' for metric '{name}'")
        fn = METRIC_TYPES[p["type"]]
        value, win, method = fn(df, channels, phases, p)

        flagged: list[str] = []
        if win is not None:
            used = _used_channels(channels, p)
            flagged = sorted(ch for ch, spans in suspect.items() if ch in used and any(
                s <= win.end_min and e >= win.start_min for s, e in spans))
        value_excl = None
        if flagged:
            kept = [c for c in channels if c.name not in flagged]
            try:
                value_excl, _, _ = fn(df, kept, phases, p)
            except ValueError:  # every channel in a group was suspect
                value_excl = None

        out.append(Metric(name=name, value=value, unit=p["unit"], method=method, window=win,
                          suspect_channels=flagged, value_excluding_suspect=value_excl))
    return out
