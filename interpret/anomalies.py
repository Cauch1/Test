"""Python anomaly checks. Each check is a separate function returning a list of
Anomaly records; detect_anomalies() runs them all. Thresholds come from
config/anomaly_thresholds.yaml.

The model only explains what these functions find. Anything it adds itself is
labelled model-suggested and unverified in Stage B.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .channels import ChannelInfo
from .metrics import resolve_window
from .schema import Anomaly, Phase


# ---------- helpers ----------

def _samples(seconds: float, dt: float) -> int:
    return max(1, int(round(seconds / dt)))


def _dt(df: pd.DataFrame) -> float:
    return float(np.median(np.diff(df["time_s"])))


def _true_runs(mask) -> list[tuple[int, int]]:
    """Index ranges [start, end) where mask is True."""
    m = np.asarray(mask, dtype=bool).astype(np.int8)
    edges = np.diff(np.concatenate(([0], m, [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _duration_s(t: np.ndarray, i0: int, i1: int) -> float:
    return float(t[i1 - 1] - t[i0])


def _r(x: float) -> float:
    return round(float(x), 2)


def neighbours(ch: ChannelInfo, channels: list[ChannelInfo], group_by: list[str]) -> list[str]:
    """Other channels sharing every attribute in group_by."""
    return [c.name for c in channels
            if c.name != ch.name and all(getattr(c, a) == getattr(ch, a) for a in group_by)]


def _make(kind: str, ch: str, t: np.ndarray, i0: int, i1: int, evidence: dict) -> Anomaly:
    return Anomaly(id="", channel=ch, type=kind, start_min=_r(t[i0] / 60),
                   end_min=_r(t[i1 - 1] / 60), evidence={k: _r(v) for k, v in evidence.items()})


# ---------- checks ----------

def detect_flatline(df, ch: ChannelInfo, channels, cfg, group_by, phases=None) -> list[Anomaly]:
    """A channel that has stopped moving when its neighbours have not. Two conditions:
    1. Neighbours are changing (a transient) and the channel std is below max_std.
    2. Neighbours show ordinary sensor noise and the channel shows none at all
       (std below frozen_max_std). This catches a sensor frozen at a plausible
       value during a stable phase, which condition 1 and the drift check miss.
    """
    others = neighbours(ch, channels, group_by)
    if not others:
        return []
    t, dt = df["time_s"].to_numpy(), _dt(df)
    w = _samples(cfg["window_s"], dt)
    std = df[ch.name].rolling(w, center=True, min_periods=w // 2).std()
    nb_std = df[others].rolling(w, center=True, min_periods=w // 2).std().median(axis=1)
    moving = (std < cfg["max_std_degC"]) & (nb_std > cfg["neighbour_min_std_degC"])
    frozen = (std < cfg["frozen_max_std_degC"]) & (nb_std > cfg["neighbour_noise_floor_degC"])
    mask = moving | frozen

    out = []
    for i0, i1 in _true_runs(mask.to_numpy()):
        if _duration_s(t, i0, i1) >= cfg["min_duration_s"]:
            # A centred window only reads flat once it lies wholly in the flat
            # stretch, so widen the run by half a window at each end.
            # Evidence comes from the core run, before widening.
            evidence = {
                "channel_std_degC": std.iloc[i0:i1].max(),
                "neighbour_median_std_degC": nb_std.iloc[i0:i1].median(),
                "value_degC": df[ch.name].iloc[i0:i1].median(),
            }
            i0, i1 = max(0, i0 - w // 2), min(len(t), i1 + w // 2)
            out.append(_make("flatline", ch.name, t, i0, i1, evidence))
    return out


def detect_drift(df, ch: ChannelInfo, channels, cfg, group_by, phases=None) -> list[Anomaly]:
    """Change in a channel's offset from its neighbours, relative to a reference window.

    offset(t) = channel - median of the other neighbours (leave-one-out, so a
    bad channel cannot drag its own reference). Normal spatial gradients (for
    example a bus front-to-rear offset of several degC) are a constant offset
    and cancel out; only a change from the reference-window offset is flagged.
    Checked from the start of the reference window to the end of the run, so
    the developing gradient during a pull-down is not mistaken for drift.
    """
    others = neighbours(ch, channels, group_by)
    if len(others) < 2:
        return []
    ref = resolve_window(phases or [], cfg["reference_window"])
    if ref is None:
        return []
    t, dt = df["time_s"].to_numpy(), _dt(df)
    t_min = t / 60.0
    offset = df[ch.name] - df[others].median(axis=1)
    offset = offset.rolling(_samples(cfg["smoothing_s"], dt), center=True, min_periods=1).mean()
    in_ref = (t_min >= ref.start_min) & (t_min <= ref.end_min)
    baseline = float(offset[in_ref].median())
    change = offset - baseline
    mask = (change.abs() > cfg["threshold_degC"]) & (t_min >= ref.start_min)

    out = []
    for i0, i1 in _true_runs(mask.to_numpy()):
        if _duration_s(t, i0, i1) >= cfg["min_duration_s"]:
            seg = change.iloc[i0:i1]
            out.append(_make("drift", ch.name, t, i0, i1, {
                "baseline_offset_degC": baseline,
                "max_abs_offset_change_degC": seg.abs().max(),
                "mean_offset_change_degC": seg.mean(),
                "threshold_degC": cfg["threshold_degC"],
            }))
    return out


def detect_spike(df, ch: ChannelInfo, channels, cfg, group_by, phases=None) -> list[Anomaly]:
    """Excursion from a long centred rolling median that recovers within max_duration_s.

    An excursion at the very start or end of the record cannot be shown to
    recover, so it is not reported as a spike.

    A rolling median ignores short excursions and follows a monotonic pull-down
    exactly, so the residual is close to zero except during spikes.
    """
    t, dt = df["time_s"].to_numpy(), _dt(df)
    base = df[ch.name].rolling(_samples(cfg["baseline_window_s"], dt), center=True,
                               min_periods=1).median()
    resid = df[ch.name] - base
    mask = resid.abs() > cfg["threshold_degC"]

    y = df[ch.name].to_numpy()
    rw = _samples(cfg["recovery_window_s"], dt)
    out = []
    for i0, i1 in _true_runs(mask.to_numpy()):
        dur = _duration_s(t, i0, i1) + dt
        # Recovery test: a spike returns to where it started. The edges of a
        # long step also poke above the threshold briefly, but do not recover.
        # The pre-spike trend is extrapolated across the gap (np.polyfit, a
        # straight-line fit) so a door opening mid pull-down still counts as
        # recovered even though the cabin kept cooling meanwhile.
        recovered = False
        b0, a1 = max(0, i0 - rw), min(len(y), i1 + rw)
        if i0 - b0 >= 2 and a1 > i1:
            ok = ~np.isnan(y[b0:i0])
            if ok.sum() >= 2:
                slope, icpt = np.polyfit(t[b0:i0][ok], y[b0:i0][ok], 1)
                predicted = slope * t[i1:a1] + icpt
                recovered = abs(np.nanmedian(y[i1:a1] - predicted)) <= cfg["recovery_tolerance_degC"]
        if dur <= cfg["max_duration_s"] and recovered:
            seg = resid.iloc[i0:i1]
            out.append(_make("spike", ch.name, t, i0, i1, {
                "peak_excursion_degC": seg.iloc[int(seg.abs().to_numpy().argmax())],
                "baseline_degC": base.iloc[i0],
                "duration_s": dur,
            }))
    return out


def detect_missing_or_out_of_range(df, ch: ChannelInfo, cfg) -> list[Anomaly]:
    """NaN runs and values outside the physical sensor range (from channel config)."""
    t = df["time_s"].to_numpy()
    s = df[ch.name]
    out = []
    for i0, i1 in _true_runs(s.isna().to_numpy()):
        if i1 - i0 >= cfg["min_nan_run_samples"]:
            out.append(_make("missing", ch.name, t, i0, i1, {"nan_samples": i1 - i0}))
    if ch.sensor_range:
        lo, hi = ch.sensor_range
        for i0, i1 in _true_runs(((s < lo) | (s > hi)).to_numpy()):
            seg = s.iloc[i0:i1]
            out.append(_make("out_of_range", ch.name, t, i0, i1, {
                "min_value": seg.min(), "max_value": seg.max(),
                "range_low": lo, "range_high": hi,
            }))
    return out


def detect_time_gaps(df, cfg) -> list[Anomaly]:
    """Gaps in the time base affect every channel, so they are reported once as 'all'."""
    t = df["time_s"].to_numpy()
    steps = np.diff(t)
    limit = cfg["max_gap_factor"] * np.median(steps)
    return [Anomaly(id="", channel="all", type="missing", start_min=_r(t[i] / 60),
                    end_min=_r(t[i + 1] / 60), evidence={"gap_s": _r(steps[i])})
            for i in np.flatnonzero(steps > limit)]


def detect_anomalies(df: pd.DataFrame, channels: list[ChannelInfo], cfg: dict,
                     phases: list[Phase] | None = None) -> tuple[list[Anomaly], list[str]]:
    """Run every check. Returns (anomalies, notes); notes say which checks were skipped."""
    group_by = cfg["neighbour_group_by"]
    notes: list[str] = []
    if resolve_window(phases or [], cfg["drift"]["reference_window"]) is None:
        notes.append(f"Drift check skipped: reference window {cfg['drift']['reference_window']} "
                     "not found in this run")

    found: list[Anomaly] = detect_time_gaps(df, cfg["missing"])
    for ch in channels:
        found += detect_missing_or_out_of_range(df, ch, cfg["missing"])
        for check, key in ((detect_flatline, "flatline"), (detect_drift, "drift"),
                           (detect_spike, "spike")):
            found += check(df, ch, channels, cfg[key], group_by, phases)
    found.sort(key=lambda a: (a.start_min, a.channel))
    for k, a in enumerate(found, start=1):
        a.id = f"A{k:02d}"  # stable IDs the commentary can refer to
    return found, notes


def suspect_spans(anomalies: list[Anomaly], types: list[str]) -> dict[str, list[tuple[float, float]]]:
    """Channel -> list of (start_min, end_min) where it is suspect for metrics."""
    out: dict[str, list[tuple[float, float]]] = {}
    for a in anomalies:
        if a.type in types:
            out.setdefault(a.channel, []).append((a.start_min, a.end_min))
    return out
