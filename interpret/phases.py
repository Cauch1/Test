"""Infer run phases (soak, pull-down, stabilised, heat-up) from rate of change.

Method:
1. Smooth the cabin reference with a centred rolling mean.
2. Rate = d(smoothed)/dt in degC/min. Label each sample falling, rising or flat
   against the transient rate threshold.
3. Merge segments shorter than the minimum duration into their neighbour.
4. Name segments: flat before any transient is soak, flat after one is
   stabilised, falling is pull-down, rising is heat-up.
5. Refine the start of each transient. A centred smoother starts "falling"
   about half a window early, which would inflate time-to-target. The start is
   moved to the first sample that departs from the plateau by more than
   departure_degC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import Phase


def _runs(labels: np.ndarray) -> list[list]:
    """Run-length encode labels into [label, start_idx, end_idx_exclusive]."""
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(labels)]))
    return [[labels[s], int(s), int(e)] for s, e in zip(starts, ends)]


def _coalesce(segs: list[list]) -> list[list]:
    out: list[list] = []
    for seg in segs:
        if out and out[-1][0] == seg[0]:
            out[-1][2] = seg[2]
        else:
            out.append(list(seg))
    return out


def _merge_short(segs: list[list], t: np.ndarray, min_s: float) -> list[list]:
    """Absorb segments shorter than min_s into the previous one (or next, if first)."""
    segs = _coalesce(segs)
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for k, (lab, i0, i1) in enumerate(segs):
            dur = t[min(i1, len(t) - 1)] - t[i0]
            if dur < min_s:
                if k > 0:
                    segs[k - 1][2] = i1
                else:
                    segs[k + 1][1] = i0
                del segs[k]
                segs = _coalesce(segs)
                changed = True
                break
    return segs


def detect_phases(time_s, reference, cfg: dict) -> list[Phase]:
    """Return contiguous phases covering the whole run. cfg is config/phases.yaml."""
    t = np.asarray(time_s, dtype=float)
    y = pd.Series(np.asarray(reference, dtype=float)).interpolate(limit_direction="both")
    dt = float(np.median(np.diff(t)))

    win = max(1, int(round(cfg["smoothing_window_s"] / dt)))
    smooth = y.rolling(win, center=True, min_periods=1).mean().to_numpy()
    # Rate as a centred difference across a whole window, not between adjacent
    # samples: adjacent differences of a noisy signal flicker around the
    # threshold and delay the stabilised start by several minutes.
    half = max(1, win // 2)
    idx = np.arange(len(t))
    lo, hi = np.clip(idx - half, 0, len(t) - 1), np.clip(idx + half, 0, len(t) - 1)
    rate = (smooth[hi] - smooth[lo]) / (t[hi] - t[lo]) * 60.0  # degC/s -> degC/min

    thr = cfg["transient_rate_degC_per_min"]
    labels = np.where(rate < -thr, "falling", np.where(rate > thr, "rising", "flat"))
    segs = _merge_short(_runs(labels), t, cfg["min_phase_duration_s"])

    # Step 5: refine transient starts using lightly smoothed data.
    dep_win = max(1, int(round(cfg["departure_smoothing_s"] / dt)))
    y_dep = y.rolling(dep_win, center=True, min_periods=1).mean().to_numpy()
    for k in range(1, len(segs)):
        if segs[k][0] == "flat" or segs[k - 1][0] != "flat":
            continue
        p0, i0 = segs[k - 1][1], segs[k][1]
        plateau = np.median(y_dep[p0:max(p0 + 1, i0 - win // 2)])
        lo, hi = max(p0, i0 - win // 2), min(len(t), i0 + win)
        departed = np.flatnonzero(np.abs(y_dep[lo:hi] - plateau) > cfg["departure_degC"])
        if departed.size:
            new_start = lo + int(departed[0])
            segs[k - 1][2] = segs[k][1] = new_start

    phases: list[Phase] = []
    seen_transient = False
    counts: dict[str, int] = {}
    for lab, i0, i1 in segs:
        if lab == "flat":
            name = "stabilised" if seen_transient else "soak"
        else:
            name = "pull-down" if lab == "falling" else "heat-up"
            seen_transient = True
        counts[name] = counts.get(name, 0) + 1
        pid = name if counts[name] == 1 else f"{name}_{counts[name]}"
        end = t[i1] if i1 < len(t) else t[-1]
        phases.append(Phase(id=pid, name=name, start_min=t[i0] / 60, end_min=end / 60))
    return phases
