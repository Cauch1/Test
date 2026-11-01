"""Synthetic 3-hour pull-down with 12 thermocouples and closed-form answers.

Every clean channel follows the same first-order response:
    T(t) = T_final + (T_soak - T_final) * exp(-(t - t0) / tau)   for t >= t0
    T(t) = T_soak                                                 for t <  t0
Only T_final differs by location, so every metric has an analytic value
(see EXPECTED below) to check the code against.

Layout: 2 rows x 2 sides x 3 heights (foot, breath, head) = 12 channels.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

T_SOAK = 43.0
T0_MIN = 10.0        # soak ends, pull-down starts
TAU_MIN = 20.0       # first-order time constant
DURATION_MIN = 180.0
FINAL = {"foot": 25.0, "breath": 22.0, "head": 21.0}
ROW_OFFSET = 1.5     # row 2 settles 1.5 degC warmer than row 1

# Planted faults (used from step 3; switched off for the analytic Stage A tests)
DRIFT_CH, DRIFT_START_MIN, DRIFT_DEGC = "tc_r2_r_foot", 90.0, 3.0
FLAT_CH, FLAT_START_MIN = "tc_r1_r_head", 30.0
SPIKE_CH, SPIKE_START_MIN, SPIKE_S, SPIKE_DEGC = "tc_r2_l_breath", 120.0, 60.0, 6.0


def channel_names() -> list[str]:
    return [f"tc_r{r}_{s}_{h}" for r in (1, 2) for s in ("l", "r") for h in ("foot", "breath", "head")]


def final_temp(name: str) -> float:
    row, height = int(name[4]), name.split("_")[-1]
    return FINAL[height] + ROW_OFFSET * (row - 1)


def make_pulldown(faults: bool = False, noise_sd: float = 0.0, seed: int = 0,
                  dt_s: float = 1.0) -> pd.DataFrame:
    t_s = np.arange(0.0, DURATION_MIN * 60 + dt_s / 2, dt_s)
    t_min = t_s / 60.0
    decay = np.where(t_min < T0_MIN, 1.0, np.exp(-(t_min - T0_MIN) / TAU_MIN))
    rng = np.random.default_rng(seed)

    data = {"time_s": t_s}
    for name in channel_names():
        tf = final_temp(name)
        y = tf + (T_SOAK - tf) * decay
        if noise_sd:
            y = y + rng.normal(0.0, noise_sd, size=y.size)
        if faults:
            if name == DRIFT_CH:  # linear drift away from neighbours
                ramp = np.clip((t_min - DRIFT_START_MIN) / (DURATION_MIN - DRIFT_START_MIN), 0, 1)
                y = y + DRIFT_DEGC * ramp
            if name == FLAT_CH:   # sensor freezes at its current reading
                y = np.where(t_min >= FLAT_START_MIN, y[np.searchsorted(t_min, FLAT_START_MIN)], y)
            if name == SPIKE_CH:  # door opening: step up then recovery
                on = (t_min >= SPIKE_START_MIN) & (t_min < SPIKE_START_MIN + SPIKE_S / 60)
                y = np.where(on, y + SPIKE_DEGC, y)
        data[name] = y
    return pd.DataFrame(data)


# ---------- closed-form expected values ----------

def _mean_decay(a_min: float, b_min: float) -> float:
    """Time-mean of exp(-(t - t0)/tau) over [a, b]."""
    ea = math.exp(-(a_min - T0_MIN) / TAU_MIN)
    eb = math.exp(-(b_min - T0_MIN) / TAU_MIN)
    return TAU_MIN * (ea - eb) / (b_min - a_min)


def expected(phase_cfg: dict, target: float = 25.0, last_min: float = 30.0) -> dict:
    breath_final = np.mean([final_temp(n) for n in channel_names() if n.endswith("breath")])
    amp = T_SOAK - breath_final  # 20.25 degC

    # Pull-down start: reference leaves the soak plateau by departure_degC.
    pd_start = T0_MIN - TAU_MIN * math.log(1 - phase_cfg["departure_degC"] / amp)
    # Stabilised start: |dT/dt| = amp/tau * exp(-(t-t0)/tau) falls to the rate threshold.
    # The centred rolling mean and the centred window difference each scale an
    # exponential's rate by sinh(x)/x, x = half-window / tau, which delays the crossing.
    x = phase_cfg["smoothing_window_s"] / 2 / 60 / TAU_MIN
    gain = (math.sinh(x) / x) ** 2
    stab_start = T0_MIN + TAU_MIN * math.log(
        gain * amp / (TAU_MIN * phase_cfg["transient_rate_degC_per_min"]))
    # Time to target, measured from the detected pull-down start.
    t_target = T0_MIN + TAU_MIN * math.log(amp / (target - breath_final)) - pd_start

    a, b = DURATION_MIN - last_min, DURATION_MIN
    md = _mean_decay(a, b)
    return {
        "pull_down_start_min": pd_start,
        "stabilised_start_min": stab_start,
        "time_to_target": t_target,
        "stabilised_cabin_mean": breath_final + amp * md,
        # row 1 breath minus row 2 breath: (22 + 21e) - (23.5 + 19.5e)
        "front_rear_gradient": -ROW_OFFSET + ROW_OFFSET * md,
        # head minus foot, same in each row: (21 + 22e) - (25 + 18e)
        "foot_head_gradient": (FINAL["head"] - FINAL["foot"]) * (1 - md),
    }
