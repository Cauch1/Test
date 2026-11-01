"""Each anomaly check finds its planted fault and leaves clean channels alone."""

import pytest
import yaml

from interpret.anomalies import (detect_anomalies, detect_drift, detect_flatline,
                                 detect_missing_or_out_of_range, detect_spike, detect_time_gaps)
from interpret.summary import build_summary

from .conftest import ROOT
from .synthetic import (DRIFT_CH, DRIFT_DEGC, DRIFT_START_MIN, DURATION_MIN, FLAT_CH,
                        FLAT_START_MIN, ROW_OFFSET, SPIKE_CH, SPIKE_DEGC, SPIKE_START_MIN,
                        TAU_MIN, T0_MIN, channel_names, final_temp, make_pulldown)


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load((ROOT / "config" / "anomaly_thresholds.yaml").read_text())


def _run(check, df, channels, cfg, key):
    """Run one check over every channel; return {channel: [anomalies]} for those that fired."""
    hits = {}
    for ch in channels:
        found = check(df, ch, channels, cfg[key], cfg["neighbour_group_by"])
        if found:
            hits[ch.name] = found
    return hits


@pytest.fixture(scope="module")
def clean_noisy():
    return make_pulldown(faults=False, noise_sd=0.05, seed=2)


# ---------- flatline ----------

def test_flatline_found(faulty_df, channels, cfg):
    hits = _run(detect_flatline, faulty_df, channels, cfg, "flatline")
    assert set(hits) == {FLAT_CH}
    a = hits[FLAT_CH][0]
    assert a.start_min == pytest.approx(FLAT_START_MIN, abs=0.5)
    # Frozen value = clean curve at the freeze time: 21 + 22 * exp(-(30-10)/20)
    frozen = final_temp(FLAT_CH) + (43 - final_temp(FLAT_CH)) * 2.718281828 ** (-(FLAT_START_MIN - T0_MIN) / TAU_MIN)
    assert a.evidence["value_degC"] == pytest.approx(frozen, abs=0.05)
    assert a.evidence["channel_std_degC"] < cfg["flatline"]["max_std_degC"]


def test_flatline_clean(clean_noisy, channels, cfg):
    assert _run(detect_flatline, clean_noisy, channels, cfg, "flatline") == {}


# ---------- drift ----------

def test_drift_found(faulty_df, channels, cfg):
    hits = _run(detect_drift, faulty_df, channels, cfg, "drift")
    # The frozen head channel also deviates from its neighbours, which is correct.
    assert set(hits) == {DRIFT_CH, FLAT_CH}
    a = hits[DRIFT_CH][0]
    # Deviation = row offset + ramp; crosses the threshold where
    # 0.5 + 3 * (t - 90) / 90 = 1.5  ->  t = 120 min
    t_cross = DRIFT_START_MIN + (cfg["drift"]["threshold_degC"] - ROW_OFFSET) / DRIFT_DEGC * (DURATION_MIN - DRIFT_START_MIN)
    assert a.start_min == pytest.approx(t_cross, abs=1.5)
    assert a.end_min == DURATION_MIN
    assert a.evidence["max_abs_deviation_degC"] == pytest.approx(ROW_OFFSET + DRIFT_DEGC, abs=0.15)


def test_drift_clean(clean_noisy, channels, cfg):
    assert _run(detect_drift, clean_noisy, channels, cfg, "drift") == {}


# ---------- spike ----------

def test_spike_found(faulty_df, channels, cfg):
    hits = _run(detect_spike, faulty_df, channels, cfg, "spike")
    assert set(hits) == {SPIKE_CH}
    (a,) = hits[SPIKE_CH]
    assert a.start_min == pytest.approx(SPIKE_START_MIN, abs=1 / 60)
    assert a.evidence["duration_s"] == pytest.approx(60, abs=2)
    assert a.evidence["peak_excursion_degC"] == pytest.approx(SPIKE_DEGC, abs=0.25)


def test_spike_clean(clean_noisy, channels, cfg):
    assert _run(detect_spike, clean_noisy, channels, cfg, "spike") == {}


def test_long_step_is_not_a_spike(clean_noisy, channels, cfg):
    df = clean_noisy.copy()
    t_min = df["time_s"] / 60
    df.loc[(t_min >= 100) & (t_min < 115), "tc_r1_l_foot"] += 6.0  # 15 min, too long for a spike
    assert "tc_r1_l_foot" not in _run(detect_spike, df, channels, cfg, "spike")


def test_spike_during_early_pull_down(clean_noisy, channels, cfg):
    """Cabin falls about 1 degC/min at 12 min; the spike must still count as recovered."""
    df = clean_noisy.copy()
    t_min = df["time_s"] / 60
    df.loc[(t_min >= 12) & (t_min < 13), "tc_r1_l_breath"] += 6.0
    hits = _run(detect_spike, df, channels, cfg, "spike")
    assert set(hits) == {"tc_r1_l_breath"}
    assert hits["tc_r1_l_breath"][0].start_min == pytest.approx(12.0, abs=1 / 60)


# ---------- missing and out of range ----------

def test_missing_and_out_of_range_found(clean_noisy, channels, cfg):
    df = clean_noisy.copy()
    df.loc[1000:1059, "tc_r1_l_foot"] = float("nan")   # 60 NaN samples
    df.loc[5000:5004, "tc_r2_l_head"] = 999.0           # open-circuit thermocouple reading
    df = df.drop(index=range(7000, 7030)).reset_index(drop=True)  # 30 s logging gap

    by_ch = {c.name: c for c in channels}
    miss = detect_missing_or_out_of_range(df, by_ch["tc_r1_l_foot"], cfg["missing"])
    assert [(a.type, a.evidence["nan_samples"]) for a in miss] == [("missing", 60)]
    oor = detect_missing_or_out_of_range(df, by_ch["tc_r2_l_head"], cfg["missing"])
    assert [(a.type, a.evidence["max_value"]) for a in oor] == [("out_of_range", 999.0)]
    (gap,) = detect_time_gaps(df, cfg["missing"])
    assert gap.channel == "all" and gap.evidence["gap_s"] == 31.0


def test_missing_clean(clean_noisy, channels, cfg):
    assert detect_time_gaps(clean_noisy, cfg["missing"]) == []
    for ch in channels:
        assert detect_missing_or_out_of_range(clean_noisy, ch, cfg["missing"]) == []


# ---------- whole run ----------

def test_clean_run_has_no_anomalies(clean_noisy, channels, cfg):
    assert detect_anomalies(clean_noisy, channels, cfg) == []


def test_ids_are_sequential(faulty_df, channels, cfg):
    ids = [a.id for a in detect_anomalies(faulty_df, channels, cfg)]
    assert ids == [f"A{k:02d}" for k in range(1, len(ids) + 1)]


def test_faulty_channels_excluded_and_gradient_fails(faulty_df, channels, spec, phase_cfg, cfg, run_args):
    """The frozen head and drifting foot sensors masked a failing gradient. With
    exclusion, the gradient matches the closed form for the remaining channels."""
    s = build_summary(faulty_df, channels, spec=spec, phase_cfg=phase_cfg, anomaly_cfg=cfg, **run_args)
    fh = next(m for m in s.metrics if m.name == "foot_head_gradient")
    assert set(fh.excluded_channels) == {FLAT_CH, DRIFT_CH}

    import math
    a, b = DURATION_MIN - 30, DURATION_MIN
    md = TAU_MIN * (math.exp(-(a - T0_MIN) / TAU_MIN) - math.exp(-(b - T0_MIN) / TAU_MIN)) / 30
    heads = [final_temp(n) for n in channel_names() if n.endswith("head") and n != FLAT_CH]
    feet = [final_temp(n) for n in channel_names() if n.endswith("foot") and n != DRIFT_CH]
    expected = (sum(heads) / len(heads) - sum(feet) / len(feet)) * (1 - md)
    assert fh.value == pytest.approx(expected, abs=0.02)

    crit = next(c for c in s.criteria if c.metric == "foot_head_gradient")
    assert crit.status == "FAIL"
    assert abs(fh.value_all_channels) < 3.0  # all-channel value would have passed

    flags = {c.name: c.quality.flags for c in s.channels}
    assert flags[FLAT_CH] == ["drift", "flatline"] and flags[SPIKE_CH] == ["spike"]
    assert flags["tc_r1_l_foot"] == []
