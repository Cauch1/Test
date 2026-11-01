"""Each anomaly check finds its planted fault and leaves clean channels alone,
and suspect channels drive the dual-verdict criteria."""

import math

import numpy as np
import pytest
import yaml

from interpret.anomalies import (detect_anomalies, detect_drift, detect_flatline,
                                 detect_missing_or_out_of_range, detect_spike, detect_time_gaps)
from interpret.channels import select
from interpret.metrics import group_mean
from interpret.phases import detect_phases
from interpret.summary import build_summary

from .conftest import ROOT
from .synthetic import (DRIFT_CH, DRIFT_DEGC, DRIFT_START_MIN, DURATION_MIN, FINAL, FLAT_CH,
                        FLAT_START_MIN, ROW_OFFSET, SPIKE_CH, SPIKE_DEGC, SPIKE_START_MIN,
                        T0_MIN, T_SOAK, TAU_MIN, final_temp, make_pulldown)


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load((ROOT / "config" / "anomaly_thresholds.yaml").read_text())


def _phases(df, channels, phase_cfg):
    return detect_phases(df["time_s"], group_mean(df, select(channels, {"height": "breath"})), phase_cfg)


def _run(check, df, channels, cfg, key, phases=None):
    """Run one check over every channel; return {channel: [anomalies]} for those that fired."""
    hits = {}
    for ch in channels:
        found = check(df, ch, channels, cfg[key], cfg["neighbour_group_by"], phases)
        if found:
            hits[ch.name] = found
    return hits


def _clean_value(name, t_min):
    """Closed-form clean curve for one channel."""
    tf = final_temp(name)
    return tf + (T_SOAK - tf) * math.exp(-max(0.0, t_min - T0_MIN) / TAU_MIN)


@pytest.fixture(scope="module")
def clean_noisy():
    return make_pulldown(faults=False, noise_sd=0.05, seed=2)


@pytest.fixture(scope="module")
def frozen_plausible(clean_noisy):
    """A breath sensor freezes at 100 min at exactly its correct stabilised value."""
    df = clean_noisy.copy()
    t_min = df["time_s"] / 60
    df.loc[t_min >= 100, "tc_r1_l_breath"] = _clean_value("tc_r1_l_breath", 100.0)
    return df


# ---------- flatline ----------

def test_flatline_found(faulty_df, channels, cfg):
    hits = _run(detect_flatline, faulty_df, channels, cfg, "flatline")
    assert set(hits) == {FLAT_CH}
    (a,) = hits[FLAT_CH]
    assert a.start_min == pytest.approx(FLAT_START_MIN, abs=0.5)
    assert a.end_min == DURATION_MIN  # frozen-with-no-noise condition carries it to the end
    assert a.evidence["value_degC"] == pytest.approx(_clean_value(FLAT_CH, FLAT_START_MIN), abs=0.05)
    assert a.evidence["channel_std_degC"] <= cfg["flatline"]["max_std_degC"]


def test_frozen_at_plausible_value(frozen_plausible, channels, cfg, phase_cfg):
    """Neighbours are not changing and the frozen value is correct, so neither
    the moving-neighbour condition nor the drift check can see it. The
    no-sensor-noise condition must."""
    phases = _phases(frozen_plausible, channels, phase_cfg)
    hits = _run(detect_flatline, frozen_plausible, channels, cfg, "flatline", phases)
    assert set(hits) == {"tc_r1_l_breath"}
    (a,) = hits["tc_r1_l_breath"]
    assert a.start_min == pytest.approx(100.0, abs=0.5)
    assert a.end_min == DURATION_MIN
    assert a.evidence["channel_std_degC"] == 0.0
    # Offset does not change, so drift correctly stays silent.
    assert _run(detect_drift, frozen_plausible, channels, cfg, "drift", phases) == {}


def test_flatline_clean(clean_noisy, channels, cfg):
    assert _run(detect_flatline, clean_noisy, channels, cfg, "flatline") == {}


def test_flatline_clean_noise_free(clean_df, channels, cfg):
    """With no noise anywhere, a quiet channel is not frozen relative to its neighbours."""
    assert _run(detect_flatline, clean_df, channels, cfg, "flatline") == {}


# ---------- drift ----------

def test_drift_found(faulty_df, channels, cfg, phase_cfg):
    phases = _phases(faulty_df, channels, phase_cfg)
    hits = _run(detect_drift, faulty_df, channels, cfg, "drift", phases)
    assert set(hits) == {DRIFT_CH}
    (a,) = hits[DRIFT_CH]

    # Closed form. Offset = own - median(others) = (28 + 15e) - (25 + 18e) + ramp
    # = 3 - 3e + ramp. Baseline = its median over the reference window.
    stab = next(p for p in phases if p.id == "stabilised")
    ref_t = np.linspace(stab.start_min, stab.start_min + 10, 601)
    e = lambda t: np.exp(-(t - T0_MIN) / TAU_MIN)
    baseline = np.median(ROW_OFFSET - ROW_OFFSET * e(ref_t))
    t = np.linspace(DRIFT_START_MIN, DURATION_MIN, 100_001)
    ramp = DRIFT_DEGC * (t - DRIFT_START_MIN) / (DURATION_MIN - DRIFT_START_MIN)
    change = ROW_OFFSET - ROW_OFFSET * e(t) + ramp - baseline
    t_cross = t[np.argmax(change > cfg["drift"]["threshold_degC"])]

    assert a.start_min == pytest.approx(t_cross, abs=0.5)
    assert a.end_min == DURATION_MIN
    assert a.evidence["baseline_offset_degC"] == pytest.approx(baseline, abs=0.05)
    assert a.evidence["max_abs_offset_change_degC"] == pytest.approx(change[-1], abs=0.15)


def test_drift_ignores_large_steady_row_offset(clean_noisy, channels, cfg, phase_cfg):
    """3 degC front-to-rear offset is constant, so it is not drift."""
    phases = _phases(clean_noisy, channels, phase_cfg)
    assert _run(detect_drift, clean_noisy, channels, cfg, "drift", phases) == {}


def test_drift_explicit_reference_window(faulty_df, channels, cfg):
    """An absolute reference window works without detected phases."""
    cfg2 = {**cfg, "drift": {**cfg["drift"], "reference_window": {"start_min": 75, "end_min": 85}}}
    hits = _run(detect_drift, faulty_df, channels, cfg2, "drift", phases=[])
    assert set(hits) == {DRIFT_CH}


def test_drift_skipped_without_reference_window(clean_noisy, channels, cfg):
    anomalies, notes = detect_anomalies(clean_noisy, channels, cfg, phases=[])
    assert anomalies == []
    assert notes and "Drift check skipped" in notes[0]


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

def test_clean_run_has_no_anomalies(clean_noisy, channels, cfg, phase_cfg):
    anomalies, notes = detect_anomalies(clean_noisy, channels, cfg, _phases(clean_noisy, channels, phase_cfg))
    assert anomalies == [] and notes == []


def test_ids_are_sequential(faulty_df, channels, cfg, phase_cfg):
    anomalies, _ = detect_anomalies(faulty_df, channels, cfg, _phases(faulty_df, channels, phase_cfg))
    assert [a.id for a in anomalies] == [f"A{k:02d}" for k in range(1, len(anomalies) + 1)]


# ---------- suspect channels and criteria ----------

def _md(last_min=30.0):
    """Time-mean of the decay term over the last 30 min."""
    a, b = DURATION_MIN - last_min, DURATION_MIN
    return TAU_MIN * (math.exp(-(a - T0_MIN) / TAU_MIN) - math.exp(-(b - T0_MIN) / TAU_MIN)) / last_min


def _faulty_summary(faulty_df, channels, spec, phase_cfg, cfg, run_args, auto):
    return build_summary(faulty_df, channels, spec=spec, phase_cfg=phase_cfg,
                         anomaly_cfg={**cfg, "auto_exclude": auto}, **run_args)


def test_flag_only_gives_review_required(faulty_df, channels, spec, phase_cfg, cfg, run_args):
    """Frozen head and drifting foot make the gradient pass with all channels.
    Without them it fails. Flag-only mode must not pick a side."""
    s = _faulty_summary(faulty_df, channels, spec, phase_cfg, cfg, run_args, auto=False)
    fh = next(m for m in s.metrics if m.name == "foot_head_gradient")
    assert set(fh.suspect_channels) == {FLAT_CH, DRIFT_CH}

    # All channels, paired head - foot per seat (last 30 min):
    # r1_l and r2_l clean: -4(1 - md); r1_r: frozen head 29.05 vs clean foot;
    # r2_r: clean head vs foot + drift ramp (mean 2.5 degC over 150-180 min).
    md = _md()
    clean_pair = (FINAL["head"] - FINAL["foot"]) * (1 - md)
    frozen_pair = _clean_value(FLAT_CH, FLAT_START_MIN) - (FINAL["foot"] + (T_SOAK - FINAL["foot"]) * md)
    drift_pair = clean_pair - DRIFT_DEGC * (165 - DRIFT_START_MIN) / (DURATION_MIN - DRIFT_START_MIN)
    assert fh.value == pytest.approx((2 * clean_pair + frozen_pair + drift_pair) / 4, abs=0.03)
    # Excluding suspect: only the two clean pairs remain, so exactly the clean value.
    assert fh.value_excluding_suspect == pytest.approx(clean_pair, abs=0.02)

    crit = next(c for c in s.criteria if c.metric == "foot_head_gradient")
    assert crit.all_channels.verdict == "PASS"
    assert crit.excluding_suspect.verdict == "FAIL"
    assert crit.status == "REVIEW_REQUIRED" and crit.basis == "all_channels"
    assert FLAT_CH in crit.note and DRIFT_CH in crit.note

    # Criteria with no suspect channels are unaffected.
    others = {c.metric: c for c in s.criteria if c.metric != "foot_head_gradient"}
    assert all(c.excluding_suspect is None and c.status in ("PASS", "FAIL") for c in others.values())


def test_auto_exclude_uses_excluded_value(faulty_df, channels, spec, phase_cfg, cfg, run_args):
    s = _faulty_summary(faulty_df, channels, spec, phase_cfg, cfg, run_args, auto=True)
    crit = next(c for c in s.criteria if c.metric == "foot_head_gradient")
    assert crit.status == "FAIL" and crit.basis == "excluding_suspect"
    assert "all-channel verdict PASS" in crit.note


def test_agreeing_verdicts_keep_pass_or_fail(frozen_plausible, channels, spec, phase_cfg, cfg, run_args):
    """Frozen breath sensor at a correct value: both versions agree, so no review flag."""
    s = build_summary(frozen_plausible, channels, spec=spec, phase_cfg=phase_cfg, anomaly_cfg=cfg, **run_args)
    crit = next(c for c in s.criteria if c.metric == "stabilised_cabin_mean")
    assert crit.excluding_suspect is not None
    assert crit.status == crit.all_channels.verdict == crit.excluding_suspect.verdict


def test_channel_flags(faulty_df, channels, spec, phase_cfg, cfg, run_args):
    s = _faulty_summary(faulty_df, channels, spec, phase_cfg, cfg, run_args, auto=False)
    flags = {c.name: c.quality.flags for c in s.channels}
    assert flags[FLAT_CH] == ["flatline"]
    assert flags[DRIFT_CH] == ["drift"]
    assert flags[SPIKE_CH] == ["spike"]
    assert flags["tc_r1_l_foot"] == []
