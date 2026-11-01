"""Stage A: phases, metrics, criteria and the written JSON, checked against closed-form values."""

import json

import pytest

from interpret.adapter import run_stage_a
from interpret.schema import Summary
from interpret.summary import build_summary

from .synthetic import DURATION_MIN, expected, make_pulldown

ONE_SAMPLE_MIN = 1 / 60  # 1 Hz data, so boundaries are only known to one sample


@pytest.fixture(scope="module")
def summary(clean_df, channels, spec, phase_cfg):
    from datetime import date
    return build_summary(clean_df, channels, test_id="T0001", facility="CWT2",
                         run_date=date(2026, 10, 30), spec=spec, phase_cfg=phase_cfg)


@pytest.fixture(scope="module")
def exp(phase_cfg):
    return expected(phase_cfg)


def _metric(summary, name):
    return next(m for m in summary.metrics if m.name == name)


def test_phases_detected_in_order(summary):
    assert [p.name for p in summary.test.phases] == ["soak", "pull-down", "stabilised"]


def test_phase_boundaries_match_closed_form(summary, exp):
    soak, pull, stab = summary.test.phases
    assert soak.start_min == 0
    # 10 s departure smoothing averages a convex curve, so it lags by ~1 s;
    # allow 2 samples plus 2 dp rounding.
    assert pull.start_min == pytest.approx(exp["pull_down_start_min"], abs=2 * ONE_SAMPLE_MIN + 0.005)
    assert stab.start_min == pytest.approx(exp["stabilised_start_min"], abs=0.1)
    assert stab.end_min == DURATION_MIN


@pytest.mark.parametrize("name, tol", [
    ("time_to_target", 0.02),          # minutes
    ("stabilised_cabin_mean", 0.01),   # degC, rounding to 2 dp
    ("front_rear_gradient", 0.01),
    ("foot_head_gradient", 0.01),
])
def test_metrics_match_closed_form(summary, exp, name, tol):
    assert _metric(summary, name).value == pytest.approx(exp[name], abs=tol)


def test_criteria_status_and_margin(summary, spec):
    by_name = {c.metric: c for c in summary.criteria}
    # Placeholder limits: 60 min, 24 degC, |2| degC, |3| degC.
    # Bus-like 3 degC row offset fails the 2 degC front-to-rear limit.
    expected = {"time_to_target": "PASS", "stabilised_cabin_mean": "PASS",
                "front_rear_gradient": "FAIL", "foot_head_gradient": "FAIL"}
    assert {k: v.status for k, v in by_name.items()} == expected
    fh = by_name["foot_head_gradient"]
    assert fh.excluding_suspect is None and fh.basis == "all_channels"
    assert fh.all_channels.measured == pytest.approx(abs(_metric(summary, "foot_head_gradient").value), abs=0.01)
    assert fh.all_channels.margin == pytest.approx(3.0 - fh.all_channels.measured, abs=0.01)
    assert fh.all_channels.margin < 0


def test_channel_stats(summary):
    ch = next(c for c in summary.channels if c.name == "tc_r1_l_breath")
    assert ch.max == 43.0
    assert ch.location_group == "breath height, row 1"
    assert ch.phase_means["soak"] == 43.0
    assert ch.quality.nan_fraction == 0.0
    assert set(ch.phase_means) == {"soak", "pull-down", "stabilised"}


def test_counts(summary):
    assert summary.counts.n_channels == 12
    assert summary.counts.duration_min == DURATION_MIN
    assert summary.counts.sample_interval_s == 1.0


def test_metric_none_when_target_never_reached(clean_df, channels, spec, phase_cfg):
    from datetime import date
    spec2 = json.loads(json.dumps(spec))
    spec2["metrics"]["time_to_target"]["threshold"] = 10.0  # never reached
    s = build_summary(clean_df, channels, test_id="T0001", facility="CWT2",
                      run_date=date(2026, 10, 30), spec=spec2, phase_cfg=phase_cfg)
    crit = next(c for c in s.criteria if c.metric == "time_to_target")
    assert _metric(s, "time_to_target").value is None
    assert crit.status == "FAIL" and crit.note


def test_noisy_faulty_data_still_finds_three_phases(faulty_df, channels, spec, phase_cfg, exp):
    from datetime import date
    s = build_summary(faulty_df, channels, test_id="T0001", facility="CWT2",
                      run_date=date(2026, 10, 30), spec=spec, phase_cfg=phase_cfg)
    assert [p.name for p in s.test.phases] == ["soak", "pull-down", "stabilised"]
    assert s.test.phases[1].start_min == pytest.approx(exp["pull_down_start_min"], abs=0.1)


def test_json_round_trip_and_no_raw_data(tmp_path, clean_df, run_args):
    s, path = run_stage_a(clean_df, spec_path="specs/bvs_2_7.yaml", out_dir=tmp_path, **run_args)
    text = path.read_text()
    assert Summary.model_validate_json(text) == s
    # Summary must stay small: per-channel stats, not samples.
    assert len(text) < 20_000
    assert "time_s" not in text


def test_unknown_field_rejected():
    with pytest.raises(Exception):
        Summary.model_validate({"bogus": 1})


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_noise_does_not_bias_metrics(channels, spec, phase_cfg, exp, seed):
    """Noise only (no faults): metrics stay close to the closed-form values."""
    from datetime import date
    df = make_pulldown(faults=False, noise_sd=0.05, seed=seed)
    s = build_summary(df, channels, test_id="T0001", facility="CWT2",
                      run_date=date(2026, 10, 30), spec=spec, phase_cfg=phase_cfg)
    assert s.test.phases[2].start_min == pytest.approx(exp["stabilised_start_min"], abs=1.0)
    assert _metric(s, "time_to_target").value == pytest.approx(exp["time_to_target"], abs=0.1)
    for name in ("stabilised_cabin_mean", "front_rear_gradient", "foot_head_gradient"):
        assert _metric(s, name).value == pytest.approx(exp[name], abs=0.02)
