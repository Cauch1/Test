"""Stage B: placeholder templates, digit and key checks, fill, final number
assertion, structure, privacy guards, clients, CLI."""

import copy
import json

import pytest
from docx import Document
from pptx import Presentation

from interpret import interpret as interp_mod
from interpret import validate as validate_mod
from interpret.cli import main, run_after_compute
from interpret.client import AnthropicClient, InterpretationError, MockClient
from interpret.commentary import Commentary
from interpret.deliverables import HEADING
from interpret.interpret import PrivacyError, build_payload, interpret, load_cfg
from interpret.placeholders import MISSING, Entry, _num, build_catalogue
from interpret.summary import build_summary, write_summary
from interpret.validate import (CommentaryValidationError, check_numbers, quoted_numbers,
                                validate)

CFG = load_cfg()
DEC = CFG["display_decimals"]


@pytest.fixture(scope="module")
def summary(faulty_df, channels, spec, phase_cfg):
    import yaml
    from datetime import date
    from .conftest import ROOT
    cfg = yaml.safe_load((ROOT / "config" / "anomaly_thresholds.yaml").read_text())
    return build_summary(faulty_df, channels, test_id="T0001", facility="CWT2",
                         run_date=date(2026, 10, 30), spec=spec, phase_cfg=phase_cfg, anomaly_cfg=cfg)


@pytest.fixture
def good(summary) -> dict:
    """Mock template reply: prose and placeholder keys only."""
    return MockClient(summary)._from_summary()


def _with(data: dict, text: str, field: int = 0) -> Commentary:
    d = copy.deepcopy(data)
    d["results_commentary"][field] += " " + text
    return Commentary.model_validate(d)


def _check(c, summary, **kw):
    return validate(c, summary, decimals=DEC, **kw)


def _fails(c, summary, kind, match=None):
    with pytest.raises(CommentaryValidationError) as e:
        _check(c, summary)
    assert any(p["kind"] == kind for p in e.value.log["problems"]), e.value
    if match:
        assert match in str(e.value), str(e.value)
    return e.value


# ---------- the happy path ----------

def test_mock_template_passes_and_fills(summary, good):
    log, filled = _check(Commentary.model_validate(good), summary)
    assert log["passed"] and log["placeholders_used"] > 40
    assert all(v == 0 for v in log["counts"].values())
    text = " ".join(filled.results_commentary)
    assert "{" not in text and "51.1 min" in text           # time to target, 1 dp
    fh = next(cc for cc in filled.criteria_commentary if cc.criterion == "Foot-to-head gradient")
    assert "2.6 degC" in fh.comment and "4.0 degC" in fh.comment and "REVIEW REQUIRED" in fh.comment
    assert "Row 1 right head" in fh.comment                  # suspect channels by display name


# ---------- placeholder keys ----------

def test_unknown_key_fails_with_sentence(summary, good):
    err = _fails(_with(good, "The peak was {metric.peak_temperature}."), summary, "placeholder",
                 "unknown key {metric.peak_temperature}")
    assert 'Sentence: "The peak was {metric.peak_temperature}."' in str(err)


def test_malformed_braces_fail(summary, good):
    _fails(_with(good, "The peak was {metric.time_to_target."), summary, "placeholder", "malformed")


# ---------- digits outside placeholders ----------

def test_digit_in_prose_fails(summary, good):
    err = _fails(_with(good, "The coolest seat reached 19.4 degC."), summary, "digit",
                 "digits outside a placeholder: 19.4")
    assert 'Sentence: "The coolest seat reached 19.4 degC."' in str(err)


def test_correct_number_written_directly_still_fails(summary, good):
    """Even a true value is rejected: the model must use the key."""
    _fails(_with(good, "The time to target was 51.09 min."), summary, "digit")


def test_allowlisted_ids_pass(summary, good):
    c = _with(good, "Anomaly A01 on tc_r1_r_head was seen in test T0001 at CWT2 under BVS 2.7.")
    log, _ = _check(c, summary)
    assert log["counts"]["digit"] == 0


def test_config_allowlist(summary, good):
    c = _with(good, "The method followed ISO 16750.")
    _fails(c, summary, "digit")
    _check(c, summary, extra_allowlist=["ISO 16750"])


def test_spelled_number_warns(summary, good):
    log, _ = _check(_with(good, "Two sensors were suspect."), summary)
    assert log["passed"] and log["counts"]["number_word"] == 1


# ---------- naming warning ----------

def test_channel_value_without_channel_name_warns(summary, good):
    log, _ = _check(_with(good, "One head sensor averaged {channel.tc_r1_r_head.mean.stabilised}."), summary)
    assert log["passed"] and log["counts"]["naming"] == 1
    log, _ = _check(_with(good, "{channel.tc_r1_r_head.name} averaged "
                                "{channel.tc_r1_r_head.mean.stabilised}."), summary)
    assert log["counts"]["naming"] == 0


def test_anomaly_value_named_by_id_or_own_explanation(summary, good):
    a = summary.anomalies[0].id
    log, _ = _check(_with(good, f"Anomaly {a} started at {{anomaly.{a}.start}}."), summary)
    assert log["counts"]["naming"] == 0
    log, _ = _check(_with(good, f"A fault started at {{anomaly.{a}.start}}."), summary)
    assert log["counts"]["naming"] == 1


# ---------- fill and formatting ----------

def test_formatting():
    assert _num(-2.614, "degC", DEC) == "-2.6 degC"
    assert _num(-0.02, "degC", DEC) == "0.0 degC"   # no negative zero
    assert _num(60.0, "s", DEC) == "60 s"
    assert _num(12, None, DEC) == "12"
    assert _num(None, "degC", DEC) == MISSING


def test_catalogue_covers_summary(summary):
    cat = build_catalogue(summary, DEC)
    for m in summary.metrics:
        assert f"metric.{m.name}" in cat
    for c in summary.criteria:
        assert {f"criteria.{c.id}.{f}" for f in ("limit", "margin", "status", "measured")} <= set(cat)
    for ch in summary.channels:
        assert cat[f"channel.{ch.name}.name"].text == ch.display_name
    for a in summary.anomalies:
        assert f"anomaly.{a.id}.start" in cat


def test_final_assertion_catches_bad_fill(summary, good, monkeypatch):
    """If the fill code ever writes a value not in the summary, the old number
    checker stops it."""
    real = validate_mod.build_catalogue

    def broken(s, d):
        cat = real(s, d)
        cat["metric.time_to_target"] = Entry("99.9 min", "broken")
        return cat
    monkeypatch.setattr(validate_mod, "build_catalogue", broken)
    _fails(Commentary.model_validate(good), summary, "final_assertion", "99.9 min not found")


# ---------- unit-aware number checker (used as the final assertion) ----------

def test_quoted_numbers_units_and_ranges():
    got = [(q.value, q.unit) for q in quoted_numbers("Between 29.72 and 180 min it read 23.51degC; 12 channels.")]
    assert got == [(29.72, "min"), (180.0, "min"), (23.51, "degC"), (12.0, None)]


def test_number_checker_unit_mismatch(summary, good):
    _, filled = _check(Commentary.model_validate(good), summary)
    d = filled.model_dump()
    d["results_commentary"][0] += " The cabin mean was 23.51 min."
    probs = check_numbers(Commentary.model_validate(d), summary, scoped=False)
    assert probs and "only with unit degC" in probs[0].detail


# ---------- structure ----------

def test_missing_anomaly_explanation_fails(summary, good):
    d = copy.deepcopy(good)
    d["anomaly_explanations"].pop(0)
    with pytest.raises(CommentaryValidationError, match="A01 explained 0 times"):
        _check(Commentary.model_validate(d), summary)


def test_invented_anomaly_id_fails(summary, good):
    d = copy.deepcopy(good)
    d["anomaly_explanations"].append(dict(d["anomaly_explanations"][0], anomaly_id="A99"))
    with pytest.raises(CommentaryValidationError, match="A99 is not a Python-detected anomaly"):
        _check(Commentary.model_validate(d), summary)


def test_criteria_by_id_and_paragraph_count(summary, good):
    d = copy.deepcopy(good)
    d["results_commentary"] = d["results_commentary"][:1]
    d["criteria_commentary"][0]["criterion"] = "Time to target"  # a name, not the id
    with pytest.raises(CommentaryValidationError) as e:
        _check(Commentary.model_validate(d), summary)
    assert "1 paragraphs" in str(e.value) and "'time_to_target' commented 0 times" in str(e.value)


def test_model_suggested_must_be_labelled_unverified(good):
    d = copy.deepcopy(good)
    d["model_suggested_anomalies"][0]["status"] = "confirmed"
    with pytest.raises(Exception):
        Commentary.model_validate(d)


# ---------- style ----------

def test_style_warns_by_default_and_fails_when_configured(summary, good):
    c = _with(good, "The cabin settled — eventually — as expected.")
    log, _ = _check(c, summary)
    assert log["passed"] and log["counts"]["style"] == 1
    with pytest.raises(CommentaryValidationError):
        _check(c, summary, style_fatal=True)


# ---------- Stage B orchestration and audit trail ----------

def test_failed_validation_writes_raw_and_log_only(tmp_path, summary, good):
    path = write_summary(summary, tmp_path)
    bad = copy.deepcopy(good)
    bad["results_commentary"][1] += " The rear row peaked at 31.2 degC."
    with pytest.raises(CommentaryValidationError):
        interpret(path, client=MockClient(summary, canned=bad))
    assert (tmp_path / "T0001_interpretation_raw.json").exists()
    log = json.loads((tmp_path / "T0001_validation_log.json").read_text())
    assert log["passed"] is False and log["counts"]["digit"] == 1
    assert log["problems"][0]["sentence"] == "The rear row peaked at 31.2 degC."
    assert not (tmp_path / "T0001_interpretation.json").exists()


def test_successful_run_saves_template_and_filled(tmp_path, summary):
    import hashlib
    path = write_summary(summary, tmp_path)
    interpret(path, offline=True)
    out = json.loads((tmp_path / "T0001_interpretation.json").read_text())
    assert out["meta"]["summary_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "{metric.time_to_target}" in " ".join(out["template"]["results_commentary"])
    assert "{" not in " ".join(out["commentary"]["results_commentary"])


# ---------- privacy and payload ----------

def test_payload_has_summary_and_key_list(summary):
    payload = build_payload(summary, CFG)
    assert "time_s" not in payload
    assert "{metric.time_to_target}" in payload and "<placeholders>" in payload
    assert len(payload) < 40_000


def test_forbidden_term_blocks_send(summary, monkeypatch):
    monkeypatch.setattr(interp_mod, "forbidden_terms", lambda cfg: ["cwt2"])  # case-insensitive
    with pytest.raises(PrivacyError, match="cwt2"):
        build_payload(summary, CFG)


# ---------- real client, without network ----------

def test_real_client_needs_env_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(InterpretationError, match="ANTHROPIC_API_KEY"):
        AnthropicClient(CFG)


def test_real_client_request(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    kw = AnthropicClient(CFG).request_kwargs("SYSTEM", "USER")
    assert kw["model"] == "claude-sonnet-5-5"
    assert kw["output_format"] is Commentary
    assert kw["messages"] == [{"role": "user", "content": "USER"}]
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]


def test_commentary_schema_converts_for_structured_output():
    from anthropic.lib._parse._transform import to_json_output_format
    fmt = to_json_output_format(Commentary)
    assert fmt["type"] == "json_schema" and fmt["schema"]["additionalProperties"] is False


# ---------- CLI end to end ----------

@pytest.fixture
def csv_path(tmp_path, faulty_df):
    p = tmp_path / "run.csv"
    faulty_df.to_csv(p, index=False)
    return p


def _cli(csv_path, out, *extra):
    return main([str(csv_path), "--test-id", "T0001", "--facility", "CWT2",
                 "--run-date", "2026-10-30", "--out", str(out), *extra])


def test_cli_offline_end_to_end(tmp_path, csv_path):
    out = tmp_path / "out"
    assert _cli(csv_path, out, "--interpret", "--offline") == 0
    assert {p.name for p in out.iterdir()} == {
        "T0001_summary.json", "T0001_interpretation_raw.json", "T0001_validation_log.json",
        "T0001_interpretation.json", "T0001_draft_commentary.pptx", "T0001_draft_commentary.docx"}

    titles = [s.shapes.title.text for s in Presentation(out / "T0001_draft_commentary.pptx").slides]
    assert titles and all(t.startswith(HEADING) for t in titles)
    doc_text = "\n".join(p.text for p in Document(out / "T0001_draft_commentary.docx").paragraphs)
    assert HEADING in doc_text and "REVIEW REQUIRED" in doc_text and "{" not in doc_text


def test_cli_without_interpret_writes_summary_only(tmp_path, csv_path):
    out = tmp_path / "out"
    assert _cli(csv_path, out) == 0
    assert {p.name for p in out.iterdir()} == {"T0001_summary.json"}


def test_hook_without_interpret_does_nothing(tmp_path, faulty_df):
    import argparse
    from datetime import date
    args = argparse.Namespace(interpret=False, offline=False, spec="specs/bvs_2_7.yaml")
    assert run_after_compute(args, faulty_df, test_id="T0001", facility="CWT2",
                             run_date=date(2026, 10, 30), out_dir=tmp_path) is None
    assert list(tmp_path.iterdir()) == []
