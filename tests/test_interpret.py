"""Stage B: number validator, structure checks, privacy guards, clients, CLI."""

import copy
import json

import pytest
from docx import Document
from pptx import Presentation

from interpret import interpret as interp_mod
from interpret.cli import main, run_after_compute
from interpret.client import AnthropicClient, InterpretationError, MockClient
from interpret.commentary import Commentary
from interpret.deliverables import HEADING
from interpret.interpret import PrivacyError, build_payload, interpret, load_cfg
from interpret.summary import build_summary, write_summary
from interpret.validate import (CommentaryValidationError, extract_numbers, summary_numbers,
                                validate)

from .synthetic import make_pulldown


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
    """Mock commentary built only from summary values."""
    return MockClient(summary)._from_summary()


def _with_paragraph(data: dict, text: str) -> Commentary:
    d = copy.deepcopy(data)
    d["results_commentary"][0] += " " + text
    return Commentary.model_validate(d)


def _metric(summary, name):
    return next(m for m in summary.metrics if m.name == name)


# ---------- number extraction ----------

def test_extract_ignores_identifiers():
    text = "Channel tc_r2_r_foot (A03) read 25.08 degC; gradient −2.61 degC; row 1."
    assert extract_numbers(text) == [25.08, -2.61, 1.0]


def test_summary_numbers_include_strings(summary):
    nums = summary_numbers(summary)
    assert 2.7 in nums          # from "BVS 2.7"
    assert 2026.0 in nums       # from the run date
    assert 12.0 in nums         # channel count


# ---------- number validator ----------

def test_accepts_correct_numbers(summary, good):
    log = validate(Commentary.model_validate(good), summary)
    assert log["passed"]


def test_rejects_made_up_number_and_names_sentence(summary, good):
    c = _with_paragraph(good, "The coolest seat reached 19.4 degC at 95 min.")
    with pytest.raises(CommentaryValidationError) as e:
        validate(c, summary)
    msg = str(e.value)
    assert "19.4 degC not found" in msg and "95 min not found" in msg
    assert 'Sentence: "The coolest seat reached 19.4 degC at 95 min."' in msg
    assert "results_commentary[0]" in msg


@pytest.mark.parametrize("quoted, ok", [
    (23.6, True),    # 23.51 + 0.09: inside the 0.1 absolute tolerance
    (23.7, False),   # 23.51 + 0.19: outside
])
def test_absolute_tolerance(summary, good, quoted, ok):
    assert _metric(summary, "stabilised_cabin_mean").value == 23.51
    c = _with_paragraph(good, f"The mean was {quoted} degC.")
    if ok:
        validate(c, summary)
    else:
        with pytest.raises(CommentaryValidationError):
            validate(c, summary)


def test_relative_tolerance_on_large_values(summary, good):
    """0.5 % beats 0.1 units above 20: 2026 +/- 10.13 (from the run date year)."""
    validate(_with_paragraph(good, "Value 2035 was noted."), summary)
    with pytest.raises(CommentaryValidationError):
        validate(_with_paragraph(good, "Value 2040 was noted."), summary)


def test_sign_and_unicode_minus_accepted(summary, good):
    validate(_with_paragraph(good, "The gradient was −2.61 degC, a magnitude of 2.61 degC."), summary)


def _false_pass_rate(summary, good, values, unit, strict=True):
    """Share of invented values that the checker would accept in a sentence
    naming no channel or anomaly."""
    from interpret.validate import matches
    if not strict:
        allowed = summary_numbers(summary)
        return sum(matches(v, allowed, 0.005, 0.1) for v in values) / len(values)
    passed = 0
    for v in values:
        c = _with_paragraph(good, f"A reading of {v:.2f} {unit} was noted.")
        try:
            validate(c, summary)
            passed += 1
        except CommentaryValidationError:
            pass
    return passed / len(values)


@pytest.mark.parametrize("lo, hi, max_rate", [(0, 5, 0.45), (20, 45, 0.25)])
def test_unit_and_scope_cut_false_passes(summary, good, lo, hi, max_rate):
    """Invented degC values: the unit- and scope-aware check accepts far fewer
    than the loose any-number check. Rates are measured on the faulty fixture."""
    values = [lo + (hi - lo) * k / 200 for k in range(200)]
    loose = _false_pass_rate(summary, good, values, "degC", strict=False)
    strict = _false_pass_rate(summary, good, values, "degC", strict=True)
    print(f"degC {lo}-{hi}: loose {loose:.0%}, strict {strict:.0%}")
    assert strict < loose
    assert strict <= max_rate


def test_channel_value_needs_channel_named(summary, good):
    # Frozen head sensor: its stabilised mean (about 29.05 degC) is unlike any global value.
    ch = next(c for c in summary.channels if c.name == "tc_r1_r_head")
    v = ch.phase_means["stabilised"]
    with pytest.raises(CommentaryValidationError, match="values of tc_r1_r_head.*does not name"):
        validate(_with_paragraph(good, f"One head sensor averaged {v:g} degC."), summary)
    validate(_with_paragraph(good, f"Channel tc_r1_r_head averaged {v:g} degC."), summary)


def test_wrong_unit_fails(summary, good):
    with pytest.raises(CommentaryValidationError, match="only with unit degC"):
        validate(_with_paragraph(good, "The cabin mean was 23.51 min."), summary)


def test_anomaly_values_need_id_outside_explanation(summary, good):
    a = summary.anomalies[0]
    sentence = f"The fault lasted from {a.start_min:g} to {a.end_min:g} min."
    with pytest.raises(CommentaryValidationError, match=f"values of {a.id}.*does not name"):
        validate(_with_paragraph(good, sentence), summary)
    validate(_with_paragraph(good, f"Anomaly {a.id} lasted from {a.start_min:g} to {a.end_min:g} min."), summary)


def test_anomaly_values_allowed_in_own_explanation(summary, good):
    d = copy.deepcopy(good)
    a = summary.anomalies[0]
    d["anomaly_explanations"][0]["evidence"] = f"It ran between {a.start_min:g} and {a.end_min:g} min."
    validate(Commentary.model_validate(d), summary)


def test_numbers_in_every_field_are_checked(summary, good):
    d = copy.deepcopy(good)
    # 47.3 is far from every summary value. (A small number such as 0.3 would
    # pass: with 0.1 absolute tolerance it lands near some summary value.)
    d["anomaly_explanations"][0]["suggested_check"] = "Check the sensor against a 47.3 degC reference."
    d["model_suggested_anomalies"][0]["reason"] = "A dip of 7.7 degC was seen."
    with pytest.raises(CommentaryValidationError) as e:
        validate(Commentary.model_validate(d), summary)
    assert "suggested_check" in str(e.value) and "model_suggested_anomalies" in str(e.value)


# ---------- structure ----------

def test_missing_anomaly_explanation_fails(summary, good):
    d = copy.deepcopy(good)
    d["anomaly_explanations"].pop(0)
    with pytest.raises(CommentaryValidationError, match="A01 explained 0 times"):
        validate(Commentary.model_validate(d), summary)


def test_invented_anomaly_id_fails(summary, good):
    d = copy.deepcopy(good)
    extra = dict(d["anomaly_explanations"][0], anomaly_id="A99")
    d["anomaly_explanations"].append(extra)
    with pytest.raises(CommentaryValidationError, match="A99 is not a Python-detected anomaly"):
        validate(Commentary.model_validate(d), summary)


def test_paragraph_count_and_criteria_coverage(summary, good):
    d = copy.deepcopy(good)
    d["results_commentary"] = d["results_commentary"][:1]
    d["criteria_commentary"].pop()
    with pytest.raises(CommentaryValidationError) as e:
        validate(Commentary.model_validate(d), summary)
    assert "1 paragraphs" in str(e.value) and "commented 0 times" in str(e.value)


def test_model_suggested_must_be_labelled_unverified(good):
    d = copy.deepcopy(good)
    d["model_suggested_anomalies"][0]["status"] = "confirmed"
    with pytest.raises(Exception):
        Commentary.model_validate(d)


# ---------- style ----------

def test_style_warns_by_default_and_fails_when_configured(summary, good):
    c = _with_paragraph(good, "The cabin settled — eventually — near 23.51degC.")
    log = validate(c, summary)
    kinds = {p["detail"] for p in log["problems"]}
    assert log["passed"] and {"contains an em or en dash", "no space between value and unit"} <= kinds
    with pytest.raises(CommentaryValidationError):
        validate(c, summary, style_fatal=True)


# ---------- Stage B orchestration and audit trail ----------

def test_failed_validation_writes_raw_and_log_only(tmp_path, summary, good):
    path = write_summary(summary, tmp_path)
    bad = copy.deepcopy(good)
    bad["results_commentary"][1] += " The rear row peaked at 31.2 degC."
    with pytest.raises(CommentaryValidationError):
        interpret(path, client=MockClient(summary, canned=bad))
    assert (tmp_path / "T0001_interpretation_raw.json").exists()
    log = json.loads((tmp_path / "T0001_validation_log.json").read_text())
    assert log["passed"] is False and log["problems"][0]["sentence"] == "The rear row peaked at 31.2 degC."
    assert not (tmp_path / "T0001_interpretation.json").exists()


def test_successful_run_records_summary_hash(tmp_path, summary):
    import hashlib
    path = write_summary(summary, tmp_path)
    interpret(path, offline=True)
    out = json.loads((tmp_path / "T0001_interpretation.json").read_text())
    assert out["meta"]["summary_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert out["status"].startswith("DRAFT")


# ---------- privacy ----------

def test_payload_is_summary_only(summary):
    payload = build_payload(summary, load_cfg())
    assert "time_s" not in payload
    assert len(payload) < 20_000


def test_forbidden_term_blocks_send(summary, monkeypatch):
    monkeypatch.setattr(interp_mod, "forbidden_terms", lambda cfg: ["cwt2"])  # case-insensitive
    with pytest.raises(PrivacyError, match="cwt2"):
        build_payload(summary, load_cfg())


# ---------- real client, without network ----------

def test_real_client_needs_env_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(InterpretationError, match="ANTHROPIC_API_KEY"):
        AnthropicClient(load_cfg())


def test_real_client_request(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    cfg = load_cfg()
    kw = AnthropicClient(cfg).request_kwargs("SYSTEM", "USER")
    assert kw["model"] == "claude-sonnet-5-5"
    assert kw["output_format"] is Commentary
    assert kw["messages"] == [{"role": "user", "content": "USER"}]
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]


def test_commentary_schema_converts_for_structured_output():
    from anthropic.lib._parse._transform import to_json_output_format
    fmt = to_json_output_format(Commentary)
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["additionalProperties"] is False


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
    names = {p.name for p in out.iterdir()}
    assert names == {"T0001_summary.json", "T0001_interpretation_raw.json", "T0001_validation_log.json",
                     "T0001_interpretation.json", "T0001_draft_commentary.pptx",
                     "T0001_draft_commentary.docx"}

    titles = [s.shapes.title.text for s in Presentation(out / "T0001_draft_commentary.pptx").slides]
    assert titles and all(t.startswith(HEADING) for t in titles)
    doc_text = [p.text for p in Document(out / "T0001_draft_commentary.docx").paragraphs]
    assert HEADING in doc_text
    assert any("REVIEW_REQUIRED" in t or "engineer review is required" in t for t in doc_text)


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
