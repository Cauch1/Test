"""Stage B: summary JSON -> model -> validated commentary, with an audit trail.

The model returns templates with placeholder keys; validate() checks them and
Python fills in the values. Files written next to the summary JSON:
  <test_id>_interpretation_raw.json   raw API response, always written
  <test_id>_validation_log.json       every check result, always written
  <test_id>_interpretation.json       filled commentary plus the template, only if checks pass
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .client import AnthropicClient, MockClient
from .commentary import Commentary
from .schema import Summary
from .placeholders import build_catalogue, catalogue_text
from .validate import CommentaryValidationError, validate

ROOT = Path(__file__).resolve().parent.parent


class PrivacyError(Exception):
    """The payload contains something that must never be sent to the API."""


def load_cfg(path: str | Path = ROOT / "config" / "interpret.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def forbidden_terms(cfg: dict) -> list[str]:
    """Client names and other banned strings, from a local untracked file."""
    p = ROOT / cfg["forbidden_terms_file"]
    if not p.exists():
        return []
    return [str(t) for t in (yaml.safe_load(p.read_text()) or {}).get("terms", [])]


def build_payload(summary: Summary, cfg: dict) -> str:
    """The user message: the summary JSON and nothing else.

    Hard rule 2 is enforced here: the summary schema holds no samples, and any
    configured forbidden term (client names) anywhere in it stops the run.
    """
    text = summary.model_dump_json(indent=2)
    hits = [t for t in forbidden_terms(cfg) if re.search(re.escape(t), text, re.IGNORECASE)]
    if hits:
        raise PrivacyError(f"Summary contains forbidden term(s) {hits}; fix the test ID or "
                           "channel names before sending")
    if len(text) > cfg["max_payload_chars"]:
        raise PrivacyError(f"Summary is {len(text)} characters, above max_payload_chars; "
                           "it should hold statistics, not samples")
    keys = catalogue_text(build_catalogue(summary, cfg["display_decimals"]))
    return ("Write the commentary for this test summary, using only the placeholder keys "
            "listed below for any value.\n\n<summary>\n" + text + "\n</summary>\n\n"
            "<placeholders>\n" + keys + "\n</placeholders>")


def make_client(summary: Summary, cfg: dict, offline: bool):
    return MockClient(summary) if offline else AnthropicClient(cfg)


def interpret(summary_path: str | Path, *, offline: bool = False, cfg: dict | None = None,
              client=None) -> Commentary:
    """Run Stage B on a written summary. Raises CommentaryValidationError on a
    failed check, after the raw response and log are on disk."""
    cfg = cfg or load_cfg()
    summary_path = Path(summary_path)
    summary_text = summary_path.read_text()
    summary = Summary.model_validate_json(summary_text)
    out_dir, tid = summary_path.parent, summary.test.test_id

    system = (ROOT / cfg["prompt_file"]).read_text()
    user = build_payload(summary, cfg)
    client = client or make_client(summary, cfg, offline)
    result = client.complete(system, user)

    (out_dir / f"{tid}_interpretation_raw.json").write_text(result.raw)
    meta = {
        "model": result.model,
        "stop_reason": result.stop_reason,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Hash ties the commentary to the exact summary it was written from.
        "summary_sha256": hashlib.sha256(summary_text.encode()).hexdigest(),
        "prompt_file": cfg["prompt_file"],
    }
    log_path = out_dir / f"{tid}_validation_log.json"
    try:
        log, filled = validate(result.commentary, summary, rel=cfg["tolerance"]["relative"],
                               abs_tol=cfg["tolerance"]["absolute"],
                               style_fatal=cfg["style_checks_fatal"],
                               decimals=cfg["display_decimals"],
                               extra_allowlist=cfg.get("digit_allowlist", []))
    except CommentaryValidationError as e:
        log_path.write_text(json.dumps({**meta, **e.log}, indent=2))
        raise
    log_path.write_text(json.dumps({**meta, **log}, indent=2))
    (out_dir / f"{tid}_interpretation.json").write_text(json.dumps(
        {"meta": meta, "status": "DRAFT, engineer review required",
         "commentary": filled.model_dump(),
         "template": result.commentary.model_dump()}, indent=2))
    return filled
