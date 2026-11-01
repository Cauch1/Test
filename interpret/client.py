"""Model clients for Stage B. Both return a ClientResult with the raw response
text (saved to disk for audit) and the parsed Commentary.

AnthropicClient: the real API, via the official SDK's messages.parse(), which
  sends the Commentary schema as a structured-output format and returns a
  validated pydantic object.
MockClient: no network. Builds a canned reply from the summary's own values,
  so the full pipeline (including the number validator) runs offline. Pass
  canned= to return a fixed reply instead, for testing validator failures.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from .commentary import UNVERIFIED, Commentary
from .schema import Summary


class InterpretationError(Exception):
    """The API call did not produce usable commentary (refusal, truncation, no key)."""


@dataclass
class ClientResult:
    raw: str                     # raw response JSON, saved verbatim
    commentary: Commentary
    model: str
    stop_reason: Optional[str] = None


class AnthropicClient:
    def __init__(self, cfg: dict):
        import anthropic  # imported here so --offline runs without the SDK installed

        # Hard rule: the key comes from the environment only, never from code or config.
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise InterpretationError("ANTHROPIC_API_KEY is not set. Set it, or run with --offline.")
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=key)
        self.cfg = cfg

    def request_kwargs(self, system: str, user: str) -> dict:
        """Everything sent to the API, separated out so tests can inspect it offline."""
        kwargs = dict(
            model=self.cfg["model"],
            max_tokens=self.cfg["max_tokens"],
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=Commentary,  # structured output: reply must match this schema
            output_config={"effort": self.cfg["effort"]},
        )
        if self.cfg.get("refusal_fallback"):
            # If a safety classifier declines, the API reruns the request on a
            # fallback model it picks by refusal category, inside the same call.
            kwargs.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        return kwargs

    def complete(self, system: str, user: str) -> ClientResult:
        kwargs = self.request_kwargs(system, user)
        api = self.client.beta.messages if "betas" in kwargs else self.client.messages
        try:
            resp = api.parse(**kwargs)
        except self._anthropic.APIStatusError as e:
            raise InterpretationError(f"API error {e.status_code}: {e.message}") from e
        except self._anthropic.APIConnectionError as e:
            raise InterpretationError(f"Could not reach the API: {e}") from e

        if resp.stop_reason == "refusal":
            raise InterpretationError(f"Model declined the request: {resp.stop_details}")
        if resp.stop_reason == "max_tokens":
            raise InterpretationError("Reply truncated at max_tokens; raise max_tokens in config/interpret.yaml")
        if resp.parsed_output is None:
            raise InterpretationError("Reply did not parse against the Commentary schema")
        return ClientResult(raw=resp.to_json(), commentary=resp.parsed_output,
                            model=resp.model, stop_reason=resp.stop_reason)


class MockClient:
    """Offline stand-in. Quotes only values present in the summary."""

    model = "mock"

    def __init__(self, summary: Summary, canned: Optional[dict] = None):
        self.summary = summary
        self.canned = canned

    def complete(self, system: str, user: str) -> ClientResult:
        data = self.canned if self.canned is not None else self._from_summary()
        commentary = Commentary.model_validate(data)
        return ClientResult(raw=json.dumps({"mock": True, "content": data}, indent=2),
                            commentary=commentary, model=self.model, stop_reason="end_turn")

    def _from_summary(self) -> dict:
        s = self.summary
        phases = {p.id: p for p in s.test.phases}
        metrics = {m.name: m for m in s.metrics}
        pull = phases.get("pull-down")
        stab = phases.get("stabilised")

        para1 = (f"Test {s.test.test_id} was conducted in {s.test.facility} against {s.test.spec_ref}. "
                 f"Data from {s.counts.n_channels} channels were logged at a "
                 f"{s.counts.sample_interval_s:g} s interval for {s.counts.duration_min:g} min.")
        if pull and stab:
            para1 += (f" The pull-down started at {pull.start_min:g} min and the cabin was "
                      f"stabilised from {stab.start_min:g} min.")
        lines = []
        for m in s.metrics:
            if m.value is not None:
                lines.append(f"The {m.name.replace('_', ' ')} was {m.value:g} {m.unit}.")
        para2 = " ".join(lines) or "No metrics could be computed for this run."

        crit = []
        for c in s.criteria:
            chk = c.all_channels
            if chk.measured is None:
                text = f"The {c.name} criterion could not be assessed because the metric was not computed."
            elif c.status == "REVIEW_REQUIRED":
                ex = c.excluding_suspect
                text = (f"The {c.name} result was {chk.measured:g} {c.unit} with all channels and "
                        f"{ex.measured:g} {c.unit} without suspect channels, so engineer review is required.")
            else:
                text = (f"The {c.name} result was {chk.measured:g} {c.unit} against a limit of "
                        f"{c.limit:g} {c.unit}, a margin of {chk.margin:g} {c.unit}.")
            crit.append({"criterion": c.name, "comment": text})

        expl = []
        for a in s.anomalies:
            ev = ", ".join(f"{k.replace('_', ' ')} {v:g}" for k, v in a.evidence.items())
            expl.append({
                "anomaly_id": a.id,
                "likely_cause": f"Mock explanation of a {a.type} on {a.channel}.",
                "evidence": f"Detected from {a.start_min:g} min to {a.end_min:g} min; {ev}.",
                "confidence": "low",
                "suggested_check": f"Inspect the {a.channel} sensor and its wiring.",
            })

        return {
            "results_commentary": [para1, para2],
            "criteria_commentary": crit,
            "anomaly_explanations": expl,
            "model_suggested_anomalies": [{
                "channel": "none",
                "description": "Mock suggestion for pipeline testing.",
                "reason": "The mock client always adds one suggestion.",
                "status": UNVERIFIED,
            }],
        }
