"""Model clients for Stage B. Both return a ClientResult with the raw response
text (saved to disk for audit) and the parsed Commentary.

AnthropicClient: the real API, via the official SDK's messages.parse(), which
  sends the Commentary schema as a structured-output format and returns a
  validated pydantic object.
MockClient: no network. Builds a canned template reply from the summary's
  keys, so the full pipeline (placeholder checks, fill, final number
  assertion) runs offline. Pass canned= to return a fixed reply instead, for
  testing validator failures.
"""

from __future__ import annotations

import json
import os
import re
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
    """Offline stand-in. Writes templates with placeholder keys, like the real model."""

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
        """A template reply: prose plus placeholder keys, no numeric values."""
        s = self.summary
        ids = {p.id for p in s.test.phases}

        para1 = ("Test {test.id} was conducted in {test.facility} against {test.spec_ref} on "
                 "{test.run_date}. Data from {counts.n_channels} channels were logged at a "
                 "{counts.sample_interval} interval for {counts.duration}.")
        if {"pull-down", "stabilised"} <= ids:
            para1 += (" The pull-down started at {phase.pull-down.start} and the cabin was "
                      "stabilised from {phase.stabilised.start}.")
        lines = [f"The {m.name.replace('_', ' ')} was {{metric.{m.name}}}."
                 for m in s.metrics if m.value is not None]
        para2 = " ".join(lines) or "No metrics could be computed for this run."

        crit = []
        for c in s.criteria:
            k = f"criteria.{c.id}"
            if c.all_channels.measured is None:
                text = f"The {{{k}.name}} criterion could not be assessed because the metric was not computed."
            elif c.status == "REVIEW_REQUIRED":
                text = (f"The {{{k}.name}} result was {{{k}.measured}} with all channels and "
                        f"{{{k}.measured_excluding}} without suspect channels "
                        f"({{metric.{c.metric}.suspect_channels}}), so the status is {{{k}.status}}.")
            else:
                text = (f"The {{{k}.name}} result was {{{k}.measured}} against a limit of "
                        f"{{{k}.limit}}, a margin of {{{k}.margin}}.")
            crit.append({"criterion": c.id, "comment": text})

        expl = []
        for a in s.anomalies:
            k = f"anomaly.{a.id}"
            label = lambda key: re.sub(r"_(degC|s|min)$", "", key).replace("_", " ")
            ev = ", ".join(f"{label(key)} {{{k}.evidence.{key}}}" for key in a.evidence)
            expl.append({
                "anomaly_id": a.id,
                "likely_cause": f"Mock explanation of a {{{k}.type}} on {{{k}.channel}}.",
                "evidence": f"Detected from {{{k}.start}} to {{{k}.end}}; {ev}.",
                "confidence": "low",
                "suggested_check": f"Inspect the {{{k}.channel}} sensor and its wiring.",
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
