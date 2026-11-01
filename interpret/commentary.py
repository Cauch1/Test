"""Pydantic models for the Stage B output. The SDK turns Commentary into a JSON
schema and the API constrains the model's reply to it (structured outputs).

Count limits (2 to 5 paragraphs, one comment per criterion, one explanation per
anomaly) are checked in validate.py, not in the schema, so a violation is
reported with a readable reason instead of being silently impossible.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

UNVERIFIED = "model-suggested, unverified"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CriterionComment(_Strict):
    criterion: str  # must match a criterion name in the summary
    comment: str    # one sentence: the margin, or the reason for failure


class AnomalyExplanation(_Strict):
    anomaly_id: str  # must match a Python-detected anomaly ID, for example A01
    likely_cause: str
    evidence: str
    confidence: Literal["low", "medium", "high"]
    suggested_check: str


class ModelSuggestedAnomaly(_Strict):
    channel: str
    description: str
    reason: str
    status: Literal["model-suggested, unverified"]


class Commentary(_Strict):
    results_commentary: list[str]  # 2 to 5 paragraphs
    criteria_commentary: list[CriterionComment]
    anomaly_explanations: list[AnomalyExplanation]
    model_suggested_anomalies: list[ModelSuggestedAnomaly]
