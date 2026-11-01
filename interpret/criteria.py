"""Evaluate spec criteria against computed metrics.

Margin sign convention: positive means inside the limit, negative means
outside. For '<=' margin = limit - measured; for '>=' margin = measured - limit.

Each criterion is checked twice when its metric has suspect channels: once with
all channels and once with the suspect ones left out.
- Flag-only mode (default): if the two verdicts differ, status is
  REVIEW_REQUIRED and the engineer decides. If they agree, that verdict stands.
- Auto-exclude mode (opt-in): status is the excluding-suspect verdict.
"""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Optional

import yaml

from .schema import Check, Criterion, Metric

_OPS = {"<=": operator.le, ">=": operator.ge, "<": operator.lt, ">": operator.gt}


def load_spec(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _check(value: Optional[float], c: dict) -> Check:
    if value is None:
        return Check(measured=None, margin=None, verdict="FAIL")
    measured = abs(value) if c.get("abs", False) else value
    margin = c["limit"] - measured if c["comparator"] in ("<=", "<") else measured - c["limit"]
    passed = _OPS[c["comparator"]](measured, c["limit"])
    return Check(measured=measured, margin=margin, verdict="PASS" if passed else "FAIL")


def evaluate_criteria(metrics: list[Metric], spec: dict, auto_exclude: bool = False) -> list[Criterion]:
    by_name = {m.name: m for m in metrics}
    out = []
    for c in spec["criteria"]:
        if c["metric"] not in by_name:
            raise ValueError(f"Criterion '{c['name']}' refers to unknown metric '{c['metric']}'")
        m = by_name[c["metric"]]

        full = _check(m.value, c)
        excl = _check(m.value_excluding_suspect, c) if m.suspect_channels else None

        note = None
        if excl is None:
            status, basis = full.verdict, "all_channels"
        elif auto_exclude:
            status, basis = excl.verdict, "excluding_suspect"
            note = f"Auto-exclusion of {', '.join(m.suspect_channels)}; all-channel verdict {full.verdict}"
        elif excl.verdict != full.verdict:
            status, basis = "REVIEW_REQUIRED", "all_channels"
            note = (f"Verdict depends on suspect channels {', '.join(m.suspect_channels)}: "
                    f"{full.verdict} with all channels, {excl.verdict} without them")
        else:
            status, basis = full.verdict, "all_channels"
        if m.value is None:
            note = "Metric could not be computed for this run"

        out.append(Criterion(id=c.get("id", c["metric"]), name=c["name"], metric=c["metric"], comparator=c["comparator"],
                             use_abs=bool(c.get("abs", False)), limit=c["limit"], unit=m.unit,
                             all_channels=full, excluding_suspect=excl, status=status,
                             basis=basis, note=note))
    return out
