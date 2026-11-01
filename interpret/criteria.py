"""Evaluate spec criteria against computed metrics.

Margin sign convention: positive means inside the limit, negative means
outside. For '<=' margin = limit - measured; for '>=' margin = measured - limit.
"""

from __future__ import annotations

import operator
from pathlib import Path

import yaml

from .schema import Criterion, Metric

_OPS = {"<=": operator.le, ">=": operator.ge, "<": operator.lt, ">": operator.gt}


def load_spec(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def evaluate_criteria(metrics: list[Metric], spec: dict) -> list[Criterion]:
    by_name = {m.name: m for m in metrics}
    out = []
    for c in spec["criteria"]:
        if c["metric"] not in by_name:
            raise ValueError(f"Criterion '{c['name']}' refers to unknown metric '{c['metric']}'")
        m = by_name[c["metric"]]
        use_abs = bool(c.get("abs", False))
        common = dict(name=c["name"], metric=c["metric"], comparator=c["comparator"],
                      use_abs=use_abs, limit=c["limit"], unit=m.unit)

        if m.value is None:
            out.append(Criterion(**common, measured=None, margin=None, status="FAIL",
                                 note="Metric could not be computed for this run"))
            continue

        measured = abs(m.value) if use_abs else m.value
        if c["comparator"] in ("<=", "<"):
            margin = c["limit"] - measured
        else:
            margin = measured - c["limit"]
        passed = _OPS[c["comparator"]](measured, c["limit"])
        out.append(Criterion(**common, measured=measured, margin=margin,
                             status="PASS" if passed else "FAIL"))
    return out
