"""Checks on the model's commentary before anyone sees it.

1. Numbers: every number in every text field must exist in the summary JSON,
   within max(0.5 % of the value, 0.1 units). Fatal.
2. Structure: every Python anomaly explained exactly once, no invented anomaly
   IDs, one comment per criterion, 2 to 5 results paragraphs. Fatal.
3. Style: em dashes and a missing space between value and unit. Logged as
   warnings; fatal only when style_checks_fatal is true in config.

Numbers are compared by absolute value: "a gradient of 3.83 degC" correctly
quotes a summary value of -3.83.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from .commentary import Commentary
from .schema import Summary

# A number not preceded by a letter, underscore, digit or point, so the "2" in
# tc_r2_l_foot and the "01" in A01 are not read as numbers. Letters may follow
# (a unit written without a space, 23.51degC, still reads as 23.51). Accepts
# the Unicode minus sign the model sometimes writes.
NUMBER = re.compile(r"(?<![\w.])[-+−]?\d+(?:\.\d+)?(?!\.?\d)")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
UNIT_NO_SPACE = re.compile(r"\d(?:degC|kW|W|min|s|h|%|kPa|Pa|m/s|V|A)\b")
DASHES = ("—", "–")  # em dash, en dash


class CommentaryValidationError(Exception):
    """Raised when the commentary fails a fatal check. str() lists every problem."""


@dataclass
class Problem:
    kind: str       # number, structure, style
    field: str      # where in the commentary
    detail: str
    sentence: str = ""


def extract_numbers(text: str) -> list[float]:
    return [float(m.replace("−", "-")) for m in NUMBER.findall(text)]


def summary_numbers(summary: Summary) -> set[float]:
    """Every number in the summary: numeric fields, plus numbers inside strings
    (spec reference, location groups, method text, dates)."""
    found: set[float] = set()

    def walk(x):
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            found.add(abs(float(x)))
        elif isinstance(x, str):
            found.update(abs(v) for v in extract_numbers(x))
        elif isinstance(x, dict):
            for k, v in x.items():
                walk(k)
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(json.loads(summary.model_dump_json()))
    return found


def matches(value: float, allowed: set[float], rel: float, abs_tol: float) -> bool:
    v = abs(value)
    return any(abs(v - a) <= max(rel * a, abs_tol) for a in allowed)


def _sentences(text: str) -> list[str]:
    return [s for s in SENTENCE_END.split(text.strip()) if s]


def _text_fields(c: Commentary):
    """Yield (field path, text) for every free-text field."""
    for i, p in enumerate(c.results_commentary):
        yield f"results_commentary[{i}]", p
    for i, cc in enumerate(c.criteria_commentary):
        yield f"criteria_commentary[{i}] ({cc.criterion})", cc.comment
    for i, a in enumerate(c.anomaly_explanations):
        for name in ("likely_cause", "evidence", "suggested_check"):
            yield f"anomaly_explanations[{i}] ({a.anomaly_id}).{name}", getattr(a, name)
    for i, m in enumerate(c.model_suggested_anomalies):
        for name in ("description", "reason"):
            yield f"model_suggested_anomalies[{i}].{name}", getattr(m, name)


def check_numbers(c: Commentary, summary: Summary, rel: float = 0.005,
                  abs_tol: float = 0.1) -> list[Problem]:
    allowed = summary_numbers(summary)
    problems = []
    for field, text in _text_fields(c):
        for sentence in _sentences(text):
            for v in extract_numbers(sentence):
                if not matches(v, allowed, rel, abs_tol):
                    problems.append(Problem("number", field,
                                            f"{v:g} not found in summary", sentence))
    return problems


def check_structure(c: Commentary, summary: Summary) -> list[Problem]:
    problems = []
    n = len(c.results_commentary)
    if not 2 <= n <= 5:
        problems.append(Problem("structure", "results_commentary", f"{n} paragraphs, expected 2 to 5"))

    expected_ids = [a.id for a in summary.anomalies]
    got_ids = [a.anomaly_id for a in c.anomaly_explanations]
    for aid in expected_ids:
        if got_ids.count(aid) != 1:
            problems.append(Problem("structure", "anomaly_explanations",
                                    f"{aid} explained {got_ids.count(aid)} times, expected once"))
    for aid in set(got_ids) - set(expected_ids):
        problems.append(Problem("structure", "anomaly_explanations",
                                f"{aid} is not a Python-detected anomaly; new findings belong "
                                "in model_suggested_anomalies"))

    expected_crit = [cr.name for cr in summary.criteria]
    got_crit = [cc.criterion for cc in c.criteria_commentary]
    for name in expected_crit:
        if got_crit.count(name) != 1:
            problems.append(Problem("structure", "criteria_commentary",
                                    f"'{name}' commented {got_crit.count(name)} times, expected once"))
    for name in set(got_crit) - set(expected_crit):
        problems.append(Problem("structure", "criteria_commentary", f"unknown criterion '{name}'"))
    return problems


def check_style(c: Commentary) -> list[Problem]:
    problems = []
    for field, text in _text_fields(c):
        for sentence in _sentences(text):
            if any(d in sentence for d in DASHES):
                problems.append(Problem("style", field, "contains an em or en dash", sentence))
            if UNIT_NO_SPACE.search(sentence):
                problems.append(Problem("style", field, "no space between value and unit", sentence))
    return problems


def validate(c: Commentary, summary: Summary, *, rel: float = 0.005, abs_tol: float = 0.1,
             style_fatal: bool = False) -> dict:
    """Run every check. Returns the log dict; raises CommentaryValidationError
    (after building the log) if any fatal check fails."""
    problems = check_numbers(c, summary, rel, abs_tol) + check_structure(c, summary) + check_style(c)
    fatal_kinds = {"number", "structure"} | ({"style"} if style_fatal else set())
    fatal = [p for p in problems if p.kind in fatal_kinds]
    log = {"passed": not fatal, "problems": [asdict(p) for p in problems],
           "tolerance": {"relative": rel, "absolute": abs_tol}}
    if fatal:
        lines = [f"[{p.kind}] {p.field}: {p.detail}" + (f'\n    Sentence: "{p.sentence}"' if p.sentence else "")
                 for p in fatal]
        err = CommentaryValidationError(f"{len(fatal)} commentary check(s) failed:\n" + "\n".join(lines))
        err.log = log  # type: ignore[attr-defined]
        raise err
    return log
