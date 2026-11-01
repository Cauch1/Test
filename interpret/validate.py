"""Checks on the model's commentary before anyone sees it.

The model writes templates: prose with placeholder keys such as
{metric.time_to_target}, and no numeric values of its own. Order of checks:

1. Placeholders (fatal): every key exists in the catalogue built from the
   summary; no malformed braces; no digit left in the model's own text
   except allowlisted IDs and spec references (test ID, facility, spec
   reference, anomaly IDs, channel IDs, digit_allowlist in config).
2. Structure (fatal): every Python anomaly explained exactly once, no
   invented anomaly IDs, one comment per criterion ID, 2 to 5 paragraphs.
3. Fill: Python replaces each key with the formatted value and unit.
4. Final assertion (fatal): the unit-aware number check runs on the filled
   text. Every number must match a summary value with the same unit, within
   max(0.5 % of the value, 0.1 units). This guards the fill and formatting
   code; with correct placeholders it cannot fail.
5. Warnings (logged): channel or anomaly values quoted in a sentence that
   does not name the channel or anomaly; spelled-out numbers; em dashes and
   unit spacing (fatal only when style_checks_fatal is true).

Numbers are compared by absolute value: "a gradient of 3.83 degC" correctly
quotes a summary value of -3.83.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from .commentary import Commentary
from .placeholders import (DIGIT, NUMBER_WORDS, PLACEHOLDER, allowlist, build_catalogue,
                           fill, strip_allowed)
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
    return [float(m.replace("\u2212", "-")) for m in NUMBER.findall(text)]


# Units the checker recognises after a number, with spellings normalised.
UNIT_AFTER = re.compile(
    r"\s?(degC|deg C|\u00b0C|kW|W|minutes|mins|min|seconds|secs|sec|s|hours|h|%|kPa|Pa|m/s|V|A)\b")
UNIT_ALIASES = {"deg C": "degC", "\u00b0C": "degC", "minutes": "min", "mins": "min",
                "seconds": "s", "secs": "s", "sec": "s", "hours": "h"}
# Text allowed between two numbers for the second number's unit to carry back.
LIST_GAP = re.compile(r"^\s*(?:,|and|to|or|-|\u2013|,\s*and|,\s*or)\s*$")


@dataclass
class Quoted:
    value: float
    unit: str | None  # None when no unit follows (counts, row numbers, years)
    text: str


def quoted_numbers(sentence: str) -> list[Quoted]:
    """Numbers in a sentence with the unit that follows each one."""
    found = []
    for m in NUMBER.finditer(sentence):
        u = UNIT_AFTER.match(sentence, m.end())
        unit = UNIT_ALIASES.get(u.group(1), u.group(1)) if u else None
        found.append((m, unit))
    # Carry a unit back through lists and ranges: "3.0 and 4.0 degC".
    for i in range(len(found) - 2, -1, -1):
        m, unit = found[i]
        nxt, nxt_unit = found[i + 1]
        if unit is None and nxt_unit and LIST_GAP.match(sentence[m.end():nxt.start()]):
            found[i] = (m, nxt_unit)
    return [Quoted(float(m.group().replace("\u2212", "-")), unit, m.group()) for m, unit in found]


@dataclass(frozen=True)
class Known:
    value: float
    unit: str | None
    scope: str  # "global", "channel:<name>", "anomaly:<id>"
    source: str  # where in the summary, for error messages


def _evidence_unit(key: str, channel_unit: str) -> str | None:
    if key.endswith("_degC"):
        return "degC"
    if key.endswith("_s"):
        return "s"
    if key.endswith("_min"):
        return "min"
    if key in ("min_value", "max_value", "range_low", "range_high"):
        return channel_unit
    return None


def known_values(summary: Summary) -> list[Known]:
    """Every number in the summary, tagged with its unit and scope."""
    out: list[Known] = []
    add = lambda v, unit, scope, src: v is not None and out.append(Known(abs(float(v)), unit, scope, src))
    units = {ch.name: ch.unit for ch in summary.channels}

    # Numbers inside any text field are global, with the unit that follows them
    # if any: "BVS 2.7", "Time to 25 degC at breath height", "row 1", the date.
    def strings(x):
        if isinstance(x, str):
            yield x
        elif isinstance(x, dict):
            for k, v in x.items():
                yield from strings(k)
                yield from strings(v)
        elif isinstance(x, list):
            for v in x:
                yield from strings(v)
    for text in strings(json.loads(summary.model_dump_json())):
        for q in quoted_numbers(text):
            add(q.value, q.unit, "global", f'text "{text[:40]}"')
    for p in summary.test.phases:
        add(p.start_min, "min", "global", f"phase {p.id} start")
        add(p.end_min, "min", "global", f"phase {p.id} end")
    c = summary.counts
    add(c.n_channels, None, "global", "counts.n_channels")
    add(c.n_samples, None, "global", "counts.n_samples")
    add(c.duration_min, "min", "global", "counts.duration_min")
    add(c.sample_interval_s, "s", "global", "counts.sample_interval_s")

    for m in summary.metrics:
        add(m.value, m.unit, "global", f"metric {m.name}")
        add(m.value_excluding_suspect, m.unit, "global", f"metric {m.name} excluding suspect")
        if m.window:
            add(m.window.start_min, "min", "global", f"metric {m.name} window start")
            add(m.window.end_min, "min", "global", f"metric {m.name} window end")
    for cr in summary.criteria:
        add(cr.limit, cr.unit, "global", f"criterion {cr.name} limit")
        for label, chk in (("all channels", cr.all_channels), ("excluding suspect", cr.excluding_suspect)):
            if chk:
                add(chk.measured, cr.unit, "global", f"criterion {cr.name} measured, {label}")
                add(chk.margin, cr.unit, "global", f"criterion {cr.name} margin, {label}")

    for ch in summary.channels:
        scope = f"channel:{ch.name}"
        add(ch.min, ch.unit, scope, f"{ch.name} min")
        add(ch.max, ch.unit, scope, f"{ch.name} max")
        for pid, v in ch.phase_means.items():
            add(v, ch.unit, scope, f"{ch.name} {pid} mean")
    for a in summary.anomalies:
        scope = f"anomaly:{a.id}"
        add(a.start_min, "min", scope, f"{a.id} start")
        add(a.end_min, "min", scope, f"{a.id} end")
        for k, v in a.evidence.items():
            add(v, _evidence_unit(k, units.get(a.channel, "degC")), scope, f"{a.id} {k}")
    return out


def summary_numbers(summary: Summary) -> set[float]:
    """Every number in the summary, ignoring unit and scope (the old, loose check)."""
    return {k.value for k in known_values(summary)}


def matches(value: float, allowed, rel: float, abs_tol: float) -> bool:
    v = abs(value)
    return any(abs(v - a) <= max(rel * a, abs_tol) for a in allowed)


def scopes_for(sentence: str, summary: Summary, base: set[str]) -> set[str]:
    """Scopes a sentence unlocks by naming channels or anomaly IDs."""
    scopes = {"global"} | base
    for ch in summary.channels:
        if ch.name in sentence:
            scopes.add(f"channel:{ch.name}")
    for a in summary.anomalies:
        if re.search(rf"\b{a.id}\b", sentence) or a.channel in sentence:
            scopes.add(f"anomaly:{a.id}")
    return scopes


def _sentences(text: str) -> list[str]:
    return [s for s in SENTENCE_END.split(text.strip()) if s]


def _text_fields(c: Commentary, summary: Summary | None = None):
    """Yield (field path, text, base scopes) for every free-text field.

    Inside an anomaly explanation, that anomaly and its channel are in scope
    without being named again.
    """
    channel_of = {a.id: a.channel for a in summary.anomalies} if summary else {}
    for i, p in enumerate(c.results_commentary):
        yield f"results_commentary[{i}]", p, set()
    for i, cc in enumerate(c.criteria_commentary):
        yield f"criteria_commentary[{i}] ({cc.criterion})", cc.comment, set()
    for i, a in enumerate(c.anomaly_explanations):
        base = {f"anomaly:{a.anomaly_id}"}
        if a.anomaly_id in channel_of:
            base.add(f"channel:{channel_of[a.anomaly_id]}")
        for name in ("likely_cause", "evidence", "suggested_check"):
            yield f"anomaly_explanations[{i}] ({a.anomaly_id}).{name}", getattr(a, name), base
    for i, m in enumerate(c.model_suggested_anomalies):
        for name in ("description", "reason"):
            yield f"model_suggested_anomalies[{i}].{name}", getattr(m, name), {f"channel:{m.channel}"}


def check_numbers(c: Commentary, summary: Summary, rel: float = 0.005,
                  abs_tol: float = 0.1, scoped: bool = True,
                  ignore: list[str] | None = None) -> list[Problem]:
    """Unit-aware number check. scoped=False accepts channel and anomaly values
    in any sentence (used on filled templates, where provenance is known).
    ignore: allowlisted tokens (IDs, spec references) whose digits are not values."""
    known = known_values(summary)
    every_scope = {k.scope for k in known}
    problems = []
    for field, text, base in _text_fields(c, summary):
        for sentence in _sentences(text):
            scopes = scopes_for(sentence, summary, base) if scoped else every_scope
            for q in quoted_numbers(strip_allowed(sentence, ignore or [])):
                same_unit = [k for k in known if k.unit == q.unit]
                in_scope = [k.value for k in same_unit if k.scope in scopes]
                if matches(q.value, in_scope, rel, abs_tol):
                    continue
                problems.append(Problem("number", field, _explain(q, same_unit, known, rel, abs_tol), sentence))
    return problems


def _explain(q: Quoted, same_unit: list[Known], known: list[Known], rel: float, abs_tol: float) -> str:
    """Say why a number failed, and point at the fix when one exists."""
    label = f"{q.text} {q.unit}" if q.unit else q.text
    out_of_scope = sorted((k for k in same_unit if matches(q.value, [k.value], rel, abs_tol)),
                          key=lambda k: abs(abs(q.value) - k.value))
    if out_of_scope:
        names = list(dict.fromkeys(k.scope.split(":", 1)[1] for k in out_of_scope))
        shown = ", ".join(names[:4]) + (f" and {len(names) - 4} more" if len(names) > 4 else "")
        return (f"{label} matches values of {shown}, but the sentence does not name any of them; "
                "name the channel or anomaly in the same sentence")
    other_unit = [k for k in known if matches(q.value, [k.value], rel, abs_tol)]
    if other_unit:
        k = other_unit[0]
        return (f"{label} matches {k.source} only with unit {k.unit or 'none'}; "
                "quote the unit as the summary gives it")
    return f"{label} not found in summary"


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

    expected_crit = [cr.id for cr in summary.criteria]
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
    for field, text, _ in _text_fields(c):
        for sentence in _sentences(text):
            if any(d in sentence for d in DASHES):
                problems.append(Problem("style", field, "contains an em or en dash", sentence))
            if UNIT_NO_SPACE.search(sentence):
                problems.append(Problem("style", field, "no space between value and unit", sentence))
    return problems


def check_placeholders(c: Commentary, summary: Summary, cat: dict, tokens: list[str]) -> list[Problem]:
    problems = []
    owner_channel = {a.id: a.channel for a in summary.anomalies}
    for field, text, base in _text_fields(c, summary):
        for sentence in _sentences(text):
            keys = PLACEHOLDER.findall(sentence)
            for k in keys:
                if k not in cat:
                    problems.append(Problem("placeholder", field, f"unknown key {{{k}}}", sentence))
            own = strip_allowed(sentence, tokens)
            if "{" in own or "}" in own:
                problems.append(Problem("placeholder", field, "malformed placeholder braces", sentence))
            if DIGIT.search(own):
                stray = " ".join(sorted(set(re.findall(r"\S*\d\S*", own))))
                problems.append(Problem("digit", field, f"digits outside a placeholder: {stray}", sentence))
            for w in NUMBER_WORDS.findall(own):
                problems.append(Problem("number_word", field, f"spelled-out number '{w}'", sentence))

            # Naming warning: a channel's or anomaly's value should sit in a
            # sentence that names it, so the reader knows what it refers to.
            named = set(base)
            for k in keys:
                parts = k.split(".")
                if parts[0] == "channel" and parts[-1] == "name":
                    named.add(f"channel:{parts[1]}")
                if parts[0] == "anomaly" and parts[-1] == "channel":
                    named |= {f"anomaly:{parts[1]}", f"channel:{owner_channel.get(parts[1])}"}
            for ch in summary.channels:
                if re.search(rf"(?<![\w.]){re.escape(ch.name)}(?![\w])", sentence):
                    named.add(f"channel:{ch.name}")
            for a in summary.anomalies:
                if re.search(rf"\b{a.id}\b", PLACEHOLDER.sub(" ", sentence)):  # ID written in prose
                    named.add(f"anomaly:{a.id}")
                if f"channel:{a.channel}" in named:
                    named.add(f"anomaly:{a.id}")
            for k in keys:
                e = cat.get(k)
                if not e or not e.owner or k.endswith((".name", ".channel")):
                    continue
                kind = "anomaly" if k.startswith("anomaly.") else "channel"
                if f"{kind}:{e.owner}" not in named:
                    problems.append(Problem("naming", field,
                                            f"{{{k}}} used without naming {e.owner} in the sentence", sentence))
    return problems


def fill_commentary(c: Commentary, summary: Summary, cat: dict) -> Commentary:
    """Replace every key with its value; criterion IDs become criterion names."""
    names = {cr.id: cr.name for cr in summary.criteria}
    d = c.model_dump()
    d["results_commentary"] = [fill(p, cat) for p in d["results_commentary"]]
    for cc in d["criteria_commentary"]:
        cc["criterion"] = names.get(cc["criterion"], cc["criterion"])
        cc["comment"] = fill(cc["comment"], cat)
    for a in d["anomaly_explanations"]:
        for f in ("likely_cause", "evidence", "suggested_check"):
            a[f] = fill(a[f], cat)
    for m in d["model_suggested_anomalies"]:
        for f in ("description", "reason"):
            m[f] = fill(m[f], cat)
    return Commentary.model_validate(d)


FATAL = {"placeholder", "digit", "structure", "final_assertion"}


def validate(c: Commentary, summary: Summary, *, rel: float = 0.005, abs_tol: float = 0.1,
             style_fatal: bool = False, decimals: dict | None = None,
             extra_allowlist: list[str] | None = None) -> tuple[dict, Commentary | None]:
    """Run every check on a template commentary.

    Returns (log, filled commentary). Raises CommentaryValidationError, with
    the log attached, if any fatal check fails.
    """
    decimals = decimals or {"default": 2}
    cat = build_catalogue(summary, decimals)
    tokens = allowlist(summary, extra_allowlist or [])
    problems = check_placeholders(c, summary, cat, tokens) + check_structure(c, summary)

    filled = None
    if not any(p.kind == "placeholder" for p in problems):  # filling needs every key to exist
        filled = fill_commentary(c, summary, cat)
        for p in check_numbers(filled, summary, rel, abs_tol, scoped=False, ignore=tokens):
            p.kind = "final_assertion"
            problems.append(p)
        problems += check_style(filled)

    fatal_kinds = FATAL | ({"style"} if style_fatal else set())
    fatal = [p for p in problems if p.kind in fatal_kinds]
    counts = {k: sum(p.kind == k for p in problems)
              for k in ("placeholder", "digit", "structure", "final_assertion", "naming",
                        "number_word", "style")}
    log = {"passed": not fatal, "counts": counts,
           "placeholders_used": sum(len(PLACEHOLDER.findall(t)) for _, t, _ in _text_fields(c)),
           "problems": [asdict(p) for p in problems],
           "tolerance": {"relative": rel, "absolute": abs_tol}}
    if fatal:
        lines = [f"[{p.kind}] {p.field}: {p.detail}" + (f'\n    Sentence: "{p.sentence}"' if p.sentence else "")
                 for p in fatal]
        err = CommentaryValidationError(f"{len(fatal)} commentary check(s) failed:\n" + "\n".join(lines))
        err.log = log  # type: ignore[attr-defined]
        raise err
    return log, filled
