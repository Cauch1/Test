"""Placeholder keys: the model writes {metric.time_to_target}, Python writes 51.1 min.

The model never writes a numeric value. It writes keys from a catalogue built
from the summary; this module checks the keys, fills them with formatted
values and units, and reports any digit left outside a placeholder.

Key families (ids are summary ids: metric names, criterion ids, channel names,
phase ids, anomaly ids):
  test.id | test.facility | test.spec_ref | test.run_date
  counts.n_channels | counts.duration | counts.sample_interval
  phase.<id>.start | phase.<id>.end
  metric.<id> | metric.<id>.excluding_suspect | metric.<id>.window_start |
      metric.<id>.window_end | metric.<id>.suspect_channels
  criteria.<id>.name | .limit | .comparator | .measured | .margin | .status |
      .measured_excluding | .margin_excluding | .verdict_all | .verdict_excluding
  channel.<id>.name | .location | .min | .max | .mean.<phase id>
  anomaly.<id>.channel | .type | .start | .end | .evidence.<key>
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .schema import Summary

PLACEHOLDER = re.compile(r"\{([A-Za-z_]+(?:\.[A-Za-z0-9_\-]+)+)\}")
DIGIT = re.compile(r"\d")
NUMBER_WORDS = re.compile(
    r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|"
    r"fifty|hundred|dozen|half)\b", re.IGNORECASE)
COMPARATOR_TEXT = {"<=": "at most", "<": "below", ">=": "at least", ">": "above"}
MISSING = "not computed"


@dataclass
class Entry:
    text: str                      # what Python writes in place of the key
    description: str               # shown to the model in the catalogue
    owner: Optional[str] = None    # channel name or anomaly id the value belongs to


def _num(value: Optional[float], unit: Optional[str], decimals: dict) -> str:
    """Format a value with the unit's display decimals and a space before the unit."""
    if value is None:
        return MISSING
    d = decimals.get(unit or "none", decimals["default"])
    text = f"{value:.{d}f}"
    if float(text) == 0:
        text = text.lstrip("-")  # no "-0.0"
    return f"{text} {unit}" if unit else text


def _evidence_unit(key: str, channel_unit: str) -> Optional[str]:
    for suffix, unit in (("_degC", "degC"), ("_s", "s"), ("_min", "min")):
        if key.endswith(suffix):
            return unit
    return channel_unit if key in ("min_value", "max_value", "range_low", "range_high") else None


def build_catalogue(summary: Summary, decimals: dict) -> dict[str, Entry]:
    n = lambda v, u: _num(v, u, decimals)
    cat: dict[str, Entry] = {}
    labels = {c.name: c.display_name for c in summary.channels}

    t = summary.test
    cat["test.id"] = Entry(t.test_id, "test ID")
    cat["test.facility"] = Entry(t.facility, "facility")
    cat["test.spec_ref"] = Entry(t.spec_ref, "specification reference")
    cat["test.run_date"] = Entry(f"{t.run_date.day} {t.run_date:%B %Y}", "run date")

    c = summary.counts
    cat["counts.n_channels"] = Entry(str(c.n_channels), "number of channels")
    cat["counts.duration"] = Entry(n(c.duration_min, "min"), "record duration")
    cat["counts.sample_interval"] = Entry(n(c.sample_interval_s, "s"), "sample interval")

    for p in t.phases:
        cat[f"phase.{p.id}.start"] = Entry(n(p.start_min, "min"), f"{p.name} phase start time")
        cat[f"phase.{p.id}.end"] = Entry(n(p.end_min, "min"), f"{p.name} phase end time")

    for m in summary.metrics:
        k = f"metric.{m.name}"
        cat[k] = Entry(n(m.value, m.unit), f"{m.name}, all channels ({m.method})")
        if m.suspect_channels:
            cat[f"{k}.excluding_suspect"] = Entry(n(m.value_excluding_suspect, m.unit),
                                                  f"{m.name} with suspect channels excluded")
            cat[f"{k}.suspect_channels"] = Entry(", ".join(labels[s] for s in m.suspect_channels),
                                                 f"suspect channels for {m.name}")
        if m.window:
            cat[f"{k}.window_start"] = Entry(n(m.window.start_min, "min"), f"{m.name} window start")
            cat[f"{k}.window_end"] = Entry(n(m.window.end_min, "min"), f"{m.name} window end")

    for cr in summary.criteria:
        k = f"criteria.{cr.id}"
        cat[f"{k}.name"] = Entry(cr.name, "criterion name")
        cat[f"{k}.limit"] = Entry(n(cr.limit, cr.unit), f"limit ({cr.comparator}"
                                  + (", applied to the magnitude" if cr.use_abs else "") + ")")
        cat[f"{k}.comparator"] = Entry(COMPARATOR_TEXT[cr.comparator], "limit direction in words")
        cat[f"{k}.measured"] = Entry(n(cr.all_channels.measured, cr.unit), "measured value, all channels")
        cat[f"{k}.margin"] = Entry(n(cr.all_channels.margin, cr.unit),
                                   "margin, all channels (positive = inside the limit)")
        cat[f"{k}.status"] = Entry(cr.status.replace("_", " "), "final status")
        cat[f"{k}.verdict_all"] = Entry(cr.all_channels.verdict, "verdict with all channels")
        if cr.excluding_suspect:
            ex = cr.excluding_suspect
            cat[f"{k}.measured_excluding"] = Entry(n(ex.measured, cr.unit), "measured, suspect channels excluded")
            cat[f"{k}.margin_excluding"] = Entry(n(ex.margin, cr.unit), "margin, suspect channels excluded")
            cat[f"{k}.verdict_excluding"] = Entry(ex.verdict, "verdict, suspect channels excluded")

    for ch in summary.channels:
        k = f"channel.{ch.name}"
        cat[f"{k}.name"] = Entry(ch.display_name, "channel display name", ch.name)
        cat[f"{k}.location"] = Entry(ch.location_group, "location group", ch.name)
        cat[f"{k}.min"] = Entry(n(ch.min, ch.unit), "minimum over the run", ch.name)
        cat[f"{k}.max"] = Entry(n(ch.max, ch.unit), "maximum over the run", ch.name)
        for pid, v in ch.phase_means.items():
            cat[f"{k}.mean.{pid}"] = Entry(n(v, ch.unit), f"mean over {pid}", ch.name)

    units = {ch.name: ch.unit for ch in summary.channels}
    for a in summary.anomalies:
        k = f"anomaly.{a.id}"
        cat[f"{k}.channel"] = Entry(labels.get(a.channel, "all channels"), "channel display name", a.id)
        cat[f"{k}.type"] = Entry(a.type.replace("_", " "), "anomaly type", a.id)
        cat[f"{k}.start"] = Entry(n(a.start_min, "min"), "start time", a.id)
        cat[f"{k}.end"] = Entry(n(a.end_min, "min"), "end time", a.id)
        for key, v in a.evidence.items():
            cat[f"{k}.evidence.{key}"] = Entry(n(v, _evidence_unit(key, units.get(a.channel, "degC"))),
                                               f"evidence: {key}", a.id)
    return cat


def catalogue_text(cat: dict[str, Entry]) -> str:
    """The key list sent to the model: keys and descriptions, no values."""
    return "\n".join(f"{{{k}}}  {e.description}" for k, e in cat.items())


def allowlist(summary: Summary, extra: list[str]) -> list[str]:
    """Tokens that may contain digits in the model's own text: IDs and spec references."""
    tokens = [summary.test.test_id, summary.test.facility, summary.test.spec_ref, *extra]
    tokens += [a.id for a in summary.anomalies] + [c.name for c in summary.channels]
    return sorted(set(tokens), key=len, reverse=True)  # longest first, so "BVS 2.7" goes before "2.7"


def strip_allowed(text: str, tokens: list[str]) -> str:
    text = PLACEHOLDER.sub(" ", text)
    for tok in tokens:
        text = re.sub(rf"(?<![\w.]){re.escape(tok)}(?![\w])", " ", text)
    return text


def fill(text: str, cat: dict[str, Entry]) -> str:
    return PLACEHOLDER.sub(lambda m: cat[m.group(1)].text, text)
