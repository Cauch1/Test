"""The single seam between the existing pipeline and this package.

The existing pipeline calls run_stage_a() after its compute step, passing the
DataFrame it already has (channels renamed to standard names by the YAML
channel matching). Nothing else in the existing code needs to know about
this package until --interpret is wired in at step 4.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from .channels import load_channel_info
from .summary import build_summary, load_yaml, write_summary

ROOT = Path(__file__).resolve().parent.parent


def run_stage_a(df: pd.DataFrame, *, test_id: str, facility: str, run_date: date,
                spec_path: str | Path, out_dir: str | Path,
                channels_path: str | Path = ROOT / "config" / "channel_locations.yaml",
                phases_path: str | Path = ROOT / "config" / "phases.yaml",
                anomalies_path: str | Path = ROOT / "config" / "anomaly_thresholds.yaml"):
    """Build and write the summary JSON. Returns (summary, path)."""
    channels = load_channel_info(channels_path)
    summary = build_summary(
        df, channels, test_id=test_id, facility=facility, run_date=run_date,
        spec=load_yaml(spec_path), phase_cfg=load_yaml(phases_path)["phase_detection"],
        anomaly_cfg=load_yaml(anomalies_path),
    )
    return summary, write_summary(summary, out_dir)
