"""Channel location metadata and selectors.

A selector is a small dict from the spec YAML, for example
{height: breath, row: first}. It picks every channel whose metadata matches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel


class ChannelInfo(BaseModel):
    name: str
    display_name: Optional[str] = None  # report label; defaults to name
    unit: str = "degC"
    height: str  # foot, breath, head, ...
    row: int
    side: Optional[str] = None
    sensor_range: Optional[tuple[float, float]] = None

    @property
    def label(self) -> str:
        return self.display_name or self.name

    @property
    def location_group(self) -> str:
        return f"{self.height} height, row {self.row}"


def load_channel_info(path: str | Path) -> list[ChannelInfo]:
    """Read config/channel_locations.yaml into ChannelInfo objects."""
    data = yaml.safe_load(Path(path).read_text())
    return [ChannelInfo(name=name, **meta) for name, meta in data["channels"].items()]


def select(channels: list[ChannelInfo], selector: dict) -> list[str]:
    """Return the names of channels matching every key in the selector.

    row accepts an integer, or 'first' / 'last' so one spec works for a car
    (2 rows) and a bus (10 rows) without editing.
    """
    rows = sorted({c.row for c in channels})
    wanted = dict(selector)
    if wanted.get("row") == "first":
        wanted["row"] = rows[0]
    elif wanted.get("row") == "last":
        wanted["row"] = rows[-1]

    names = [
        c.name for c in channels
        if all(getattr(c, key) == value for key, value in wanted.items())
    ]
    if not names:
        raise ValueError(f"Selector {selector} matched no channels")
    return names
