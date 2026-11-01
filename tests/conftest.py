from datetime import date
from pathlib import Path

import pytest
import yaml

from interpret.channels import load_channel_info

from .synthetic import make_pulldown

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def phase_cfg():
    return yaml.safe_load((ROOT / "config" / "phases.yaml").read_text())["phase_detection"]


@pytest.fixture(scope="session")
def spec():
    return yaml.safe_load((ROOT / "specs" / "bvs_2_7.yaml").read_text())


@pytest.fixture(scope="session")
def channels():
    return load_channel_info(ROOT / "config" / "channel_locations.yaml")


@pytest.fixture(scope="session")
def clean_df():
    return make_pulldown(faults=False, noise_sd=0.0)


@pytest.fixture(scope="session")
def faulty_df():
    """The step-3 fixture: noise plus drift, flatline and door-open spike."""
    return make_pulldown(faults=True, noise_sd=0.05, seed=1)


@pytest.fixture
def run_args():
    return dict(test_id="T0001", facility="CWT2", run_date=date(2026, 10, 30))
