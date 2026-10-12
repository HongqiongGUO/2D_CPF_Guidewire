"""Keep each test's numerical artifacts in its own output directory."""

from pathlib import Path
from uuid import uuid4
import pytest


@pytest.fixture
def output_dir():
    path = Path(__file__).resolve().parents[1] / "outputs/tests" / uuid4().hex
    path.mkdir(parents=True)
    return path
