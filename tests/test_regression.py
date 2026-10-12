"""Compare dynamics and contact against independently stored numerical data."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from cpf2d.config import default_scenario
from cpf2d.contact import wall_response
from cpf2d.geometry import vessel
from cpf2d.recording import load_trajectory
from cpf2d.simulation import run

REFERENCE = Path(__file__).resolve().parents[1] / "reference"


def test_fixture_integrity():
    manifest = json.loads((REFERENCE / "manifest.json").read_text(encoding="utf-8"))
    for name, metadata in manifest["fixtures"].items():
        assert (
            hashlib.sha256((REFERENCE / "fixtures" / name).read_bytes()).hexdigest()
            == metadata["sha256"]
        )


@pytest.mark.parametrize("case", ["free", "single_bend", "s_bend", "insertion"])
def test_recorded_trajectory(case, output_dir):
    config = replace(default_scenario(case), duration=0.2)
    run(config, output_dir)
    actual = load_trajectory(output_dir / "trajectory.npz")
    with np.load(REFERENCE / "fixtures" / f"{case}_trajectory.npz", allow_pickle=False) as expected:
        np.testing.assert_array_equal(actual["step"], expected["step"])
        for name in ("position", "velocity", "directors", "omega"):
            np.testing.assert_allclose(
                actual[name], expected[name], atol=2e-10, rtol=2e-8, err_msg=f"{case}: {name}"
            )


@pytest.mark.parametrize("case,code", [("single_bend", 1), ("s_bend", 2)])
def test_contacted_equilibrium_force(case, code):
    with np.load(REFERENCE / "fixtures" / f"{case}_contact.npz", allow_pickle=False) as data:
        for position, expected in zip(data["position"], data["force"]):
            actual = wall_response(
                position, data["rest_lengths"], vessel(case), code, 0.035, 800.0
            )[0]
            np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=1e-12)
