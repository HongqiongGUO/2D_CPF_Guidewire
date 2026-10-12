"""Configuration errors and recording correctness at off-stride endpoints."""

from dataclasses import replace
from pathlib import Path
import json
import pytest
from cpf2d.config import Parameters, default_scenario, load_config
from cpf2d.recording import prepare_output, load_trajectory
from cpf2d.simulation import run


@pytest.mark.parametrize(
    "name,case",
    [
        ("free_rod", "free"),
        ("single_bend", "single_bend"),
        ("s_bend", "s_bend"),
        ("insertion", "insertion"),
    ],
)
def test_configs_match_builtin_defaults(name, case):
    root = Path(__file__).resolve().parents[1]
    assert load_config(root / "configs" / f"{name}.toml") == default_scenario(case)


@pytest.mark.parametrize("changes", [{"n": 1}, {"dt": 0.0}, {"radius": 0.1}, {"damping": -1.0}])
def test_invalid_physics_rejected(changes):
    with pytest.raises(ValueError):
        Parameters(**changes)


def test_misspelled_toml_rejected(output_dir):
    path = output_dir / "config.toml"
    path.write_text('[scenario]\ncase="free"\nduraton=1.0\n')
    with pytest.raises(TypeError):
        load_config(path)


def test_output_is_not_silently_replaced(output_dir):
    prepare_output(output_dir)
    with pytest.raises(FileExistsError):
        prepare_output(output_dir)


@pytest.mark.parametrize("case", ["free", "insertion"])
def test_off_stride_last_endpoint_saved(case, output_dir):
    config = replace(default_scenario(case), duration=0.001)
    run(config, output_dir)
    data = load_trajectory(output_dir / "trajectory.npz")
    assert data["step"].tolist() == [0, 20]
    assert data["time"][-1] == pytest.approx(0.001)
    summary = json.loads((output_dir / "summary.json").read_text())
    assert len(summary["history"]) == 2
    if case == "insertion":
        assert not summary["complete_sequence"]
