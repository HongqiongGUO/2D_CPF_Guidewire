"""Validated physical parameters and portable TOML scenario configuration."""

from dataclasses import dataclass, fields
from pathlib import Path
import math
import tomllib


@dataclass(frozen=True)
class Parameters:
    n: int = 64
    dt: float = 0.00005
    length: float = 1.0
    radius: float = 0.02
    density: float = 1.0
    youngs_modulus: float = 100.0
    shear_modulus: float = 100.0 / 3.0
    damping: float = 0.5
    half_width: float = 0.055
    penalty_per_length: float = 800.0
    geometry_spacing: float = 0.005
    axial_scale: float = 16.0
    speed_tolerance: float = 2e-9
    force_tolerance: float = 2e-10
    torque_tolerance: float = 2e-11

    def __post_init__(self):
        if type(self.n) is not int or self.n < 2:
            raise ValueError("n must be an integer >= 2")
        for field in fields(self):
            value = getattr(self, field.name)
            if not math.isfinite(value) or value < 0 or (field.name != "damping" and value == 0):
                raise ValueError(f"{field.name} must be finite and positive (damping may be zero)")
        if self.half_width <= self.radius:
            raise ValueError("half_width must exceed radius")


@dataclass(frozen=True)
class Scenario:
    case: str
    parameters: Parameters = Parameters()
    duration: float | None = 100.0
    sample_interval: float = 0.1
    displacement: float = 0.0
    ramp_time: float = 2.0
    base_x: float = -1.15
    perturbation: float = 0.005
    perturbation_time: float = 10.0
    minimum_settle: float = 20.0
    maximum_settle: float = 180.0

    def __post_init__(self):
        if self.case not in ("free", "single_bend", "s_bend", "insertion"):
            raise ValueError(f"Unknown scenario: {self.case}")
        for field in fields(self):
            if field.name in ("case", "parameters"):
                continue
            value = getattr(self, field.name)
            if field.name == "duration" and value is None and self.case == "insertion":
                continue
            if value is None or not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
            if field.name not in ("base_x", "displacement", "perturbation") and value <= 0:
                raise ValueError(f"{field.name} must be positive")
        for name in ("sample_interval", "duration", "ramp_time", "perturbation_time"):
            value = getattr(self, name)
            if value is not None:
                steps = value / self.parameters.dt
                if steps < 1 or not math.isclose(steps, round(steps), abs_tol=1e-7, rel_tol=1e-10):
                    raise ValueError(f"{name} must be an integer multiple of dt")
        if self.maximum_settle < self.minimum_settle:
            raise ValueError("maximum_settle must be >= minimum_settle")
        if self.case == "insertion":
            if not self.base_x < 0 < self.base_x + self.parameters.length:
                raise ValueError("The initial insertion rod must cross the fixed entry x=0")
            if self.displacement < 0 or self.perturbation < 0:
                raise ValueError("Insertion and perturbation must be nonnegative")
            if self.base_x + self.displacement + self.perturbation >= 0:
                raise ValueError("The pusher must remain upstream of the entry")
            if not math.isclose(
                2 / self.parameters.dt, round(2 / self.parameters.dt), abs_tol=1e-7
            ):
                raise ValueError("The two-unit settling interval must be a multiple of dt")
            if not math.isclose(
                2 / self.sample_interval, round(2 / self.sample_interval), abs_tol=1e-7
            ):
                raise ValueError("sample_interval must divide the two-unit settling interval")


def default_scenario(case):
    if case == "free":
        return Scenario(case, duration=24.0, displacement=0.0002)
    if case == "insertion":
        return Scenario(
            case,
            Parameters(n=80, length=1.2),
            duration=None,
            sample_interval=0.2,
            displacement=0.95,
            ramp_time=120.0,
        )
    return Scenario(case)


def load_config(path):
    """Read an explicit scenario and reject misspelled configuration keys."""
    from dataclasses import replace

    with Path(path).open("rb") as file:
        data = tomllib.load(file)
    unknown = set(data) - {"scenario", "parameters"}
    if unknown:
        raise ValueError(f"Unknown TOML sections: {sorted(unknown)}")
    scenario = data.get("scenario", {})
    if "case" not in scenario:
        raise ValueError("Missing scenario.case")
    defaults = default_scenario(scenario["case"])
    p = replace(defaults.parameters, **data.get("parameters", {}))
    return replace(defaults, parameters=p, **scenario)
