"""Shared test parameters and rod construction."""

from dataclasses import replace
from ..config import Parameters
from ..rod import make_rod as build_rod

P = Parameters()
H = P.half_width - P.radius
PENALTY = P.penalty_per_length
LIMITS = dict(
    beam_reference_relative=0.02,
    beam_dt_relative=0.001,
    impact_reference_relative=0.01,
    impact_energy_relative=0.002,
    geometry_distance=1e-10,
    geometry_normal=1e-8,
    feed_final_relative=1e-6,
    feed_peak_strain=1e-5,
    feed_final_speed=2e-9,
    planarity=1e-12,
)


def make_rod(n, length=1.0, start=(0.0, 0.0, 0.0)):
    return build_rod(replace(P, n=n, length=length), start=start)[0]


def energy(r, t):
    r.compute_internal_forces_and_torques(t)
    return (
        float(r.compute_translational_energy() + r.compute_rotational_energy()),
        float(r.compute_bending_energy() + r.compute_shear_energy()),
    )
