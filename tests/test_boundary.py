"""Planarity, straight rest strain and spatial release of material nodes."""

import numpy as np
import elastica as ea
from cpf2d.config import Parameters
from cpf2d.rod import make_rod, Simulator
from cpf2d.boundary import Feeder, PlanarBase


def constraint_system(case="s_bend", feeder=False):
    p = Parameters(n=16)
    rod, base = make_rod(
        p, start=(-0.3, 0.0, 0.0) if feeder else (0.0, 0.0, 0.0), case="free" if feeder else case
    )
    drift = dict(raw_position_z=0.0, raw_velocity_z=0.0, raw_forbidden_omega=0.0)
    control = dict(start=0.0, duration=1.0, du=0.01, offset=0.0, base_directors=base)
    sim = Simulator()
    sim.append(rod)
    sim.constrain(rod).using(
        Feeder if feeder else PlanarBase,
        control=control,
        drift=drift,
        constrained_position_idx=(0,),
        constrained_director_idx=(0,),
    )
    sim.finalize()
    return rod, sim, control, drift


def test_straight_rest_state_and_planarity():
    rod, sim, control, drift = constraint_system()
    np.testing.assert_array_equal(rod.rest_kappa, 0.0)
    np.testing.assert_array_equal(rod.rest_sigma, 0.0)
    np.testing.assert_allclose(rod.rest_lengths, 1 / 16, atol=1e-14)
    rod.velocity_collection[2] = 0.1
    rod.omega_collection[1:] = 0.2
    sim.constrain_values(0.0)
    sim.constrain_rates(0.0)
    stepper = ea.PositionVerlet()
    t = 0.0
    for _ in range(20):
        t = stepper.step(sim, t, 5e-5)
    np.testing.assert_array_equal(rod.position_collection[2], 0.0)
    np.testing.assert_array_equal(rod.omega_collection[1:], 0.0)
    assert np.isfinite(rod.position_collection).all()


def test_feeder_releases_material_by_spatial_position():
    rod, sim, control, drift = constraint_system(feeder=True)
    lengths = rod.rest_lengths.copy()
    mass = rod.mass.copy()
    rod.position_collection[1, 1:] = 0.02
    rod.velocity_collection[1, 1:] = 0.03
    sim.constrain_values(0.0)
    sim.constrain_rates(0.0)
    upstream = rod.position_collection[0] < 0
    np.testing.assert_array_equal(rod.position_collection[1, upstream], 0.0)
    np.testing.assert_array_equal(rod.velocity_collection[1, upstream], 0.0)
    assert np.all(rod.velocity_collection[1, ~upstream] == 0.03)
    # Release a material node across the fixed opening without changing its identity.
    i = np.flatnonzero(upstream)[-1]
    rod.position_collection[0, i] = 0.001
    rod.position_collection[1, i] = 0.015
    sim.constrain_values(0.5)
    assert rod.position_collection[1, i] == 0.015
    np.testing.assert_array_equal(rod.rest_lengths, lengths)
    np.testing.assert_array_equal(rod.mass, mass)
