"""Canonical wall impact check."""

import time
import json
import numpy as np
import elastica as ea
from ..rod import Simulator
from ..geometry import vessel
from ..contact import LumenWall, contact_observation
from ..recording import write_result
from .common import H, PENALTY, make_rod, energy


def impact(output, dt):
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    name = f"impact_dt{dt:g}"
    n = 16
    length = 0.4
    v0 = 0.1
    r = make_rod(n, length, start=(0.2, H, 0.0))
    r.velocity_collection[1] = v0
    curve = vessel("straight")
    sim = Simulator()
    sim.append(r)
    sim.add_forcing_to(r).using(
        LumenWall, centerline=curve, clearance=H, stiffness=PENALTY, graph_case=0
    )
    sim.finalize()
    stepper = ea.PositionVerlet()
    t = 0.0
    mass = float(r.mass.sum())
    K = PENALTY * length
    w = np.sqrt(K / mass)
    duration = np.pi / w
    rows = []
    min_load = 0.0
    outward = 0.0
    tangent = 0.0

    def observe():
        nonlocal min_load, outward, tangent
        ke, ee = energy(r, t)
        obs = contact_observation(r.position_collection, r.rest_lengths, curve, H, PENALTY, 0)
        U = 0.5 * PENALTY * np.sum(obs[8] * obs[3] ** 2)
        min_load = min(min_load, float(obs[4].min()))
        outward = max(outward, float(np.max(np.sum(obs[5] * obs[2], axis=1))))
        tangent = max(tangent, float(np.max(np.abs(np.sum(obs[5] * obs[7], axis=1)))))
        rows.append(
            [
                t,
                np.average(r.position_collection[1], weights=r.mass) - H,
                np.average(r.velocity_collection[1], weights=r.mass),
                obs[3].max(),
                ke,
                ee,
                U,
                np.max(
                    np.abs(
                        r.position_collection[1]
                        - np.average(r.position_collection[1], weights=r.mass)
                    )
                ),
            ]
        )

    observe()
    for _ in range(round(2.5 * duration / dt)):
        t = float(stepper.step(sim, t, dt))
        observe()
    h = np.asarray(rows)
    release = np.flatnonzero((h[:-1, 1] > 0) & (h[1:, 1] <= 0))[0]
    a, b = h[release : release + 2]
    # Scalar event-time interpolation only; every saved state is an actual timestep.
    measured_duration = float(a[0] - a[1] * (b[0] - a[0]) / (b[1] - a[1]))
    initial_energy = 0.5 * mass * v0 * v0
    result = dict(
        component="contact",
        name=name,
        n=n,
        dt=dt,
        penalty=PENALTY,
        mass=mass,
        reference=dict(
            omega=w,
            maximum_penetration=v0 / w,
            contact_duration=duration,
            rebound_velocity=-v0,
            energy=initial_energy,
        ),
        maximum_penetration=float(h[:, 3].max()),
        contact_duration=measured_duration,
        rebound_velocity=float(h[-1, 2]),
        penetration_error_relative=float(h[:, 3].max() / (v0 / w) - 1),
        duration_error_relative=float(measured_duration / duration - 1),
        rebound_error_relative=float(abs(h[-1, 2] + v0) / v0),
        maximum_energy_error_relative=float(
            np.max(np.abs(h[:, 4:7].sum(axis=1) - initial_energy)) / initial_energy
        ),
        final_energy_error_relative=float((h[-1, 4:7].sum() - initial_energy) / initial_energy),
        departure_from_rigid_translation=float(h[:, 7].max()),
        minimum_normal_load=min_load,
        maximum_outward_force=outward,
        maximum_tangential_force=tangent,
        all_finite=bool(np.isfinite(h).all()),
        runtime_seconds=time.perf_counter() - started,
    )
    np.savez_compressed(output / f"{name}.npz", history=h)
    write_result(output, f"{name}.json", result)
    print(json.dumps(result), flush=True)
    return result
