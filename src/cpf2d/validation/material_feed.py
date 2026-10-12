"""Canonical material feed check."""

import time
import json
import numpy as np
import elastica as ea
from ..boundary import Feeder
from ..rod import Simulator
from ..recording import write_result
from .common import P, make_rod, energy


def feeder(output):
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    dt = 5e-5
    n = 80
    length = 1.2
    base = -0.302
    feed = 0.005
    ramp = 10.0
    end = 50.0
    r = make_rod(n, length, start=(base, 0.0, 0.0))
    rest = r.rest_lengths.copy()
    masses = r.mass.copy()
    material = np.r_[0.0, np.cumsum(rest)]
    control = dict(
        start=0.0,
        duration=ramp,
        du=feed,
        offset=0.0,
        base_directors=r.director_collection[:, :, 0].copy(),
    )
    drift = dict(raw_position_z=0.0, raw_velocity_z=0.0, raw_forbidden_omega=0.0)
    sim = Simulator()
    sim.append(r)
    sim.constrain(r).using(
        Feeder,
        control=control,
        drift=drift,
        constrained_position_idx=(0,),
        constrained_director_idx=(0,),
    )
    # Straight free rod: the fixed spatial guide is the boundary under test.
    # Lumen clearance .035 is never approached; adding a wall would give zero force.
    sim.dampen(r).using(ea.AnalyticalLinearDamper, uniform_damping_constant=P.damping, time_step=dt)
    sim.finalize()
    sim.constrain_values(0.0)
    sim.constrain_rates(0.0)
    stepper = ea.PositionVerlet()
    t = 0.0
    history = []
    positions = []
    sections = np.array([0.0, 0.5])
    initial_s = np.interp(sections, r.position_collection[0], material)
    initial_x = r.position_collection[0].copy()

    def observe():
        ke, ee = energy(r, t)
        x = r.position_collection[0]
        if np.any(np.diff(x) <= 0):
            raise RuntimeError("Nonmonotone material coordinates")
        section_s = np.interp(sections, x, material)
        # Material labels crossing FIXED spatial planes, independent of the tip.
        passage = initial_s - section_s
        reconstructed = np.interp(section_s, material, x)
        speed = np.linalg.norm(r.velocity_collection, axis=0).max()
        force = np.linalg.norm(r.internal_forces[:, 1:], axis=0).max()
        q = np.clip(t / ramp, 0, 1)
        command = feed * (10 * q**3 - 15 * q**4 + 6 * q**5)
        history.append(
            [
                t,
                command,
                x[0] - base,
                *passage,
                x[-1] - initial_x[-1],
                np.max(np.abs(r.sigma[2])),
                np.max(np.abs(r.dilatation - 1)),
                speed,
                force,
                ke,
                ee,
                np.max(np.abs(r.position_collection[1:])),
                np.max(np.abs(reconstructed - sections)),
            ]
        )
        positions.append(r.position_collection.copy())

    observe()
    sample = round(0.05 / dt)
    for step in range(1, round(end / dt) + 1):
        t = float(stepper.step(sim, t, dt))
        if step % sample == 0:
            observe()
        if step % round(10 / dt) == 0:
            print("feeder t", round(t, 2), "passage", history[-1][3:5], flush=True)
    h = np.asarray(history)
    pos = np.asarray(positions)
    result = dict(
        component="feeder",
        dt=dt,
        n=n,
        length=length,
        base=base,
        feed=feed,
        ramp_time=ramp,
        total_time=t,
        damping=P.damping,
        entry_plane=0.0,
        distal_plane=0.5,
        entry_material_passage=float(h[-1, 3]),
        distal_material_passage=float(h[-1, 4]),
        entry_relative_error=float(abs(h[-1, 3] - feed) / feed),
        distal_relative_error=float(abs(h[-1, 4] - feed) / feed),
        maximum_transient_passage_error=float(np.max(np.abs(h[:, 3:5] - h[:, 1, None]))),
        maximum_entry_distal_difference=float(np.max(np.abs(h[:, 3] - h[:, 4]))),
        maximum_axial_strain=float(h[:, 6].max()),
        final_axial_strain=float(h[-1, 6]),
        maximum_stretch=float(h[:, 7].max()),
        final_speed=float(h[-1, 8]),
        final_force_residual=float(h[-1, 9]),
        maximum_transverse_motion=float(h[:, 12].max()),
        maximum_fixed_section_error=float(h[:, 13].max()),
        node_ids_crossed_entry=np.flatnonzero(
            (initial_x < 0) & (r.position_collection[0] >= 0)
        ).tolist(),
        node_ids_crossed_distal=np.flatnonzero(
            (initial_x < 0.5) & (r.position_collection[0] >= 0.5)
        ).tolist(),
        rest_lengths_unchanged=bool(np.array_equal(rest, r.rest_lengths)),
        mass_unchanged=bool(np.array_equal(masses, r.mass)),
        all_finite=bool(np.isfinite(h).all()),
        runtime_seconds=time.perf_counter() - started,
    )
    np.savez_compressed(
        output / "feeder.npz",
        history=h,
        position=pos,
        material_coordinate=material,
        fixed_sections=sections,
    )
    write_result(output, "feeder.json", result)
    print(json.dumps(result), flush=True)
    return result
