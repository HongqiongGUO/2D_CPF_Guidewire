"""PositionVerlet scenario orchestration and endpoint sampling."""

from dataclasses import asdict
import time
import numpy as np
import elastica as ea
from .rod import make_rod, Simulator
from .boundary import PlanarBase, Feeder
from .geometry import vessel
from .contact import LumenWall, OpenLumen, contact_observation, wall_response
from .metrics import DampingMonitor, kinetic_state
from .recording import write_result

PHASES = (
    "Insertion",
    "Relaxation before perturbation",
    "Perturbation",
    "Relaxation after perturbation",
)


def run_relaxation(config, output):
    p = config.parameters
    dt = p.dt
    case = config.case
    curve = vessel(case, p.geometry_spacing)
    code = {"free": 0, "single_bend": 1, "s_bend": 2}[case]
    r, base_directors = make_rod(p, case=case)
    control = dict(
        start=0.0, duration=config.ramp_time, du=config.displacement, base_directors=base_directors
    )
    drift = dict(raw_position_z=0.0, raw_velocity_z=0.0, raw_forbidden_omega=0.0)
    ledger = dict(
        dissipation=0.0,
        input_work=0.0,
        peak_node_speed=0.0,
        peak_kinetic=0.0,
        maximum_penetration=0.0,
        maximum_outward_contact_force=0.0,
        maximum_tangential_force=0.0,
        minimum_damping_loss=0.0,
    )
    sim = Simulator()
    sim.append(r)
    sim.constrain(r).using(
        PlanarBase,
        control=control,
        drift=drift,
        constrained_position_idx=(0,),
        constrained_director_idx=(0,),
    )
    if case != "free":
        sim.add_forcing_to(r).using(
            LumenWall,
            centerline=curve,
            graph_case=code,
            clearance=p.half_width - p.radius,
            stiffness=p.penalty_per_length,
            ledger=ledger,
        )
    sim.dampen(r).using(
        DampingMonitor, ledger=ledger, uniform_damping_constant=p.damping, time_step=dt
    )
    sim.finalize()
    # Apply the fixed base before measuring initial stored energy.
    sim.constrain_values(0.0)
    sim.constrain_rates(0.0)
    r.compute_internal_forces_and_torques(0.0)
    # Midpoint traction work only for the existing free-rod base displacement.
    original = sim.constrain_rates

    def rates(t):
        original(t)
        if config.displacement != 0 and 0 < t < config.ramp_time:
            q = t / config.ramp_time
            v = config.displacement * (30 * q * q - 60 * q**3 + 30 * q**4) / config.ramp_time
            acc = config.displacement * (60 * q - 180 * q * q + 120 * q**3) / config.ramp_time**2
            reaction = r.mass[0] * (acc + p.damping * v) - r.internal_forces[0, 0]
            ledger["input_work"] += float(reaction * v * dt)

    sim.constrain_rates = rates

    history = []
    positions = []
    velocities = []
    directors = []
    angular = []
    contact_points = []
    contact_active = []
    sample_steps = []

    def sample(step, t):
        r.compute_internal_forces_and_torques(t)
        if case == "free":
            obs = None
            wall_energy = 0.0
            force = np.zeros_like(r.position_collection)
            pen = 0.0
            points = np.empty((0, 2))
            active = np.zeros(0, dtype=bool)
        else:
            obs = contact_observation(
                r.position_collection,
                r.rest_lengths,
                curve,
                p.half_width - p.radius,
                p.penalty_per_length,
                code,
            )
            points = obs[0]
            active = obs[4] > 0
            force, wall_energy, pen, _, _ = wall_response(
                r.position_collection,
                r.rest_lengths,
                curve,
                code,
                p.half_width - p.radius,
                p.penalty_per_length,
            )
        kinetic = float(r.compute_translational_energy() + r.compute_rotational_energy())
        assert abs(kinetic - kinetic_state(r)[0]) < 1e-14
        potential = float(r.compute_bending_energy() + r.compute_shear_energy() + wall_energy)
        net = r.internal_forces + force
        finite = all(
            np.isfinite(a).all()
            for a in (
                r.position_collection,
                r.velocity_collection,
                r.director_collection,
                r.omega_collection,
            )
        )
        assert finite and np.isfinite(kinetic + potential)
        history.append(
            dict(
                step=step,
                time=t,
                kinetic=kinetic,
                potential=potential,
                wall_energy=float(wall_energy),
                mechanical_energy=kinetic + potential,
                max_speed=float(kinetic_state(r)[1]),
                force_residual=float(np.linalg.norm(net[:, 1:], axis=0).max()),
                torque_residual=float(abs(r.internal_torques[:, 1:]).max()),
                max_penetration=float(pen),
                dissipation=ledger["dissipation"],
                input_work=ledger["input_work"],
                finite=finite,
            )
        )
        positions.append(r.position_collection.copy())
        velocities.append(r.velocity_collection.copy())
        directors.append(r.director_collection.copy())
        angular.append(r.omega_collection.copy())
        contact_points.append(points.copy())
        contact_active.append(active.copy())
        sample_steps.append(step)

    sample(0, 0.0)
    start = time.perf_counter()
    stepper = ea.PositionVerlet()
    t = 0.0
    stride = round(config.sample_interval / dt)
    total = round(config.duration / dt)
    for i in range(1, total + 1):
        t = float(stepper.step(sim, t, dt))
        if i % stride == 0 or i == total:
            sample(i, t)
        if i % round(10 / dt) == 0:
            print(case, dt, "t", round(t, 3), "speed", history[-1]["max_speed"], flush=True)
    np.savez_compressed(
        output / "trajectory.npz",
        position=positions,
        velocity=velocities,
        directors=directors,
        omega=angular,
        contact_points=contact_points,
        contact_active=contact_active,
        time=[h["time"] for h in history],
        step=sample_steps,
        dt=dt,
    )
    result = dict(
        case=case,
        dt=dt,
        parameters=asdict(p),
        duration=history[-1]["time"],
        sequence=(
            "small base displacement then relaxation"
            if case == "free"
            else "release of curved initial shape with fixed base"
        ),
        ledger=ledger,
        raw_planarity=drift,
        history=history,
        runtime_seconds=time.perf_counter() - start,
    )
    write_result(output, "summary.json", result)
    print("completed", case, result["runtime_seconds"], flush=True)
    return result


def run_insertion(config, output):
    p = config.parameters
    DT, SAMPLE = p.dt, config.sample_interval
    BASE, INSERT, INSERT_TIME = config.base_x, config.displacement, config.ramp_time
    PERTURB, PERTURB_TIME = config.perturbation, config.perturbation_time
    MIN_SETTLE, MAX_SETTLE = config.minimum_settle, config.maximum_settle
    rod, base_directors = make_rod(p, start=(BASE, 0.0, 0.0))
    curve = vessel("s_bend", p.geometry_spacing)
    control = dict(
        start=0.0,
        duration=INSERT_TIME,
        du=INSERT,
        offset=0.0,
        base_directors=rod.director_collection[:, :, 0].copy(),
    )
    drift = dict(raw_position_z=0.0, raw_velocity_z=0.0, raw_forbidden_omega=0.0)
    ledger = dict(
        dissipation=0.0,
        peak_node_speed=0.0,
        peak_kinetic=0.0,
        maximum_penetration=0.0,
        maximum_outward_contact_force=0.0,
        maximum_tangential_force=0.0,
        minimum_damping_loss=0.0,
    )
    sim = Simulator()
    sim.append(rod)
    sim.constrain(rod).using(
        Feeder,
        control=control,
        drift=drift,
        constrained_position_idx=(0,),
        constrained_director_idx=(0,),
    )
    sim.add_forcing_to(rod).using(OpenLumen, curve=curve, p=p, ledger=ledger)
    sim.dampen(rod).using(
        DampingMonitor, ledger=ledger, uniform_damping_constant=p.damping, time_step=DT
    )
    sim.finalize()
    stepper = ea.PositionVerlet()
    sim.constrain_values(0.0)
    sim.constrain_rates(0.0)
    t = 0.0
    step = 0
    phase = 0
    history = []
    positions = []
    velocities = []
    directors = []
    contacts = []
    offsets = [0]
    angular = []
    checkpoints = {}
    started = time.perf_counter()

    def observe():
        rod.compute_internal_forces_and_torques(t)
        first = max(0, int(np.searchsorted(rod.position_collection[0], 0.0)) - 1)
        pos = rod.position_collection[:, first:]
        ds = rod.rest_lengths[first:]
        obs = contact_observation(pos, ds, curve, p.half_width - p.radius, p.penalty_per_length, 2)
        active = (obs[4] > 1e-12) & (obs[0][:, 0] >= 0)
        loads = obs[4].reshape(-1, 33).sum(axis=1)
        element_active = np.flatnonzero(loads > max(1e-11, 1e-4 * loads.max()))
        zones = (
            0 if not element_active.size else 1 + int(np.count_nonzero(np.diff(element_active) > 1))
        )
        wall = np.zeros_like(rod.position_collection)
        wall[:, first:] = wall_response(
            pos, ds, curve, 2, p.half_width - p.radius, p.penalty_per_length
        )[0]
        forces = rod.internal_forces + wall
        downstream = rod.position_collection[0] >= 0
        free_angles = 0.5 * (rod.position_collection[0, 1:] + rod.position_collection[0, :-1]) >= 0
        residual = max(float(abs(forces[0, 1:]).max()), float(abs(forces[1, downstream]).max()))
        torque = float(abs(rod.internal_torques[0, free_angles]).max())
        speed = float(np.linalg.norm(rod.velocity_collection, axis=0).max())
        finite = all(
            np.isfinite(a).all()
            for a in (
                rod.position_collection,
                rod.velocity_collection,
                rod.director_collection,
                rod.omega_collection,
            )
        )
        if not finite:
            raise RuntimeError("Nonfinite insertion state")
        if np.any(np.diff(rod.position_collection[0]) <= 0):
            raise RuntimeError("Rod folded back; spatial feeder indexing no longer applies")
        row = dict(
            time=t,
            step=step,
            phase=phase,
            phase_name=PHASES[phase],
            proximal_displacement=float(rod.position_collection[0, 0] - BASE),
            contact_zones=zones,
            active_elements=(element_active + first).tolist(),
            max_node_speed=speed,
            force_residual=residual,
            torque_residual=torque,
            max_penetration=float(obs[3].max()),
            tip=rod.position_collection[:2, -1].tolist(),
            base=rod.position_collection[:2, 0].tolist(),
            settled=bool(
                speed < p.speed_tolerance
                and residual < p.force_tolerance
                and torque < p.torque_tolerance
            ),
            finite=finite,
        )
        kinetic = float(rod.compute_translational_energy() + rod.compute_rotational_energy())
        elastic = float(rod.compute_bending_energy() + rod.compute_shear_energy())
        wall_energy = float(0.5 * p.penalty_per_length * np.sum(obs[8] * obs[3] ** 2))
        row.update(
            kinetic=kinetic,
            elastic=elastic,
            wall_energy=wall_energy,
            mechanical_energy=kinetic + elastic + wall_energy,
            dissipation=ledger["dissipation"],
        )
        assert np.isfinite(kinetic + elastic + wall_energy)
        angular.append(rod.omega_collection.copy())
        points = obs[0][active]
        history.append(row)
        positions.append(rod.position_collection.copy())
        velocities.append(rod.velocity_collection.copy())
        directors.append(rod.director_collection.copy())
        contacts.extend(points.tolist())
        offsets.append(len(contacts))
        return row

    def advance(duration):
        nonlocal t, step
        for _ in range(round(duration / DT)):
            t = float(stepper.step(sim, t, DT))
            step += 1
            if step % round(SAMPLE / DT) == 0:
                observe()
            if step % round(10 / DT) == 0:
                h = history[-1]
                print(
                    PHASES[phase],
                    "t",
                    round(t, 2),
                    "push",
                    round(h["proximal_displacement"], 6),
                    "zones",
                    h["contact_zones"],
                    "speed",
                    h["max_node_speed"],
                    "tip",
                    h["tip"],
                    flush=True,
                )
        if history[-1]["step"] != step:
            observe()

    def settle():
        start = t
        consecutive = 0
        while t - start < MAX_SETTLE - 1e-6:
            advance(2.0)
            good = history[-1]["settled"]
            consecutive = consecutive + 1 if good else 0
            if t - start >= MIN_SETTLE - 1e-6 and consecutive >= 3:
                return history[-1]
        return {**history[-1], "settled": False}

    observe()
    if config.duration is not None:
        advance(config.duration)
        if history[-1]["step"] != step:
            observe()
        save_insertion(
            output, p, history, positions, velocities, directors, angular, contacts, offsets
        )
        result = dict(
            case="insertion",
            complete_sequence=False,
            parameters=asdict(p),
            history=history,
            ledger=ledger,
            raw_planarity=drift,
        )
        write_result(output, "summary.json", result)
        return result
    advance(INSERT_TIME)
    checkpoints["end_insertion"] = history[-1]
    phase = 1
    checkpoints["before_perturbation"] = settle()
    if not checkpoints["before_perturbation"]["settled"]:
        raise RuntimeError("Insertion did not settle; perturbation not applied")
    if checkpoints["before_perturbation"]["contact_zones"] < 3:
        raise RuntimeError("Distributed three-zone contact was not reached")
    phase = 2
    control.update(start=t, duration=PERTURB_TIME, du=PERTURB, offset=INSERT)
    advance(PERTURB_TIME)
    checkpoints["end_perturbation"] = history[-1]
    phase = 3
    checkpoints["after_perturbation"] = settle()
    save_insertion(output, p, history, positions, velocities, directors, angular, contacts, offsets)
    result = dict(
        case="insertion",
        complete_sequence=True,
        parameters=asdict(p),
        history=history,
        initial_base_x=BASE,
        initial_tip_x=BASE + p.length,
        fixed_entry_x=0.0,
        feeder="Straight frictionless spatial guide for x<0; axial motion free except prescribed upstream end.",
        insertion_distance=INSERT,
        insertion_duration=INSERT_TIME,
        perturbation_size=PERTURB,
        perturbation_duration=PERTURB_TIME,
        checkpoints=checkpoints,
        distal_perturbation_displacement=(
            np.array(checkpoints["after_perturbation"]["tip"])
            - checkpoints["before_perturbation"]["tip"]
        ).tolist(),
        final_settled=checkpoints["after_perturbation"]["settled"],
        sample_interval=SAMPLE,
        integrator="PyElastica PositionVerlet",
        raw_planarity=drift,
        ledger=ledger,
        runtime_seconds=time.perf_counter() - started,
    )
    write_result(output, "summary.json", result)
    print("Simulation complete", time.perf_counter() - started, flush=True)
    return result


def save_insertion(
    output, p, history, positions, velocities, directors, angular, contacts, offsets
):
    np.savez_compressed(
        output / "trajectory.npz",
        position=positions,
        velocity=velocities,
        directors=directors,
        omega=angular,
        contact_points=np.asarray(contacts).reshape(-1, 2),
        contact_offsets=offsets,
        time=[h["time"] for h in history],
        step=[h["step"] for h in history],
        phase=[h["phase"] for h in history],
        dt=p.dt,
    )


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    write_result(output, "config.json", asdict(config))
    result = (
        run_insertion(config, output)
        if config.case == "insertion"
        else run_relaxation(config, output)
    )
    return result
