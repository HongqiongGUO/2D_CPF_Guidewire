"""Canonical cantilever check."""

import time
import json
import numpy as np
import elastica as ea
from scipy.linalg import expm, svd
from scipy.optimize import brentq, minimize_scalar
from ..boundary import PlanarBase
from ..rod import Simulator
from ..recording import write_result
from .common import P, make_rod, energy


def beam_reference(r, length=1.0):
    """Continuum Timoshenko eigenvalue, constant-coefficient analytic transfer matrix.

    Q=S(y'-theta), M=B theta'; Q'=-m*w^2*y, M'=-Q-j*w^2*theta.
    Clamp y=theta=0; free end Q=M=0. No rod-solver result enters the root.
    """
    B = float(r.bend_matrix[0, 0, 0])
    S = float(r.shear_matrix[1, 1, 0])
    m = float(r.mass.sum() / length)
    j = float(r.mass_second_moment_of_inertia[0, 0, 0] / r.rest_lengths[0])
    beta = brentq(lambda b: np.cos(b) * np.cosh(b) + 1.0, 1.5, 2.0)
    w_eb = beta**2 * np.sqrt(B / m) / length**2

    def matrix(w):
        return np.array(
            [[0, 1, 1 / S, 0], [0, 0, 0, 1 / B], [-m * w * w, 0, 0, 0], [0, -j * w * w, -1, 0]]
        )

    def boundary(w):
        return expm(matrix(w) * length)[2:, 2:]

    w = brentq(lambda z: np.linalg.det(boundary(z)), 0.8 * w_eb, 1.2 * w_eb, xtol=1e-14)
    _, _, vt = svd(boundary(w))
    state = np.r_[0.0, 0.0, vt[-1]]
    state /= (expm(matrix(w) * length) @ state)[0]
    return (
        dict(
            omega=w,
            frequency=w / (2 * np.pi),
            omega_euler_bernoulli=w_eb,
            frequency_euler_bernoulli=w_eb / (2 * np.pi),
            beta1=beta,
            EI=B,
            shear_stiffness=S,
            mass_per_length=m,
            rotary_inertia_per_length=j,
            shear_correction=S / (P.shear_modulus * np.pi * P.radius**2),
        ),
        matrix(w),
        state,
    )


def frequency_fit(t, y, reference):
    # Continuous least-squares sinusoid, not a bin-limited FFT estimate.
    use = t >= 0.5 / reference
    ts = t[use]
    ys = y[use]

    def fit(f):
        X = np.column_stack(
            (np.sin(2 * np.pi * f * ts), np.cos(2 * np.pi * f * ts), np.ones_like(ts))
        )
        c = np.linalg.lstsq(X, ys, rcond=None)[0]
        return float(np.mean((ys - X @ c) ** 2)), c

    result = minimize_scalar(
        lambda f: fit(f)[0],
        bounds=(0.95 * reference, 1.06 * reference),
        method="bounded",
        options={"xatol": 1e-14},
    )
    mse, c = fit(result.x)
    return float(result.x), float(np.sqrt(mse) / np.hypot(c[0], c[1])), c


def beam(output, n, dt):
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    name = f"beam_n{n}_dt{dt:g}"
    r = make_rod(n)
    ref, A, state = beam_reference(r)
    sim = Simulator()
    sim.append(r)
    control = dict(
        start=1e100, duration=1.0, du=0.0, base_directors=r.director_collection[:, :, 0].copy()
    )
    drift = dict(raw_position_z=0.0, raw_velocity_z=0.0, raw_forbidden_omega=0.0)
    sim.constrain(r).using(
        PlanarBase,
        control=control,
        drift=drift,
        constrained_position_idx=(0,),
        constrained_director_idx=(0,),
    )
    sim.finalize()
    # A velocity-mode start keeps the initial centerline/rest strain exactly zero.
    amplitude = 1e-4
    x = r.position_collection[0]
    shape = np.array([expm(A * xx) @ state for xx in x])
    theta = np.array([(expm(A * xx) @ state)[1] for xx in 0.5 * (x[:-1] + x[1:])])
    r.velocity_collection[1] = amplitude * ref["omega"] * shape[:, 0]
    r.omega_collection[0] = amplitude * ref["omega"] * theta
    sim.constrain_values(0.0)
    sim.constrain_rates(0.0)
    stepper = ea.PositionVerlet()
    t = 0.0
    rows = []
    steps = round(4 / ref["frequency"] / dt)
    sample = max(1, round(0.02 / dt))

    def observe():
        ke, ee = energy(r, t)
        rows.append(
            [
                t,
                r.position_collection[1, -1],
                ke,
                ee,
                np.linalg.norm(r.velocity_collection, axis=0).max(),
                np.abs(r.position_collection[2]).max(),
            ]
        )

    observe()
    for step in range(1, steps + 1):
        t = float(stepper.step(sim, t, dt))
        if step % sample == 0 or step == steps:
            observe()
        if step % round(10 / dt) == 0:
            print(name, "t", round(t, 2), flush=True)
    rows = np.asarray(rows)
    if not np.isfinite(rows).all():
        raise RuntimeError("Nonfinite beam")
    freq, fit_residual, coeff = frequency_fit(rows[:, 0], rows[:, 1], ref["frequency"])
    np.savez_compressed(
        output / f"{name}.npz",
        history=rows,
        final_position=r.position_collection,
        reference_mode=shape,
        fit_coefficients=coeff,
    )
    result = dict(
        component="rod",
        name=name,
        n=n,
        dt=dt,
        duration=t,
        amplitude=amplitude,
        reference=ref,
        frequency=freq,
        reference_error_relative=(freq / ref["frequency"] - 1),
        sinusoid_fit_residual_relative=fit_residual,
        maximum_energy_error_relative=float(
            np.max(np.abs(rows[:, 2] + rows[:, 3] - (rows[0, 2] + rows[0, 3])))
            / (rows[0, 2] + rows[0, 3])
        ),
        peak_speed=float(rows[:, 4].max()),
        out_of_plane=float(rows[:, 5].max()),
        runtime_seconds=time.perf_counter() - started,
        all_finite=True,
    )
    write_result(output, f"{name}.json", result)
    print(json.dumps(result), flush=True)
    return result
