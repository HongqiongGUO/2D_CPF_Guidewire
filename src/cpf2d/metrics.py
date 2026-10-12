"""Passive kinetic-energy and dissipation observations."""

import numpy as np
import elastica as ea
from numba import njit


@njit(cache=True)
def kinetic_speed(mass, v, J, omega, stretch):
    kinetic = 0.0
    vmax = 0.0
    for i in range(mass.size):
        speed2 = 0.0
        for a in range(3):
            speed2 += v[a, i] ** 2
        vmax = max(vmax, np.sqrt(speed2))
        kinetic += 0.5 * mass[i] * speed2
    for i in range(omega.shape[1]):
        for a in range(3):
            for b in range(3):
                kinetic += 0.5 * omega[a, i] * J[a, b, i] * omega[b, i] / stretch[i]
    return kinetic, vmax


def kinetic_state(r):
    return kinetic_speed(
        r.mass,
        r.velocity_collection,
        r.mass_second_moment_of_inertia,
        r.omega_collection,
        r.dilatation,
    )


class DampingMonitor(ea.AnalyticalLinearDamper):
    def __init__(self, ledger, **kw):
        self.ledger = ledger
        super().__init__(**kw)

    def dampen_rates(self, system, time):
        before, speed = kinetic_state(system)
        super().dampen_rates(system, time)
        after, _ = kinetic_state(system)
        if not np.isfinite(before + after + speed):
            raise RuntimeError("Nonfinite dynamic state")
        self.ledger["dissipation"] += before - after
        self.ledger["peak_node_speed"] = max(self.ledger["peak_node_speed"], speed)
        self.ledger["peak_kinetic"] = max(self.ledger["peak_kinetic"], before, after)
        self.ledger["minimum_damping_loss"] = min(
            self.ledger["minimum_damping_loss"], before - after
        )


def transmission(before, after, du):
    d = np.array(after["tip"]) - np.array(before["tip"])
    t = np.array(before["tip_tangent"])
    angle = np.arctan2(
        np.sin(after["tip_angle"] - before["tip_angle"]),
        np.cos(after["tip_angle"] - before["tip_angle"]),
    )
    return dict(
        du=du,
        delta_s_tip=float(d @ t),
        tau=float(d @ t / du),
        delta_x_tip=float(d[0]),
        delta_y_tip=float(d[1]),
        gain_x=float(d[0] / du),
        gain_y=float(d[1] / du),
        gain_angle=float(angle / du),
        delta_tip_angle=float(angle),
        delta_length=float(after["total_length"] - before["total_length"]),
        length_change_per_du=float((after["total_length"] - before["total_length"]) / du),
        vessel_arc_gain=(after["tip_vessel_arc"] - before["tip_vessel_arc"]) / du,
        vessel_tangent_gain=float(d[:2] @ np.asarray(before["tip_vessel_tangent"]) / du),
    )
