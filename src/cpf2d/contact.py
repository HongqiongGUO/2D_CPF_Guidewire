"""Unilateral normal penalty integrated along reference rod length."""

import numpy as np
import elastica as ea
from numba import njit
from .geometry import project, project_graph


@njit(cache=True)
def contact_observation(position, rest_lengths, centerline, clearance, stiffness, graph_case=-1):
    # Fixed 33-point composite Simpson line quadrature, including the free tip.
    nq = 33
    count = rest_lengths.size
    points = np.zeros((nq * count, 2))
    weights = np.zeros(nq * count)
    for i in range(count):
        for j in range(nq):
            u = j / (nq - 1)
            points[nq * i + j] = (1 - u) * position[:2, i] + u * position[:2, i + 1]
            coefficient = 1.0 if j == 0 or j == nq - 1 else 4.0 if j % 2 else 2.0
            weights[nq * i + j] = rest_lengths[i] * coefficient / (3 * (nq - 1))
    q, normal, distance, arc, tangent = (
        project(points, centerline)
        if graph_case < 0
        else project_graph(points, centerline, graph_case)
    )
    penetration = np.maximum(distance - clearance, 0.0)
    load = stiffness * weights * penetration
    force = -load[:, None] * normal
    return points, q, normal, penetration, load, force, arc, tangent, weights


@njit(cache=True)
def wall_response(position, ds, curve, code, clearance, stiffness):
    """Nodal forces, potential, penetration, outward and tangential force maxima."""
    force = np.zeros_like(position)
    mid = (position[:2, 1:] + position[:2, :-1]).T * 0.5
    dm = (project(mid, curve) if code < 0 else project_graph(mid, curve, code))[2]
    half = 0.5 * np.sqrt(np.sum((position[:2, 1:] - position[:2, :-1]) ** 2, axis=0))
    selected = np.flatnonzero(dm + half > clearance)
    points = np.zeros((selected.size * 33, 2))
    for a, i in enumerate(selected):
        for j in range(33):
            u = j / 32
            points[a * 33 + j] = (1 - u) * position[:2, i] + u * position[:2, i + 1]
    if not selected.size:
        return force, 0.0, 0.0, 0.0, 0.0
    _, normal, distance, _, _ = (
        project(points, curve) if code < 0 else project_graph(points, curve, code)
    )
    energy = 0.0
    penetration = 0.0
    outward = 0.0
    tangential = 0.0
    for a, i in enumerate(selected):
        for j in range(33):
            idx = 33 * a + j
            u = j / 32
            delta = max(distance[idx] - clearance, 0.0)
            w = ds[i] * (1.0 if j == 0 or j == 32 else 4.0 if j % 2 else 2.0) / 96
            load = stiffness * w * delta
            fx = -load * normal[idx, 0]
            fy = -load * normal[idx, 1]
            outward = max(outward, fx * normal[idx, 0] + fy * normal[idx, 1])
            tangential = max(tangential, abs(-fx * normal[idx, 1] + fy * normal[idx, 0]))
            energy += 0.5 * stiffness * w * delta**2
            penetration = max(penetration, delta)
            force[0, i] += (1 - u) * fx
            force[0, i + 1] += u * fx
            force[1, i] += (1 - u) * fy
            force[1, i + 1] += u * fy
    return force, energy, penetration, outward, tangential


@njit(cache=True)
def add_contact(position, rest_lengths, external, centerline, clearance, stiffness, graph_case=-1):
    external += wall_response(position, rest_lengths, centerline, graph_case, clearance, stiffness)[
        0
    ]


def update_contact_ledger(ledger, response):
    if ledger is None:
        return
    for key, value in zip(
        ("maximum_penetration", "maximum_outward_contact_force", "maximum_tangential_force"),
        response[2:],
    ):
        ledger[key] = max(ledger[key], float(value))


class LumenWall(ea.NoForces):
    def __init__(self, centerline, clearance, stiffness, graph_case, ledger=None):
        super().__init__()
        self.centerline, self.clearance = centerline, clearance
        self.stiffness, self.graph_case, self.ledger = stiffness, graph_case, ledger

    def apply_forces(self, system, time=0.0):
        response = wall_response(
            system.position_collection,
            system.rest_lengths,
            self.centerline,
            self.graph_case,
            self.clearance,
            self.stiffness,
        )
        if not np.isfinite(response[0]).all():
            raise RuntimeError("Nonfinite contact force")
        system.external_forces += response[0]
        update_contact_ledger(self.ledger, response)


class OpenLumen(ea.NoForces):
    """S-bend contact downstream of x=0, including the crossing element."""

    def __init__(self, curve, p, ledger):
        super().__init__()
        self.curve, self.p, self.ledger = curve, p, ledger

    def apply_forces(self, system, time=0.0):
        x = system.position_collection[0]
        if np.any(np.diff(x) <= 0):
            raise RuntimeError("Spatial feeder requires a monotone-x rod")
        first = max(0, int(np.searchsorted(x, 0.0)) - 1)
        response = wall_response(
            system.position_collection[:, first:],
            system.rest_lengths[first:],
            self.curve,
            2,
            self.p.half_width - self.p.radius,
            self.p.penalty_per_length,
        )
        if not np.isfinite(response[0]).all():
            raise RuntimeError("Nonfinite contact force")
        system.external_forces[:, first:] += response[0]
        update_contact_ledger(self.ledger, response)
