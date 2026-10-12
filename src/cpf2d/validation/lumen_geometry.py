"""Canonical lumen geometry check."""

import csv
import json
import numpy as np
from ..geometry import vessel, project_graph
from ..contact import contact_observation, add_contact
from ..recording import write_result
from .common import H, PENALTY


def geometry(output):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for case, code, xs in [
        ("straight", 0, [0.2, 0.8]),
        ("single_bend", 1, [0.3, 0.8]),
        ("s_bend", 2, [0.12, 0.2375, 0.7125]),
    ]:
        curve = vessel(case)
        for x in xs:
            if code == 0:
                y = dy = 0.0
            elif code == 1:
                y = 0.32 * x * x
                dy = 0.64 * x
            else:
                a = 2 * np.pi * x / 0.95
                y = 0.08 * np.sin(a) ** 3
                dy = 0.24 * (2 * np.pi / 0.95) * np.sin(a) ** 2 * np.cos(a)
            base = np.array([x, y])
            normal = np.array([-dy, 1.0]) / np.hypot(1.0, dy)
            for side in (-1, 1):
                for d in (0.0, H - 0.01, H, H + 0.003):
                    point = base + side * d * normal
                    q, n, dist, arc, tan = project_graph(point[None, :], curve, code)
                    # An actual short straight probe element, with its middle sample
                    # exactly at the hand-constructed normal-offset test point.
                    direction = np.array([1.0, dy]) / np.hypot(1.0, dy)
                    pos = np.zeros((3, 2))
                    pos[:2, 0] = point - 0.5e-6 * direction
                    pos[:2, 1] = point + 0.5e-6 * direction
                    obs = contact_observation(pos, np.array([1e-6]), curve, H, PENALTY, code)
                    f = np.zeros_like(pos)
                    add_contact(pos, np.array([1e-6]), f, curve, H, PENALTY, code)
                    distributed = np.zeros_like(pos)
                    for j, u in enumerate(np.linspace(0, 1, 33)):
                        distributed[:2, 0] += (1 - u) * obs[5][j]
                        distributed[:2, 1] += u * obs[5][j]
                    rows.append(
                        dict(
                            case=case,
                            x=x,
                            side=side,
                            offset=d,
                            expected_clearance=H - d,
                            clearance=float(H - dist[0]),
                            clearance_error=float(abs(d - dist[0])),
                            projection_error=float(np.linalg.norm(q[0] - base)),
                            normal_error=float(np.linalg.norm(n[0] - side * normal)) if d else 0.0,
                            penetration=float(obs[3][16]),
                            expected_penetration=max(d - H, 0.0),
                            penetration_error=float(abs(obs[3][16] - max(d - H, 0.0))),
                            minimum_normal_load=float(obs[4].min()),
                            maximum_outward_force=float(np.max(np.sum(obs[5] * obs[2], axis=1))),
                            maximum_tangential_force=float(
                                np.max(np.abs(np.sum(obs[5] * obs[7], axis=1)))
                            ),
                            force_assembly_error=float(np.max(np.abs(f - distributed))),
                            center_force_error=float(
                                np.linalg.norm(
                                    obs[5][16]
                                    + PENALTY * obs[8][16] * max(d - H, 0.0) * side * normal
                                )
                            ),
                            false_contact=bool(d < H - 1e-12 and np.max(np.abs(f)) > 1e-16),
                        )
                    )
    # Non-grazing exact straight-wall tests also exercise broadphase and force magnitude.
    straight = []
    for y in (0.0, H - 0.001, H + 0.002, -H - 0.002):
        pos = np.array([[0.2, 0.3], [y, y], [0.0, 0.0]])
        f = np.zeros_like(pos)
        add_contact(pos, np.array([0.1]), f, vessel("straight"), H, PENALTY, 0)
        expected = np.array([0.0, -np.sign(y) * PENALTY * 0.1 * max(abs(y) - H, 0.0), 0.0])
        straight.append(
            dict(
                y=y,
                force=f.sum(axis=1).tolist(),
                expected_force=expected.tolist(),
                error=float(np.max(np.abs(f.sum(axis=1) - expected))),
            )
        )
    with (output / "geometry_cases.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    result = dict(
        component="geometry",
        probes=len(rows),
        straight_force_cases=straight,
        maximum_clearance_error=max(r["clearance_error"] for r in rows),
        maximum_projection_error=max(r["projection_error"] for r in rows),
        maximum_normal_error=max(r["normal_error"] for r in rows),
        maximum_penetration_error=max(r["penetration_error"] for r in rows),
        minimum_normal_load=min(r["minimum_normal_load"] for r in rows),
        maximum_outward_force=max(r["maximum_outward_force"] for r in rows),
        maximum_tangential_force=max(r["maximum_tangential_force"] for r in rows),
        maximum_force_assembly_error=max(r["force_assembly_error"] for r in rows),
        maximum_center_force_error=max(r["center_force_error"] for r in rows),
        false_contacts=sum(r["false_contact"] for r in rows),
    )
    write_result(output, "geometry.json", result)
    print(json.dumps(result), flush=True)
    return result
