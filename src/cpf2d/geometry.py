"""Canonical lumen graphs and nearest-point projection."""

import numpy as np
from numba import njit


def vessel(case, spacing=0.005):
    x = np.linspace(-0.25, 1.5, int(round(1.75 / spacing)) + 1)
    if case in ("free", "straight"):
        y = np.zeros_like(x)
    elif case == "single_bend":
        y = 0.32 * np.maximum(x, 0.0) ** 2
    elif case == "s_bend":
        phase = np.clip(x, 0.0, 0.95) / 0.95
        y = 0.08 * np.sin(2 * np.pi * phase) ** 3
    else:
        raise ValueError(case)
    return np.column_stack((x, y))


@njit(cache=True)
def project(points, centerline):
    """Global nearest point on the fixed, short polyline; ties choose first segment."""
    count = points.shape[0]
    closest = np.zeros((count, 2))
    normals = np.zeros((count, 2))
    distance = np.zeros(count)
    arc = np.zeros(count)
    tangent = np.zeros((count, 2))
    lengths = np.sqrt(np.sum((centerline[1:] - centerline[:-1]) ** 2, axis=1))
    cumulative = np.zeros(centerline.shape[0])
    for j in range(lengths.size):
        cumulative[j + 1] = cumulative[j] + lengths[j]
    for i in range(count):
        # Seed from the x-containing segment. The global x-distance bound below
        # still searches every segment that could improve this nearest point.
        seed = min(
            centerline.shape[0] - 2, max(0, np.searchsorted(centerline[:, 0], points[i, 0]) - 1)
        )
        ex = centerline[seed + 1, 0] - centerline[seed, 0]
        ey = centerline[seed + 1, 1] - centerline[seed, 1]
        px = points[i, 0] - centerline[seed, 0]
        py = points[i, 1] - centerline[seed, 1]
        u = min(1.0, max(0.0, (px * ex + py * ey) / (ex * ex + ey * ey)))
        dx = px - u * ex
        dy = py - u * ey
        best = dx * dx + dy * dy
        closest[i, 0] = centerline[seed, 0] + u * ex
        closest[i, 1] = centerline[seed, 1] + u * ey
        arc[i] = cumulative[seed] + u * lengths[seed]
        tangent[i, 0] = ex / lengths[seed]
        tangent[i, 1] = ey / lengths[seed]
        # For these monotone-x graphs, segments outside this x interval cannot
        # improve the seed distance. This is the same exact lower-bound search.
        radius = np.sqrt(best)
        first = max(0, np.searchsorted(centerline[:, 0], points[i, 0] - radius) - 1)
        stop = min(
            centerline.shape[0] - 1,
            np.searchsorted(centerline[:, 0], points[i, 0] + radius, side="right"),
        )
        for j in range(first, stop):
            bound = max(centerline[j, 0] - points[i, 0], points[i, 0] - centerline[j + 1, 0], 0.0)
            if bound * bound > best:
                continue
            ex = centerline[j + 1, 0] - centerline[j, 0]
            ey = centerline[j + 1, 1] - centerline[j, 1]
            length2 = ex * ex + ey * ey
            px = points[i, 0] - centerline[j, 0]
            py = points[i, 1] - centerline[j, 1]
            t = min(1.0, max(0.0, (px * ex + py * ey) / length2))
            qx = centerline[j, 0] + t * ex
            qy = centerline[j, 1] + t * ey
            dx = points[i, 0] - qx
            dy = points[i, 1] - qy
            d2 = dx * dx + dy * dy
            length = lengths[j]
            if d2 < best:
                best = d2
                closest[i, 0] = qx
                closest[i, 1] = qy
                arc[i] = cumulative[j] + t * length
                tangent[i, 0] = ex / length
                tangent[i, 1] = ey / length
        distance[i] = np.sqrt(best)
        if distance[i] > 1e-15:
            normals[i] = (points[i] - closest[i]) / distance[i]
    return closest, normals, distance, arc, tangent


@njit(cache=True)
def graph_values(x, case):
    if case == 1:
        if x <= 0:
            return 0.0, 0.0, 0.0
        return 0.32 * x * x, 0.64 * x, 0.64
    if case == 2 and 0 < x < 0.95:
        k = 2 * np.pi / 0.95
        s = np.sin(k * x)
        c = np.cos(k * x)
        return 0.08 * s**3, 0.24 * k * s * s * c, 0.24 * k * k * (2 * s * c * c - s**3)
    return 0.0, 0.0, 0.0


@njit(cache=True)
def project_graph(points, centerline, case):
    """Refine the global polyline seed onto the exact smooth canonical graph."""
    q, n, d, arc, tangent = project(points, centerline)
    lengths = np.sqrt(np.sum((centerline[1:] - centerline[:-1]) ** 2, axis=1))
    cumulative = np.zeros(centerline.shape[0])
    for j in range(lengths.size):
        cumulative[j + 1] = cumulative[j] + lengths[j]
    for i in range(points.shape[0]):
        x = q[i, 0]
        px = points[i, 0]
        py = points[i, 1]
        for iteration in range(20):
            y, dy, ddy = graph_values(x, case)
            f = x - px + (y - py) * dy
            derivative = 1 + dy * dy + (y - py) * ddy
            if abs(f) < 2e-15 or abs(derivative) < 1e-12:
                break
            step = f / derivative
            old = (x - px) ** 2 + (y - py) ** 2
            candidate = min(1.5, max(-0.25, x - step))
            for line_search in range(12):
                cy, _, _ = graph_values(candidate, case)
                if (candidate - px) ** 2 + (cy - py) ** 2 <= old + 1e-26:
                    break
                step *= 0.5
                candidate = min(1.5, max(-0.25, x - step))
            if abs(candidate - x) < 1e-15:
                break
            x = candidate
        y, dy, ddy = graph_values(x, case)
        q[i, 0] = x
        q[i, 1] = y
        dx = px - x
        delta_y = py - y
        d[i] = np.sqrt(dx * dx + delta_y * delta_y)
        if d[i] > 1e-15:
            n[i, 0] = dx / d[i]
            n[i, 1] = delta_y / d[i]
        else:
            n[i] = 0.0
        tangent[i, 0] = 1 / np.sqrt(1 + dy * dy)
        tangent[i, 1] = dy / np.sqrt(1 + dy * dy)
        # The arclength coordinate is the high-resolution plotting/seed polyline
        # coordinate at x; force and tangent use the exact smooth graph above.
        j = min(centerline.shape[0] - 2, max(0, np.searchsorted(centerline[:, 0], x) - 1))
        fraction = (x - centerline[j, 0]) / (centerline[j + 1, 0] - centerline[j, 0])
        arc[i] = cumulative[j] + fraction * lengths[j]
    return q, n, d, arc, tangent
