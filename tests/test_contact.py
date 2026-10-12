"""Force, potential, projection and broadphase regression checks."""

import unittest
import numpy as np
from cpf2d.geometry import project, project_graph, vessel
from cpf2d.contact import contact_observation, add_contact


class PhysicsChecks(unittest.TestCase):
    def test_inside_and_two_sided_containment(self):
        cl = np.array([[-2.0, 0.0], [2.0, 0.0]])
        for y, expected in ((0.0, 0.0), (0.03, 0.0), (0.08, -0.04), (-0.08, 0.04)):
            pos = np.array([[0.0, 1.0], [y, y], [0.0, 0.0]])
            obs = contact_observation(pos, np.array([1.0]), cl, 0.04, 1.0)
            self.assertAlmostEqual(obs[5][:, 1].sum(), expected, places=12)
            self.assertGreaterEqual(obs[4][0], 0.0)
            self.assertEqual(obs[5][:, 0].sum(), 0.0)

    def test_global_projection_and_rotated_lumen(self):
        cl = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        q, n, d, arc, t = project(np.array([[1.1, 0.7], [-0.1, 0.05]]), cl)
        np.testing.assert_allclose(q, [[1.0, 0.7], [0.0, 0.0]], atol=1e-14)
        np.testing.assert_allclose(arc, [1.7, 0.0], atol=1e-14)
        angle = 0.43
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        points = np.array([[0.2, 0.08], [0.6, -0.1]])
        q1, n1, d1, _, _ = project(points, np.array([[0.0, 0.0], [1.0, 0.0]]))
        q2, n2, d2, _, _ = project(points @ rot.T, np.array([[0.0, 0.0], [1.0, 0.0]]) @ rot.T)
        np.testing.assert_allclose(q2, q1 @ rot.T, atol=1e-13)
        np.testing.assert_allclose(n2, n1 @ rot.T, atol=1e-13)
        np.testing.assert_allclose(d2, d1, atol=1e-13)

    def test_contact_force_is_negative_penalty_energy_gradient(self):
        cl = np.array([[-1.0, -0.2], [0.0, 0.0], [1.0, 0.2]])
        pos = np.array([[0.1, 0.4, 0.7], [0.12, 0.21, 0.25], [0.0, 0.0, 0.0]])
        ds = np.array([0.3, 0.3])
        clearance = 0.035
        k = 7.0
        f = np.zeros_like(pos)
        add_contact(pos, ds, f, cl, clearance, k)

        def energy(p):
            obs = contact_observation(p, ds, cl, clearance, k)
            return 0.5 * k * np.dot(obs[8], obs[3] ** 2)

        for axis in (0, 1):
            for node in range(3):
                plus = pos.copy()
                minus = pos.copy()
                plus[axis, node] += 1e-7
                minus[axis, node] -= 1e-7
                gradient = (energy(plus) - energy(minus)) / 2e-7
                self.assertAlmostEqual(f[axis, node], -gradient, places=8)
        obs = contact_observation(pos, ds, cl, clearance, k)
        np.testing.assert_allclose(f.sum(axis=1)[:2], obs[5].sum(axis=0), atol=1e-14)

    def test_projection_pruning_matches_unpruned_reference(self):
        rng = np.random.default_rng(1729)
        points = rng.uniform([-0.4, -0.2], [1.7, 0.65], size=(120, 2))
        for case in ("straight", "single_bend", "s_bend"):
            cl = vessel(case)
            edge = np.diff(cl, axis=0)
            delta = points[:, None, :] - cl[None, :-1, :]
            u = np.clip(np.einsum("ijk,jk->ij", delta, edge) / np.sum(edge**2, axis=1), 0, 1)
            q = cl[None, :-1, :] + u[:, :, None] * edge
            d2 = np.sum((points[:, None, :] - q) ** 2, axis=2)
            best = np.argmin(d2, axis=1)
            actual = project(points, cl)
            np.testing.assert_allclose(actual[0], q[np.arange(points.shape[0]), best], atol=2e-13)
            np.testing.assert_allclose(
                actual[2], np.sqrt(d2[np.arange(points.shape[0]), best]), atol=2e-13
            )

    def test_exact_smooth_curve_normal_offsets_project_back(self):
        x = np.linspace(0.03, 1.15, 40)
        for case, code in (("single_bend", 1), ("s_bend", 2)):
            if code == 1:
                y = 0.32 * x * x
                slope = 0.64 * x
            else:
                k = 2 * np.pi / 0.95
                inside = x < 0.95
                y = np.where(inside, 0.08 * np.sin(k * x) ** 3, 0.0)
                slope = np.where(inside, 0.24 * k * np.sin(k * x) ** 2 * np.cos(k * x), 0.0)
            normal = np.column_stack((-slope, np.ones(x.size)))
            normal /= np.linalg.norm(normal, axis=1)[:, None]
            center = np.column_stack((x, y))
            for delta in (-0.025, 0.025):
                point = center + delta * normal
                q, n, d, _, _ = project_graph(point, vessel(case), code)
                np.testing.assert_allclose(q, center, atol=2e-11)
                np.testing.assert_allclose(d, abs(delta), atol=2e-12)

    def test_smooth_curve_penalty_work(self):
        cl = vessel("s_bend")
        pos = np.array([[0.22, 0.24, 0.26], [0.13, 0.13, 0.13], [0.0, 0.0, 0.0]])
        ds = np.array([0.02, 0.02])
        k = 13.0
        clearance = 0.035
        force = np.zeros_like(pos)
        add_contact(pos, ds, force, cl, clearance, k, 2)

        def energy(p):
            o = contact_observation(p, ds, cl, clearance, k, 2)
            return 0.5 * k * np.dot(o[8], o[3] ** 2)

        for axis in (0, 1):
            for node in range(3):
                plus = pos.copy()
                minus = pos.copy()
                plus[axis, node] += 1e-7
                minus[axis, node] -= 1e-7
                grad = (energy(plus) - energy(minus)) / 2e-7
                self.assertAlmostEqual(force[axis, node], -grad, places=8)
