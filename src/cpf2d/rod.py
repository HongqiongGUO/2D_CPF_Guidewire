"""Rod construction; initially curved geometry retains a straight rest state."""

import numpy as np
import elastica as ea
from .config import Parameters
from .geometry import vessel


class Simulator(ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.Damping):
    pass


def make_rod(p=Parameters(), start=(0.0, 0.0, 0.0), case="free"):
    r = ea.CosseratRod.straight_rod(
        p.n,
        np.asarray(start, dtype=float),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        p.length,
        p.radius,
        p.density,
        youngs_modulus=p.youngs_modulus,
        shear_modulus=p.shear_modulus,
    )
    base_directors = r.director_collection[:, :, 0].copy()
    if case in ("single_bend", "s_bend"):
        curve = vessel(case, p.geometry_spacing)
        curve = curve[curve[:, 0] >= -1e-12]
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
        material = np.linspace(0.0, p.length, p.n + 1)
        if p.length > arc[-1]:
            raise ValueError("Rod length exceeds the canonical initialization curve")
        r.position_collection[:2] = np.stack(
            [np.interp(material, arc, curve[:, k]) for k in range(2)]
        )
        tangent = np.diff(r.position_collection, axis=1)
        tangent /= np.linalg.norm(tangent, axis=0)
        r.director_collection[0] = np.array([0.0, 0.0, 1.0])[:, None]
        r.director_collection[1] = np.cross(tangent.T, np.array([0.0, 0.0, 1.0])).T
        r.director_collection[2] = tangent
    elif case not in ("free", "straight", "insertion"):
        raise ValueError(f"Unknown rod geometry: {case}")
    r.shear_matrix[2, 2] *= p.axial_scale
    return r, base_directors
