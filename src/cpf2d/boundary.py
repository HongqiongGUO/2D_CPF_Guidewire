"""Planar clamping and a fixed spatial guide with material passage."""

import numpy as np
import elastica as ea


class PlanarBase(ea.ConstraintBase):
    def __init__(self, fixed_position, fixed_directors, control, drift, **kwargs):
        super().__init__(**kwargs)
        self.origin = fixed_position.copy()
        self.base_directors = control["base_directors"].copy()
        self.control = control
        self.drift = drift

    def prescribed(self, t):
        elapsed = t - self.control["start"]
        duration = self.control["duration"]
        q = np.clip(elapsed / duration, 0.0, 1.0)
        fraction = 10 * q**3 - 15 * q**4 + 6 * q**5
        derivative = (30 * q**2 - 60 * q**3 + 30 * q**4) / duration if 0 < q < 1 else 0.0
        return self.control["du"] * fraction, self.control["du"] * derivative

    def constrain_values(self, system, time):
        self.drift["raw_position_z"] = max(
            self.drift["raw_position_z"], float(np.max(np.abs(system.position_collection[2])))
        )
        u, _ = self.prescribed(time)
        system.position_collection[2] = 0.0
        system.position_collection[:, 0] = self.origin + u * np.array([1.0, 0.0, 0.0])
        system.director_collection[:, :, 0] = self.base_directors

    def constrain_rates(self, system, time):
        self.drift["raw_velocity_z"] = max(
            self.drift["raw_velocity_z"], float(np.max(np.abs(system.velocity_collection[2])))
        )
        self.drift["raw_forbidden_omega"] = max(
            self.drift["raw_forbidden_omega"], float(np.max(np.abs(system.omega_collection[1:])))
        )
        _, v = self.prescribed(time)
        system.velocity_collection[2] = 0.0
        system.velocity_collection[:, 0] = [v, 0.0, 0.0]
        # d1 is the world z axis. Only rotation about d1 is planar.
        system.omega_collection[1:] = 0.0
        system.omega_collection[:, 0] = 0.0


class Feeder(PlanarBase):
    """A spatial straight guide x<0. Axial motion is free except at the pusher.

    Its downstream opening stays at x=0. Material nodes leave the guide as their
    actual coordinates cross zero; rest lengths and node identities never change.
    """

    def prescribed(self, t):
        u, v = super().prescribed(t)
        return self.control["offset"] + u, v

    def constrain_values(self, system, time):
        super().constrain_values(system, time)
        outside = system.position_collection[0] < 0
        guided = 0.5 * (system.position_collection[0, 1:] + system.position_collection[0, :-1]) < 0
        system.position_collection[1, outside] = 0.0
        system.director_collection[:, :, guided] = self.base_directors[:, :, None]

    def constrain_rates(self, system, time):
        super().constrain_rates(system, time)
        outside = system.position_collection[0] < 0
        guided = 0.5 * (system.position_collection[0, 1:] + system.position_collection[0, :-1]) < 0
        system.velocity_collection[1, outside] = 0.0
        system.omega_collection[0, guided] = 0.0
