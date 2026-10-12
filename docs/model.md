# Mechanical formulation

## Rod and units

The solver is a discrete, shearable and extensible Cosserat rod implemented by PyElastica. Positions are constrained to the world XY plane. Each material cross-section retains a director triad: the first material director is the world Z axis, so only the first component of material angular velocity is allowed. The rod can stretch, shear and bend in the plane.

The natural configuration is a uniform straight rod: rest shear/stretch strain and rest curvature remain zero. A curved initial centerline is an initially strained configuration, not a curved stress-free rod. The default normalized parameters are:

| Quantity | Value |
|---|---:|
| Radius | 0.02 |
| Density | 1 |
| Young's modulus | 100 |
| Shear modulus | 100 / 3 |
| Axial stiffness multiplier | 16 |
| Uniform velocity damping | 0.5 |
| Lumen half-width | 0.055 |
| Centerline clearance | 0.035 |
| Normal penalty per reference length | 800 |
| Timestep | 0.00005 |

The axial multiplier acts only on the axial entry of the shear/stretch stiffness matrix. It does not rescale bending or transverse shear stiffness. The contact radius is the fixed nominal radius; it is not adjusted using instantaneous element dilatation. Values are normalized model quantities, without a calibration to a particular guidewire or clinical anatomy.

## Lumen geometry

The rigid lumen is a constant-distance neighborhood of a graph centerline. The three centerlines are a straight line, `y = 0.32 max(x, 0)^2`, and `y = 0.08 sin(2 pi x / 0.95)^3` for `0 < x < 0.95` with zero outside this interval. The graph domain is `[-0.25, 1.5]`.

A polyline with nominal spacing 0.005 supplies the nearest-point seed. Newton iteration with a bounded line search refines the projection onto the analytic graph. Contact normals and distances use the analytic graph; reported arclength uses the seed polyline. This projection is validated within the tubular neighborhood of the supplied graphs, not for arbitrary self-intersecting anatomy.

## Contact

Let `d` be the distance from a rod quadrature point to the lumen centerline, `h = half_width - radius`, and `delta = max(d - h, 0)`. The contact potential and force are

```text
U_contact = 1/2 k integral(delta^2 ds_ref)
f_q = -k w_q delta_q n_q
```

Here `n` points from the closest centerline point toward the rod, and `w` is a reference-length quadrature weight. Each element uses the fixed 33-point composite Simpson rule, including endpoints. Linear shape functions transfer each quadrature force to the element's two nodes. A distance bound skips elements that cannot contact either wall.

The law is repulsive and frictionless. It contributes no explicit contact torque, tangential force, adhesion, restitution parameter, or contact damping. The finite penalty permits a small penetration. The read-only quadrature observer independently checks force assembly and contact potential; all dynamic wall forces share one kernel in `contact.py`.

## Constraints and insertion

The proximal clamp prescribes node 0 position and director 0. A displacement `u` over duration `T` uses the quintic ramp `u (10 q^3 - 15 q^4 + 6 q^5)`, where `q` is elapsed time divided by `T`, clamped to `[0, 1]`.

Insertion starts with a finite rod from x=-1.15 to x=0.05. A spatial guide occupies x<0; its opening remains at x=0. Inside the guide, node transverse positions and velocities are zero. Element directors are fixed when their spatial midpoint lies upstream; the corresponding planar angular velocity is zero. Axial motion remains free except at the proximal pusher. Material nodes and elements are released as they cross the opening. Rest lengths, mass and node identities stay unchanged.

The insertion scenario advances by 0.95 over 120 time units, relaxes, advances a further 0.005 over 10 units, and relaxes again. Settling requires three consecutive checks of speed, unconstrained force residual and torque residual, after at least 20 time units. Checks are two units apart; the maximum relaxation duration is 180. The proximal end must remain upstream, and the rod must remain monotone in x for this guide's indexing to apply.

## Integration and observations

The dynamics use PyElastica PositionVerlet and its analytical linear damping operator. Energy and contact telemetry observe the actual solver state. Saved positions, velocities, directors and angular velocities are endpoint samples; videos render those samples directly. No trajectory interpolation, smoothing or motion amplification is applied. See [limitations](limitations.md) for the distinction between settled-state behavior and transient validation.
