# Validation scope

The verification suite uses independent analytic references and explicit acceptance limits. It runs from source without reading precomputed scenario outputs. Acceptance limits are written to the output directory before the numerical runs.

| Component | Independent comparison | Acceptance |
|---|---|---|
| Rod | First cantilever frequency against a continuum Timoshenko transfer-matrix solution | Fine-mesh relative error <2%; timestep difference <0.1%; error decreases with mesh refinement |
| Contact | Rigid translation into a unilateral linear wall spring | Peak penetration, release duration and rebound relative errors <1%; peak energy error <0.2%; refinement improves energy, penetration and duration errors |
| Geometry | 56 analytic normal-offset probes and four straight-wall force cases | Distance error <1e-10; normal error <1e-8; no false interior contacts; correct force direction and assembly |
| Feeder | Material labels crossing two fixed spatial planes | Final relative passage error <1e-6; peak axial strain <1e-5; final speed <2e-9; unchanged mass and rest lengths |

## Cantilever

A straight rod has a fixed proximal node and first element director, with a free distal end. It starts with a small first-mode velocity perturbation and runs without damping or contact for four periods. A continuous least-squares sinusoid fit measures the frequency after half a period. Cases are 64 elements at dt=1e-4 and 5e-5, then 128 elements at dt=5e-5.

The reference uses the rod's actual bending stiffness `B`, transverse shear stiffness `S`, line mass `m` and rotary inertia `j`. A matrix exponential solves

```text
y' = theta + Q/S
theta' = M/B
Q' = -m omega^2 y
M' = -Q - j omega^2 theta
```

with `y(0)=theta(0)=Q(L)=M(L)=0`. The root does not fit rod solver data. At finite mesh, fixing director 0 fixes an orientation half an element from the geometric endpoint, so mesh convergence matters. The reference first frequency is approximately 0.05590575; the measured relative error is approximately 1.57% for 64 elements and 0.784% for 128.

## Wall impact

An unconstrained horizontal rod of length 0.4 and 16 elements starts at the wall with normal velocity 0.1. There is no damping. Uniform penetration gives the reference oscillator `omega=sqrt(k L/M)`, peak penetration `v0/omega`, contact duration `pi/omega` and rebound velocity `-v0`. Every saved state is an actual timestep; only the scalar release event time uses interpolation. Halving dt from 5e-5 to 2.5e-5 reduces the observed peak energy error from approximately 0.0398% to 0.00995%.

## Geometry and feeding

The geometry probes are constructed using exact graph coordinates and normals. They include both sides, the centerline, interior, grazing and penetrated points. Independent force assembly and potential-gradient tests cover the production penalty law.

The feeder check pushes a straight rod by 0.005 over 10 time units and holds to t=50. It locates material coordinates at fixed planes x=0 and x=0.5 independently of tip displacement. Material node 20 crosses the entry, and the final passage at both planes agrees with the commanded feed. This checks material passage through the spatial guide; it does not prove work consistency during release into curved contact.

## Regression and continuous integration

`tests/` also compares four short trajectories and two contacted force states against compact preserved numerical fixtures. Checks cover planarity, straight rest strain, spatial guide release, input validation, output preservation and sampling of off-stride endpoints. The fixture manifest records SHA-256 hashes and the mechanics backend commit.

GitHub Actions installs the locked environment and runs tests on Windows and Linux. The full canonical suite runs on Linux. Reference summaries distinguish preserved scenario results from freshly run component checks. **Passing component and regression checks does not certify coupled insertion transients.**
