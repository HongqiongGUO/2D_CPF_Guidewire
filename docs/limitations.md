# Limitations

The model is planar and uses rigid, analytic, constant-width lumens with frictionless normal penalty contact. It does not represent three-dimensional anatomy, torsional steering, compliant vessel walls, fluid coupling, plasticity, material failure or a calibrated commercial guidewire. The fixed nominal contact radius and the factor-16 axial stiffness adjustment are explicit modeling choices.

The spatial guide assumes the rod remains monotone in x. It constrains upstream nodes and element orientations and releases them at the fixed opening. It is a finite-rod pusher, not an infinite material-injection boundary. Configurations that fold back in x are rejected.

**Detailed coupled S-bend insertion transients are not timestep-convergence certified.** A retained comparison at penalty 200 between dt=1e-4 and 5e-5 had a maximum transient node-position difference of about 0.00731648 and a maximum phase RMS difference of about 0.000398138. Refinement from the coarser pair did not show the required reduction. Settled shape and local tip response differences were much smaller, about 1.95e-11 and 5.92e-12. Those results do not establish converged contact timing or dynamic trajectories.

The shipped working scenario uses penalty 800 and dt=5e-5. Stable runs and small settling residuals at that configuration do not resolve the transient convergence issue at another penalty or prove convergence at the working penalty. Component validation checks isolated regimes; it neither identifies the cause of the coupled discrepancy nor removes it.

The quintic input, damping observations and contact potential support diagnostic energy reporting. They do not supply a complete work-consistency proof for constraint activation and release in the curved insertion problem. No physical interpretation should be inferred from visual smoothness alone.
