# Reference data

This directory contains small numerical fixtures and expected scalar results. It contains no environment, third-party source checkout, complete trajectory archive or solver cache.

- `fixtures/*_trajectory.npz`: actual endpoint samples through t=0.2 for each of the four scenarios, at dt=5e-5 and penalty 800. The insertion samples cover only the beginning of its ramp.
- `fixtures/*_contact.npz`: two contacted shapes per curved lumen, with forces independently evaluated using the preserved quadrature kernel at penalty 800.
- `manifest.json`: hashes of each fixture, its source numerical archive, the preserved model source, and the PyElastica commit.
- `expected_metrics.json`: scalar scenario baselines and component validation results, with their provenance distinguished in the file.

The trajectory and contact fixtures are regression references, not analytic truth or proof of full transient convergence. Their source hashes establish identity without requiring private file paths. Fresh canonical checks use independent analytic references and never read these fixtures. Full numerical trajectories are regenerated using the commands in [reproduction](../docs/reproduction.md).
