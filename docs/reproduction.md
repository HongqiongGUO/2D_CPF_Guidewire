# Reproduction

## Environment

Use Git and uv. The tested interpreter is CPython 3.12.14. `uv.lock` locks dependencies, and the PyElastica dependency identifies the exact upstream commit `09d8e0ed4ae5658c9c4824b7a9a0e937d91bcd6d`. The first install requires internet access to PyPI and GitHub. No local checkout of the mechanics backend is required.

```sh
uv sync --locked
uv run cpf2d --help
uv run pytest
```

The commands are the same in PowerShell, Bash and Zsh. Python modules can also be invoked with `uv run python -m cpf2d`. Importing the package does not create output directories or launch simulations. Numba and Matplotlib manage their normal caches; headless systems may set `MPLBACKEND=Agg`.

## Scenarios

| Configuration | Model | Complete sequence |
|---|---|---|
| `configs/free_rod.toml` | 64 elements, length 1 | 0.0002 proximal ramp over 2 units; relax to t=24 |
| `configs/single_bend.toml` | 64 elements, length 1 | Release curved initial geometry; run to t=100 |
| `configs/s_bend.toml` | 64 elements, length 1 | Release curved initial geometry; run to t=100 |
| `configs/insertion.toml` | 80 elements, length 1.2 | Insert, relax, perturb, relax |

Run from the repository root, using a new or empty output directory:

```sh
uv run cpf2d run --config configs/insertion.toml --output outputs/insertion
uv run cpf2d render outputs/insertion
```

Full contact runs can take tens of minutes on a desktop CPU. The exact runtime depends on hardware and compilation caches. A quick installation check uses an explicit duration:

```sh
uv run cpf2d run --case insertion --duration 0.2 --output outputs/insertion_check
```

For insertion, `--duration` stops the initial ramp after that amount of model time and records `complete_sequence=false`. It does not accelerate the ramp or replace a complete experiment. Other scenarios run to the requested duration. Duration and sample interval must be integer multiples of `dt`; an endpoint is always saved even if it falls between normal sample times. Changing `dt`, mesh or stiffness requires a new stability and convergence assessment.

## Output format

Each run writes:

| File | Content |
|---|---|
| `config.json` | Fully resolved scenario and physical parameters |
| `environment.json` | Interpreter, platform, package versions and source hashes |
| `summary.json` | Diagnostics, telemetry and sampled history; insertion checkpoints |
| `trajectory.npz` | Actual numerical states and sample-to-step mapping |

NPZ arrays use `position/velocity: (frames, 3, nodes)`, `directors: (frames, 3, 3, elements)`, `omega: (frames, 3, elements)`, `time/step: (frames,)`, and scalar `dt`. These archives contain no pickled objects and can be loaded with `numpy.load(..., allow_pickle=False)`.

For the first three scenarios, `contact_points` has shape `(frames, 33*elements, 2)`, with a boolean `contact_active` mask; the free rod has zero contact points. Insertion stores only active downstream points as a flat `(points, 2)` array, with `contact_offsets` of length `frames+1` and a `phase` array. Phase integers 0–3 denote the physical sequence: insertion, relaxation, perturbation and relaxation. They are physical event labels.

No generated trajectory or video is required to import or run the package. Output directories and caches are excluded from Git. Tests retain their numerical artifacts under `outputs/tests/` for inspection; commands do not automatically clear earlier results.

## Rendering

```sh
uv run cpf2d render outputs/insertion
uv sync --locked --extra video
uv run cpf2d render outputs/insertion --output outputs/insertion/motion.mp4 --fps 24
```

A PNG shows the last saved state. MP4 rendering uses every recorded frame; it requires a system FFmpeg or the optional `imageio-ffmpeg` extra. The renderer uses Matplotlib's bundled font, with no operating-system font paths. A companion frame CSV maps video frames to solver steps and model time. Playback speed depends on sample spacing and the chosen FPS; it does not change simulated time.

## Validation

```sh
uv run cpf2d validate --suite quick --output outputs/quick_checks
uv run cpf2d validate --suite all --output outputs/canonical_checks
```

The quick suite checks analytic geometry and wall impact. The full suite additionally runs three cantilever discretizations and straight material feeding, and produces JSON, CSV and three figures. Failed acceptance criteria return a nonzero process exit code. See [validation](validation.md).
