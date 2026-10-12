"""JSON serialization, runtime provenance and portable endpoint records."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path


def write_result(output, name, value):
    path = Path(output) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def provenance():
    versions = {}
    for name in ("cpf2d", "pyelastica", "numpy", "numba", "llvmlite", "scipy", "matplotlib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "uninstalled source"
    sources = {
        p.relative_to(Path(__file__).parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.rglob("*.py"))
    }
    return dict(
        python=platform.python_version(),
        platform=platform.platform(),
        packages=versions,
        source_sha256=sources,
    )


def prepare_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {path}. Choose a new output directory."
        )
    path.mkdir(parents=True, exist_ok=True)
    write_result(path, "environment.json", provenance())
    return path


def load_trajectory(path):
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    n = len(data["time"])
    if n == 0 or data["position"].shape[0] != n or data["position"].ndim != 3:
        raise ValueError("Invalid trajectory dimensions")
    if data["position"].shape[1] != 3 or data["step"].shape != (n,):
        raise ValueError("Invalid position or step shape")
    if not all(np.isfinite(value).all() for value in data.values()):
        raise ValueError("Nonfinite trajectory")
    if n > 1 and (np.any(np.diff(data["time"]) <= 0) or np.any(np.diff(data["step"]) <= 0)):
        raise ValueError("Sample times and solver steps must increase")
    if not np.allclose(data["time"], data["step"] * data["dt"], rtol=2e-9, atol=1e-9):
        raise ValueError("Recorded time does not match solver step and dt")
    return data
