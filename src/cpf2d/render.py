"""Portable figures and optional videos made from actual recorded states."""

import csv
import json
from pathlib import Path
import shutil
import numpy as np
from .geometry import graph_values
from .recording import load_trajectory, write_result


def render_run(run_directory, output=None, fps=24):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    run_directory = Path(run_directory)
    target = Path(output) if output else run_directory / "preview.png"
    if target.exists():
        raise FileExistsError(f"Render target already exists: {target}")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if target.suffix.lower() not in (".png", ".mp4"):
        raise ValueError("Choose a .png or .mp4 output")
    data = load_trajectory(run_directory / "trajectory.npz")
    config = json.loads((run_directory / "config.json").read_text(encoding="utf-8"))
    case = config["case"]
    fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
    fig.patch.set_facecolor("#fafcfb")
    ax.set_facecolor("#fafcfb")
    if case != "free":
        code = 1 if case == "single_bend" else 2
        x = np.linspace(0, 1.12, 700)
        graph = np.array([graph_values(xx, code) for xx in x])
        center = np.column_stack((x, graph[:, 0]))
        normal = np.column_stack((-graph[:, 1], np.ones(len(x))))
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        width = config["parameters"]["half_width"]
        for sign in (-1, 1):
            wall = center + sign * width * normal
            ax.plot(wall[:, 0], wall[:, 1], color="#a2b5b8", lw=1.5)
    if case == "insertion":
        ax.axvline(0, color="#a2b5b8", ls=":", lw=1, label="Fixed entry")
    (line,) = ax.plot([], [], color="#147d8b", lw=2, label="Rod centerline")
    contact = ax.scatter([], [], color="#d97935", s=9, label="Active contact samples", zorder=3)
    position = data["position"]
    ax.update_datalim(position[:, :2, :].transpose(0, 2, 1).reshape(-1, 2))
    ax.autoscale_view()
    ax.margins(0.06, 0.25)
    ax.set_aspect("equal", adjustable="box")
    if case == "free":
        ax.set_ylim(-0.04, 0.04)
    ax.set(xlabel="x (model length)", ylabel="y (model length)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=8)

    def frame(i):
        i %= len(data["time"])
        line.set_data(position[i, 0], position[i, 1])
        if "contact_offsets" in data:
            a, b = data["contact_offsets"][i : i + 2]
            points = data["contact_points"][a:b]
        else:
            points = data["contact_points"][i][data["contact_active"][i]]
        contact.set_offsets(points.reshape(-1, 2))
        ax.set_title(f'{case.replace("_", " ").title()}  |  t = {data["time"][i]:.3f}')

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        if target.suffix.lower() == ".png":
            frame(-1)
            fig.savefig(target, dpi=180)
        else:
            executable = shutil.which("ffmpeg")
            if executable is None:
                try:
                    import imageio_ffmpeg
                except ImportError as error:
                    raise ValueError(
                        "Video requires ffmpeg or: uv sync --locked --extra video"
                    ) from error
                executable = imageio_ffmpeg.get_ffmpeg_exe()
            plt.rcParams["animation.ffmpeg_path"] = executable
            writer = FFMpegWriter(
                fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "18"]
            )
            with writer.saving(fig, str(target), dpi=120):
                for i in range(len(data["time"])):
                    frame(i)
                    writer.grab_frame()
            with target.with_suffix(".frames.csv").open("w", newline="", encoding="utf-8") as file:
                rows = csv.writer(file)
                rows.writerow(["frame", "solver_step", "solver_time"])
                rows.writerows(
                    (i, int(step), float(t))
                    for i, (step, t) in enumerate(zip(data["step"], data["time"]))
                )
            write_result(
                target.parent,
                target.stem + ".video.json",
                dict(
                    frames=len(data["time"]),
                    fps=fps,
                    interpolation=False,
                    motion_amplification=1.0,
                    source="trajectory.npz",
                ),
            )
    finally:
        plt.close(fig)
    print(target)
    return target
