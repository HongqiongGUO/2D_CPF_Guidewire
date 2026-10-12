"""Acceptance criteria and figures for a complete canonical run."""

import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from .common import LIMITS
from ..recording import write_result


def summarize(output):
    def read(name):
        return json.loads((output / name).read_text(encoding="utf-8"))

    def load(result):
        with np.load(output / (result["name"] + ".npz"), allow_pickle=False) as data:
            return data["history"]

    beams = [
        read(f"beam_n{n}_dt{dt:g}.json") for n, dt in ((64, 0.0001), (64, 0.00005), (128, 0.00005))
    ]
    impacts = [read(f"impact_dt{dt:g}.json") for dt in (5e-5, 2.5e-5)]
    g = read("geometry.json")
    f = read("feeder.json")
    b0, b1, b2 = beams
    i0, i1 = impacts
    dt_difference = abs(b0["frequency"] / b1["frequency"] - 1)
    mesh_difference = abs(b1["frequency"] / b2["frequency"] - 1)
    rod_pass = bool(
        abs(b2["reference_error_relative"]) < LIMITS["beam_reference_relative"]
        and dt_difference < LIMITS["beam_dt_relative"]
        and abs(b2["reference_error_relative"]) < abs(b1["reference_error_relative"])
        and all(b["all_finite"] and b["out_of_plane"] < LIMITS["planarity"] for b in beams)
    )
    contact_pass = bool(
        all(
            i["all_finite"]
            and i["minimum_normal_load"] >= 0
            and i["maximum_outward_force"] <= 1e-14
            and i["maximum_tangential_force"] <= 1e-14
            and max(
                abs(i["penetration_error_relative"]),
                abs(i["duration_error_relative"]),
                i["rebound_error_relative"],
            )
            < LIMITS["impact_reference_relative"]
            and i["maximum_energy_error_relative"] < LIMITS["impact_energy_relative"]
            for i in impacts
        )
        and i1["maximum_energy_error_relative"] < i0["maximum_energy_error_relative"]
        and abs(i1["penetration_error_relative"]) < abs(i0["penetration_error_relative"])
        and abs(i1["duration_error_relative"]) < abs(i0["duration_error_relative"])
    )
    geometry_pass = bool(
        max(
            g["maximum_clearance_error"],
            g["maximum_projection_error"],
            g["maximum_penetration_error"],
        )
        < LIMITS["geometry_distance"]
        and g["maximum_normal_error"] < LIMITS["geometry_normal"]
        and g["false_contacts"] == 0
        and g["minimum_normal_load"] >= 0
        and g["maximum_outward_force"] <= 1e-14
        and g["maximum_tangential_force"] <= 1e-14
        and g["maximum_force_assembly_error"] <= 1e-14
        and g["maximum_center_force_error"] <= 1e-14
        and max(s["error"] for s in g["straight_force_cases"]) <= 1e-14
    )
    feeder_pass = bool(
        f["all_finite"]
        and max(f["entry_relative_error"], f["distal_relative_error"])
        < LIMITS["feed_final_relative"]
        and f["maximum_axial_strain"] < LIMITS["feed_peak_strain"]
        and f["final_speed"] < LIMITS["feed_final_speed"]
        and f["maximum_transverse_motion"] < LIMITS["planarity"]
        and f["maximum_fixed_section_error"] < 1e-12
        and f["rest_lengths_unchanged"]
        and f["mass_unchanged"]
        and len(f["node_ids_crossed_entry"]) > 0
    )
    passes = dict(
        rod_dynamics=rod_pass,
        normal_contact=contact_pass,
        vessel_geometry=geometry_pass,
        fixed_entry_feeder=feeder_pass,
    )
    result = dict(
        components=passes,
        all_canonical_pass=all(passes.values()),
        rod_timestep_frequency_difference_relative=dt_difference,
        rod_mesh_frequency_difference_relative=mesh_difference,
        impact_energy_error_reduction=i0["maximum_energy_error_relative"]
        / i1["maximum_energy_error_relative"],
        combined_transient_certified=False,
        limitation="Component checks do not certify timestep convergence of coupled insertion transients.",
    )
    write_result(output, "validation.json", result)
    rows = []

    def add(component, test, metric, value, reference, unit, status):
        rows.append(
            dict(
                component=component,
                test=test,
                metric=metric,
                value=value,
                reference=reference,
                absolute_error=abs(value - reference),
                relative_error_percent=100 * (value / reference - 1) if reference else "",
                unit=unit,
                component_pass=status,
            )
        )

    for b in beams:
        add(
            "rod",
            b["name"],
            "first_frequency",
            b["frequency"],
            b["reference"]["frequency"],
            "1/model_time",
            rod_pass,
        )
        add(
            "rod",
            b["name"],
            "maximum_relative_energy_error",
            b["maximum_energy_error_relative"],
            0.0,
            "fraction",
            rod_pass,
        )
    add(
        "rod",
        "dt_refinement",
        "relative_frequency_difference",
        dt_difference,
        0.0,
        "fraction",
        rod_pass,
    )
    add(
        "rod",
        "mesh_refinement",
        "relative_frequency_difference",
        mesh_difference,
        0.0,
        "fraction",
        rod_pass,
    )
    for i in impacts:
        for key, unit in [
            ("maximum_penetration", "model_length"),
            ("contact_duration", "model_time"),
            ("rebound_velocity", "model_length/model_time"),
        ]:
            add("contact", i["name"], key, i[key], i["reference"][key], unit, contact_pass)
        add(
            "contact",
            i["name"],
            "maximum_relative_energy_error",
            i["maximum_energy_error_relative"],
            0.0,
            "fraction",
            contact_pass,
        )
    for key, unit in [
        ("maximum_clearance_error", "model_length"),
        ("maximum_penetration_error", "model_length"),
        ("maximum_normal_error", "dimensionless"),
        ("maximum_outward_force", "model_force"),
        ("false_contacts", "count"),
    ]:
        add("geometry", "56_probes_and_4_force_checks", key, g[key], 0.0, unit, geometry_pass)
    for key, reference, unit in [
        ("entry_material_passage", f["feed"], "model_length"),
        ("distal_material_passage", f["feed"], "model_length"),
        ("maximum_axial_strain", 0.0, "dimensionless"),
        ("maximum_fixed_section_error", 0.0, "model_length"),
        ("final_speed", 0.0, "model_length/model_time"),
    ]:
        add("feeder", "straight_feed", key, f[key], reference, unit, feeder_pass)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "figure.dpi": 120,
            "savefig.dpi": 180,
        }
    )
    colors = ["#6e7d8b", "#149c95", "#c5603b"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for b, c in zip(beams, colors):
        h = load(b)
        axes[0].plot(h[:, 0], h[:, 1], color=c, lw=1.2, label=f"{b['n']} elements, dt={b['dt']:g}")
        axes[1].scatter(
            1 / b["n"],
            100 * b["reference_error_relative"],
            color=c,
            s=65,
            marker="o" if b["dt"] == 5e-5 else "x",
            label=f"{b['n']} elements, dt={b['dt']:g}",
        )
    t = load(b2)[:, 0]
    axes[0].plot(
        t,
        b2["amplitude"] * np.sin(b2["reference"]["omega"] * t),
        "k--",
        lw=1,
        label="Continuum first mode",
    )
    axes[0].set(
        xlabel="Time (model units)",
        ylabel="Tip transverse displacement",
        title="Small-amplitude cantilever",
    )
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0].legend(fontsize=8)
    axes[1].plot(
        [1 / b1["n"], 1 / b2["n"]],
        [100 * b1["reference_error_relative"], 100 * b2["reference_error_relative"]],
        color=colors[1],
        lw=1,
    )
    axes[1].axhline(0, color="black", ls="--", lw=1, label="Timoshenko reference")
    axes[1].set(
        xlabel="Element length / rod length",
        ylabel="First-frequency error (%)",
        title="Mesh and timestep comparison",
    )
    axes[1].legend(fontsize=8)
    fig.savefig(output / "cantilever.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    tt = np.linspace(0, load(i1)[-1, 0], 1000)
    omega = i1["reference"]["omega"]
    T = i1["reference"]["contact_duration"]
    delta = np.where(tt < T, 0.1 / omega * np.sin(omega * tt), 0.0)
    velocity = np.where(tt < T, 0.1 * np.cos(omega * tt), -0.1)
    for i, c in zip(impacts, colors[1:]):
        h = load(i)
        label = f"dt={i['dt']:g}"
        axes[0].plot(h[:, 0], h[:, 3], color=c, label=label)
        axes[1].plot(h[:, 0], h[:, 2], color=c, label=label)
        axes[2].plot(
            h[:, 0],
            100 * (h[:, 4:7].sum(axis=1) / i["reference"]["energy"] - 1),
            color=c,
            label=label,
        )
    axes[0].plot(tt, delta, "k--", lw=1, label="Unilateral spring reference")
    axes[1].plot(tt, velocity, "k--", lw=1)
    axes[2].axhline(0, color="black", ls="--", lw=1)
    for ax, title, y in zip(
        axes,
        ["Penetration", "Impact and rebound", "Kinetic + elastic + contact energy"],
        ["Penetration (model length)", "Normal velocity", "Energy error (%)"],
    ):
        ax.set(xlabel="Time (model units)", ylabel=y, title=title)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    fig.savefig(output / "wall_impact.png")
    plt.close(fig)

    h = np.load(output / "feeder.npz")["history"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), layout="constrained")
    axes[0].plot(
        h[:, 0], h[:, 1], color="black", ls="--", lw=2, label="Imposed proximal displacement"
    )
    axes[0].plot(h[:, 0], h[:, 3], color=colors[1], label="Material crossing fixed entry")
    axes[0].plot(
        h[:, 0],
        h[:, 4],
        color=colors[2],
        ls=":",
        lw=2,
        label="Material crossing fixed distal plane",
    )
    axes[0].set(
        xlabel="Time (model units)",
        ylabel="Feed / material passage",
        title="Independent material-label measurement",
    )
    axes[0].legend(fontsize=8)
    axes[1].plot(h[:, 0], h[:, 3] - h[:, 1], color=colors[1], label="Entry passage minus input")
    axes[1].plot(h[:, 0], h[:, 4] - h[:, 1], color=colors[2], label="Distal passage minus input")
    axes[1].set(
        xlabel="Time (model units)",
        ylabel="Passage error (model length)",
        title="Transient compression relaxes after feeding",
    )
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].legend(fontsize=8)
    fig.savefig(output / "fixed_entry_feed.png")
    plt.close(fig)

    return result
