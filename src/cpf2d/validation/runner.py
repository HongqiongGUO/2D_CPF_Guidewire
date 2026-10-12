"""Canonical validation independent of previously generated simulation data."""

from .common import LIMITS
from ..recording import write_result


def validate(output, suite="all"):
    from .lumen_geometry import geometry
    from .wall_impact import impact

    write_result(output, "limits.json", LIMITS)
    g = geometry(output)
    impacts = [impact(output, dt) for dt in (5e-5, 2.5e-5)]
    if suite == "quick":
        geometry_ok = (
            max(
                g["maximum_clearance_error"],
                g["maximum_projection_error"],
                g["maximum_penetration_error"],
            )
            < LIMITS["geometry_distance"]
            and g["maximum_normal_error"] < LIMITS["geometry_normal"]
            and g["false_contacts"] == 0
            and g["minimum_normal_load"] >= 0
            and max(
                g["maximum_outward_force"],
                g["maximum_tangential_force"],
                g["maximum_force_assembly_error"],
                g["maximum_center_force_error"],
            )
            <= 1e-14
            and max(s["error"] for s in g["straight_force_cases"]) <= 1e-14
        )
        impact_ok = all(
            i["all_finite"]
            and i["minimum_normal_load"] >= 0
            and max(i["maximum_outward_force"], i["maximum_tangential_force"]) <= 1e-14
            and max(
                abs(i["penetration_error_relative"]),
                abs(i["duration_error_relative"]),
                i["rebound_error_relative"],
            )
            < LIMITS["impact_reference_relative"]
            and i["maximum_energy_error_relative"] < LIMITS["impact_energy_relative"]
            for i in impacts
        )
        impact_ok &= (
            impacts[1]["maximum_energy_error_relative"]
            < impacts[0]["maximum_energy_error_relative"]
        )
        passed = bool(geometry_ok and impact_ok)
        write_result(
            output,
            "validation.json",
            dict(
                suite="quick",
                passed=passed,
                components=dict(vessel_geometry=bool(geometry_ok), normal_contact=bool(impact_ok)),
                omitted=["rod_dynamics", "fixed_entry_feeder"],
                combined_transient_certified=False,
            ),
        )
    else:
        from .cantilever import beam
        from .material_feed import feeder
        from .report import summarize

        for n, dt in ((64, 1e-4), (64, 5e-5), (128, 5e-5)):
            beam(output, n, dt)
        feeder(output)
        passed = summarize(output)["all_canonical_pass"]
    print(f'{suite} canonical validation: {"PASS" if passed else "FAIL"}', flush=True)
    return passed
