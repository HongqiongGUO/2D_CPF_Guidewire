"""Command-line interface; imports never launch a simulation."""

import argparse
from dataclasses import replace
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="cpf2d", description="Planar Cosserat guidewire simulation"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run one configured physical scenario")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--case", choices=("free", "single_bend", "s_bend", "insertion"))
    source.add_argument("--config", type=Path)
    run.add_argument("--output", type=Path, required=True, help="New or empty output directory")
    run.add_argument(
        "--duration", type=float, help="Override duration; insertion runs only its initial ramp"
    )
    run.add_argument("--dt", type=float, help="Override solver timestep")
    validate = sub.add_parser("validate", help="Run canonical physics checks")
    validate.add_argument("--suite", choices=("quick", "all"), default="all")
    validate.add_argument("--output", type=Path, required=True)
    render = sub.add_parser("render", help="Render actual saved endpoint states")
    render.add_argument("run_directory", type=Path)
    render.add_argument(
        "--output", type=Path, help="PNG or MP4 path (default: preview.png in run directory)"
    )
    render.add_argument("--fps", type=int, default=24)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            from .config import load_config, default_scenario
            from .recording import prepare_output
            from .simulation import run as simulate

            config = load_config(args.config) if args.config else default_scenario(args.case)
            overrides = {}
            if args.duration is not None:
                overrides["duration"] = args.duration
            if args.dt is not None:
                overrides["parameters"] = replace(config.parameters, dt=args.dt)
            config = replace(config, **overrides)
            result = simulate(config, prepare_output(args.output))
            if result.get("complete_sequence") and not result["final_settled"]:
                print("The sequence finished without satisfying the settling criteria.")
                return 1
        elif args.command == "validate":
            from .recording import prepare_output
            from .validation.runner import validate

            return 0 if validate(prepare_output(args.output), args.suite) else 1
        else:
            from .render import render_run

            render_run(args.run_directory, args.output, args.fps)
    except (ValueError, TypeError, FileExistsError, FileNotFoundError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
