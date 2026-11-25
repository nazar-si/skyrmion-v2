"""Headless Magnum.np experiment for a coupled skyrmion-vortex pair."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from scripts.mm.compression import FILE_NAME, SavedFrame, load_frame
from scripts.mm.experiment import BC, Backend, Experiment
from scripts.mm.magnum import MagnumSystemBuilder
from scripts.mm.runner import run_magnum_experiment


OUTPUT_ROOT = Path(__file__).resolve().parent / "08-SkV-pair"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a periodic Magnum.np skyrmion-vortex model. Spatial CLI values "
            "are in domain-wall lengths lw; material constants and time are SI."
        )
    )
    parser.add_argument("--name", required=True, help="Short run name used in the output folder.")
    parser.add_argument("--description", default=None)
    parser.add_argument("--load-dir", type=Path, help="Prior 08-SkV-pair run directory.")
    parser.add_argument(
        "--load-step",
        type=int,
        default=-1,
        help="Absolute saved integrator step; -1 selects the latest frame.",
    )

    parser.add_argument("--skyrmion-position", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--vortex-position", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--tip-position", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--gamma", type=float, help="Reduced vortex-skyrmion strength.")
    parser.add_argument("--beta", type=float, help="Reduced nonuniform DMI strength.")
    parser.add_argument("--epsilon", type=float, help="Reduced constant DMI strength.")
    parser.add_argument("--radius", type=float, help="Initial skyrmion radius in lw.")
    parser.add_argument("--delta", type=float, help="Initial wall width in lw.")
    parser.add_argument("--dt", type=float, help="Integrator time step in seconds.")
    parser.add_argument("--steps", type=int, help="Number of additional integration steps.")
    parser.add_argument("--stride", type=int, help="Save and preview interval in steps.")

    parser.add_argument("--half-width", type=float, help="Domain half-width in lw.")
    parser.add_argument("--half-height", type=float, help="Domain half-height in lw.")
    parser.add_argument("--cell-size", type=float, help="Cubic cell size in lw.")
    parser.add_argument("--tip-height", type=float, help="Electric tip height in lw.")
    parser.add_argument("--ms", type=float, help="Saturation magnetization in A/m.")
    parser.add_argument("--exchange", type=float, help="Exchange constant A in J/m.")
    parser.add_argument("--anisotropy", type=float, help="Effective anisotropy K in J/m^3.")
    parser.add_argument("--damping", type=float, help="Gilbert damping alpha.")
    parser.add_argument("--vortex-drag", type=float, help="Vortex drag in N s/m.")
    parser.add_argument(
        "--superconductor-thickness",
        type=float,
        help="Superconductor thickness in metres; use 0 for the thin-film profile.",
    )
    parser.add_argument("--flux-quantum", type=float, help="Positive flux quantum in webers.")
    parser.add_argument("--vortex-polarity", type=int, choices=(-1, 1))
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cuda", "cpu"),
        default="auto",
        help="auto prefers MPS, then CUDA, then CPU.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Open and update a local Matplotlib window at the save stride.",
    )
    return parser


def notebook_defaults(name: str, description: str = "") -> Experiment:
    return Experiment(
        name=name,
        description=description,
        backend=Backend.MAGNUM,
        width=7.0,
        height=7.0,
        dx=0.025,
        Ms=1e5,
        A=1e-11,
        K=1000.0,
        eps=0.45,
        gamma=0.15,
        beta=0.0,
        h=1.0,
        chi=1.0,
        eta=0.0,
        R0=0.7,
        w0=0.8,
        bc=BC.PERIODIC,
        damping=0.1,
        skyrmion_to=(0.0, 0.0),
        field_to=(0.0, 0.0),
        vortex_to=(0.0, 0.0),
        dt=1e-11,
        steps=10_000,
        stride=10,
        device="auto",
        phi0=2.067833848e-15,
        vortex_sigma=2e-15,
        superconductor_thickness=10e-9,
        vortex_polarity=-1,
    )


def _apply_overrides(experiment: Experiment, args: argparse.Namespace) -> None:
    scalar_fields = {
        "gamma": "gamma",
        "beta": "beta",
        "epsilon": "eps",
        "radius": "R0",
        "delta": "w0",
        "dt": "dt",
        "steps": "steps",
        "stride": "stride",
        "half_width": "width",
        "half_height": "height",
        "cell_size": "dx",
        "tip_height": "h",
        "ms": "Ms",
        "exchange": "A",
        "anisotropy": "K",
        "damping": "damping",
        "vortex_drag": "vortex_sigma",
        "superconductor_thickness": "superconductor_thickness",
        "flux_quantum": "phi0",
        "vortex_polarity": "vortex_polarity",
    }
    for argument, field in scalar_fields.items():
        value = getattr(args, argument)
        if value is not None:
            setattr(experiment, field, value)

    position_fields = {
        "skyrmion_position": "skyrmion_to",
        "vortex_position": "vortex_to",
        "tip_position": "field_to",
    }
    for argument, field in position_fields.items():
        value = getattr(args, argument)
        if value is not None:
            setattr(experiment, field, tuple(value))


def create_experiment(args: argparse.Namespace) -> tuple[Experiment, SavedFrame | None]:
    initial_frame = None
    if args.load_dir is None:
        experiment = notebook_defaults(args.name, args.description or "")
    else:
        source_dir = args.load_dir.expanduser().resolve()
        source = Experiment.load(source_dir)
        if source.backend != Backend.MAGNUM:
            raise ValueError(f"Cannot restart non-Magnum experiment in {source_dir}.")
        data_file = source_dir / FILE_NAME
        initial_frame = load_frame(data_file, args.load_step)
        experiment = replace(source)
        experiment.name = args.name
        if args.description is not None:
            experiment.description = args.description
        experiment.source_dir = str(source_dir)
        experiment.source_file = str(data_file.resolve())
        experiment.source_step = initial_frame.step
        experiment.source_time = initial_frame.time

    experiment.backend = Backend.MAGNUM
    experiment.bc = BC.PERIODIC
    experiment.device = args.device
    experiment.status = "created"
    experiment.completed_step = None
    experiment.completed_time = None
    _apply_overrides(experiment, args)

    if initial_frame is not None:
        experiment.vortex_to = tuple(
            float(value / experiment.domain_wall_length)
            for value in initial_frame.vortex_position
        )
    return experiment, initial_frame


def main() -> None:
    args = build_parser().parse_args()
    experiment, initial_frame = create_experiment(args)
    simulation = MagnumSystemBuilder().from_experiment(
        experiment,
        device=args.device,
        initial_frame=initial_frame,
    )
    info = run_magnum_experiment(
        simulation,
        OUTPUT_ROOT,
        preview=args.preview,
    )
    print(f"{info['status']}: {info['run_dir']}")
    print(f"saved step {info['completed_step']} at t={info['completed_time']:.6g} s")


if __name__ == "__main__":
    main()
