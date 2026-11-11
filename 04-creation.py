import numpy as np
from scripts.mm.const import get_file
from scripts.mm.experiment import BC, Experiment
from scripts.mm.runner import run_experiment
from scripts.mm.system import SystemBuilder
from scripts.compiled import Problem
import scripts.scientific 
import matplotlib.pyplot as plt 

from scripts.refined import refined_search

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, help="From which file to start")
parser.add_argument("--frame", type=int, help="From which frame to start if --from is passed")
parser.add_argument("--count", type=int, help="Number of frames to model")
parser.add_argument("--step", type=int, help="Number of steps between MM snapshots")
parser.add_argument("--name", type=str, help="Prefix for exact experiment name")
parser.add_argument("--size", type=float, help="Size of a system")
parser.add_argument("--dmi", type=float, help="DMI")
parser.add_argument("-r", '--regular', action='store_true', help="Start in mode without swinging")
parser.add_argument("beta", type=float, help="Field strength")


def create_experiment(args) -> Experiment:
    beta = args.beta
    size = args.size if args.size else 5
    epsilon = args.dmi if args.dmi else 0.3
    return Experiment(
        name= args.name + (("-swing-end" if args.dir else "-swing-start") if not args.regular else ""),
        description="Check swing creation conditions",

        width=size,
        height=size,
        dx=0.025,

        Ms=1,
        A=1,
        K=1,
        damping=0.05,
        bc=BC.PERIODIC,

        eps=epsilon,
        gamma=0,
        beta=beta if args.dir or args.regular else -beta,
        h=1,

        chi=1,
        eta=0,
        # No skyrmion initially
        R0=0,
        w0=0,

        skyrmion_to=(0, 0),
        field_to=(0, 0),
        vortex_to=(0, 0),
    )


def main():
    args = parser.parse_args()
    builder = SystemBuilder()
    drive_parent = "04-creation"

    is_full_run = args.dir or args.regular

    if args.dir:
        exp = create_experiment(args)
        _, system, driver = builder.from_directory(f'./04-creation/skyrmion_system/{args.dir}', magnetization_index=args.frame if args.frame else -1, override_exp=exp)
    else:
        exp = create_experiment(args)
        system, driver = builder.from_empty(exp)

    def run():
        return driver.drive(
            system,
            dirname=drive_parent,
            output_step=True,
            t=1,
            n=1000,
            steps=args.step if args.step else (5000 if is_full_run else 1000),
            n_threads=12,
        )
    info = run_experiment(run, exp, drive_parent=drive_parent, stop_at_omf_step=args.count if args.count else (10000 if is_full_run else 2000))
if __name__ == "__main__":
    main()