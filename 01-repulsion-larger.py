from scripts.mm.const import get_file
from scripts.mm.experiment import Experiment
from scripts.mm.runner import run_experiment
from scripts.mm.system import SystemBuilder
import matplotlib.pyplot

def create_experiment() -> Experiment:
    return Experiment(
        name="beta=-2-rotation",
        description="Segmented run for repulsion with larger amount of steps before refitting with larger field.",

        width=7,
        height=7,
        dx=0.025,

        Ms=1,
        A=1,
        K=1,
        damping=0.05,

        eps=0.5,
        gamma=0,
        beta=-3,
        h=1,

        chi=1,
        eta=0,
        R0=1,
        w0=0.745,

        skyrmion_to=(0, 0),
        field_to=(5, 0),
        vortex_to=(0, 0),
    )


def main():
    builder = SystemBuilder()
    drive_parent = "01-repulsion"

    info = {
        "final_dir": "01-repulsion/skyrmion_system/drive-7-beta-2-rotation"
    }

    index = 1
    while True:
        if index == 1 and False:
            exp = create_experiment()
        else:
            exp, system, driver = builder.from_directory(
                info["final_dir"],
                magnetization_index=-1,
            )
            if index != 1:
                builder.fit_skyrmion_parameters(exp, system.m)
                # shift coordinate system to skyrmion in center 
                exp.vortex_to = (exp.vortex_to[0] - exp.skyrmion_to[0], exp.vortex_to[1] - exp.skyrmion_to[1])
                exp.field_to = (exp.field_to[0] - exp.skyrmion_to[0], exp.field_to[1] - exp.skyrmion_to[1])
                exp.skyrmion_to = (0, 0)
                system, driver = builder.from_experiment(exp)
        def run():
            return driver.drive(
                system,
                dirname=drive_parent,
                output_step=True,
                t=1,
                n=1000,
                steps=10000,
                n_threads=12,
            )
        print(index, "Experiment:", exp)
        index += 1 
        info = run_experiment(run, exp, drive_parent=drive_parent, stop_at_omf_step=200000)
        print(info)
if __name__ == "__main__":
    main()