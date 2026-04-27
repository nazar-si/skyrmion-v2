from scripts.mm.const import get_file
from scripts.mm.experiment import BC, Experiment
from scripts.mm.runner import run_experiment
from scripts.mm.system import SystemBuilder
import matplotlib.pyplot

def create_experiment() -> Experiment:
    return Experiment(
        name="beta=3-rotation",
        description="Run for attraction with beta=3.",

        width=7,
        height=7,
        dx=0.025,

        Ms=1,
        A=1,
        K=1,
        damping=0.05,
        bc=BC.PERIODIC,

        eps=0.5,
        gamma=0,
        beta=2,
        h=1,

        chi=1,
        eta=0,
        R0=1,
        w0=0.745,

        skyrmion_to=(3, 0),
        field_to=(0, 0),
        vortex_to=(0, 0),
    )


def main():
    builder = SystemBuilder()
    drive_parent = "02-attraction"

    # exp = create_experiment()
    # system, driver = builder.from_experiment(exp)
    exp, system, driver = builder.from_directory(f"./{drive_parent}/skyrmion_system/drive-4", magnetization_index=-1)
    def run():
        return driver.drive(
            system,
            dirname=drive_parent,
            output_step=True,
            t=1,
            n=1000,
            steps=15000,
            n_threads=12,
        )
    info = run_experiment(run, exp, drive_parent=drive_parent, stop_at_omf_step=500000)
    print(info)
if __name__ == "__main__":
    main()