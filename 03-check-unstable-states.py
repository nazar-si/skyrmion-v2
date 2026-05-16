import numpy as np
from scripts.mm.const import get_file
from scripts.mm.experiment import BC, Experiment
from scripts.mm.runner import run_experiment
from scripts.mm.system import SystemBuilder
from scripts.compiled import Problem
import scripts.scientific 
import matplotlib.pyplot as plt 

from scripts.refined import refined_search

def create_experiment() -> Experiment:
    return Experiment(
        name="beta=-5-unstable-check",
        description="Checking behavior of unstable solutions for beta=-5",

        width=5,
        height=5,
        dx=0.025,

        Ms=1,
        A=1,
        K=1,
        damping=0.05,
        bc=BC.PERIODIC,

        eps=0.3,
        gamma=0,
        beta=-8.8,
        h=1,

        chi=-1,
        eta=0,
        R0=1,
        w0=0.745,

        skyrmion_to=(0, 0),
        field_to=(0, 0),
        vortex_to=(0, 0),
    )


def main():
    builder = SystemBuilder()
    drive_parent = "03-unstable-modes"

    exp = create_experiment()
    problem = Problem(epsilon=exp.eps, beta=exp.beta, chi=exp.chi, height=exp.h, helicity=exp.chi, method='DOP853')
    answers = refined_search(problem, -np.pi/2 + 1e-3, np.pi/2 - 1e-3, 10000, 10, exclude_negative=False)
    ans = answers[3]
    # plt.plot(ans.r, ans.theta)
    # plt.xlim([0,6])
    # plt.show()
    
    interp = problem.interpolate(ans)

    system, driver = builder.from_envelope(exp, interp.theta)
    def run():
        return driver.drive(
            system,
            dirname=drive_parent,
            output_step=True,
            t=1,
            n=1000,
            steps=1000,
            n_threads=12,
        )
    info = run_experiment(run, exp, drive_parent=drive_parent, stop_at_omf_step=20000)
if __name__ == "__main__":
    main()