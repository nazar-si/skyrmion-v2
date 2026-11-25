from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from scripts.mm.compression import load_frame
from scripts.mm.experiment import Backend, Experiment
from scripts.mm.magnum import MagnumSystemBuilder
from scripts.mm.runner import run_magnum_experiment


def test_one_step_cpu_magnum_run(tmp_path: Path) -> None:
    experiment = Experiment(
        name="smoke",
        backend=Backend.MAGNUM,
        width=0.1,
        height=0.1,
        dx=0.1,
        Ms=1e5,
        A=1e-11,
        K=1000.0,
        eps=0.45,
        gamma=0.15,
        beta=0.0,
        h=1.0,
        R0=0.7,
        w0=0.8,
        damping=0.1,
        dt=1e-15,
        steps=1,
        stride=1,
        device="cpu",
    )
    simulation = MagnumSystemBuilder().from_experiment(experiment, device="cpu")
    result = run_magnum_experiment(simulation, tmp_path)

    assert result["status"] == "completed"
    assert result["completed_step"] == 1
    run_dir = Path(result["run_dir"])
    assert (run_dir / Experiment.FILE_NAME).is_file()
    with h5py.File(result["data_file"]) as file:
        assert file["step"][:].tolist() == [0, 1]
        assert file["time"][-1] > 0

    frame = load_frame(result["data_file"], 1)
    restarted_experiment = Experiment.load(run_dir)
    restarted_experiment.steps = 0
    restarted = MagnumSystemBuilder().from_experiment(
        restarted_experiment,
        device="cpu",
        initial_frame=frame,
    )
    assert restarted.start_step == 1
    assert restarted.time() == frame.time
    assert np.allclose(restarted.vortex_position(), frame.vortex_position)
