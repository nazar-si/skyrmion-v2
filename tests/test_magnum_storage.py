from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from scripts.mm.compression import (
    FORMAT,
    H5TrajectoryWriter,
    load_frame,
    pack_spherical_12bit,
    unpack_spherical_12bit,
)
from scripts.mm.experiment import Backend, Experiment
from scripts.mm.runner import create_magnum_run_directory


def _unit_vectors(shape: tuple[int, ...], seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=shape + (3,))
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def test_spherical_12bit_round_trip() -> None:
    original = _unit_vectors((20, 30))
    packed = pack_spherical_12bit(original)
    restored = unpack_spherical_12bit(packed)

    assert packed.dtype == np.uint8
    assert packed.shape == original.shape
    assert restored.dtype == np.float32
    angular_error = np.arccos(np.clip(np.sum(original * restored, axis=-1), -1, 1))
    assert float(angular_error.max()) < 0.002


def test_h5_trajectory_appends_and_loads_by_absolute_step(tmp_path: Path) -> None:
    first = _unit_vectors((4, 5), seed=1)
    second = _unit_vectors((4, 5), seed=2)
    path = tmp_path / "data.h5"
    writer = H5TrajectoryWriter(path, first.shape, domain_wall_length=1e-7)
    writer.append(first, (1e-7, 2e-7), step=120, time=1.2e-9)
    writer.append(second, (3e-7, 4e-7), step=130, time=1.3e-9)
    writer.close("interrupted")

    selected = load_frame(path, 120)
    latest = load_frame(path, -1)
    assert selected.step == 120
    assert selected.time == 1.2e-9
    assert np.allclose(selected.vortex_position, (1e-7, 2e-7))
    assert latest.step == 130
    with h5py.File(path) as file:
        assert file.attrs["format"] == FORMAT
        assert file.attrs["status"] == "interrupted"
        assert file["m"].shape == (2, 4, 5, 3)


def test_experiment_migrates_v1_oommf_metadata(tmp_path: Path) -> None:
    old_data = {
        "version": 1,
        "name": "legacy",
        "driver": "time",
        "bc": "xy",
        "skyrmion_to": [0, 0],
        "field_to": [0, 0],
        "vortex_to": [0, 0],
    }
    with (tmp_path / Experiment.FILE_NAME).open("w") as file:
        yaml.safe_dump(old_data, file)

    experiment = Experiment.load(tmp_path)
    assert experiment.version == 2
    assert experiment.backend == Backend.OOMMF
    assert experiment.name == "legacy"


def test_magnum_directory_format(tmp_path: Path) -> None:
    experiment = Experiment(name="A useful test")
    first = create_magnum_run_directory(tmp_path, experiment)
    second = create_magnum_run_directory(tmp_path, experiment)
    assert first.name.startswith("01-")
    assert first.name.endswith("-a-useful-test")
    assert second.name.startswith("02-")


def test_experiment_records_reduced_and_derived_si_values(tmp_path: Path) -> None:
    experiment = Experiment(A=1e-11, K=1000.0, Ms=1e5, gamma=0.15, width=7, dx=0.025)
    path = experiment.save(tmp_path)
    with path.open() as file:
        data = yaml.safe_load(file)

    assert data["width"] == 7
    assert data["derived_si"]["domain_wall_length_m"] == pytest.approx(1e-7)
    assert data["derived_si"]["cell_size_m"] == pytest.approx(2.5e-9)
    assert Experiment.load(tmp_path).width == 7
