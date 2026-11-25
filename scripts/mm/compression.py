"""Compact, restartable HDF5 storage for unit-vector trajectories.

Each magnetization vector is represented by two 12-bit spherical angles. The
two integers are packed into three bytes, so the uncompressed representation is
24 bits per vector instead of 96 bits for three float32 components.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


BITS = 12
LEVELS = 2**BITS - 1
FORMAT = "spherical-theta-phi-12bit-v1"
FILE_NAME = "data.h5"


def pack_spherical_12bit(magnetization: np.ndarray) -> np.ndarray:
    """Normalize and pack Cartesian unit vectors into three uint8 values."""
    m = np.asarray(magnetization)
    if m.ndim < 1 or m.shape[-1] != 3:
        raise ValueError(f"Expected (..., 3) magnetization, got {m.shape}.")

    norm = np.linalg.norm(m, axis=-1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError("Cannot encode a zero-length magnetization vector.")
    unit = m / norm

    theta = np.arccos(np.clip(unit[..., 2], -1.0, 1.0))
    phi = np.arctan2(unit[..., 1], unit[..., 0])
    theta_q = np.rint(theta * (LEVELS / np.pi)).astype(np.uint16)
    phi_q = np.rint((phi + np.pi) * (LEVELS / (2 * np.pi))).astype(np.uint16)

    packed = np.empty(theta_q.shape + (3,), dtype=np.uint8)
    packed[..., 0] = theta_q >> 4
    packed[..., 1] = ((theta_q & 0x0F) << 4) | (phi_q >> 8)
    packed[..., 2] = phi_q & 0xFF
    return packed


def unpack_spherical_12bit(packed: np.ndarray) -> np.ndarray:
    """Decode packed spherical angles into float32 Cartesian unit vectors."""
    data = np.asarray(packed)
    if data.ndim < 1 or data.shape[-1] != 3 or data.dtype != np.uint8:
        raise ValueError(f"Expected uint8 (..., 3) packed data, got {data.dtype} {data.shape}.")

    theta_q = (data[..., 0].astype(np.uint16) << 4) | (data[..., 1] >> 4)
    phi_q = ((data[..., 1].astype(np.uint16) & 0x0F) << 8) | data[..., 2]
    theta = theta_q.astype(np.float64) * (np.pi / LEVELS)
    phi = phi_q.astype(np.float64) * (2 * np.pi / LEVELS) - np.pi

    sin_theta = np.sin(theta)
    return np.stack(
        (sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.cos(theta)),
        axis=-1,
    ).astype(np.float32)


@dataclass(frozen=True)
class SavedFrame:
    magnetization: np.ndarray
    vortex_position: np.ndarray
    step: int
    time: float
    index: int


class H5TrajectoryWriter:
    """Append saved frames and flush each one so interrupts preserve progress."""

    def __init__(
        self,
        path: str | Path,
        frame_shape: tuple[int, int, int],
        *,
        domain_wall_length: float,
    ) -> None:
        if len(frame_shape) != 3 or frame_shape[-1] != 3:
            raise ValueError("frame_shape must be (ny, nx, 3).")

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.path, "w")
        self._file.attrs.update(
            {
                "format": FORMAT,
                "bits_per_angle": BITS,
                "axis_order": "frame,y,x,component",
                "magnetization_units": "1",
                "vortex_position_units": "m",
                "time_units": "s",
                "step_units": "integrator_step",
                "domain_wall_length_m": float(domain_wall_length),
                "status": "running",
            }
        )

        shape = (0,) + frame_shape
        maxshape = (None,) + frame_shape
        self._m = self._file.create_dataset(
            "m",
            shape=shape,
            maxshape=maxshape,
            dtype=np.uint8,
            chunks=(1,) + frame_shape,
            compression="gzip",
            compression_opts=5,
            shuffle=True,
        )
        self._steps = self._file.create_dataset(
            "step", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=True
        )
        self._times = self._file.create_dataset(
            "time", shape=(0,), maxshape=(None,), dtype=np.float64, chunks=True
        )
        self._vortex = self._file.create_dataset(
            "vortex_position",
            shape=(0, 2),
            maxshape=(None, 2),
            dtype=np.float64,
            chunks=True,
        )
        self._file.flush()

    def append(
        self,
        magnetization: np.ndarray,
        vortex_position: np.ndarray | tuple[float, float],
        *,
        step: int,
        time: float,
    ) -> int:
        packed = pack_spherical_12bit(magnetization)
        if packed.shape != self._m.shape[1:]:
            raise ValueError(f"Frame shape {packed.shape} does not match {self._m.shape[1:]}.")
        position = np.asarray(vortex_position, dtype=np.float64)
        if position.shape != (2,):
            raise ValueError("vortex_position must contain exactly x and y.")

        index = len(self._steps)
        self._m.resize(index + 1, axis=0)
        self._steps.resize(index + 1, axis=0)
        self._times.resize(index + 1, axis=0)
        self._vortex.resize(index + 1, axis=0)
        self._m[index] = packed
        self._steps[index] = int(step)
        self._times[index] = float(time)
        self._vortex[index] = position
        self._file.attrs["saved_frames"] = index + 1
        self._file.attrs["last_step"] = int(step)
        self._file.attrs["last_time_s"] = float(time)
        self._file.flush()
        return index

    def close(self, status: str = "completed") -> None:
        if self._file.id.valid:
            self._file.attrs["status"] = status
            self._file.flush()
            self._file.close()

    def __enter__(self) -> H5TrajectoryWriter:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close("failed" if exc_type is not None else "completed")


def _resolve_frame_index(steps: np.ndarray, step: int) -> int:
    if len(steps) == 0:
        raise IndexError("The trajectory contains no saved frames.")
    if step == -1:
        return len(steps) - 1
    matches = np.flatnonzero(steps == step)
    if len(matches) == 0:
        available = ", ".join(str(int(value)) for value in steps)
        raise KeyError(f"Step {step} was not saved. Available steps: {available}")
    return int(matches[-1])


def load_frame(path: str | Path, step: int = -1) -> SavedFrame:
    """Load one frame by absolute integrator step (``-1`` selects latest)."""
    path = Path(path)
    with h5py.File(path, "r") as file:
        file_format = file.attrs.get("format", "")
        if file_format != FORMAT:
            raise ValueError(f"Unsupported trajectory format {file_format!r} in {path}.")
        steps = file["step"][:]
        index = _resolve_frame_index(steps, step)
        magnetization = unpack_spherical_12bit(file["m"][index])
        return SavedFrame(
            magnetization=magnetization,
            vortex_position=file["vortex_position"][index].astype(np.float64),
            step=int(steps[index]),
            time=float(file["time"][index]),
            index=index,
        )


def load_magnetization(path: str | Path) -> np.ndarray:
    """Decode every magnetization frame, primarily for analysis notebooks."""
    with h5py.File(path, "r") as file:
        if file.attrs.get("format", "") != FORMAT:
            raise ValueError(f"Unsupported trajectory format in {path}.")
        return unpack_spherical_12bit(file["m"][:])
