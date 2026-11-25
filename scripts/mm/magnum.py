"""Magnum.np construction for coupled skyrmion-vortex experiments."""

from __future__ import annotations

from dataclasses import dataclass

import magnumnp as mp
import numpy as np
from scipy import special
import torch

from scripts.mm.compression import SavedFrame
from scripts.mm.experiment import BC, Backend, Experiment


def select_device(requested: str = "auto") -> torch.device:
    """Resolve ``auto`` to MPS locally, CUDA remotely, and CPU otherwise."""
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


class PearlVortex:
    """Rigid Pearl-vortex field and overdamped motion on a magnetic plane."""

    def __init__(
        self,
        state: mp.State,
        pos: tuple[float, float] = (0.0, 0.0),
        lambdaP: float = 200e-9,
        phi0: float = 2.067833848e-15,
        sigma: float = 1.0,
        z: float = 2.5e-9,
        ds: float = 10e-9,
        polarity: int = -1,
    ) -> None:
        if lambdaP <= 0 or ds < 0 or z <= 0 or sigma <= 0:
            raise ValueError("Require lambdaP, z and sigma > 0, and ds >= 0.")
        if polarity not in (-1, 1):
            raise ValueError("Vortex polarity must be -1 or +1.")
        self.pos = state.m.new_tensor(pos)
        self.lambdaP = float(lambdaP)
        self.ds = float(ds)
        self.z = float(z)
        self.flux = polarity * float(phi0)
        self.sigma = float(sigma)
        self._mesh = state.mesh
        x, y, _ = self._mesh.SpatialCoordinate()
        self._x = x[:, :1, :1]
        self._y = y[:1, :, :1]
        pbc_x, pbc_y = (bool(value) for value in self._mesh.pbc[:2])
        if pbc_x != pbc_y or (pbc_x and not self._mesh.is_equidistant):
            raise ValueError("Periodic vortex sampling requires a uniform xy-periodic mesh.")
        self._periodic = pbc_x
        self._h_cache = None

    def screening(self, q):
        if self.ds == 0:
            return 1 + 2 * q * self.lambdaP

        lambdaL = np.sqrt(self.lambdaP * self.ds)
        tau = np.hypot(q, 1 / lambdaL)
        return (lambdaL * tau) ** 2 + self.lambdaP * q * (
            (tau * self.ds) / np.tanh(tau * self.ds / 2)
        )

    def build_profile(self, state: mp.State, r_max: float, Nr: int = 8193) -> None:
        if self._periodic:
            self._build_periodic_profile(state)
            self._h_cache = None
            return

        if r_max <= 0 or Nr < 2:
            raise ValueError("Require r_max > 0 and at least two radial samples.")
        q_max = 32 / self.z
        n_panels = int(np.ceil(q_max * max(r_max / (2 * np.pi), self.lambdaP / 2)))
        edges = np.linspace(0, q_max, n_panels + 1)
        nodes, weights = np.polynomial.legendre.leggauss(8)
        half_width = q_max / (2 * n_panels)
        q = (edges[:-1, None] + half_width * (1 + nodes)).ravel()
        weight = (
            half_width
            * weights[None, :]
            * (q * np.exp(-q * self.z) / self.screening(q)).reshape(n_panels, -1)
        ).ravel()
        weight *= self.flux / (2 * np.pi)

        rho = np.linspace(0, r_max, Nr)
        profile = np.empty((Nr, 2))
        for start in range(0, Nr, 64):
            radii = rho[start : start + 64]
            qr = radii[:, None] * q
            profile[start : start + len(radii), 0] = special.j1(qr) @ weight
            profile[start : start + len(radii), 1] = special.j0(qr) @ weight
        profile[1:, 0] /= rho[1:]
        profile[0, 0] = weight @ (q / 2)
        self._table = state.m.new_tensor(profile)
        self.dr = r_max / (Nr - 1)
        self._h_cache = None

    def _build_periodic_profile(self, state: mp.State) -> None:
        nx, ny = self._mesh.n[:2]
        dx, dy = self._mesh.dx[:2]
        oversampling = 4
        shape = (oversampling * nx, oversampling * ny)
        self._periodic_step = (dx / oversampling, dy / oversampling)
        kx = 2 * np.pi * np.fft.fftfreq(shape[0], self._periodic_step[0])[:, None]
        ky = 2 * np.pi * np.fft.rfftfreq(shape[1], self._periodic_step[1])[None, :]
        q = np.hypot(kx, ky)
        amplitude = self.flux / (dx * dy) * oversampling**2
        bz = amplitude * np.exp(-q * self.z) / self.screening(q)
        bz *= (abs(kx) < np.pi / dx) & (abs(ky) < np.pi / dy)
        q[0, 0] = 1
        profile = np.empty((*shape, 3))
        spectra = (-1j * kx * bz / q, -1j * ky * bz / q, bz)
        for component, spectrum in enumerate(spectra):
            profile[..., component] = np.fft.irfft2(spectrum, s=shape, axes=(0, 1))
        self._periodic_table = state.m.new_tensor(profile)
        self._periodic_indices = (
            torch.arange(nx, device=state.m.device)[:, None, None] * oversampling,
            torch.arange(ny, device=state.m.device)[None, :, None] * oversampling,
        )

    def _sample_periodic(self, x, y, pos, on_mesh: bool):
        step_x, step_y = self._periodic_step
        table = self._periodic_table
        if on_mesh:
            ix, iy = self._periodic_indices
            u = (self._x[0, 0, 0] - pos[0]) / step_x
            v = (self._y[0, 0, 0] - pos[1]) / step_y
        else:
            ix, iy = 0, 0
            u, v = (x - pos[0]) / step_x, (y - pos[1]) / step_y
        u0, v0 = u.floor(), v.floor()
        fx, fy = (u - u0).unsqueeze(-1), (v - v0).unsqueeze(-1)
        i0, j0 = (ix + u0.long()) % table.shape[0], (iy + v0.long()) % table.shape[1]
        i1, j1 = (i0 + 1) % table.shape[0], (j0 + 1) % table.shape[1]
        lower = torch.lerp(table[i0, j0], table[i1, j0], fx)
        upper = torch.lerp(table[i0, j1], table[i1, j1], fx)
        return torch.lerp(lower, upper, fy)

    def B(self, state, pos=None):
        """Sample Cartesian B in tesla on the mesh or at ``(..., 2)`` points."""
        pos = self.pos if pos is None else pos
        on_mesh = not isinstance(state, torch.Tensor)
        if on_mesh:
            if state.mesh is not self._mesh:
                raise ValueError("State must use the mesh used to construct the vortex.")
            x, y = self._x, self._y
        else:
            x, y = state.unbind(dim=-1)
        if self._periodic:
            return self._sample_periodic(x, y, pos, on_mesh)

        dx, dy = x - pos[0], y - pos[1]
        u2 = (dx / self.dr).square() + (dy / self.dr).square()
        radius = u2.clamp(min=1).sqrt()
        u = radius.clamp(max=len(self._table) - 1)
        index = u.long().clamp(max=len(self._table) - 2)
        fraction = torch.where(u2 < 1, u2, u - index)
        values = torch.lerp(self._table[index], self._table[index + 1], fraction.unsqueeze(-1))
        br_over_r, bz = values.unbind(dim=-1)
        br_over_r = br_over_r * ((len(self._table) - 1) / radius).clamp(max=1)
        return torch.stack((br_over_r * dx, br_over_r * dy, bz), dim=-1)

    def H(self, state: mp.State):
        """Return A/m, reusing the field across LLG stages at one position."""
        if self.pos.requires_grad or torch.is_inference(self.pos):
            return self.B(state) / mp.constants.mu_0
        if state.mesh is not self._mesh:
            raise ValueError("State must use the mesh used to construct the vortex.")
        cached = self._h_cache
        if cached is None or cached[0] is not self.pos or cached[1] != self.pos._version:
            cached = (self.pos, self.pos._version, self.B(state) / mp.constants.mu_0)
            self._h_cache = cached
        return cached[2]

    def interaction_energy(self, state: mp.State, pos):
        """Zeeman energy relative to the uniformly +z magnetized state."""
        bx, by, bz = self.B(state, pos).unbind(dim=-1)
        mx, my, mz = state.m.unbind(dim=-1)
        relative_dot = mx * bx + my * by + (mz - 1) * bz
        density = -state.material["Ms"] * relative_dot.unsqueeze(-1)
        return (density * state.mesh.cell_volumes).sum()

    def velocity(self, state: mp.State):
        with torch.enable_grad():
            pos = self.pos.detach().requires_grad_(True)
            force = -torch.autograd.grad(self.interaction_energy(state, pos), pos)[0]
        return force / self.sigma

    def advance(self, state: mp.State, dt: float):
        velocity = self.velocity(state)
        self.pos = (self.pos + velocity * dt).detach()
        return velocity


@dataclass
class MagnumSimulation:
    experiment: Experiment
    state: mp.State
    vortex: PearlVortex
    llg: mp.LLGSolver
    device: torch.device
    start_step: int
    mesh_shape: tuple[int, int, int]

    def magnetization_for_storage(self) -> np.ndarray:
        """Return ``(ny, nx, 3)`` unit vectors for HDF5 and plotting."""
        data = self.state.m.detach().cpu().squeeze(2).numpy()
        return data.swapaxes(0, 1)

    def vortex_position(self) -> np.ndarray:
        return self.vortex.pos.detach().cpu().numpy().astype(np.float64, copy=True)

    def time(self) -> float:
        return float(self.state.t.detach().cpu().item())


class MagnumSystemBuilder:
    """Build a fresh or restarted Magnum simulation from an :class:`Experiment`."""

    def from_experiment(
        self,
        experiment: Experiment,
        *,
        device: str | None = None,
        initial_frame: SavedFrame | None = None,
    ) -> MagnumSimulation:
        self._validate(experiment)
        resolved_device = select_device(device or experiment.device)
        torch.set_default_device(resolved_device)
        torch.set_default_dtype(torch.float32)

        lw = experiment.domain_wall_length
        dx_m = experiment.dx * lw
        nx = int(round(2 * experiment.width / experiment.dx)) + 1
        ny = int(round(2 * experiment.height / experiment.dx)) + 1
        mesh_shape = (nx, ny, 1)
        mesh = mp.Mesh(
            mesh_shape,
            (dx_m, dx_m, dx_m),
            (-experiment.width * lw, -experiment.height * lw, 0),
            pbc=(1, 1, 0),
        )
        state = mp.State(mesh)
        state.material = {
            "Ms": experiment.Ms,
            "A": experiment.A,
            "Ku": experiment.K,
            "Ku_axis": (0, 0, 1.0),
            "alpha": experiment.damping,
        }
        state.material["Di"] = self._dmi_field(state, experiment)
        state.m = self._initial_skyrmion(state, experiment)

        start_step = 0
        if initial_frame is not None:
            expected = (ny, nx, 3)
            if initial_frame.magnetization.shape != expected:
                raise ValueError(
                    f"Saved frame shape {initial_frame.magnetization.shape} does not match mesh {expected}."
                )
            restored = initial_frame.magnetization.swapaxes(0, 1)[:, :, None, :]
            state.m = torch.as_tensor(restored, dtype=state.dtype, device=state.device)
            state.t = initial_frame.time
            start_step = initial_frame.step

        vortex_position = (
            tuple(float(value) for value in initial_frame.vortex_position)
            if initial_frame is not None
            else tuple(value * lw for value in experiment.vortex_to)
        )
        vortex = PearlVortex(
            state,
            pos=vortex_position,
            lambdaP=experiment.pearl_length,
            phi0=experiment.phi0,
            sigma=experiment.vortex_sigma,
            z=dx_m / 2,
            ds=experiment.superconductor_thickness,
            polarity=experiment.vortex_polarity,
        )
        vortex.build_profile(state, r_max=np.hypot(nx * dx_m, ny * dx_m))
        llg = mp.LLGSolver(
            [
                mp.ExchangeField(),
                mp.UniaxialAnisotropyField(),
                mp.InterfaceDMIField(),
                mp.ExternalField(vortex.H),
            ]
        )
        return MagnumSimulation(
            experiment=experiment,
            state=state,
            vortex=vortex,
            llg=llg,
            device=resolved_device,
            start_step=start_step,
            mesh_shape=mesh_shape,
        )

    @staticmethod
    def _validate(experiment: Experiment) -> None:
        if experiment.backend != Backend.MAGNUM:
            raise ValueError("MagnumSystemBuilder requires backend=magnum.")
        if experiment.bc != BC.PERIODIC:
            raise ValueError("Magnum skyrmion-vortex experiments require periodic boundaries.")
        if experiment.width <= 0 or experiment.height <= 0 or experiment.dx <= 0:
            raise ValueError("width, height, and dx must be positive reduced lengths.")
        if experiment.R0 <= 0 or experiment.w0 <= 0:
            raise ValueError("Skyrmion radius and delta must be positive.")
        if experiment.dt <= 0 or experiment.steps < 0 or experiment.stride <= 0:
            raise ValueError("dt and stride must be positive, and steps cannot be negative.")
        if experiment.vortex_polarity not in (-1, 1):
            raise ValueError("vortex_polarity must be -1 or +1.")
        _ = experiment.pearl_length

    @staticmethod
    def _dmi_field(state: mp.State, experiment: Experiment):
        x, y, _ = state.mesh.SpatialCoordinate()
        lw = experiment.domain_wall_length
        tip_x, tip_y = (value * lw for value in experiment.field_to)
        height = experiment.h * lw
        r2 = (x - tip_x) ** 2 + (y - tip_y) ** 2
        reduced = experiment.eps + (
            experiment.beta * experiment.h * lw**3 / (r2 + height**2) ** 1.5
        )
        return 2 * experiment.dmi_scale * reduced

    @staticmethod
    def _initial_skyrmion(state: mp.State, experiment: Experiment):
        x, y, _ = state.mesh.SpatialCoordinate()
        lw = experiment.domain_wall_length
        center_x, center_y = (value * lw for value in experiment.skyrmion_to)
        radius = experiment.R0 * lw
        delta = experiment.w0 * lw
        r = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        theta = experiment.chi * 2 * torch.atan(
            np.sinh(radius / delta) / torch.sinh(r / delta)
        )
        phi = torch.atan2(y - center_y, x - center_x) + experiment.eta
        return torch.stack(
            (
                torch.sin(theta) * torch.cos(phi),
                torch.sin(theta) * torch.sin(phi),
                torch.cos(theta),
            ),
            dim=-1,
        )
