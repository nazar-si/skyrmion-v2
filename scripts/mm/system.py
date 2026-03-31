import numpy as np
import discretisedfield as df
import micromagneticmodel as mm
import oommfc as mc
from pathlib import Path

from scripts.mm.experiment import Experiment, Driver
from scripts.mm.const import get_file, list_magnetization_files


class SystemBuilder:
    """
    Builds micromagnetic system either from an Experiment object
    or from files on disk (experiment.yaml + state files).
    """

    def __init__(self, base_path: str | Path = "."):
        self.base_path = Path(base_path)

        self.system: mm.System | None = None
        self.driver: mc.TimeDriver | None = None

    # -----------------------
    # Fresh system from Experiment
    # -----------------------
    def from_experiment(self, exp: Experiment | None) -> tuple[mm.System, mc.TimeDriver]:
        assert exp is not None, "Experiment object required"

        A_ex = exp.A
        K_an = exp.K
        l_w = np.sqrt(A_ex / K_an) if K_an != 0 else 1

        W = exp.width * l_w
        H = exp.height * l_w
        dL = exp.dx * l_w
        Ms = exp.Ms * (1e5 if exp.driver == Driver.TIME else 1)
        eps_0 = exp.eps
        chi = exp.chi
        eta = exp.eta
        R0 = exp.R0 * l_w
        w0 = exp.w0 * l_w
        
        beta = exp.beta * l_w ** 2
        h = exp.h * l_w
        alpha = exp.damping

        D = np.sqrt(A_ex * K_an)

        # --- mesh ---
        p1 = (-W, -H, 0)
        p2 = (W, H, dL)
        cell = (dL, dL, dL)

        region = df.Region(p1=p1, p2=p2)
        mesh = df.Mesh(region=region, cell=cell, bc="xy")

        to = (
            exp.field_to[0] * l_w,
            exp.field_to[1] * l_w,
        )

        skyrmion_to = (
            exp.skyrmion_to[0] * l_w,
            exp.skyrmion_to[1] * l_w,
        )

        def EM_DMI(pos):
            x, y, z = pos
            r2 = (x - to[0]) ** 2 + (y - to[1]) ** 2
            eps = eps_0 + beta * h / (r2 + h**2) ** (3 / 2)
            return (0, 0, -2 * eps * D)

        EM_field = df.Field(mesh, nvdim=3, value=EM_DMI)

        def m_initial(pos):
            x, y, z = pos
            r = np.sqrt((x - skyrmion_to[0])**2 + (y - skyrmion_to[1])**2)
            theta = chi * 2 * np.arctan(np.sinh(R0 / w0) / np.sinh(r / w0)) if r != 0 else np.pi
            phi = np.atan2(y - skyrmion_to[1], x - skyrmion_to[0])
            mx = np.sin(theta) * np.cos(phi + eta) 
            my = np.sin(theta) * np.sin(phi + eta)
            mz = np.cos(theta) 
            return (mx, my, mz)


        system = mm.System(name="skyrmion_system")
        system.m = df.Field(mesh, nvdim=3, value=m_initial, norm=Ms)
        system.energy = (
            mm.Exchange(A=A_ex)
            + mm.UniaxialAnisotropy(K=K_an, u=(0, 0, 1))
            + mm.DMI(D=EM_field, crystalclass="Cnv_z")
        )
        system.dynamics = mm.Damping(alpha=alpha) + mm.Precession(gamma0=mm.consts.gamma0)

        mc.runner.runner = mc.oommf.DockerOOMMFRunner(image="field/oommf:latest")
        driver = mc.TimeDriver()

        self.system = system
        self.driver = driver
        return system, driver

    # -----------------------
    # Load system from disk
    # -----------------------
    def from_directory(
        self,
        directory: str | Path,
        *,
        magnetization_index: int | None = -1,
        magnetization_chunk: int | list[int] | None = None,
    ) -> tuple[Experiment, mm.System, mc.TimeDriver]:
        """
        Load `Experiment` + rebuild system + restore magnetization from a saved .omf.

        Parameters
        ----------
        directory:
            Path to a drive folder (e.g. `.../drive-0-...`) that contains `experiment.yaml`
            and magnetization `.omf` files.
        magnetization_index:
            Index into sorted magnetization files (sorted by last numeric suffix).
            Default `-1` = latest.
            Ignored if `magnetization_chunk` is provided.
        magnetization_chunk:
            Chunk/step number (the last numeric suffix), e.g. `2000` for `...-0002000.omf`.
            You can also pass `[2000]` to match your earlier convention.

        Returns
        -------
        (experiment, system, driver)
        """
        directory = Path(directory).resolve()
        exp = Experiment.load(directory)
        system, driver = self.from_experiment(exp)

        if magnetization_chunk is not None:
            m_file = get_file(directory, magnetization_chunk if isinstance(magnetization_chunk, list) else [int(magnetization_chunk)])
        else:
            if magnetization_index is None:
                magnetization_index = -1
            m_files = list_magnetization_files(directory)
            if not m_files:
                raise FileNotFoundError(f"No magnetization .omf files found in {directory}")
            m_file = m_files[magnetization_index]

        m_file = m_file.resolve()
        system.m = df.Field.from_file(str(m_file))

        # record provenance
        exp.source_dir = str(directory)
        exp.source_file = str(m_file)

        self.system = system
        self.driver = driver
        return exp, system, driver

    # -----------------------
    # Fitting helpers
    # -----------------------

    def fit_skyrmion_parameters(
        self,
        exp: Experiment,
        magnetization: df.Field,
    ) -> Experiment:
        """
        Fit R0, w0, eta, skyrmion_to (x,y) from a given magnetization field by
        minimizing vector-valued least-squares error against the analytical skyrmion ansatz.

        Notes
        -----
        - `chi` is treated as fixed (taken from `exp`).
        - The error is computed as sum of squared per-spin differences:
          `(mx - mx_target)^2 + (my - my_target)^2 + (mz - mz_target)^2`.
        - For speed, the fit can subsample the grid (see `sample_stride`).
        """
        from scipy.optimize import least_squares

        # Convert field to numpy arrays: discretisedfield gives (nx, ny, 3) for sel('z').
        m = magnetization.sel("z").array
        if m.ndim != 3 or m.shape[-1] != 3:
            raise ValueError(f"Unexpected magnetization array shape: {m.shape}")

        mx_t = m[:, :, 0]
        my_t = m[:, :, 1]
        mz_t = m[:, :, 2]

        mesh = magnetization.mesh
        (x1, y1, _), (x2, y2, _) = mesh.region.pmin, mesh.region.pmax
        cell_x, cell_y = mesh.cell[0], mesh.cell[1]

        nx, ny = mx_t.shape
        xs = np.linspace(x1 + cell_x / 2, x2 - cell_x / 2, nx)
        ys = np.linspace(y1 + cell_y / 2, y2 - cell_y / 2, ny)
        X, Y = np.meshgrid(xs, ys, indexing="ij")  # (nx, ny)

        # Model magnitude (matches from_experiment(..., norm=Ms))
        Ms_model = exp.Ms * (1e5 if exp.driver == Driver.TIME else 1)

        chi = exp.chi

        # Initial guess (robust-ish even under deformation):
        # core is where mz is minimal.
        m_mag = np.sqrt(mx_t**2 + my_t**2 + mz_t**2)
        Ms_est = float(np.median(m_mag))
        if not np.isfinite(Ms_est) or Ms_est <= 0:
            Ms_est = float(Ms_model)

        # weight core based on how far mz is from +Ms
        core_weight = np.clip((Ms_est - mz_t) / (2 * Ms_est), 0.0, 1.0)
        wsum = float(core_weight.sum())
        if wsum > 0:
            x0_init = float((core_weight * X).sum() / wsum)
            y0_init = float((core_weight * Y).sum() / wsum)
        else:
            i0, j0 = np.unravel_index(np.argmin(mz_t), mz_t.shape)
            x0_init, y0_init = float(xs[i0]), float(ys[j0])

        # Rough size guess: use weighted radius of core.
        r_init = np.sqrt((X - x0_init) ** 2 + (Y - y0_init) ** 2)
        if wsum > 0:
            r_mean = float((core_weight * r_init).sum() / wsum)
        else:
            r_mean = float(np.median(r_init))

        # Ensure positive nonzero guesses.
        eps = 1e-9
        R0_init = max(float(r_mean), eps)
        w0_init = max(float(r_mean), eps)

        # Phase offset eta guess: align in-plane angle with geometric angle.
        phi_geo = np.arctan2(Y - y0_init, X - x0_init)
        phi_m = np.arctan2(my_t, mx_t)
        dphi = np.angle(np.exp(1j * (phi_m - phi_geo)))
        eta_init = float(np.sum(dphi * core_weight) / (np.sum(core_weight) + 1e-12))

        # Subsample for speed.
        sample_stride = 4
        sx = slice(None, None, sample_stride)
        sy = slice(None, None, sample_stride)

        Xs = X[sx, sy]
        Ys = Y[sx, sy]
        mx_s = mx_t[sx, sy]
        my_s = my_t[sx, sy]
        mz_s = mz_t[sx, sy]

        # Bounds in physical units.
        r_span = float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
        R0_min, R0_max = 1e-6, max(2 * r_span, 1e-3)
        w0_min, w0_max = 1e-6, max(2 * r_span, 1e-3)

        lower = [x1, y1, R0_min, w0_min, -np.pi]
        upper = [x2, y2, R0_max, w0_max, np.pi]
        p0 = [x0_init, y0_init, R0_init, w0_init, eta_init]

        def model_m(params: np.ndarray):
            x0, y0, R0_phys, w0_phys, eta = params
            r = np.sqrt((Xs - x0) ** 2 + (Ys - y0) ** 2)
            # Avoid division by zero inside sinh.
            w0_phys = max(float(w0_phys), eps)
            R0_phys = max(float(R0_phys), eps)
            ratio = R0_phys / w0_phys
            # theta(r): match from_experiment m_initial
            theta = np.empty_like(r)
            nonzero = r > 1e-12
            theta[nonzero] = chi * 2 * np.arctan(
                np.sinh(ratio) / np.sinh(r[nonzero] / w0_phys)
            )
            theta[~nonzero] = np.pi

            phi = np.arctan2(Ys - y0, Xs - x0)
            mx_hat = np.sin(theta) * np.cos(phi + eta)
            my_hat = np.sin(theta) * np.sin(phi + eta)
            mz_hat = np.cos(theta)

            return (
                Ms_model * mx_hat,
                Ms_model * my_hat,
                Ms_model * mz_hat,
            )

        def residuals(params: np.ndarray) -> np.ndarray:
            mx_m, my_m, mz_m = model_m(params)
            # Vector least-squares error summed over spins.
            res = np.stack([mx_m - mx_s, my_m - my_s, mz_m - mz_s], axis=-1)
            return res.reshape(-1)

        res = least_squares(
            residuals,
            x0=np.array(p0, dtype=float),
            bounds=(np.array(lower, dtype=float), np.array(upper, dtype=float)),
            max_nfev=200,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )

        x0_fit, y0_fit, R0_fit, w0_fit, eta_fit = res.x

        # Convert center + radii back to Experiment reduced units (divide by l_w).
        A_ex = exp.A
        K_an = exp.K
        l_w = np.sqrt(A_ex / K_an) if K_an != 0 else 1.0

        exp.skyrmion_to = (float(x0_fit / l_w), float(y0_fit / l_w))
        exp.R0 = float(R0_fit / l_w)
        exp.w0 = float(w0_fit / l_w)
        exp.eta = float(eta_fit)
        return exp