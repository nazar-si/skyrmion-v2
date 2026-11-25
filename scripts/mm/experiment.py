from dataclasses import asdict, dataclass
import re
from pathlib import Path
from typing import Type, TypeVar
from enum import Enum
import yaml
import unicodedata

class Driver(Enum):
    TIME = "time"
    ENERGY = "energy"


class Backend(Enum):
    OOMMF = "oommf"
    MAGNUM = "magnum"


class BC(Enum):
    PERIODIC = "xy"
    NEUMANN = ""

T = TypeVar("T", bound="Experiment")

@dataclass
class Experiment:
    """Serializable parameters and provenance for a micromagnetic run.

    Spatial inputs are stored in reduced units of ``lw = sqrt(A / K)``. Material
    constants, time, and vortex material parameters are stored in SI units. This
    preserves the older OOMMF files while making every conversion used by the
    Magnum runner explicit and reproducible.
    """

    version: int        = 2
    
    FILE_NAME           = "experiment.yaml"
    name: str           = ""
    description: str    = ""
    backend: Backend    = Backend.OOMMF
    driver: Driver      = Driver.TIME  # default just to avoid init issues
    
    width: float        = 0.025
    height: float       = 0.0
    dx: float           = 0.0
    
    Ms: float           = 1
    A: float            = 0.0
    K: float            = 0.0
    eps: float          = 0.0
    gamma: float        = 0
    beta: float         = 0.0
    h: float            = 1

    chi: float          = 1
    eta: float          = 0
    R0: float           = 0.0
    w0: float           = 0.0
    bc: BC              = BC.PERIODIC

    damping: float      = 5e-2

    skyrmion_to: tuple[float, float] = (0, 0)
    field_to: tuple[float, float] = (0, 0)
    vortex_to: tuple[float, float] = (0, 0)

    # Magnum time integration and vortex parameters (SI unless noted).
    dt: float           = 1e-11
    steps: int          = 10_000
    stride: int         = 10
    device: str         = "auto"
    phi0: float         = 2.067833848e-15
    vortex_sigma: float = 2e-15
    superconductor_thickness: float = 10e-9
    vortex_polarity: int = -1

    # provenance (optional)
    source_dir: str     = ""
    source_file: str    = ""
    source_step: int | None = None
    source_time: float | None = None

    # Filled by a runner. These fields make interrupted runs self-describing.
    status: str         = "created"
    completed_step: int | None = None
    completed_time: float | None = None

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        data = asdict(self)

        # Enum → value
        for key in ("backend", "driver", "bc"):
            value = getattr(self, key)
            if isinstance(value, Enum):
                data[key] = value.value

        # tuples → lists (YAML-safe)
        for key in ["skyrmion_to", "field_to", "vortex_to"]:
            data[key] = list(data[key])
        data["derived_si"] = self.derived_si

        file_path = directory / self.FILE_NAME

        with open(file_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

        return file_path
    
    @classmethod
    def load(cls: Type[T], directory: str | Path) -> T:
        directory = Path(directory)
        file_path = directory / cls.FILE_NAME

        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid experiment data in {file_path}")
        # Derived values are written for auditability and recomputed from inputs.
        data.pop("derived_si", None)

        # --- version hook ---
        version = data.get("version", 1)
        if version != 2:
            data = cls._migrate(data, version)

        # restore enums
        if "backend" in data:
            data["backend"] = Backend(data["backend"])
        if "driver" in data:
            data["driver"] = Driver(data["driver"])

        if "bc" in data:
            data["bc"] = BC(data["bc"])

        # lists → tuples
        for key in ["skyrmion_to", "field_to", "vortex_to"]:
            if key in data:
                data[key] = tuple(data[key])

        return cls(**data)
    
    @staticmethod
    def _migrate(data: dict, version: int) -> dict:
        """Migrate older OOMMF metadata without changing its physical values."""
        if version != 1:
            raise ValueError(f"Unsupported experiment version: {version}")
        data.setdefault("backend", Backend.OOMMF.value)
        data["version"] = 2
        return data

    @property
    def domain_wall_length(self) -> float:
        """Domain-wall length ``sqrt(A / K)`` in metres."""
        if self.A <= 0 or self.K <= 0:
            raise ValueError("A and K must be positive to define the domain-wall length.")
        return (self.A / self.K) ** 0.5

    @property
    def dmi_scale(self) -> float:
        """Natural interfacial DMI scale ``sqrt(A K)`` in J/m^2."""
        if self.A <= 0 or self.K <= 0:
            raise ValueError("A and K must be positive to define the DMI scale.")
        return (self.A * self.K) ** 0.5

    @property
    def pearl_length(self) -> float:
        """Pearl length corresponding to the reduced interaction ``gamma``."""
        if self.gamma <= 0:
            raise ValueError("gamma must be positive to define the Pearl length.")
        return self.Ms * self.phi0 / (8 * 3.141592653589793 * self.gamma * self.dmi_scale)

    @property
    def derived_si(self) -> dict[str, float | list[float] | None]:
        """Physical values derived from the reduced inputs and stored in YAML."""
        if self.A <= 0 or self.K <= 0:
            return {}
        lw = self.domain_wall_length
        pearl_length = self.pearl_length if self.gamma > 0 else None
        return {
            "domain_wall_length_m": lw,
            "dmi_scale_j_per_m2": self.dmi_scale,
            "pearl_length_m": pearl_length,
            "domain_half_width_m": self.width * lw,
            "domain_half_height_m": self.height * lw,
            "cell_size_m": self.dx * lw,
            "skyrmion_position_m": [value * lw for value in self.skyrmion_to],
            "vortex_position_m": [value * lw for value in self.vortex_to],
            "electric_tip_position_m": [value * lw for value in self.field_to],
            "electric_tip_height_m": self.h * lw,
            "skyrmion_radius_m": self.R0 * lw,
            "skyrmion_delta_m": self.w0 * lw,
            "uniform_interface_dmi_j_per_m2": 2 * self.dmi_scale * self.eps,
            "nonuniform_dmi_prefactor_j_m": (
                2 * self.dmi_scale * self.beta * self.h * lw**3
            ),
        }

    @property
    def slug(self) -> str:
        name = unicodedata.normalize("NFKD", self.name)
        name = name.encode("ascii", "ignore").decode("ascii")

        slug = name.lower()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"\s+", "-", slug)
        slug = re.sub(r"-{2,}", "-", slug)
        return slug.strip("-")
