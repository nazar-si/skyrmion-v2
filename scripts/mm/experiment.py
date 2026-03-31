from dataclasses import dataclass, asdict
import re
from typing import Literal
from pathlib import Path
from typing import Type, TypeVar
from enum import Enum
import yaml
import unicodedata

class Driver(Enum):
    TIME = "time"
    ENERGY = "energy"

T = TypeVar("T", bound="Experiment")

@dataclass
class Experiment:
    # for migration purposes
    version: int        = 1
    
    FILE_NAME           = "experiment.yaml"
    name: str           = ""
    description: str    = ""
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

    damping: float      = 5e-2

    skyrmion_to: tuple[int, int] = (0, 0)
    field_to: tuple[int, int] = (0, 0)
    vortex_to: tuple[int, int] = (0, 0)

    # provenance (optional)
    source_dir: str     = ""
    source_file: str    = ""

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        data = asdict(self)

        # Enum → value
        if isinstance(self.driver, Enum):
            data["driver"] = self.driver.value

        # tuples → lists (YAML-safe)
        for key in ["skyrmion_to", "field_to", "vortex_to"]:
            data[key] = list(data[key])

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

        # --- version hook (future migrations) ---
        version = data.get("version", 1)
        if version != 1:
            data = cls._migrate(data, version)

        # restore enum
        if "driver" in data:
            data["driver"] = Driver(data["driver"])

        # lists → tuples
        for key in ["skyrmion_to", "field_to", "vortex_to"]:
            if key in data:
                data[key] = tuple(data[key])

        return cls(**data)
    
    @staticmethod
    def _migrate(data: dict, version: int) -> dict:
        """
        Handle backward compatibility here.
        """
        # Example:
        # if version == 0:
        #     data["new_field"] = default_value

        data["version"] = 1
        return data

    @property
    def slug(cls) -> str:
        name = unicodedata.normalize("NFKD", cls.name)
        name = name.encode("ascii", "ignore").decode("ascii")

        slug = name.lower()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"\s+", "-", slug)
        slug = re.sub(r"-{2,}", "-", slug)
        return slug.strip("-")
