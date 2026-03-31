from __future__ import annotations

from pathlib import Path
import re

OMF_FILE_PREFIX = "skyrmion_system-Oxs_TimeDriver-Magnetization-"
_MAGNETIZATION_STEP_RE = re.compile(r"-Magnetization-[^-]*-(\d+)\.omf$")


def list_magnetization_files(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []

    files = [p for p in folder.iterdir() if p.is_file() and p.suffix == ".omf" and p.name.startswith(OMF_FILE_PREFIX)]

    def key(p: Path) -> int:
        m = _MAGNETIZATION_STEP_RE.search(p.name)
        return int(m.group(1)) if m else -1

    return sorted(files, key=key)

def get_file(folder: str | Path, n: int | list[int]) -> Path:
    """
    Select a magnetization .omf file in `folder`.

    - If `n` is a list (e.g. [2000]), treat `n[0]` as a *chunk/step number*
      (the last numeric suffix in filenames like ...-0002000.omf).
    - If `n` is an int, treat it as an *index* into the sorted list of magnetization files.
      (supports negative indices: -1 is last)
    """
    folder = Path(folder)

    if isinstance(n, list):
        step = int(n[0])
        # Files look like: skyrmion_system-Oxs_TimeDriver-Magnetization-0000000-0002000.omf
        matches = sorted(folder.glob(f"{OMF_FILE_PREFIX}*-{step:07}.omf"))
        if not matches:
            raise FileNotFoundError(f"No magnetization file with step {step} in {folder}")
        return matches[0]

    files = list_magnetization_files(folder)
    if not files:
        raise FileNotFoundError(f"No magnetization .omf files found in {folder}")
    try:
        return files[n]
    except IndexError as exc:
        raise IndexError(f"Index {n} out of range for {len(files)} magnetization files in {folder}") from exc