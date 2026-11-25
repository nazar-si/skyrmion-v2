import os
import re
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Any, Callable

from scripts.mm.experiment import Experiment

if TYPE_CHECKING:
    from scripts.mm.magnum import MagnumSimulation


LOCK_FILE_NAME = ".experiment_lock"
_MAGNETIZATION_OMF_STEP_RE = re.compile(r"-Magnetization-[^-]*-(\d+)\.omf$")


def _list_drive_dirs(base: Path) -> list[Path]:
    if not base.exists():
        return []
    return sorted(path for path in base.rglob("drive-*") if path.is_dir())


def _pick_recent_drive_dir(
    drive_parents: list[Path], baseline: set[Path], started_at: float
) -> Path | None:
    current: list[Path] = []
    for root in drive_parents:
        current.extend(_list_drive_dirs(root))

    # First priority: newly created during this run.
    new_dirs = [p for p in current if p not in baseline]
    if new_dirs:
        return max(new_dirs, key=lambda p: p.stat().st_mtime)

    # Fallback: recently modified since run started.
    recent_dirs = [p for p in current if p.stat().st_mtime >= started_at - 0.5]
    if recent_dirs:
        return max(recent_dirs, key=lambda p: p.stat().st_mtime)

    return None


def _try_claim(folder: Path) -> bool:
    lock_file = folder / LOCK_FILE_NAME
    try:
        fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False

    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("locked\n")
    return True


def _watch_and_write(
    drive_parents: list[Path],
    baseline: set[Path],
    started_at: float,
    experiment: Experiment,
    stop_event: Event,
    shared: dict[str, Any],
    shared_lock: Lock,
) -> None:
    while not stop_event.is_set():
        selected = _pick_recent_drive_dir(drive_parents, baseline, started_at)
        if selected is not None and selected.exists() and _try_claim(selected):
            # Lock file was created by this runner, now write yaml.
            experiment.save(selected)
            with shared_lock:
                shared["drive_dir"] = selected
            return
        time.sleep(0.2)


def _max_omf_step(folder: Path) -> int | None:
    try:
        files = list(folder.iterdir())
    except FileNotFoundError:
        return None

    best: int | None = None
    for p in files:
        if not p.is_file() or p.suffix != ".omf":
            continue
        m = _MAGNETIZATION_OMF_STEP_RE.search(p.name)
        if not m:
            continue
        try:
            step = int(m.group(1))
        except ValueError:
            continue
        if best is None or step > best:
            best = step
    return best


def _monitor_and_interrupt(
    stop_event: Event,
    shared: dict[str, Any],
    shared_lock: Lock,
    stop_at_omf_step: int,
    poll_seconds: float = 0.5,
) -> None:
    while not stop_event.is_set():
        with shared_lock:
            drive_dir: Path | None = shared.get("drive_dir")
        if drive_dir is None:
            time.sleep(0.1)
            continue

        step = _max_omf_step(drive_dir)
        if step is not None and step >= stop_at_omf_step:
            with shared_lock:
                shared["stopped_by_limit"] = True
            os.kill(os.getpid(), signal.SIGINT)
            return

        time.sleep(poll_seconds)


def _rename_drive_dir(drive_dir: Path, experiment: Experiment) -> Path:
    parent = drive_dir.parent
    parts = drive_dir.name.split("-")
    number = parts[1] if len(parts) >= 2 else "unknown"
    slug = experiment.slug or "experiment"

    base_name = f"drive-{number}-{slug}"
    target = parent / base_name
    i = 1
    while target.exists():
        target = parent / f"{base_name}-{i}"
        i += 1

    drive_dir.rename(target)
    return target

def _resolve_watch_roots(drive_parent: str | Path) -> list[Path]:
    p = Path(drive_parent).expanduser()
    if p.is_absolute():
        return [p]

    roots = [(Path.cwd() / p).resolve()]
    script_root = Path(sys.argv[0]).resolve().parent
    alt = (script_root / p).resolve()
    if alt not in roots:
        roots.append(alt)
    return roots


def _stop_docker_container_for_mount(mount_source: Path) -> str | None:
    """
    Best-effort: stop a running Docker container that bind-mounts `mount_source`.
    Returns stopped container id if found.
    """
    import json
    import subprocess

    mount_source = mount_source.resolve()

    try:
        ps = subprocess.run(
            ["docker", "ps", "-q"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return None

    if ps.returncode != 0:
        return None

    ids = [line.strip() for line in ps.stdout.splitlines() if line.strip()]
    if not ids:
        return None

    best_id: str | None = None
    best_created: str | None = None

    for cid in ids:
        insp = subprocess.run(
            ["docker", "inspect", cid],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if insp.returncode != 0:
            continue
        try:
            data = json.loads(insp.stdout)[0]
        except Exception:
            continue

        mounts = data.get("Mounts", []) or []
        matches = False
        for m in mounts:
            if m.get("Type") != "bind":
                continue
            src = m.get("Source")
            if not src:
                continue
            try:
                if Path(src).resolve() == mount_source:
                    matches = True
                    break
            except Exception:
                continue

        if not matches:
            continue

        created = data.get("Created")  # ISO string
        if best_id is None or (created and best_created and created > best_created) or (created and not best_created):
            best_id = cid
            best_created = created

    if best_id is None:
        return None

    subprocess.run(["docker", "stop", "-t", "2", best_id], check=False)
    return best_id


def run_experiment(
    run_callable: Callable[[], Any],
    experiment: Experiment,
    drive_parent: str | Path,
    stop_at_omf_step: int | None = None,
    stop_docker: bool = True,
) -> Any:
    drive_parents = _resolve_watch_roots(drive_parent)
    started_at = time.time()
    baseline: set[Path] = set()
    for root in drive_parents:
        baseline.update(_list_drive_dirs(root))
    stop_event = Event()
    shared_lock = Lock()
    shared: dict[str, Any] = {"drive_dir": None, "stopped_by_limit": False}

    watcher = Thread(
        target=_watch_and_write,
        args=(drive_parents, baseline, started_at, experiment, stop_event, shared, shared_lock),
        daemon=True,
    )
    watcher.start()

    monitor: Thread | None = None
    if stop_at_omf_step is not None:
        monitor = Thread(
            target=_monitor_and_interrupt,
            args=(stop_event, shared, shared_lock, stop_at_omf_step),
            daemon=True,
        )
        monitor.start()

    result: Any = None
    stopped: str = "completed"
    try:
        result = run_callable()
    except KeyboardInterrupt:
        with shared_lock:
            stopped_by_limit = bool(shared.get("stopped_by_limit"))
        stopped = "limit" if stopped_by_limit else "interrupt"
    finally:
        stop_event.set()
        watcher.join(timeout=1.0)
        if monitor is not None:
            monitor.join(timeout=1.0)

    with shared_lock:
        drive_dir: Path | None = shared.get("drive_dir")

    stopped_container_id: str | None = None
    if stop_docker and drive_dir is not None and drive_dir.exists():
        stopped_container_id = _stop_docker_container_for_mount(drive_dir)

    final_dir: Path | None = None
    if drive_dir is not None and drive_dir.exists():
        final_dir = _rename_drive_dir(drive_dir, experiment)

    return {
        "result": result,
        "stopped": stopped,
        "drive_dir": str(drive_dir) if drive_dir is not None else None,
        "final_dir": str(final_dir) if final_dir is not None else None,
        "stopped_container_id": stopped_container_id,
    }


def create_magnum_run_directory(
    output_root: str | Path,
    experiment: Experiment,
    *,
    now: datetime | None = None,
) -> Path:
    """Create ``NN-YYMMDD-name`` without OOMMF's nested drive directories."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    numbers: list[int] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = re.match(r"^(\d+)-", path.name)
        if match:
            numbers.append(int(match.group(1)))

    number = max(numbers, default=0) + 1
    date = (now or datetime.now()).strftime("%y%m%d")
    slug = experiment.slug or "experiment"
    while True:
        target = root / f"{number:02d}-{date}-{slug}"
        try:
            target.mkdir()
        except FileExistsError:
            number += 1
            continue
        return target


class _LivePreview:
    """Small interactive Matplotlib window, imported only when requested."""

    def __init__(self, simulation: "MagnumSimulation") -> None:
        import matplotlib.pyplot as plt

        from scripts.read_field import field_image

        self._plt = plt
        self._field_image = field_image
        self._lw = simulation.experiment.domain_wall_length
        self._extent = [
            -simulation.experiment.width,
            simulation.experiment.width,
            -simulation.experiment.height,
            simulation.experiment.height,
        ]
        plt.ion()
        self._fig, self._ax = plt.subplots()
        magnetization = simulation.magnetization_for_storage()
        colors = self._colors(magnetization)
        self._image = self._ax.imshow(colors, extent=self._extent, origin="lower")
        position = simulation.vortex_position() / self._lw
        self._marker = self._ax.scatter([position[0]], [position[1]], s=12, color="black")
        self._title = self._ax.set_title("")
        self._ax.set_xlabel(r"$x/l_w$")
        self._ax.set_ylabel(r"$y/l_w$")
        plt.show(block=False)

    def _colors(self, magnetization):
        import numpy as np

        return np.clip(self._field_image(magnetization, color="spirality"), 0, 1)

    def update(self, simulation: "MagnumSimulation", step: int) -> None:
        import numpy as np

        self._image.set_data(self._colors(simulation.magnetization_for_storage()))
        position = simulation.vortex_position() / self._lw
        self._marker.set_offsets(np.asarray([position]))
        self._title.set_text(f"Step {step} · t = {simulation.time() * 1e9:.3f} ns")
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        self._plt.pause(0.001)

    def close(self) -> None:
        self._plt.ioff()


def run_magnum_experiment(
    simulation: "MagnumSimulation",
    output_root: str | Path,
    *,
    preview: bool = False,
) -> dict[str, Any]:
    """Run and incrementally save a coupled Magnum skyrmion-vortex model."""
    import logging

    import magnumnp as mp
    from tqdm import tqdm

    from scripts.mm.compression import FILE_NAME, H5TrajectoryWriter

    experiment = simulation.experiment
    # A per-step Magnum INFO message fights with tqdm in a terminal. Warnings and
    # errors remain visible, while tqdm supplies the progress indication.
    mp.logging.set_log_level(logging.WARNING)
    run_dir = create_magnum_run_directory(output_root, experiment)
    data_path = run_dir / FILE_NAME
    experiment.status = "running"
    experiment.completed_step = simulation.start_step
    experiment.completed_time = simulation.time()
    experiment.save(run_dir)

    initial_m = simulation.magnetization_for_storage()
    writer = H5TrajectoryWriter(
        data_path,
        initial_m.shape,
        domain_wall_length=experiment.domain_wall_length,
    )
    writer.append(
        initial_m,
        simulation.vortex_position(),
        step=simulation.start_step,
        time=simulation.time(),
    )

    live_preview = _LivePreview(simulation) if preview else None
    if live_preview is not None:
        live_preview.update(simulation, simulation.start_step)

    status = "completed"
    last_saved_step = simulation.start_step
    last_saved_time = simulation.time()
    final_step = simulation.start_step + experiment.steps
    try:
        for step in tqdm(
            range(simulation.start_step + 1, final_step + 1),
            total=experiment.steps,
            desc=experiment.slug or "experiment",
            unit="step",
        ):
            simulation.vortex.advance(simulation.state, experiment.dt / 2)
            simulation.llg.step(simulation.state, experiment.dt)
            simulation.vortex.advance(simulation.state, experiment.dt / 2)

            if step % experiment.stride == 0 or step == final_step:
                saved_time = simulation.time()
                writer.append(
                    simulation.magnetization_for_storage(),
                    simulation.vortex_position(),
                    step=step,
                    time=saved_time,
                )
                last_saved_step = step
                last_saved_time = saved_time
                if live_preview is not None:
                    live_preview.update(simulation, step)
    except KeyboardInterrupt:
        status = "interrupted"
    except BaseException:
        status = "failed"
        raise
    finally:
        writer.close(status)
        experiment.status = status
        experiment.completed_step = last_saved_step
        experiment.completed_time = last_saved_time
        experiment.device = str(simulation.device)
        experiment.save(run_dir)
        if live_preview is not None:
            live_preview.close()

    return {
        "status": status,
        "run_dir": str(run_dir),
        "data_file": str(data_path),
        "completed_step": last_saved_step,
        "completed_time": last_saved_time,
    }
