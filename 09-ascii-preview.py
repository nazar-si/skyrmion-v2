"""Render a saved or growing skyrmion HDF5 trajectory in an ANSI terminal."""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Literal

import h5py
import numpy as np

from scripts.mm.compression import FORMAT, unpack_spherical_12bit
from scripts.mm.experiment import Experiment


RGB = tuple[int, int, int]
ColorMode = Literal["truecolor", "256", "none"]

BLUE: RGB = (75, 145, 255)
ORANGE: RGB = (255, 145, 45)
PURPLE: RGB = (205, 95, 255)
CORE_GRAY: RGB = (135, 138, 148)
WALL_GRAY: RGB = (105, 108, 120)
VORTEX_CYAN: RGB = (60, 225, 225)
BOUNDARY_GLYPHS = ":-=+*#%@"


@dataclass(frozen=True)
class FileState:
    frame_count: int
    status: str
    domain_wall_length: float


@dataclass(frozen=True)
class PreviewFrame:
    magnetization: np.ndarray
    vortex_position: np.ndarray
    step: int
    simulation_time: float
    index: int
    frame_count: int
    status: str
    domain_wall_length: float


@dataclass(frozen=True)
class Geometry:
    half_width_lw: float
    half_height_lw: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Draw a compressed skyrmion trajectory in a terminal. A growing HDF5 "
            "file is followed live; a finished file is replayed once."
        )
    )
    parser.add_argument("file", type=Path, help="Path to a data.h5 trajectory.")
    parser.add_argument(
        "--mode",
        choices=("auto", "follow", "replay"),
        default="auto",
        help="auto follows running files and replays completed files.",
    )
    parser.add_argument("--fps", type=float, default=8.0, help="Maximum replay frame rate.")
    parser.add_argument(
        "--poll",
        type=float,
        default=0.25,
        help="Seconds between checks for a newly flushed live frame.",
    )
    parser.add_argument("--loop", action="store_true", help="Loop replay of a finished file.")
    parser.add_argument(
        "--from-start",
        action="store_true",
        help="In follow mode, show existing frames instead of jumping to the latest.",
    )
    parser.add_argument("--width", type=int, help="Maximum map width in terminal characters.")
    parser.add_argument("--height", type=int, help="Maximum map height in terminal rows.")
    parser.add_argument(
        "--half-width",
        type=float,
        help="Fallback domain half-width in lw if experiment.yaml is absent.",
    )
    parser.add_argument(
        "--half-height",
        type=float,
        help="Fallback domain half-height in lw if experiment.yaml is absent.",
    )
    parser.add_argument(
        "--color",
        choices=("auto", "truecolor", "256", "none"),
        default="auto",
        help="ANSI color mode; auto detects truecolor or 256-color support.",
    )
    parser.add_argument(
        "--no-alt-screen",
        action="store_true",
        help="Draw in the normal terminal buffer instead of a temporary screen.",
    )
    return parser


def _open_live_file(path: Path):
    """Open a flushed snapshot while another process still owns the writer."""
    try:
        return h5py.File(path, "r", locking=False)
    except TypeError:  # Compatibility with older h5py builds.
        return h5py.File(path, "r")


def read_file_state(path: Path) -> FileState:
    with _open_live_file(path) as file:
        file_format = file.attrs.get("format", "")
        if file_format != FORMAT:
            raise ValueError(f"Unsupported trajectory format {file_format!r} in {path}.")
        return FileState(
            frame_count=int(file["m"].shape[0]),
            status=str(file.attrs.get("status", "unknown")),
            domain_wall_length=float(file.attrs["domain_wall_length_m"]),
        )


def read_frame(path: Path, index: int) -> PreviewFrame:
    with _open_live_file(path) as file:
        count = int(file["m"].shape[0])
        if not 0 <= index < count:
            raise IndexError(f"Frame {index} is outside the current 0..{count - 1} range.")
        return PreviewFrame(
            magnetization=unpack_spherical_12bit(file["m"][index]),
            vortex_position=file["vortex_position"][index].astype(np.float64),
            step=int(file["step"][index]),
            simulation_time=float(file["time"][index]),
            index=index,
            frame_count=count,
            status=str(file.attrs.get("status", "unknown")),
            domain_wall_length=float(file.attrs["domain_wall_length_m"]),
        )


def load_geometry(path: Path, args: argparse.Namespace) -> Geometry:
    experiment_file = path.parent / Experiment.FILE_NAME
    if experiment_file.is_file():
        experiment = Experiment.load(path.parent)
        return Geometry(experiment.width, experiment.height)
    if args.half_width is None:
        raise FileNotFoundError(
            f"{experiment_file} is needed to place the vortex. "
            "Alternatively pass --half-width and --half-height."
        )
    return Geometry(args.half_width, args.half_height or args.half_width)


def resolve_color_mode(requested: str) -> ColorMode:
    if requested != "auto":
        return requested  # type: ignore[return-value]
    if not sys.stdout.isatty() or os.environ.get("TERM", "") == "dumb":
        return "none"
    if os.environ.get("COLORTERM", "").lower() in {"truecolor", "24bit"}:
        return "truecolor"
    return "256"


def _ansi_color(color: RGB, mode: ColorMode) -> str:
    if mode == "none":
        return ""
    red, green, blue = color
    if mode == "truecolor":
        return f"\x1b[38;2;{red};{green};{blue}m"
    r = round(red / 255 * 5)
    g = round(green / 255 * 5)
    b = round(blue / 255 * 5)
    return f"\x1b[38;5;{16 + 36 * r + 6 * g + b}m"


def _quantize_color(color: np.ndarray) -> RGB:
    # Quantization avoids emitting a distinct escape sequence for every cell.
    values = np.clip(np.rint(color / 12) * 12, 0, 255).astype(int)
    return int(values[0]), int(values[1]), int(values[2])


def _wall_orientation(magnetization: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return cosine/sine of the in-plane angle relative to grad(m_z)."""
    mx = magnetization[..., 0]
    my = magnetization[..., 1]
    mz = magnetization[..., 2]
    grad_y, grad_x = np.gradient(mz)
    denominator = np.hypot(grad_x, grad_y) * np.hypot(mx, my)
    denominator = np.clip(denominator, 1e-12, None)
    cosine = np.clip((grad_x * mx + grad_y * my) / denominator, -1, 1)
    sine = np.clip((grad_x * my - grad_y * mx) / denominator, -1, 1)
    return cosine, sine


def _boundary_color(cosine: float, sine: float, strength: float) -> RGB:
    positive = max(cosine, 0.0)
    negative = max(-cosine, 0.0)
    bloch = abs(sine)
    total = positive + negative + bloch
    if total < 1e-9:
        base = np.asarray(PURPLE, dtype=float)
    else:
        base = (
            positive * np.asarray(BLUE)
            + negative * np.asarray(ORANGE)
            + bloch * np.asarray(PURPLE)
        ) / total
    # ANSI has no alpha channel. Blend toward gray and reduce glyph density to
    # represent the translucent ends of the wall gradient.
    color = (1 - strength) * np.asarray(WALL_GRAY) + strength * base
    color *= 0.72 + 0.28 * strength
    return _quantize_color(color)


def _output_shape(
    source_shape: tuple[int, int],
    geometry: Geometry,
    args: argparse.Namespace,
) -> tuple[int, int]:
    terminal = shutil.get_terminal_size(fallback=(100, 36))
    max_width = max(10, args.width or terminal.columns)
    max_height = max(5, args.height or (terminal.lines - 3))
    ny, nx = source_shape
    width = min(nx, max_width)
    # A terminal character is roughly twice as tall as it is wide.
    physical_aspect = geometry.half_height_lw / geometry.half_width_lw
    height = round(width * physical_aspect * 0.5)
    if height > min(ny, max_height):
        height = min(ny, max_height)
        width = round(height / (physical_aspect * 0.5))
    return max(1, width), max(1, height)


def _wrapped_coordinate(value: float, half_extent: float) -> float:
    return (value + half_extent) % (2 * half_extent) - half_extent


def render_map(
    frame: PreviewFrame,
    geometry: Geometry,
    width: int,
    height: int,
    color_mode: ColorMode,
) -> str:
    m = frame.magnetization
    ny, nx = m.shape[:2]
    x_indices = np.rint(np.linspace(0, nx - 1, width)).astype(int)
    # Highest physical y belongs at the top of the terminal.
    y_indices = np.rint(np.linspace(ny - 1, 0, height)).astype(int)
    cosine, sine = _wall_orientation(m)

    vortex_lw = frame.vortex_position / frame.domain_wall_length
    vortex_x = _wrapped_coordinate(float(vortex_lw[0]), geometry.half_width_lw)
    vortex_y = _wrapped_coordinate(float(vortex_lw[1]), geometry.half_height_lw)
    vortex_x_index = (vortex_x + geometry.half_width_lw) / (2 * geometry.half_width_lw) * (nx - 1)
    vortex_y_index = (vortex_y + geometry.half_height_lw) / (2 * geometry.half_height_lw) * (ny - 1)
    vortex_column = int(np.argmin(abs(x_indices - vortex_x_index)))
    vortex_row = int(np.argmin(abs(y_indices - vortex_y_index)))

    lines: list[str] = []
    for row, source_y in enumerate(y_indices):
        pieces: list[str] = []
        active_color: RGB | None = None
        for column, source_x in enumerate(x_indices):
            if row == vortex_row and column == vortex_column:
                character, color = "V", VORTEX_CYAN
            else:
                mz = float(m[source_y, source_x, 2])
                if mz >= 0.65:
                    character, color = " ", None
                elif mz <= -0.65:
                    character, color = ".", CORE_GRAY
                else:
                    strength = float(np.clip(1 - abs(mz), 0, 1))
                    glyph_index = min(
                        len(BOUNDARY_GLYPHS) - 1,
                        int(round(strength * (len(BOUNDARY_GLYPHS) - 1))),
                    )
                    character = BOUNDARY_GLYPHS[glyph_index]
                    color = _boundary_color(
                        float(cosine[source_y, source_x]),
                        float(sine[source_y, source_x]),
                        strength,
                    )

            if color is None:
                if active_color is not None and color_mode != "none":
                    pieces.append("\x1b[0m")
                active_color = None
            elif color != active_color:
                pieces.append(_ansi_color(color, color_mode))
                active_color = color
            pieces.append(character)
        if active_color is not None and color_mode != "none":
            pieces.append("\x1b[0m")
        lines.append("".join(pieces))
    return "\n".join(lines)


def _colored_label(text: str, color: RGB, mode: ColorMode) -> str:
    if mode == "none":
        return text
    return f"{_ansi_color(color, mode)}{text}\x1b[0m"


def render_frame(
    frame: PreviewFrame,
    geometry: Geometry,
    args: argparse.Namespace,
    color_mode: ColorMode,
) -> str:
    width, height = _output_shape(frame.magnetization.shape[:2], geometry, args)
    vortex_lw = frame.vortex_position / frame.domain_wall_length
    header = (
        f"frame {frame.index + 1}/{frame.frame_count}  step {frame.step}  "
        f"t={frame.simulation_time * 1e9:.4f} ns  "
        f"vortex=({vortex_lw[0]:+.3f}, {vortex_lw[1]:+.3f}) lw  {frame.status}"
    )
    legend = "  ".join(
        (
            _colored_label("V vortex", VORTEX_CYAN, color_mode),
            _colored_label("blue +Neel", BLUE, color_mode),
            _colored_label("orange -Neel", ORANGE, color_mode),
            _colored_label("purple Bloch", PURPLE, color_mode),
            _colored_label(". core", CORE_GRAY, color_mode),
        )
    )
    return f"{header}\n{legend}\n{render_map(frame, geometry, width, height, color_mode)}"


class TerminalSession(AbstractContextManager["TerminalSession"]):
    def __init__(self, use_alt_screen: bool) -> None:
        self.interactive = sys.stdout.isatty()
        self.use_alt_screen = use_alt_screen and self.interactive

    def __enter__(self) -> TerminalSession:
        if self.use_alt_screen:
            sys.stdout.write("\x1b[?1049h")
        if self.interactive:
            sys.stdout.write("\x1b[?25l\x1b[2J\x1b[H")
            sys.stdout.flush()
        return self

    def draw(self, content: str) -> None:
        if self.interactive:
            sys.stdout.write(f"\x1b[H{content}\x1b[0m\x1b[J")
        else:
            sys.stdout.write(f"{content}\n\n")
        sys.stdout.flush()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.interactive:
            sys.stdout.write("\x1b[0m\x1b[?25h")
        if self.use_alt_screen:
            sys.stdout.write("\x1b[?1049l")
        sys.stdout.flush()


def _wait_for_file(path: Path, poll: float) -> FileState:
    while True:
        try:
            return read_file_state(path)
        except (FileNotFoundError, BlockingIOError, OSError, KeyError):
            time.sleep(poll)


def follow(
    path: Path,
    geometry: Geometry,
    args: argparse.Namespace,
    terminal: TerminalSession,
    color_mode: ColorMode,
) -> None:
    last_index = -1
    first = True
    while True:
        try:
            state = read_file_state(path)
            if state.frame_count == 0:
                time.sleep(args.poll)
                continue
            if first:
                next_index = 0 if args.from_start else state.frame_count - 1
                first = False
            elif args.from_start and last_index + 1 < state.frame_count:
                next_index = last_index + 1
            else:
                next_index = state.frame_count - 1

            if next_index != last_index:
                frame = read_frame(path, next_index)
                terminal.draw(render_frame(frame, geometry, args, color_mode))
                last_index = next_index

            if state.status != "running" and last_index >= state.frame_count - 1:
                return
        except (BlockingIOError, OSError, KeyError, IndexError):
            # Reopening during the writer's short resize/write window can race;
            # the previous flushed frame remains on screen until the next poll.
            pass
        time.sleep(args.poll)


def replay(
    path: Path,
    geometry: Geometry,
    args: argparse.Namespace,
    terminal: TerminalSession,
    color_mode: ColorMode,
) -> None:
    delay = 1 / args.fps
    while True:
        state = read_file_state(path)
        for index in range(state.frame_count):
            started = time.monotonic()
            frame = read_frame(path, index)
            terminal.draw(render_frame(frame, geometry, args, color_mode))
            time.sleep(max(0, delay - (time.monotonic() - started)))
        if not args.loop:
            return


def main() -> None:
    args = build_parser().parse_args()
    if args.fps <= 0 or args.poll <= 0:
        raise ValueError("--fps and --poll must be positive.")
    path = args.file.expanduser().resolve()
    initial_state = _wait_for_file(path, args.poll)
    geometry = load_geometry(path, args)
    color_mode = resolve_color_mode(args.color)
    mode = args.mode
    if mode == "auto":
        mode = "follow" if initial_state.status == "running" else "replay"

    try:
        with TerminalSession(use_alt_screen=not args.no_alt_screen) as terminal:
            if mode == "follow":
                follow(path, geometry, args, terminal, color_mode)
            else:
                replay(path, geometry, args, terminal, color_mode)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
