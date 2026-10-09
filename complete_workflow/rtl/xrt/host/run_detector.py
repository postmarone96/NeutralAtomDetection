#!/usr/bin/env python3
"""Run one 28x28 raw frame through the neutral-atom RTL kernel with pyxrt."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import os
import subprocess
import sys
import tempfile

from xrt_profile import summarize_native_trace

import numpy as np


FRAME_PIXELS = 28 * 28
FRAME_BYTES = FRAME_PIXELS * 2
RESULT_BYTES = 64
SITE_COUNT = 9
COUNTER_FORMAT = 0x43594331


@dataclass(frozen=True)
class CycleCounts:
    preprocess: int
    cnn: int
    pipeline: int


def unpack_cycles(words: np.ndarray) -> CycleCounts:
    if words.size < 13 or int(words[12]) != COUNTER_FORMAT:
        raise ValueError(
            "cycle counters are missing; rebuild and load the updated xclbin"
        )
    base = tuple(int(value) for value in words[9:12])
    return CycleCounts(*base)


def latency_us(cycles: int, frequency_mhz: float) -> float:
    if not math.isfinite(frequency_mhz) or frequency_mhz <= 0:
        raise ValueError("clock frequency must be positive and finite")
    return cycles / frequency_mhz


def _load_text_values(path: Path) -> np.ndarray:
    values = [int(token, 16) for token in path.read_text().split()]
    return np.asarray(values, dtype=np.uint16)


def load_frame(path: Path, frame_index: int = 0) -> np.ndarray:
    """Load one frame from uint16 binary or hexadecimal text vectors."""
    if frame_index < 0:
        raise ValueError("frame index must be non-negative")

    if path.suffix.lower() == ".txt":
        values = _load_text_values(path)
    else:
        values = np.fromfile(path, dtype="<u2")

    if values.size == 0 or values.size % FRAME_PIXELS != 0:
        raise ValueError(
            f"{path} must contain a whole number of 784-pixel frames"
        )

    frame_count = values.size // FRAME_PIXELS
    if frame_index >= frame_count:
        raise ValueError(
            f"frame index {frame_index} is outside the file's {frame_count} frames"
        )

    frame = values[
        frame_index * FRAME_PIXELS : (frame_index + 1) * FRAME_PIXELS
    ].astype("<u2", copy=True)
    if np.any(frame > 1023):
        raise ValueError("raw pixels must fit in 10 bits (0..1023)")
    return frame


def signed_10(value: int) -> int:
    value &= 0x3FF
    return value - 0x400 if value & 0x200 else value


def unpack_results(words: np.ndarray) -> list[tuple[int, int, int]]:
    if words.size < SITE_COUNT:
        raise ValueError("result buffer is too short")
    results = []
    for record_value in words[:SITE_COUNT]:
        record = int(record_value)
        results.append(
            (
                signed_10(record),
                signed_10(record >> 10),
                (record >> 20) & 1,
            )
        )
    return results


def run_kernel(
    xclbin_path: Path, frame: np.ndarray, device_index: int
) -> tuple[list[tuple[int, int, int]], CycleCounts]:
    """Run through native XRT; timestamps are recorded by its profiling plugin."""
    try:
        import pyxrt
    except ImportError as error:
        raise RuntimeError(
            "pyxrt is unavailable; source the XRT setup script on the Alveo host"
        ) from error

    device = pyxrt.device(device_index)
    xclbin = pyxrt.xclbin(str(xclbin_path))
    context = pyxrt.hw_context(device, xclbin)
    kernel = pyxrt.kernel(context, "neutral_atom_kernel")

    frame_bo = pyxrt.bo(
        device, FRAME_BYTES, pyxrt.bo.normal, kernel.group_id(0)
    )
    result_bo = pyxrt.bo(
        device, RESULT_BYTES, pyxrt.bo.normal, kernel.group_id(1)
    )

    frame_map = np.frombuffer(frame_bo.map(), dtype="<u2", count=FRAME_PIXELS)
    result_map = np.frombuffer(result_bo.map(), dtype="<u4", count=RESULT_BYTES // 4)
    frame_map[:] = frame
    result_map[:] = 0

    result_bo.sync(
        pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, RESULT_BYTES, 0
    )
    frame_bo.sync(
        pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, FRAME_BYTES, 0
    )
    run = kernel(frame_bo, result_bo)
    run.wait()
    result_bo.sync(
        pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, RESULT_BYTES, 0
    )

    return unpack_results(result_map), unpack_cycles(result_map)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one raw frame through neutral_atom_kernel using pyxrt."
    )
    parser.add_argument("xclbin", type=Path)
    parser.add_argument(
        "frame",
        type=Path,
        help="uint16 little-endian .bin or hexadecimal .txt frame file",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--freq-mhz", type=float, required=True,
        help="actual kernel clock in MHz, used to convert hardware cycles to us",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="frame to use when the input file contains multiple frames",
    )
    parser.add_argument(
        "--profile-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "build" / "profiles",
        help="parent directory for the XRT trace files and timing summary",
    )
    parser.add_argument("--xrt-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not math.isfinite(args.freq_mhz) or args.freq_mhz <= 0:
        parser.error("--freq-mhz must be positive and finite")
    return args


def run_profiled(args: argparse.Namespace) -> None:
    # XRT flushes native_trace.csv when the worker process exits.
    args.profile_dir.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="single_", dir=args.profile_dir)).resolve()
    environment = os.environ.copy()
    environment["XRT_INI_PATH"] = str(Path(__file__).with_name("xrt.ini").resolve())
    command = [
        sys.executable, str(Path(__file__).resolve()),
        str(args.xclbin.resolve()), str(args.frame.resolve()),
        "--device", str(args.device), "--freq-mhz", str(args.freq_mhz),
        "--frame-index", str(args.frame_index), "--xrt-worker",
    ]
    subprocess.run(command, cwd=output, env=environment, check=True)
    try:
        timing = summarize_native_trace(output / "native_trace.csv")
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"XRT profiling did not provide the expected transfer events: {error}. "
            f"Check the native XRT profiling plugin and files in {output}."
        ) from error
    print("XRT end-to-end (input transfer to HBM through result readback): "
          f"{timing.end_to_end_us:.3f} us")
    print(f"XRT profiling reports: {output}")


def main() -> None:
    args = parse_args()
    if not args.xrt_worker:
        run_profiled(args)
        return
    frame = load_frame(args.frame, args.frame_index)
    results, cycles = run_kernel(args.xclbin, frame, args.device)
    for site, (logit_0, logit_1, decision) in enumerate(results):
        print(
            f"site {site}: logits=({logit_0}, {logit_1}) "
            f"decision={decision}"
        )
    for label, count in (
        ("Preprocessing (first pixel to all site data ready)", cycles.preprocess),
        ("CNN (one site, last site measured)", cycles.cnn),
        ("Pipeline (first pixel to last prediction)", cycles.pipeline),
    ):
        print(f"{label}: {count} cycles = {latency_us(count, args.freq_mhz):.3f} us "
              f"at {args.freq_mhz:g} MHz")


if __name__ == "__main__":
    main()
